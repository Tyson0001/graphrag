"""
evaluate_graphrag.py
─────────────────────────────────────────────────────────────────
End-to-end RAGAS evaluation for a GraphRAG API backed by a
skills/people CSV with columns:
  id, name, core_skills, secondary_skills, soft_skills,
  years_of_experience, potential_roles, skill_summary

STEPS:
  1. Load the CSV
  2. Auto-generate diverse Q&A pairs from the data (no GPT needed)
  3. Optionally add GPT-generated questions  (set USE_GPT_QUESTIONS=True)
  4. Query your running GraphRAG API for each question
  5. Run RAGAS metrics & save results

USAGE:
  pip install ragas langchain-openai datasets requests pandas python-dotenv
  python evaluate_graphrag.py
"""

import os
import json
import logging
import random
import time
import requests
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  CONFIG  — edit these to match your setup
# ═══════════════════════════════════════════════════════════════
CSV_PATH          = "profiles.csv"        # ← path to your CSV
GRAPHRAG_BASE_URL = "http://localhost:8000"        # ← your uvicorn server
CHAT_ENDPOINT     = f"{GRAPHRAG_BASE_URL}/api/chat"

# Set to True to also generate questions via GPT (costs tokens)
USE_GPT_QUESTIONS = False
GPT_QUESTIONS_PER_ROW = 2          # how many GPT Qs per person row

# How many rule-based questions to generate total
MAX_RULE_BASED_QUESTIONS = 30

# RAGAS judge model
JUDGE_MODEL = "gpt-4o-mini"

# Retry / timeout settings for API calls
REQUEST_TIMEOUT   = 60
RETRY_COUNT       = 2
RETRY_DELAY       = 3   # seconds between retries
# ═══════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────
# 1. LOAD & VALIDATE CSV
# ───────────────────────────────────────────────────────────────
REQUIRED_COLS = {
    "id", "name", "core_skills", "secondary_skills",
    "soft_skills", "years_of_experience", "potential_roles",
    "skill_summary",
}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    df = df.fillna("")
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


# ───────────────────────────────────────────────────────────────
# 2. RULE-BASED Q&A GENERATION  (no API cost)
# ───────────────────────────────────────────────────────────────
def _skills_list(row: pd.Series, col: str) -> list[str]:
    """Split a comma/semicolon-separated skill string into a clean list."""
    raw = str(row.get(col, ""))
    if not raw.strip():
        return []
    return [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]


def generate_rule_based_qa(df: pd.DataFrame) -> list[dict]:
    """
    Create deterministic Q&A pairs directly from the CSV rows.
    Returns a list of {"question": ..., "reference": ...} dicts.
    """
    qa_pairs = []

    for _, row in df.iterrows():
        name       = str(row["name"]).strip()
        core       = _skills_list(row, "core_skills")
        secondary  = _skills_list(row, "secondary_skills")
        soft       = _skills_list(row, "soft_skills")
        roles      = _skills_list(row, "potential_roles")
        yoe        = str(row["years_of_experience"]).strip()
        summary    = str(row["skill_summary"]).strip()

        if not name:
            continue

        # — Name → skills
        if core:
            qa_pairs.append({
                "question":  f"What are the core skills of {name}?",
                "reference": f"{name}'s core skills are: {', '.join(core)}.",
            })

        # — Name → roles
        if roles:
            qa_pairs.append({
                "question":  f"What potential roles is {name} suitable for?",
                "reference": f"{name} is suitable for the following roles: {', '.join(roles)}.",
            })

        # — Name → experience
        if yoe:
            qa_pairs.append({
                "question":  f"How many years of experience does {name} have?",
                "reference": f"{name} has {yoe} years of experience.",
            })

        # — Name → soft skills
        if soft:
            qa_pairs.append({
                "question":  f"What soft skills does {name} have?",
                "reference": f"{name}'s soft skills include: {', '.join(soft)}.",
            })

        # — Name → summary
        if summary:
            qa_pairs.append({
                "question":  f"Give me a brief overview of {name}'s profile.",
                "reference": summary,
            })

        # — Name → all skills combined
        all_skills = core + secondary
        if all_skills:
            qa_pairs.append({
                "question":  f"What technical skills does {name} possess?",
                "reference": f"{name} has the following technical skills: {', '.join(all_skills)}.",
            })

    # ── Cross-row questions (skill → who has it) ──────────────
    # Build an inverted index: skill → [names]
    skill_to_people: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        for skill in _skills_list(row, "core_skills"):
            skill_to_people.setdefault(skill, []).append(name)

    for skill, people in skill_to_people.items():
        if len(people) >= 1:
            qa_pairs.append({
                "question":  f"Who has {skill} as a core skill?",
                "reference": f"The following people have {skill} as a core skill: {', '.join(people)}.",
            })

    # ── Cross-row: role → who fits ────────────────────────────
    role_to_people: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        for role in _skills_list(row, "potential_roles"):
            role_to_people.setdefault(role, []).append(name)

    for role, people in role_to_people.items():
        qa_pairs.append({
            "question":  f"Who can take on the role of {role}?",
            "reference": f"People suitable for the {role} role are: {', '.join(people)}.",
        })

    # ── Experience bucket questions ───────────────────────────
    senior = []
    for _, row in df.iterrows():
        try:
            yoe_val = float(str(row["years_of_experience"]).split("-")[0])
            if yoe_val >= 5:
                senior.append(str(row["name"]).strip())
        except ValueError:
            pass

    if senior:
        qa_pairs.append({
            "question":  "Who has 5 or more years of experience?",
            "reference": f"People with 5+ years of experience: {', '.join(senior)}.",
        })

    # Shuffle and cap
    random.seed(42)
    random.shuffle(qa_pairs)
    qa_pairs = qa_pairs[:MAX_RULE_BASED_QUESTIONS]
    logger.info(f"Generated {len(qa_pairs)} rule-based Q&A pairs")
    return qa_pairs


# ───────────────────────────────────────────────────────────────
# 3. GPT-BASED Q&A GENERATION  (optional, richer questions)
# ───────────────────────────────────────────────────────────────
def generate_gpt_qa(df: pd.DataFrame, llm_client) -> list[dict]:
    """
    Use GPT to generate more nuanced / multi-hop questions from rows.
    Only runs when USE_GPT_QUESTIONS=True.
    """
    from langchain_core.messages import HumanMessage

    qa_pairs = []
    sample_rows = df.sample(min(10, len(df)), random_state=42)

    for _, row in sample_rows.iterrows():
        prompt = f"""
You are creating evaluation questions for a RAG system that stores employee skill profiles.
Given this employee profile, generate {GPT_QUESTIONS_PER_ROW} question-answer pairs.

Profile:
  Name: {row['name']}
  Core Skills: {row['core_skills']}
  Secondary Skills: {row['secondary_skills']}
  Soft Skills: {row['soft_skills']}
  Experience: {row['years_of_experience']} years
  Potential Roles: {row['potential_roles']}
  Summary: {row['skill_summary']}

Rules:
- Questions should require retrieving from the profile above
- Mix skill-based, role-based and comparative questions
- Answers must be factual and derived ONLY from the profile above
- Return ONLY a JSON array like:
  [{{"question": "...", "reference": "..."}}, ...]
- No markdown, no preamble
"""
        try:
            response = llm_client.invoke([HumanMessage(content=prompt)])
            text = response.content.strip().replace("```json", "").replace("```", "")
            pairs = json.loads(text)
            qa_pairs.extend(pairs)
            logger.info(f"GPT generated {len(pairs)} Qs for {row['name']}")
        except Exception as e:
            logger.warning(f"GPT Q generation failed for {row['name']}: {e}")

    logger.info(f"Generated {len(qa_pairs)} GPT Q&A pairs")
    return qa_pairs


# ───────────────────────────────────────────────────────────────
# 4. QUERY YOUR GRAPHRAG API
# ───────────────────────────────────────────────────────────────
def query_graphrag(question: str) -> dict:
    """
    Sends a question to the GraphRAG API and collects:
      - answer  (assembled from SSE stream tokens)
      - contexts (source chunks returned by the API)

    The API returns Server-Sent Events (SSE).
    Adjust the payload keys and chunk parsing to match your actual
    API models — check http://localhost:8000/docs for exact schema.
    """
    payload = {
        "message": question,          # ← check models.py for field name
                                      #   alternatives: "query", "question", "content"
        "session_id": "ragas-eval",   # keeps eval calls isolated
    }

    answer_parts = []
    contexts     = []

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            with requests.post(
                CHAT_ENDPOINT,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="ignore")

                    if decoded.startswith("data: "):
                        raw = decoded[6:].strip()
                        if raw in ("[DONE]", ""):
                            break
                        try:
                            chunk = json.loads(raw)

                            # ── Collect answer tokens ─────────────
                            # Common key names — keep whichever your API uses:
                            for key in ("token", "text", "content", "delta", "answer"):
                                if key in chunk and chunk[key]:
                                    answer_parts.append(str(chunk[key]))
                                    break

                            # ── Collect source contexts ────────────
                            # GraphRAG typically returns sources/chunks
                            for src_key in ("sources", "contexts", "chunks", "documents"):
                                if src_key in chunk:
                                    for src in chunk[src_key]:
                                        if isinstance(src, dict):
                                            ctx = (
                                                src.get("content")
                                                or src.get("text")
                                                or src.get("chunk")
                                                or json.dumps(src)
                                            )
                                        else:
                                            ctx = str(src)
                                        if ctx:
                                            contexts.append(ctx)
                                    break

                        except json.JSONDecodeError:
                            # Plain-text token (non-JSON stream)
                            answer_parts.append(raw)

            break  # success — exit retry loop

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"All retries failed for: {question[:60]}")
                return {"answer": "", "contexts": ["[API error]"]}

    answer = " ".join(answer_parts).strip()

    # Fallback: if API returned no contexts, use the answer itself
    # so RAGAS doesn't crash (it requires non-empty retrieved_contexts)
    if not contexts:
        contexts = [answer] if answer else ["[no context returned by API]"]

    return {"answer": answer, "contexts": contexts}


# ───────────────────────────────────────────────────────────────
# 5. BUILD RAGAS DATASET
# ───────────────────────────────────────────────────────────────
def build_ragas_dataset(qa_pairs: list[dict]) -> list[dict]:
    test_cases = []
    total = len(qa_pairs)

    for i, qa in enumerate(qa_pairs):
        question  = qa["question"].strip()
        reference = qa["reference"].strip()

        logger.info(f"[{i+1}/{total}] → {question[:70]}")
        result = query_graphrag(question)

        test_cases.append({
            "user_input":         question,
            "response":           result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference":          reference,
        })

        logger.info(
            f"         answer: {len(result['answer'])} chars | "
            f"contexts: {len(result['contexts'])}"
        )

    return test_cases


# ───────────────────────────────────────────────────────────────
# 6. RUN RAGAS
# ───────────────────────────────────────────────────────────────
def run_ragas_eval(test_cases: list[dict]) -> pd.DataFrame:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, api_key=api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(api_key=api_key)
    )

    dataset = Dataset.from_list(test_cases)

    result = evaluate(
        dataset,
        metrics=[
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    return result.to_pandas()


# ───────────────────────────────────────────────────────────────
# 7. MAIN
# ───────────────────────────────────────────────────────────────
def main():
    # Load data
    df = load_csv(CSV_PATH)

    # Generate questions
    qa_pairs = generate_rule_based_qa(df)

    if USE_GPT_QUESTIONS:
        api_key = os.getenv("OPENAI_API_KEY")
        gpt_llm = ChatOpenAI(model=JUDGE_MODEL, api_key=api_key)
        gpt_qa  = generate_gpt_qa(df, gpt_llm)
        qa_pairs.extend(gpt_qa)
        logger.info(f"Total Q&A pairs (rule + GPT): {len(qa_pairs)}")

    # Save generated questions for review
    with open("generated_qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)
    logger.info("Q&A pairs saved → generated_qa_pairs.json")

    # Query GraphRAG API
    logger.info(f"\nQuerying GraphRAG API at: {CHAT_ENDPOINT}")
    test_cases = build_ragas_dataset(qa_pairs)

    # Save raw test cases (useful for debugging)
    with open("test_cases_debug.json", "w") as f:
        json.dump(test_cases, f, indent=2)
    logger.info("Raw test cases saved → test_cases_debug.json")

    # Run RAGAS
    logger.info("\nRunning RAGAS evaluation (this may take a few minutes)...")
    scores_df = run_ragas_eval(test_cases)

    # Save per-question breakdown
    scores_df.to_csv("ragas_results_detail.csv", index=False)
    logger.info("Per-question results saved → ragas_results_detail.csv")

    # Aggregate scores
    agg = scores_df.mean(numeric_only=True).to_dict()
    with open("ragas_scores.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Pretty print
    print("\n" + "═"*52)
    print("          GRAPHRAG RAGAS EVALUATION RESULTS")
    print("═"*52)
    metrics_display = {
        "context_precision":  "Context Precision  (0-1, higher=better)",
        "context_recall":     "Context Recall     (0-1, higher=better)",
        "answer_relevancy":   "Answer Relevancy   (0-1, higher=better)",
        "faithfulness":       "Faithfulness       (0-1, higher=better)",
    }
    for key, label in metrics_display.items():
        val = agg.get(key, agg.get(key.replace("_", " "), None))
        if val is not None:
            bar = "█" * int(val * 20)
            print(f"  {label}")
            print(f"    [{bar:<20}] {val:.4f}")
            print()
    print("═"*52)
    print(f"  Questions evaluated : {len(test_cases)}")
    print(f"  Detailed results    : ragas_results_detail.csv")
    print(f"  Score summary       : ragas_scores.json")
    print("═"*52 + "\n")


if __name__ == "__main__":
    main()