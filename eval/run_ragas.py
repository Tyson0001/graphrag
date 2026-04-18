import json
import os
import sys
import numpy as np

sys.path.insert(0, "/home/kunaljha/graphrag")
os.chdir("/home/kunaljha/graphrag")

from dotenv import load_dotenv
load_dotenv("/home/kunaljha/graphrag/.env")

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness
from ragas.dataset_schema import SingleTurnSample
from ragas import evaluate, EvaluationDataset

client = OpenAI(api_key=api_key)
llm = llm_factory("gpt-4o-mini", client=client)

with open("eval/test_cases.json") as f:
    test_cases = json.load(f)

samples = [
    SingleTurnSample(
        user_input=tc["question"],
        response=tc["answer"],
        retrieved_contexts=tc["contexts"],
        reference=tc["ground_truth"],
    )
    for tc in test_cases
]

dataset = EvaluationDataset(samples=samples)

result = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
    ],
)

df = result.to_pandas()
scores = df.mean(numeric_only=True).to_dict()

# Compute answer relevancy manually using cosine similarity
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Computing answer relevancy manually...")
relevancy_scores = []
for tc in test_cases:
    q_emb = get_embedding(tc["question"])
    a_emb = get_embedding(tc["answer"])
    sim = cosine_similarity(q_emb, a_emb)
    relevancy_scores.append(sim)

answer_relevancy = float(np.mean(relevancy_scores))
scores["answer_relevancy"] = answer_relevancy

print("\n=== RAGAS SCORES ===")
for metric, score in scores.items():
    try:
        val = float(score)
        if val != val:
            print(f"  {metric:<40} [N/A] NaN")
        else:
            bar = "█" * int(val * 20)
            print(f"  {metric:<40} [{bar:<20}] {val:.4f}")
    except:
        print(f"  {metric:<40} [ERROR] {score}")

print("\n=== PS1 EVALUATION SUMMARY ===")
cp = scores.get('llm_context_precision_with_reference', 0)
cr = scores.get('context_recall', 0)
fa = scores.get('faithfulness', 0)
ar = scores.get('answer_relevancy', 0)
print(f"  Context Precision : {cp:.2%} {'✅ PASS' if cp >= 0.9 else '⚠ needs improvement'}")
print(f"  Context Recall    : {cr:.2%} {'✅ PASS' if cr >= 0.8 else '⚠ needs improvement'}")
print(f"  Faithfulness      : {fa:.2%} {'✅ PASS' if fa >= 0.6 else '⚠ needs improvement'}")
print(f"  Answer Relevancy  : {ar:.2%} {'✅ PASS' if ar >= 0.8 else '⚠ needs improvement'}")
