from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ExplainedResult:
    answer: str
    chunks_used: List[dict]
    entities_found: List[str]
    graph_relationships: List[dict]
    intent: str
    retrieval_scores: List[dict]
    explanation_text: str

def build_explanation(
    answer: str,
    chunks: List[dict],
    graph_context: dict,
    intent: str,
    query: str,
    expanded_queries: List[str],
) -> ExplainedResult:
    """
    Generates a human-readable explanation of HOW the answer was found.
    Fulfills PS1 Explainable Information Retrieval requirement.
    """
    top_sources = list(set([
        c.get("document_title", c.get("chunk_id", "unknown"))
        for c in chunks[:3]
    ]))
    rels = graph_context.get("relationships", [])

    explanation_parts = [
        f"Your query was classified as a '{intent}' question.",
        f"It was expanded into {len(expanded_queries)} sub-queries for broader coverage.",
        f"Top {len(chunks)} passages retrieved using BM25 + semantic + graph search, fused with RRF.",
    ]

    if top_sources:
        explanation_parts.append(
            f"Key sources: {', '.join(top_sources)}."
        )

    if rels:
        rel_summary = "; ".join(
            f"{r['source']} -> {r['relation']} -> {r['target']}"
            for r in rels[:3]
        )
        explanation_parts.append(
            f"Knowledge graph links used: {rel_summary}."
        )

    return ExplainedResult(
        answer=answer,
        chunks_used=chunks,
        entities_found=graph_context.get("entities", []),
        graph_relationships=rels,
        intent=intent,
        retrieval_scores=[
            {"id": c.get("chunk_id"), "score": c.get("rrf_score", 0)}
            for c in chunks
        ],
        explanation_text=" ".join(explanation_parts),
    )