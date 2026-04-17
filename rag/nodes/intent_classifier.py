from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INTENT_LABELS = {
    "factual": ["what is", "who is", "define", "when did"],
    "comparative": ["compare", "difference", "vs", "versus", "better than"],
    "procedural": ["how to", "steps to", "guide", "process for"],
    "exploratory": ["tell me about", "explain", "overview of", "describe"],
}

class IntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchors = {}
        for intent, phrases in INTENT_LABELS.items():
            self.anchors[intent] = self.model.encode(phrases)

    def classify(self, query: str) -> str:
        q_emb = self.model.encode([query])
        scores = {}
        for intent, anchor_embs in self.anchors.items():
            sims = cosine_similarity(q_emb, anchor_embs)
            scores[intent] = float(np.max(sims))
        return max(scores, key=scores.get)

    def expand_query(self, query: str, intent: str) -> list[str]:
        expansions = [query]
        if intent == "factual":
            expansions += [
                f"definition of {query}",
                f"{query} explanation",
            ]
        elif intent == "comparative":
            expansions += [
                f"advantages of {query}",
                f"disadvantages of {query}",
            ]
        elif intent == "procedural":
            expansions += [
                f"step by step {query}",
                f"best practices for {query}",
            ]
        elif intent == "exploratory":
            expansions += [
                f"overview {query}",
                f"key concepts in {query}",
            ]
        return expansions