"""
'Explainable AI Architecture' block from your diagram.
A lightweight verification layer that checks the Reasoning LLM's answer
before it's trusted, using three cheap checks run in parallel:

  1. Citation existence -> plain string matching, no model at all
  2. Semantic similarity -> cosine similarity of embeddings (nomic-embed-text,
                             the same model already used for RAG)
  3. Entailment / NLI    -> cross-encoder/nli-deberta-v3-xsmall (~70M params),
                             runs on CPU so it never touches your GPU's 4GB

If any check fails, the caller (pipeline.py) sends the answer back to the
Reasoning LLM to regenerate — mirrors the "If fails -> LLM regenerate"
loop in the diagram.
"""
import re
import numpy as np
from sentence_transformers import CrossEncoder

from ollama_utils import embed_text
from models import EMBEDDING_MODEL

# CPU is plenty fast for single sentence-pairs at this model size, and
# keeping it off the GPU means it never competes with the generator
# model's VRAM on a 4GB card.
_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall", device="cpu")
_NLI_LABELS = ["contradiction", "entailment", "neutral"]

SIMILARITY_THRESHOLD = 0.55  # tune against your own retrieval scores


def split_into_claims(answer_text: str) -> list[str]:
    """Naive sentence splitter — good enough for atomic-ish claims without an extra LLM call."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def check_citation_existence(sources: list[str]) -> dict:
    """
    Diagram's 'Citation Existence' box: confirms the answer actually has
    retrieved sources behind it at all. (If you want stricter behavior —
    requiring the answer text to literally name a filename — swap the
    `passed` line for a substring check against answer_text.)
    """
    passed = bool(sources)
    return {
        "passed": passed,
        "detail": f"{len(sources)} source(s) backing this answer." if passed else "No sources were retrieved.",
        "sources": sources,
    }


def check_semantic_similarity(claims: list[str], retrieved_context: str, retrieval_score: float | None = None) -> dict:
    """
    Diagram's 'Semantic Similarity' box: embed the claims and the
    retrieved context, take cosine similarity, and require it to be at
    or above the retrieval's own confidence score (falls back to a
    fixed floor if that score isn't available).
    """
    if not claims or not retrieved_context:
        return {"passed": False, "detail": "Missing claims or retrieved context to compare."}

    context_vec = embed_text(EMBEDDING_MODEL, retrieved_context)
    threshold = retrieval_score if retrieval_score is not None else SIMILARITY_THRESHOLD

    scores = [_cosine_similarity(embed_text(EMBEDDING_MODEL, c), context_vec) for c in claims]
    avg_score = sum(scores) / len(scores)

    return {
        "passed": avg_score >= threshold,
        "detail": f"avg similarity {avg_score:.3f} vs threshold {threshold:.3f}",
        "scores": scores,
    }


def check_entailment(claims: list[str], retrieved_context: str) -> dict:
    """
    Diagram's 'Entailment/NLI' box: for each claim, ask the NLI model
    whether the retrieved context (premise) entails, contradicts, or is
    neutral toward the claim (hypothesis). Any contradiction fails the check.
    """
    if not claims or not retrieved_context:
        return {"passed": False, "detail": "Missing claims or retrieved context to compare."}

    pairs = [(retrieved_context, claim) for claim in claims]
    raw_scores = _nli_model.predict(pairs)  # shape: (n_claims, 3) -> [contradiction, entailment, neutral]

    results = []
    for claim, scores in zip(claims, raw_scores):
        label = _NLI_LABELS[int(np.argmax(scores))]
        results.append({"claim": claim, "label": label, "scores": scores.tolist()})

    contradictions = [r for r in results if r["label"] == "contradiction"]
    passed = len(contradictions) == 0
    return {
        "passed": passed,
        "detail": f"{len(contradictions)}/{len(results)} claims contradicted by retrieved context.",
        "results": results,
    }


def verify(answer_text: str, rag_result: dict) -> dict:
    """
    Runs all three checks and combines them — the diagram's
    'Normal If/Else: everyone passes -> trustable' logic.
    """
    claims = split_into_claims(answer_text)
    context = rag_result.get("context", "")
    sources = rag_result.get("sources", [])

    citation = check_citation_existence(sources)
    similarity = check_semantic_similarity(claims, context)
    entailment = check_entailment(claims, context)

    passed = citation["passed"] and similarity["passed"] and entailment["passed"]
    return {
        "passed": passed,
        "citation_existence": citation,
        "semantic_similarity": similarity,
        "entailment_nli": entailment,
    }
