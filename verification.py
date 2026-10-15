"""Evidence verification for answers grounded in datasheets and schematics.

The verifier combines retrieved datasheet passages with deterministic schematic
observations. Schematic facts are not treated as datasheet facts: they are
included as a separate evidence section and source.
"""
from __future__ import annotations

import re
import numpy as np
from sentence_transformers import CrossEncoder

from ollama_utils import embed_text
from models import EMBEDDING_MODEL
from schematic_checker import format_findings_for_llm

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall", device="cpu")
_NLI_LABELS = ["contradiction", "entailment", "neutral"]
SIMILARITY_THRESHOLD = 0.55


MAX_EMBED_CONTEXT_CHARS = 3500


def _bounded_embedding_context(text: str) -> str:
    """Keep embedding input bounded while retaining both ends of the evidence."""
    if len(text) <= MAX_EMBED_CONTEXT_CHARS:
        return text

    marker = "\n\n[Middle of evidence omitted for embedding.] \n\n"
    available = MAX_EMBED_CONTEXT_CHARS - len(marker)
    head_size = available * 2 // 3
    tail_size = available - head_size

    return text[:head_size] + marker + text[-tail_size:]

def split_into_claims(answer_text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def check_citation_existence(sources: list[dict | str]) -> dict:
    passed = bool(sources)
    return {
        "passed": passed,
        "detail": f"{len(sources)} source(s) backing this answer." if passed else "No datasheet or schematic evidence sources were supplied.",
        "sources": sources,
    }


def check_semantic_similarity(
    claims: list[str], retrieved_context: str, retrieval_score: float | None = None
) -> dict:
    if not claims or not retrieved_context.strip():
        return {"passed": False, "detail": "Missing claims or evidence context to compare."}
    context_for_embedding = _bounded_embedding_context(retrieved_context)
    context_vec = embed_text(EMBEDDING_MODEL, context_for_embedding)
    threshold = retrieval_score if retrieval_score is not None else SIMILARITY_THRESHOLD
    scores = [_cosine_similarity(embed_text(EMBEDDING_MODEL, claim), context_vec) for claim in claims]
    avg_score = sum(scores) / len(scores)
    return {
        "passed": avg_score >= threshold,
        "detail": f"avg similarity {avg_score:.3f} vs threshold {threshold:.3f}",
        "scores": scores,
        "threshold": threshold,
    }



_NLI_STRONG_CONTRADICTION = 0.82
_NLI_CONTRADICTION_MARGIN = 0.15
_NLI_ENTAILMENT_THRESHOLD = 0.60
_NLI_PASSAGE_CHARS = 1100

_NLI_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "are", "is", "was", "were", "be", "been", "being", "has", "have", "had",
    "into", "onto", "than", "then", "when", "where", "which", "while", "also",
    "should", "could", "would", "must", "may", "can", "cannot", "not", "only",
    "ensure", "check", "verify", "confirm", "review", "design", "evidence",
    "datasheet", "component", "components", "based", "provided", "supplied",
    "following", "using", "used", "such", "each", "other", "their", "there",
}


def _evidence_passages(text: str) -> list[str]:
    """Split evidence into short passages so NLI never compares a claim with a huge truncated premise."""
    passages: list[str] = []
    # RAG context uses --- separators between source chunks; combined evidence uses them too.
    for block in text.split("---"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= _NLI_PASSAGE_CHARS:
            passages.append(block)
            continue
        start = 0
        while start < len(block):
            passage = block[start:start + _NLI_PASSAGE_CHARS].strip()
            if passage:
                passages.append(passage)
            if start + _NLI_PASSAGE_CHARS >= len(block):
                break
            start += _NLI_PASSAGE_CHARS - 120
    return passages


def _claim_terms(text: str) -> set[str]:
    terms = set()
    for item in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+/-]*", text.lower()):
        normalized = re.sub(r"[^a-z0-9]", "", item)
        if len(normalized) >= 2 and normalized not in _NLI_STOPWORDS:
            terms.add(normalized)
    return terms


def _best_evidence_passages(claim: str, passages: list[str], limit: int = 2) -> list[str]:
    """Choose source passages sharing technical terms with a claim; avoid irrelevant-premise contradictions."""
    claim_terms = _claim_terms(claim)
    ranked = []
    for passage in passages:
        passage_terms = _claim_terms(passage)
        overlap = claim_terms & passage_terms
        if overlap:
            # Prefer exact part references, net names, units, and numerical facts over generic overlap.
            specificity = sum(
                2 if any(ch.isdigit() for ch in term) or len(term) >= 7 else 1
                for term in overlap
            )
            ranked.append((specificity, len(overlap), passage))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:limit]]


def _is_specific_technical_claim(claim: str) -> bool:
    """Detect assertions whose pins, parts, or numeric ratings must be evidence-backed."""
    return bool(re.search(
        r"\b(?:[A-Z]{1,6}\d+[A-Z0-9.-]*|GPIO\d+|pin\s*\d+|net\s+[A-Z0-9_+./-]+|"
        r"\d+(?:\.\d+)?\s*(?:mV|V|uV|µV|mA|uA|µA|A|kHz|MHz|GHz|Hz|mW|W|°C|ohms?|Ω))\b",
        claim,
        flags=re.IGNORECASE,
    ))


def check_entailment(claims: list[str], retrieved_context: str) -> dict:
    """Flag only high-confidence contradictions against relevant, short evidence passages.

    An unrelated or unmentioned claim is labelled neutral/unsupported rather than
    contradictory merely because it is absent from a long combined premise. This
    check does not prove that a neutral claim is true.
    """
    if not claims or not retrieved_context.strip():
        return {
            "passed": False,
            "detail": "Missing claims or evidence context to compare.",
            "results": [],
        }

    passages = _evidence_passages(retrieved_context)
    selected: dict[int, list[str]] = {
        index: _best_evidence_passages(claim, passages)
        for index, claim in enumerate(claims)
    }
    pending_pairs: list[tuple[str, str]] = []
    pair_claim_indices: list[int] = []
    for index, claim in enumerate(claims):
        for premise in selected[index]:
            pending_pairs.append((premise, claim))
            pair_claim_indices.append(index)

    per_claim_scores: dict[int, list[list[float]]] = {index: [] for index in range(len(claims))}
    if pending_pairs:
        raw_scores = _nli_model.predict(pending_pairs)
        for claim_index, scores in zip(pair_claim_indices, raw_scores):
            per_claim_scores[claim_index].append([float(value) for value in scores])

    results = []
    for index, claim in enumerate(claims):
        scores_list = per_claim_scores[index]
        if not scores_list:
            results.append({
                "claim": claim,
                "label": "neutral",
                "scores": [],
                "note": "No evidence passage shared sufficiently specific terms; claim not validated by NLI.",
            })
            continue

        max_contradiction = max(scores[0] for scores in scores_list)
        max_entailment = max(scores[1] for scores in scores_list)
        max_neutral = max(scores[2] for scores in scores_list)

        if (
            max_contradiction >= _NLI_STRONG_CONTRADICTION
            and max_contradiction >= max_entailment + _NLI_CONTRADICTION_MARGIN
        ):
            label = "contradiction"
        elif max_entailment >= _NLI_ENTAILMENT_THRESHOLD:
            label = "entailment"
        else:
            label = "neutral"

        results.append({
            "claim": claim,
            "label": label,
            "scores": {
                "max_contradiction": max_contradiction,
                "max_entailment": max_entailment,
                "max_neutral": max_neutral,
                "passages_compared": len(scores_list),
            },
        })

    contradictions = [result for result in results if result["label"] == "contradiction"]
    neutral = [result for result in results if result["label"] == "neutral"]
    entailed = [result for result in results if result["label"] == "entailment"]
    unsupported_technical = [
        result for result in neutral if _is_specific_technical_claim(result["claim"])
    ]
    return {
        # Generic recommendations can be neutral, but specific pin/part/rating
        # claims must be entailed by relevant evidence to pass this gate.
        "passed": len(contradictions) == 0 and len(unsupported_technical) == 0,
        "detail": (
            f"{len(contradictions)} high-confidence contradictions, "
            f"{len(entailed)} claims with entailment support, "
            f"{len(neutral)} neutral/unsupported claims, including "
            f"{len(unsupported_technical)} unsupported technical assertions among {len(results)} claims. "
            "Neutral claims are not confirmed as true."
        ),
        "results": results,
        "contradiction_count": len(contradictions),
        "entailed_count": len(entailed),
        "neutral_count": len(neutral),
        "unsupported_technical_count": len(unsupported_technical),
    }



def _format_netlist_connectivity(schematic_result: dict | None, max_chars: int = 3800) -> str:
    """Compactly serialize actual net-to-pin membership for schematic claim checks."""
    if not schematic_result:
        return ""
    lines = ["DETERMINISTIC NETLIST CONNECTIVITY (observed in the parsed file):"]
    for net_name, net in schematic_result.get("nets", {}).items():
        nodes = net.get("nodes", [])
        if not nodes:
            continue
        refs = [f"{node.get('ref')}.{node.get('pin')}" for node in nodes]
        lines.append(f"- {net_name}: {', '.join(refs)}")
        if sum(len(line) + 1 for line in lines) >= max_chars:
            lines.append("[Remaining nets omitted from verification context]")
            break
    return "\n".join(lines)[:max_chars]


def _combine_evidence(rag_result: dict, schematic_result: dict | None) -> tuple[str, list[dict | str], dict[str, str]]:
    datasheet_context = rag_result.get("context", "") or ""
    datasheet_sources = list(rag_result.get("sources", []) or [])
    schematic_context = ""
    schematic_sources: list[dict | str] = []
    if schematic_result:
        schematic_context = (
            format_findings_for_llm(schematic_result)
            + "\n\n"
            + _format_netlist_connectivity(schematic_result)
        )
        schematic_sources.append({
            "source": schematic_result.get("input_path", "EDA schematic/netlist"),
            "page": None,
            "content_type": "schematic_checker",
            "finding_count": schematic_result.get("finding_count", 0),
        })
    parts = []
    if datasheet_context.strip():
        parts.append("DATASHEET EVIDENCE (component-filtered by the RAG layer):\n" + datasheet_context)
    if schematic_context.strip():
        parts.append("SCHEMATIC/NETLIST EVIDENCE (connectivity observed in the file):\n" + schematic_context)
    return "\n\n---\n\n".join(parts), datasheet_sources + schematic_sources, {
        "datasheet_context": datasheet_context,
        "schematic_context": schematic_context,
    }


def _check_component_datasheet_match(rag_result: dict) -> dict:
    target_ids = rag_result.get("component_ids", []) or []
    if rag_result.get("filter_mode") != "strict_component_match" or not target_ids:
        return {"passed": True, "detail": "No strict component-specific datasheet gate was requested."}

    audit = {str(item.get("component")): int(item.get("matched", 0)) for item in rag_result.get("match_audit", []) if isinstance(item, dict)}
    # If a merge collapsed audit values, recover presence from the source records.
    for source in rag_result.get("sources", []) or []:
        if isinstance(source, dict) and source.get("matched_component"):
            audit[str(source["matched_component"])] = max(1, audit.get(str(source["matched_component"]), 0))
    missing = [part for part in target_ids if audit.get(str(part), 0) <= 0]
    return {
        "passed": not missing and bool(rag_result.get("found")),
        "detail": (
            "Part-matched datasheet evidence found for all target components."
            if not missing and rag_result.get("found")
            else "No exact/family-matched datasheet evidence found for: " + ", ".join(missing or target_ids)
        ),
        "missing_components": missing,
        "audit": audit,
    }


def verify(answer_text: str, rag_result: dict, schematic_result: dict | None = None) -> dict:
    """Verify evidence existence, semantic alignment, NLI, and strict source match."""
    claims = split_into_claims(answer_text)
    context, sources, evidence_parts = _combine_evidence(rag_result, schematic_result)
    citation = check_citation_existence(sources)
    similarity = check_semantic_similarity(claims, context) if context else {
        "passed": False, "detail": "No evidence context available to compare."
    }
    entailment = check_entailment(claims, context) if context else {
        "passed": False, "detail": "No evidence context available for entailment checks."
    }
    component_match = _check_component_datasheet_match(rag_result)
    passed = (
        citation["passed"]
        and similarity["passed"]
        and entailment["passed"]
        and component_match["passed"]
    )
    return {
        "passed": passed,
        "citation_existence": citation,
        "semantic_similarity": similarity,
        "entailment_nli": entailment,
        "datasheet_component_match": component_match,
        "evidence_types": {
            "datasheet_available": bool(evidence_parts["datasheet_context"].strip()),
            "schematic_available": bool(evidence_parts["schematic_context"].strip()),
            "schematic_finding_count": schematic_result.get("finding_count", 0) if schematic_result else 0,
        },
    }
