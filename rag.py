"""Source-aware RAG with optional strict component/datasheet filtering.

When a concrete part number is supplied, candidate chunks are searched only
from documents tagged as containing that part (or a conservative same-family
alias). If no matching document exists, this module returns no evidence rather
than substituting an unrelated datasheet with a high embedding score.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from models import EMBEDDING_MODEL
from data_ingest import CHROMA_PATH

RAG_SCORE_THRESHOLD = 0.40
RAG_K = 5
MAX_CONTEXT_CHARS = 6500
MAX_DOC_CHARS = 1800
SEARCH_POOL = 30

_PART_ID_RE = re.compile(
    r"\b(?:ESP32(?:-[A-Z0-9]+){1,5}|ESP32|"
    r"(?:AP|LM|LT|LTC|TL|OPA|MCP|INA|ADS|MAX|TPS|AMS|NCP|SN|AD|NE|MC|MIC|STM32|ATmega|BQ|DRV|TUSB|CH340|CP210|"
    r"BME|BMP|MPU|ICM|LSM|LIS|BMI|NRF|RP|GD32|CH32|W25Q|MX25|GD25|PCF|ADXL|SHT|DS18B|USBLC|XC6|XC62)[A-Z0-9]*"
    r"(?:-[A-Z0-9.]+)*)\b",
    re.IGNORECASE,
)


def get_retriever() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )


def _search(db, query: str, k: int):
    if not query or not query.strip():
        return []
    return db.similarity_search_with_relevance_scores(query.strip(), k=k)


def _dedupe(results):
    dedup = {}
    for doc, score in results:
        key = (
            doc.metadata.get("source"), doc.metadata.get("page"),
            doc.metadata.get("table_index"), doc.metadata.get("content_type", "text"),
            doc.page_content,
        )
        if key not in dedup or score > dedup[key][1]:
            dedup[key] = (doc, score)
    return sorted(dedup.values(), key=lambda item: item[1], reverse=True)


def extract_component_ids(text: str) -> list[str]:
    """Extract concrete candidate part identifiers from user text."""
    found, seen = [], set()
    for match in _PART_ID_RE.finditer(text or ""):
        part = match.group(0).strip(".,:;()[]")
        key = _normalize_id(part)
        if key and key not in seen:
            seen.add(key)
            found.append(part)
    return found


def _normalize_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _aliases(part: str) -> set[str]:
    """Generate conservative aliases without conflating materially different module variants."""
    raw = (part or "").strip().upper()
    aliases = {_normalize_id(raw)}
    if raw.startswith("ESP32-S3-WROOM-1U"):
        # Keep WROOM-1U (external antenna connector) distinct from WROOM-1.
        pass
    elif raw.startswith("ESP32-S3-WROOM-1"):
        # Do not let a WROOM-1 datasheet match WROOM-1U just because the ID is a prefix.
        pass
    elif raw.startswith("ESP32-S3-WROOM"):
        aliases.add("ESP32S3WROOM")
    elif raw.startswith("ESP32-S3"):
        # Do not broaden a specific module question to generic ESP32 parts.
        aliases.add("ESP32S3")
    elif raw.startswith("AP2112"):
        aliases.update({"AP2112", "AP2112K"})
    elif raw.startswith(("AMS1117", "LM1117", "LD1117")):
        prefix = re.match(r"(?:AMS|LM|LD)1117", raw).group(0)
        aliases.add(prefix)
        aliases.add(_normalize_id(prefix))
    return {item for item in aliases if item}



def _tagged_parts(metadata: dict) -> list[str]:
    raw = metadata.get("document_part_numbers") or metadata.get("component_hints") or ""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [x.strip() for x in re.split(r"[|,;]", str(raw)) if x.strip()]


def _normalized_part_match(aliases: set[str], values: list[str]) -> list[str]:
    normalized_values = {_normalize_id(value) for value in values if value}
    def prefix_match_allowed(alias: str) -> bool:
        # Prefix matching is useful for generic family IDs, but WROOM-1 and
        # WROOM-1U are not interchangeable exact identities.
        if alias.startswith("ESP32S3WROOM") and alias != "ESP32S3WROOM":
            return False
        return len(alias) >= 8

    return sorted(
        alias for alias in aliases
        if alias in normalized_values
        or (prefix_match_allowed(alias) and any(value.startswith(alias) for value in normalized_values))
    )


def _document_matches_part(doc, part: str) -> tuple[bool, str]:
    """Fail closed when the source metadata identifies a different product family."""
    aliases = _aliases(part)

    # Front-matter/title identifiers are strongest evidence for what datasheet the
    # file is actually about. If they identify another family, a mention buried in
    # later pages must not override that decision.
    primary_tags = _tagged_parts({
        "document_part_numbers": doc.metadata.get("document_primary_part_numbers", "")
    })
    if primary_tags:
        matched = _normalized_part_match(aliases, primary_tags)
        if matched:
            return True, f"primary document/title identifier matched {matched[0]}"
        return False, "front-matter identifiers identify a different part/family"

    tags = _tagged_parts(doc.metadata)
    if tags:
        matched = _normalized_part_match(aliases, tags)
        if matched:
            return True, f"document-level part tag matched {matched[0]}"
        # Do not let a filename override explicit metadata that says the document
        # describes a different part. This is important for renamed/wrong PDFs.
        return False, "document-level part tags did not match"

    # Legacy indexes lack the new metadata. In this case only a precise filename
    # may be used as identity evidence; a random part-number mention inside an
    # unrelated retrieved passage is not sufficient to claim it is the right PDF.
    source_norm = _normalize_id(Path(str(doc.metadata.get("source", ""))).stem)
    named_match = sorted(alias for alias in aliases if len(alias) >= 5 and alias in source_norm)
    if named_match:
        return True, f"source filename matched {named_match[0]} (legacy index fallback)"
    return False, "no reliable part identity found in document metadata or filename"


def _build_context(ranked, *, component_id: str | None = None) -> tuple[str, list[dict], list[float]]:
    parts, sources, scores = [], [], []
    remaining = MAX_CONTEXT_CHARS
    for doc, score in ranked:
        if remaining <= 0:
            break
        content = (doc.page_content or "").strip()
        if not content:
            continue
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "\n[passage truncated]"
        source = str(doc.metadata.get("source", "unknown source"))
        page = doc.metadata.get("page")
        try:
            page_label = f"page {int(page) + 1}" if page is not None else "page not recorded"
        except (ValueError, TypeError):
            page_label = f"page {page}"
        matched_component = component_id or doc.metadata.get("_eda_matched_component") or "general"
        header = f"[SOURCE: {Path(source).name}; {page_label}; component filter: {matched_component}; relevance={float(score):.3f}]\n"
        block = header + content
        if len(block) > remaining:
            block = block[:remaining] + "\n[context limit reached]"
        parts.append(block)
        remaining -= len(block)
        sources.append({
            "source": source,
            "page": page,
            "content_type": doc.metadata.get("content_type", "text"),
            "table_index": doc.metadata.get("table_index"),
            "document_part_numbers": doc.metadata.get("document_part_numbers", ""),
            "matched_component": matched_component if matched_component != "general" else None,
            "relevance_score": float(score),
        })
        scores.append(float(score))
    return "\n\n---\n\n".join(parts), sources, scores


def _empty_result(query: str, component_ids: list[str] | None = None) -> dict:
    return {
        "context": "",
        "sources": [],
        "scores": [],
        "found": False,
        "query": query,
        "component_ids": component_ids or [],
        "filter_mode": "strict_component_match" if component_ids else "general_semantic_search",
        "warning": (
            "No relevant datasheet passages met both the part-identity and query-relevance gates. No unrelated datasheet was substituted."
            if component_ids else "No sufficiently relevant chunks were found."
        ),
    }


def retrieve(
    query: str,
    k: int = RAG_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
    secondary_query: str | None = None,
    component_ids: list[str] | None = None,
    strict_component_filter: bool = False,
) -> dict:
    """Retrieve general evidence or strictly target known component datasheets.

    `strict_component_filter=True` ensures that all returned datasheet passages
    match at least one requested part ID at document level or by exact/family ID.
    If there are no matches, returns an empty result instead of unrelated text.
    """
    db = get_retriever()
    component_ids = list(dict.fromkeys(str(x).strip() for x in (component_ids or []) if str(x).strip()))
    strict = strict_component_filter and bool(component_ids)

    if strict:
        eligible = []
        match_audit = []
        for component_id in component_ids:
            # Keep the original user intent in the retrieval query. Searching the
            # bare part number alone often promotes cover pages and unrelated rating
            # tables just because they repeat the part ID.
            component_query = f"{component_id} {query}".strip()
            try:
                candidates = _dedupe(_search(db, component_query, max(SEARCH_POOL, k * 4)))
            except Exception as exc:
                match_audit.append({"component": component_id, "search_error": str(exc), "candidates": 0, "matched": 0})
                continue
            identity_matches = []
            relevant_matches = []
            for doc, score in candidates:
                ok, basis = _document_matches_part(doc, component_id)
                if not ok:
                    continue
                identity_matches.append((doc, score))
                # A correct product title is necessary but not sufficient: only
                # send passages that also have meaningful query relevance.
                if float(score) < score_threshold:
                    continue
                relevant_matches.append((doc, score))
                doc.metadata.setdefault("_eda_matched_component", component_id)
                doc.metadata.setdefault("_eda_match_basis", basis)
            match_audit.append({
                "component": component_id,
                "candidates": len(candidates),
                "source_identity_matches": len(identity_matches),
                "matched": len(relevant_matches),
                "min_relevance_score": score_threshold,
            })
            eligible.extend(relevant_matches[:max(2, k)])
        eligible = _dedupe(eligible)
        if not eligible:
            result = _empty_result(query, component_ids)
            result["match_audit"] = match_audit
            return result
        # Keep source-specific chunks and avoid sending dozens of repeated pages.
        eligible = eligible[:max(k * min(len(component_ids), 3), k)]
        context, sources, scores = _build_context(eligible)
        result = {
            "context": context, "sources": sources, "scores": scores,
            "found": bool(context.strip()), "query": query,
            "component_ids": component_ids,
            "filter_mode": "strict_component_match",
            "match_audit": match_audit,
            "warning": None,
        }
        return result

    primary = _dedupe(_search(db, query, k))[:k]
    ranked = primary
    if (not ranked or ranked[0][1] < score_threshold) and secondary_query:
        secondary = _dedupe(_search(db, secondary_query, max(2, k // 2)))
        ranked = _dedupe(primary + secondary)[:k]
    if not ranked or ranked[0][1] < score_threshold:
        return _empty_result(query)
    context, sources, scores = _build_context(ranked)
    if not context:
        return _empty_result(query)
    return {
        "context": context, "sources": sources, "scores": scores,
        "found": True, "query": query, "component_ids": [],
        "filter_mode": "general_semantic_search", "warning": None,
    }
