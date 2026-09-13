"""Fast, conservative knowledge-graph builder for datasheets.

The KG is intentionally small. Structured/deterministic facts are extracted
without an LLM. The LLM path is optional and disabled by default because
RAG already stores the full text/tables and Qwen inference was the ingestion
bottleneck.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from ollama_utils import chat_json
from models import EXTRACTION_MODEL
from data_ingest import load_documents, split_text

GRAPH_PATH = "knowledge_graph.graphml"

# Optional fallback. Keep False for fast ingestion.
USE_LLM_FALLBACK = False
LLM_FALLBACK_MAX_BLOCKS = 2
LLM_FALLBACK_BATCH_SIZE = 4
LLM_FALLBACK_WORKERS = 2

# Conservative part-number patterns.
PART_RE = re.compile(
    r"\b(?:LM|LT|LTC|TL|OPA|MCP|INA|ADS|MAX|TPS|NCP|SN|AD|NE|MC|MIC|STM32|ATmega|ESP32)[A-Za-z0-9_-]{2,}\b",
    re.IGNORECASE,
)

TRIPLE_SYSTEM_PROMPT = """Extract only high-confidence electronics facts from the supplied evidence.
Return STRICT JSON: {\"triples\":[{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\",\"source_id\":\"...\"}]}
Rules: never invent values; never mix device columns; never guess ambiguous table mappings; source_id must match the evidence header."""


def _part_numbers(text: str) -> list[str]:
    result, seen = [], set()
    for match in PART_RE.findall(text or ""):
        key = match.upper()
        if key not in seen:
            seen.add(key)
            result.append(match)
    return result


def _add_edge(graph, subject, relation, obj, source=None, page=None):
    if not subject or not relation or not obj:
        return
    attrs = {"relation": str(relation), "source": source or ""}
    if page is not None:
        attrs["page"] = str(page)
    graph.add_edge(str(subject), str(obj), **attrs)


def _extract_deterministic_facts(graph: nx.MultiDiGraph, chunks, table_records):
    """Extract facts that are explicit and structurally recognizable."""
    facts = 0

    # ---- Page/chunk text facts -------------------------------------------
    for chunk in chunks:
        text = chunk.page_content or ""
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        parts = _part_numbers(text)
        if not parts:
            continue

        # Explicit single-supply range, e.g. "Single Supply Operation: 3.0 V to 32 V"
        m = re.search(
            r"single\s+supply(?:\s+operation)?\s*[:\-]\s*([+-]?\d+(?:\.\d+)?)\s*V\s*(?:to|[-–])\s*([+-]?\d+(?:\.\d+)?)\s*V",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            for part in parts:
                _add_edge(graph, part, "minimum_supply_voltage", f"{m.group(1)} V", source, page)
                _add_edge(graph, part, "maximum_supply_voltage", f"{m.group(2)} V", source, page)
                facts += 2

        # Explicit split-supply rating. Only accept an explicit ± value so a
        # nearby single-supply value such as 3.0 V can never be misclassified.
        m = re.search(
            r"split\s+suppl(?:y|ies)[^\n]{0,180}?([±]\s*\d+(?:\.\d+)?)\s*V(?:dc)?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            value = re.sub(r"\s+", "", m.group(1))
            for part in parts:
                _add_edge(graph, part, "maximum_split_supply_voltage", f"{value} Vdc", source, page)
                facts += 1

        # Common explicit feature statements.
        feature_patterns = (
            (r"single\s+and\s+split\s+supply\s+operation", "supports", "single_and_split_supply_operation"),
            (r"internally\s+compensated", "feature", "internally_compensated"),
            (r"short\s+circuit\s+protected\s+outputs", "feature", "short_circuit_protected_outputs"),
            (r"true\s+differential\s+input\s+stage", "feature", "true_differential_input_stage"),
        )
        for pattern, relation, obj in feature_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                for part in parts:
                    _add_edge(graph, part, relation, obj, source, page)
                    facts += 1

        # Operating ambient range for a part when the same chunk names it.
        m = re.search(
            r"Operating\s+Ambient\s+Temperature\s+Range.*?\bLM358(?:,\s*LM358A)?\b.*?0\s*to\s*\+?70\s*°?C",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m and any(p.upper().startswith("LM358") for p in parts):
            for part in parts:
                if part.upper().startswith("LM358"):
                    _add_edge(graph, part, "operating_temperature_range", "0 °C to +70 °C", source, page)
                    facts += 1

    # ---- Table relationships ---------------------------------------------
    # Tables are always preserved in Chroma. Only create KG facts from tables
    # when the mapping is explicit enough to be safe. Otherwise do nothing.
    for record in table_records:
        page_text = record.get("page_text", "") or ""
        parts = record.get("component_hints") or _part_numbers(page_text)
        source = record.get("source")
        page = record.get("page")

        rows = record.get("rows") or []
        flat = "\n".join(" | ".join(row) for row in rows)

        # Maximum ratings: explicit "Single Supply ... 32 V" patterns.
        m = re.search(
            r"single\s+supply.*?\b(\d+(?:\.\d+)?)\s*V(?:dc)?\b",
            flat,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m and any("maximum" in r.lower() or "rating" in r.lower() for r in rows[:5]):
            for part in parts:
                _add_edge(graph, part, "maximum_supply_voltage", f"{m.group(1)} Vdc", source, page)
                facts += 1


    return facts


def PathLike(value):
    """Short source identifier used as a graph node without importing pathlib everywhere."""
    if not value:
        return "document"
    return os.path.basename(str(value))


def _is_narrative_candidate(text: str) -> bool:
    lowered = (text or "").lower()
    keywords = (
        "supports", "interface", "operational amplifier", "channels",
        "pin connection", "package", "operating temperature", "supply voltage",
    )
    return any(k in lowered for k in keywords)


def _llm_extract_batch(items):
    sections = []
    for sid, source, page, text in items:
        sections.append(
            f"[SOURCE_ID={sid}]\nSOURCE={source}\nPAGE={page + 1 if isinstance(page, int) else page}\nEVIDENCE:\n{text}"
        )
    try:
        result = chat_json(EXTRACTION_MODEL, TRIPLE_SYSTEM_PROMPT, "\n\n========\n\n".join(sections))
        triples = result.get("triples", [])
        return triples if isinstance(triples, list) else []
    except Exception:
        return []


def _add_llm_triples(graph, triples, metadata):
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        sid = str(triple.get("source_id", "")).strip()
        subject = str(triple.get("subject", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if subject and relation and obj and sid in metadata:
            meta = metadata[sid]
            _add_edge(graph, subject, relation, obj, meta.get("source"), meta.get("page"))


def build_knowledge_graph(
    data_path: str,
    *,
    chunks=None,
    table_records=None,
    max_workers: int = 2,
    batch_size: int = 4,
    use_llm_fallback: bool = USE_LLM_FALLBACK,
) -> nx.MultiDiGraph:
    if chunks is None:
        documents = load_documents(data_path)
        chunks = split_text(documents)
    table_records = table_records or []

    graph = nx.MultiDiGraph()
    facts = _extract_deterministic_facts(graph, chunks, table_records)
    print(f"Built {facts} deterministic KG facts without LLM calls.")

    if use_llm_fallback:
        # Only a tiny narrative subset is sent to Qwen.
        candidates = [
            (i, c) for i, c in enumerate(chunks)
            if _is_narrative_candidate(c.page_content)
        ][:LLM_FALLBACK_MAX_BLOCKS]
        items = []
        metadata = {}
        for i, chunk in candidates:
            sid = f"TEXT_{i}"
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page", 0)
            items.append((sid, source, page, chunk.page_content))
            metadata[sid] = {"source": source, "page": page}

        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        if batches:
            print(f"Optional LLM fallback: {len(items)} evidence blocks in {len(batches)} request(s).")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_llm_extract_batch, b) for b in batches]
                for future in as_completed(futures):
                    _add_llm_triples(graph, future.result(), metadata)

    nx.write_graphml(graph, GRAPH_PATH)
    print(
        f"Knowledge graph saved to {GRAPH_PATH} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)."
    )
    return graph


def load_knowledge_graph() -> nx.MultiDiGraph:
    if not os.path.exists(GRAPH_PATH):
        return nx.MultiDiGraph()
    return nx.read_graphml(GRAPH_PATH)


def query_knowledge_graph(
    graph: nx.MultiDiGraph,
    terms: list[str],
    max_facts: int = 40,
) -> str:
    """
    Return a compact set of useful KG facts for the component selector.

    Table-link edges are intentionally ignored here. Full tables live in
    Chroma/RAG; sending hundreds of table identifiers into the selector
    wastes context and can cause downstream model context overflows.
    """
    lines = []
    seen = set()

    for term in terms:
        term_text = str(term).strip().lower()
        if not term_text:
            continue

        matches = [
            node for node in graph.nodes
            if term_text in str(node).lower()
        ]

        for node in matches:
            for _, target, data in graph.out_edges(node, data=True):
                relation = str(data.get("relation", "related_to"))
                if relation == "has_table":
                    continue
                line = f"{node} --{relation}--> {target}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
                    if len(lines) >= max_facts:
                        return "\n".join(lines)

            for source, _, data in graph.in_edges(node, data=True):
                relation = str(data.get("relation", "related_to"))
                if relation == "has_table":
                    continue
                line = f"{source} --{relation}--> {node}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
                    if len(lines) >= max_facts:
                        return "\n".join(lines)

    return "\n".join(sorted(lines)) if lines else "No relevant knowledge graph facts found."
