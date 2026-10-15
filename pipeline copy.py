"""Question-first design assistant with strict datasheet source matching."""
from __future__ import annotations

import os
import re
from typing import Any

from rag import retrieve, extract_component_ids
from reasoning_llm import reason
from verification import verify
from schematic_checker import check_schematic, format_findings_for_llm


def _knowledge_base_ready() -> bool:
    # The source-aware pipeline queries Chroma directly; the graph is optional
    # now and should not block retrieval if graph construction was unavailable.
    return os.path.isdir("chroma")


def _is_design_review_request(question: str) -> bool:
    lowered = (question or "").lower()
    cues = (
        "review my pcb", "review the pcb", "pcb design", "schematic review",
        "connectivity issues", "power supply issues", "review my board",
        "check my board", "check this schematic", "electrical review",
        "review this circuit", "review my circuit", "netlist review",
    )
    return any(cue in lowered for cue in cues)


def _generic_esp32_needs_variant(question: str, identifiers: list[str], has_schematic: bool) -> bool:
    """Avoid assigning variant-specific electrical facts to the generic ESP32 family."""
    if has_schematic:
        return False
    normalized = {item.upper() for item in identifiers}
    if "ESP32" not in normalized:
        return False
    if any(item.startswith("ESP32-") for item in normalized):
        return False
    return bool(re.search(
        r"\b(?:datasheet|pinout|pin\s*\d+|supply|power|voltage|current|clock|frequency|boot|strapping|gpio|operating|absolute maximum|rating|specification|electrical|thermal|consumption)\b",
        question,
        flags=re.IGNORECASE,
    ))


def _datasheet_targets_from_schematic(schematic_result: dict[str, Any]) -> list[str]:
    """Select only recognizable part identifiers from parsed components.

    Generic symbol labels and connector names are not treated as part numbers,
    because requiring a datasheet for values like 'DEBUG' would make the whole
    review fail despite not identifying a manufacturer part.
    """
    targets: list[str] = []
    seen: set[str] = set()
    for component in schematic_result.get("components", []):
        ref = str(component.get("ref", "")).upper()
        value = str(component.get("value", "")).strip()
        if not ref.startswith(("U", "IC", "Q", "D", "J", "Y", "X")) or not value:
            continue
        for part_id in extract_component_ids(value):
            key = part_id.casefold()
            if key not in seen:
                seen.add(key)
                targets.append(part_id)
    return targets


def _merge_rag_results(results: list[dict], query: str, component_ids: list[str]) -> dict:
    """Merge already-filtered results only; never combine with broad/unfiltered results."""
    contexts, sources, scores = [], [], []
    seen_contexts, seen_sources = set(), set()
    match_audit = []
    for result in results:
        context = result.get("context", "")
        if context and context not in seen_contexts:
            contexts.append(context)
            seen_contexts.add(context)
        for source in result.get("sources", []):
            key = (source.get("source"), source.get("page"), source.get("table_index"), source.get("matched_component"))
            if key not in seen_sources:
                sources.append(source)
                seen_sources.add(key)
        scores.extend(result.get("scores", []))
        match_audit.extend(result.get("match_audit", []))
    context = "\n\n--- MATCHED DATASHEET FOR ANOTHER TARGET COMPONENT ---\n\n".join(contexts)
    found = bool(context.strip())
    return {
        "context": context[:9000], "sources": sources[:20], "scores": scores,
        "found": found, "query": query, "component_ids": component_ids,
        "filter_mode": "strict_component_match",
        "warning": None if found else "No part-number-matched datasheet was found for the identified component(s). Unrelated datasheets were not substituted.",
        "match_audit": match_audit,
    }


def _verification_feedback(verdict: dict) -> str:
    problems = []
    for key, label in (
        ("citation_existence", "Evidence/source existence"),
        ("semantic_similarity", "Semantic similarity"),
        ("entailment_nli", "Entailment/NLI"),
        ("datasheet_component_match", "Exact component datasheet match"),
    ):
        item = verdict.get(key, {})
        if item and not item.get("passed", False):
            problems.append(f"{label}: {item.get('detail', 'check failed')}")
    return " | ".join(problems) or "The verifier did not approve the answer."


def _empty_rag_result(query: str, warning: str, mode: str, component_ids: list[str] | None = None) -> dict:
    return {
        "context": "", "sources": [], "scores": [], "found": False,
        "query": query, "component_ids": component_ids or [],
        "filter_mode": mode, "warning": warning,
        "match_audit": [],
    }


def _missing_component_datasheets(rag_result: dict) -> list[str]:
    """Return target part IDs with no verified source match."""
    targets = [str(item) for item in (rag_result.get("component_ids") or [])]
    if not targets:
        return []
    counts: dict[str, int] = {}
    for item in rag_result.get("match_audit", []) or []:
        if isinstance(item, dict) and item.get("component"):
            counts[str(item["component"])] = max(
                counts.get(str(item["component"]), 0), int(item.get("matched", 0) or 0)
            )
    for source in rag_result.get("sources", []) or []:
        if isinstance(source, dict) and source.get("matched_component"):
            key = str(source["matched_component"])
            counts[key] = max(1, counts.get(key, 0))
    return [target for target in targets if counts.get(target, 0) <= 0]


def _structural_only_report(schematic_result: dict, missing_parts: list[str], reason: str) -> str:
    """Return deterministic netlist observations when datasheet coverage is incomplete."""
    inventory = format_findings_for_llm(schematic_result)
    missing_text = ", ".join(missing_parts) if missing_parts else "no exact part identifiers could be resolved"
    return (
        "STRUCTURAL NETLIST REVIEW — NOT AN ELECTRICAL SIGN-OFF\n\n"
        f"Parsed {schematic_result.get('component_count', 0)} components and "
        f"{schematic_result.get('net_count', 0)} nets using "
        f"{schematic_result.get('input_format', 'an auto-detected input format')}.\n\n"
        "The deterministic connectivity checks ran, but datasheet-grounded reasoning was blocked to prevent unrelated or guessed device specifications.\n\n"
        f"DATASHEET GATE: BLOCKED\nReason: {reason}\n"
        f"Missing/unverified target datasheets: {missing_text}.\n\n"
        "NETLIST OBSERVATIONS (connectivity only; uncertain findings are review prompts):\n"
        f"{inventory}\n\n"
        "Next step: place the manufacturer datasheet(s) for the exact component values in data/datasheets/, run `python ingest.py` once, and ask again."
    )


def run_pipeline(raw_requirements: str, schematic_path: str | None = None) -> str:
    raw_requirements = (raw_requirements or "").strip()
    if not raw_requirements:
        return "Enter a question or design requirement first."

    # Question gate first: a board-specific review cannot be inferred from datasheets.
    question_part_ids = extract_component_ids(raw_requirements)
    has_schematic_path = bool(schematic_path and schematic_path.strip())
    if _generic_esp32_needs_variant(raw_requirements, question_part_ids, has_schematic_path):
        return (
            "Please specify the exact ESP32 chip/module variant before I retrieve electrical specifications "
            "(for example, ESP32, ESP32-S3-WROOM-1, ESP32-C3, or a specific DevKit/module SKU). "
            "Supply limits, pin assignments, and strapping behavior differ between variants, so I will not "
            "substitute another ESP32 family's datasheet."
        )
    if _is_design_review_request(raw_requirements) and not has_schematic_path:
        specified = (
            " The question names these device identifiers: " + ", ".join(question_part_ids) + "."
            if question_part_ids else ""
        )
        return (
            "A board-specific PCB review needs the actual schematic or exported netlist; a device datasheet alone "
            "cannot show how the board is connected.\n\n"
            "No schematic or unrelated datasheet was used to guess the design." + specified + " "
            "Provide a supported netlist/schematic path. For datasheet validation, add each exact device datasheet to "
            "data/datasheets/ and run `python ingest.py` once."
        )

    if not _knowledge_base_ready():
        if has_schematic_path:
            # Structural checks do not require a vector store. Without the datasheets,
            # stop before calling a reasoning model that might fill gaps from memory.
            structural = check_schematic(schematic_path.strip())
            targets = _datasheet_targets_from_schematic(structural)
            return _structural_only_report(
                structural,
                targets,
                "No processed datasheet/vector database was found.",
            )
        return "Please process datasheets first. Add the exact manufacturer PDFs to data/datasheets/ and run `python ingest.py`."

    # Route the raw question and supplied design first. No model is allowed to
    # invent an electrical specification before source selection.
    print("Question validated for routing; selecting evidence from explicit identifiers and parsed design.")

    # Parse design before doing retrieval, then derive sources to search from actual part IDs.
    schematic_result = None
    if has_schematic_path:
        schematic_result = check_schematic(schematic_path.strip())
        print(
            f"Schematic/netlist analyzed: {schematic_result.get('component_count')} components, "
            f"{schematic_result.get('net_count')} nets, {schematic_result.get('finding_count')} findings; "
            f"format={schematic_result.get('input_format')}"
        )
        component_ids = _datasheet_targets_from_schematic(schematic_result)
        print("Device identifiers extracted from file:", component_ids or "none detected")
    else:
        component_ids = question_part_ids

    # This metadata is deterministic; the LLM is not asked to invent candidate parts.
    selection = {
        "identified_component_ids": component_ids,
        "identification_source": "parsed schematic/netlist" if schematic_result else "explicit user question",
        "note": "These are search targets, not proof that any datasheet rating is compatible with the circuit.",
    }

    if component_ids:
        print("Question route: strict component-specific datasheet retrieval")
        result = retrieve(
            raw_requirements,
            k=5,
            component_ids=component_ids,
            strict_component_filter=True,
        )
        rag_result = _merge_rag_results([result], raw_requirements, component_ids) if schematic_result else result
        print("Strict component-specific RAG found relevant chunks:", rag_result.get("found", False))
        if not rag_result.get("found", False):
            print("WARNING:", rag_result.get("warning"))
    elif schematic_result:
        rag_result = _empty_rag_result(
            raw_requirements,
            "No datasheet-searchable component part numbers were found in the provided design.",
            "strict_component_match",
        )
    else:
        # Only non-board-specific, non-part-specific educational questions use broad semantic search.
        rag_result = retrieve(raw_requirements, k=5)
        print("Question route: general educational question; general RAG found chunks:", rag_result.get("found", False))

    if rag_result.get("filter_mode") == "strict_component_match":
        missing_parts = _missing_component_datasheets(rag_result)
        if not rag_result.get("found") or missing_parts:
            gate_reason = (
                "No component-matched datasheet chunks were retrieved."
                if not rag_result.get("found")
                else "At least one explicitly identified component has no matching datasheet source."
            )
            if schematic_result:
                return _structural_only_report(
                    schematic_result,
                    missing_parts or component_ids,
                    gate_reason,
                )
            missing = ", ".join(missing_parts or component_ids) or "the requested component"
            return (
                f"DATASHEET GATE: BLOCKED — no grounded device-specific answer generated.\n\n"
                f"No relevant, part-matched datasheet passages were found for: {missing}.\n"
                "No unrelated datasheet was substituted. Add the exact manufacturer's datasheet to "
                "data/datasheets/, run `python ingest.py` to rebuild document identity metadata, and ask again."
            )

    if schematic_result and not component_ids:
        return _structural_only_report(
            schematic_result,
            [],
            "No specific manufacturer part identifiers were identified in the schematic.",
        )

    if not schematic_result and not rag_result.get("found"):
        return (
            "No relevant datasheet evidence was retrieved for this question.\n\n"
            "I did not ask the reasoning model to fill the gap from memory. Add the exact manufacturer datasheet to "
            "data/datasheets/, run `python ingest.py`, and retry."
        )

    # Generate once and verify once; do not hide failed checks through blind regeneration.
    print(
        "DEBUG: Reasoning inputs | "
        f"retrieved_context_chars={len(rag_result.get('context', ''))} | "
        f"components={len((schematic_result or {}).get('components', []))} | "
        f"findings={len((schematic_result or {}).get('findings', []))} | "
        f"datasheet_sources={len(rag_result.get('sources', []))}"
    )
    answer = reason(
        selection,
        rag_result,
        original_query=raw_requirements,
        schematic_result=schematic_result,
    )
    verdict = verify(answer, rag_result, schematic_result=schematic_result)
    print(f"Verification passed={verdict.get('passed')}")
    print("Verification details:", verdict)

    if not verdict.get("passed", False):
        answer += (
            "\n\n--- QUALITY GATE ---\n"
            "STATUS: UNVERIFIED — do not treat this response as an electrical sign-off.\n"
            f"Reasons: {_verification_feedback(verdict)}\n"
            "Review the source-matching and evidence gaps before relying on datasheet-specific ratings."
        )
    if schematic_result and not rag_result.get("found"):
        answer += (
            "\n\n--- EVIDENCE LIMITATION ---\n"
            "No exact component-matched datasheet passages were found. This report is limited to observations from the supplied netlist and preliminary review prompts; all part-specific electrical ratings remain unverified."
        )
    return answer


if __name__ == "__main__":
    print("\n========================================")
    print("          DESIGN ASSISTANT")
    print("========================================")
    if not _knowledge_base_ready():
        print("\nNo processed knowledge base was found.\nRun `python ingest.py` after adding datasheets, then `python pipeline.py`.")
    else:
        print("\nVector database found. Datasheet retrieval uses source/component identity checks.")
        print("Provide any supported exported netlist or a KiCad schematic path when prompted.")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_requirements = input("Describe your design requirements: ").strip()
                if user_requirements.lower() in {"exit", "quit"}:
                    print("\nExiting Design Assistant...")
                    break
                if not user_requirements:
                    continue
                schematic_path = input("Optional schematic/netlist path (Enter to skip): ").strip()
                print("\n" + "=" * 70 + "\nProcessing query...\n" + "=" * 70 + "\n")
                try:
                    answer = run_pipeline(user_requirements, schematic_path or None)
                    print("\n=== Design Assistant output ===\n")
                    print(answer)
                except Exception as exc:
                    print(f"\nError while processing query: {exc}")
                    print("You can continue asking another question.")
                print("\n" + "=" * 70 + "\n")
            except KeyboardInterrupt:
                print("\n\nExiting Design Assistant...")
                break
