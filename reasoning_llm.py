"""Focused reasoning for datasheet-grounded electronics design reviews."""

import re

from ollama_utils import chat_text
from models import REASONING_MODEL
from schematic_checker import format_findings_for_llm


SYSTEM_PROMPT = """You are an electronics design engineer. Answer the user's engineering question directly.

Do not start with phrases such as "To determine whether...", "We need to verify...", "The retrieved data indicates...", or an explanation of the retrieval process. Start with a concise verdict that answers the question.

For a power/regulator review, use:
1. Verdict: SUITABLE, NOT SUITABLE, or REVIEW REQUIRED.
2. A short table: Parameter | Value and conditions | Datasheet/page | Status.
3. Calculation of linear-regulator power loss when input/output/load current are available.
4. A concise conclusion separating what is supported, what is a thermal/load risk, and what cannot be concluded.

Rules:
- Use only values present in supplied excerpts or directly calculable from them and the user-provided design.
- Cite PDF filename and the printed page number from each [SOURCE: ... page N ...] header.
- Keep Min/Typ/Max and test conditions distinct.
- Do not use an AP2112-2.5 or AP2112-2.6 electrical-characteristics table for an AP2112-3.3 specification. If a 3.3V table is present, use it. Other-voltage tables should have been filtered out before this call.
- Do not conflate the ESP32-S3 chip with the ESP32-S3-WROOM-1 module.
- For the ESP32-S3-WROOM-1 supply, distinguish its recommended supply-voltage range from its external-supply current recommendation and from particular measured/characterized Wi-Fi TX peak currents.
- AP2112's 600mA maximum current rating is not, by itself, proof that it can thermally sustain the entire load on the target PCB.
- For a linear regulator, calculate P = (VIN - VOUT) * IOUT. State the chosen current and whether it is measured, specified, or an assumption.
- If using thermal resistance, state the package, the stated thermal-resistance value, ambient assumption, and that actual PCB copper/layout affects thermal performance. An estimate is not PCB sign-off.
- Never assume an input voltage. Use VIN only when explicitly stated by the user or clearly identified as a named rail in the supplied netlist; otherwise report it as unknown. The nominal output is 3.3 V only when the exact selected regulator variant supports that value. Do not mark the board safe unless thermal and load evidence supports it.
- A schematic/netlist demonstrates listed connectivity, not electrical performance. INFO findings are not confirmed faults.
- Do not discuss embedding scores, document matching counts, or the process of finding data. Do not merely summarize the documents.
- If a requested value is absent after the curated evidence, mark only that parameter UNVERIFIED and still answer all the other parts you can support.
- Keep the answer concise and actionable. Do not repeat these rules in the response.
"""


def _concise_sources(sources):
    result = []
    for source in sources or []:
        result.append({
            key: source.get(key)
            for key in (
                "source", "page", "content_type", "table_index",
                "matched_component"
            )
            if source.get(key) is not None
        })
    return result


def _clean_context_for_llm(context: str) -> str:
    """FIX 2: keep only '[SOURCE: file; page N]' so the model never sees
    relevance scores or filter text it could copy into its answer."""
    return re.sub(
        r"(\[SOURCE:[^\];]+;\s*page\s+\d+)[^\]]*\]",
        r"\1]",
        context or "",
        flags=re.IGNORECASE,
    )


def reason(
    selection: dict,
    rag_result: dict,
    feedback: str | None = None,
    original_query: str | None = None,
    schematic_result: dict | None = None,
) -> str:
    context = (
        rag_result.get("context", "")
        if rag_result.get("found")
        else "(no supporting datasheet excerpts found)"
    )
    context = _clean_context_for_llm(context)
    schematic_context = (
        format_findings_for_llm(schematic_result)
        if schematic_result
        else "(no schematic/netlist supplied)"
    )

    coverage = rag_result.get("power_evidence_coverage")
    coverage_text = (
        f"\nPOWER EVIDENCE COVERAGE FLAGS: {coverage}"
        if coverage is not None else ""
    )

    user_content = (
        f"USER QUESTION:\n{original_query or '(not provided)'}\n\n"
        f"TARGET COMPONENTS:\n{selection.get('identified_component_ids', selection)}\n\n"
        f"SCHEMATIC / NETLIST FACTS:\n{schematic_context}\n\n"
        f"CURATED DATASHEET EVIDENCE:\n{context}\n\n"
        f"SOURCE INDEX (concise):\n{_concise_sources(rag_result.get('sources', []))}"
        f"{coverage_text}\n\n"
        "Now answer the user's question directly. The verdict must reflect the actual evidence. "
        "Do not turn an evidence limitation into the main answer if other parameters are supported."
    )

    if feedback:
        user_content += (
            f"\n\nVERIFICATION FEEDBACK:\n{feedback}\n"
            "Correct unsupported or contradicted claims, but continue answering the user's actual question."
        )

    return chat_text(REASONING_MODEL, SYSTEM_PROMPT, user_content)