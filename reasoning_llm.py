"""Reasoning stage for datasheet-grounded answers."""

from ollama_utils import chat_text
from models import REASONING_MODEL

SYSTEM_PROMPT = """You are the Reasoning stage of an electronics/PCB design assistant.

You receive:
1. The original user question.
2. A component selection and knowledge-graph-derived context.
3. Datasheet excerpts retrieved from the local document store.

Produce a clear, engineer-readable answer.

Rules:
- Answer the original question directly when the retrieved datasheet context supports it.
- Treat numeric values, units, part numbers, and Min/Typ/Max values as source-controlled facts.
- Do not invent, normalize, or replace values.
- Do not mix values between part-number/device columns.
- If the evidence is insufficient or ambiguous, say so.
- Cite the supplied source information in a concise way when possible.
- Flag contradictions between requirements and retrieved specifications.
- List open questions only when they are genuinely unresolved.
"""


def reason(
    selection: dict,
    rag_result: dict,
    feedback: str | None = None,
    original_query: str | None = None,
) -> str:
    context = (
        rag_result["context"]
        if rag_result["found"]
        else "(no supporting datasheet excerpts found)"
    )

    user_content = (
        f"ORIGINAL USER QUESTION:\n{original_query or '(not provided)'}\n\n"
        f"COMPONENT SELECTION:\n{selection}\n\n"
        f"RETRIEVED DATASHEET CONTEXT:\n{context}\n\n"
        f"SOURCES:\n{rag_result.get('sources', [])}"
    )

    if feedback:
        user_content += (
            f"\n\nYour previous answer failed verification for this reason:\n{feedback}\n"
            "Revise the answer so it stays strictly grounded in the retrieved context above."
        )

    return chat_text(REASONING_MODEL, SYSTEM_PROMPT, user_content)
