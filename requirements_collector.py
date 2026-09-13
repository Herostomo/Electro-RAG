"""Requirements collector model."""

from ollama_utils import chat_json
from models import EXTRACTION_MODEL

SYSTEM_PROMPT = """You are a Requirements Collector for an electronics/PCB design assistant.

Given raw free-text requirements from a user, extract a structured specification.

Return STRICT JSON with keys:
- project_summary: string
- functional_requirements: list of strings
- electrical_constraints: object
- mechanical_constraints: list of strings
- component_hints: list of strings

Rules:
- Preserve explicit values, units, part numbers, ranges, and device names exactly.
- Never invent, reinterpret, or substitute a numerical value stated by the user.
- For a factual datasheet question, keep the requested component/parameter as a component hint when possible.
- Only include constraints actually mentioned or reasonably implied.
"""

def collect_requirements(raw_requirements: str) -> dict:
    return chat_json(EXTRACTION_MODEL, SYSTEM_PROMPT, raw_requirements)
