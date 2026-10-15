"""Deterministic requirements extraction; no model-invented electrical values."""
from __future__ import annotations

import re

_ELECTRICAL_FIELDS = (
    "input_voltage",
    "output_voltage",
    "current_limit",
    "frequency",
)
_UNKNOWN = "unknown"

def _extract_explicit_electrical_fields(original_text: str) -> dict[str, str]:
    """Extract electrical values only when the field is explicitly labelled in user text.

    This avoids trusting the LLM to assign a bare number to input voltage,
    output voltage, current, or frequency. Unlabelled values remain unknown.
    """
    voltage_value = r"([+-]?\d+(?:\.\d+)?\s*(?:mV|uV|µV|V))"
    current_value = r"([+-]?\d+(?:\.\d+)?\s*(?:mA|uA|µA|A))"
    frequency_value = r"([+-]?\d+(?:\.\d+)?\s*(?:kHz|MHz|GHz|Hz))"
    patterns = {
        "input_voltage": [
            rf"\b(?:input|vin|power\s*input|input\s*supply)\s*(?:voltage|supply|is|of|=|:)?\s*(?:is|of|=|:)?\s*{voltage_value}",
            rf"{voltage_value}\s*(?:input\s*voltage|input|vin)\b",
        ],
        "output_voltage": [
            rf"\b(?:output|vout|output\s*supply)\s*(?:voltage|is|of|=|:)?\s*(?:is|of|=|:)?\s*{voltage_value}",
            rf"{voltage_value}\s*(?:output\s*voltage|output|vout)\b",
        ],
        "current_limit": [
            rf"\b(?:maximum|max\.?|min(?:imum)?|limit(?:ed)?|rated|required)?\s*(?:input\s+|output\s+|load\s+)?current\s*(?:limit|rating|is|of|=|:|up\s+to)?\s*(?:is|of|=|:|up\s+to)?\s*{current_value}",
            rf"{current_value}\s*(?:max(?:imum)?|current\s*limit)\b",
        ],
        "frequency": [
            rf"\b(?:clock\s+|operating\s+|switching\s+|signal\s+)?frequency\s*(?:is|of|=|:)?\s*{frequency_value}",
            rf"\b(?:clock|operating|switching)\s+frequency\s*(?:is|of|=|:)?\s*{frequency_value}",
        ],
    }
    result = {field: _UNKNOWN for field in _ELECTRICAL_FIELDS}
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, original_text, flags=re.IGNORECASE)
            if match:
                # The final captured group is the value, even when the pattern
                # also has a capture for the optional unit group.
                value = next((group.strip() for group in reversed(match.groups()) if group and re.search(r"\d", group)), "")
                if value:
                    result[field] = re.sub(r"\s+", "", value)
                    break
    return result



def _normalize_electrical_constraints(original_text: str) -> dict[str, str]:
    """Only assign numeric electrical values when the field is labelled in the user text."""
    return _extract_explicit_electrical_fields(original_text)


def collect_requirements(raw_requirements: str) -> dict:
    """Return conservative structured notes without an LLM call or assumptions."""
    raw_requirements = (raw_requirements or "").strip()
    if not raw_requirements:
        raise ValueError("Requirements cannot be empty.")
    component_ids = []
    component_pattern = re.compile(
        r"\b(?:ESP32(?:-[A-Z0-9]+){1,5}|"
        r"(?:AP|LM|LT|LTC|TL|OPA|MCP|INA|ADS|MAX|TPS|AMS|STM32|ATmega|BME|BMP|MPU|ICM|NRF|RP)[A-Z0-9]*(?:-[A-Z0-9.]+)*)\b",
        re.IGNORECASE,
    )
    for match in component_pattern.finditer(raw_requirements):
        item = match.group(0)
        if item.casefold() not in {value.casefold() for value in component_ids}:
            component_ids.append(item)
    return {
        "project_summary": raw_requirements,
        "functional_requirements": [],
        "electrical_constraints": _normalize_electrical_constraints(raw_requirements),
        "mechanical_constraints": [],
        "component_hints": component_ids,
        "note": "Only explicitly labelled electrical values are populated; unspecified fields remain unknown.",
    }
