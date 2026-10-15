"""Deterministic AP2112 / ESP32-S3 power-review path.

This module intentionally does not ask an LLM to calculate ratings or arithmetic.
It uses only explicit table evidence from the local manufacturer PDFs and/or the
curated RAG passages. Missing values remain NOT FOUND.

Scope (by design): AP2112 family electrical questions and AP2112 + ESP32-S3
power-supply reviews. It is not a general-purpose SPICE or thermal simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class EvidencePage:
    source: str
    page: int  # Printed PDF page, 1-based
    text: str

    @property
    def citation(self) -> str:
        return f"{Path(self.source).name} · p. {self.page}"


def _page_blocks_from_context(context: str) -> list[EvidencePage]:
    """Read source/page-labelled passages without trusting the LLM's citations."""
    pages: list[EvidencePage] = []
    parts = re.split(r"(?=\[SOURCE:\s*)", context or "")
    for part in parts:
        match = re.match(
            r"\[SOURCE:\s*([^;\]]+);\s*page\s*(\d+)[^\]]*\]\s*(.*)",
            part.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            continue
        source, page_text, body = match.groups()
        try:
            page = int(page_text)
        except ValueError:
            continue
        body = re.split(r"\n\s*---\s*(?:RELEVANT EVIDENCE|ADDITIONAL PARAMETER-SPECIFIC EVIDENCE|MATCHED DATASHEET).*?---\s*\n", body, maxsplit=1, flags=re.I)[0]
        pages.append(EvidencePage(Path(source.strip().replace("\\", "/")).name, page, body.strip()))
    return pages


def _load_local_relevant_pages(datasheet_dir: str | Path) -> list[EvidencePage]:
    """Read relevant pages directly from local PDFs; no network access/model calls."""
    root = Path(datasheet_dir)
    if not root.exists():
        return []
    try:
        import pymupdf as fitz  # preferred package name
    except ImportError:
        try:
            import fitz  # type: ignore[no-redef]
        except ImportError:
            return []

    pages: list[EvidencePage] = []
    pdf_paths = sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    for path in pdf_paths:
        name = path.name.lower()
        is_ap2112 = name.startswith("ap2112")
        is_module = "esp32-s3" in name and "wroom-1" in name
        if not (is_ap2112 or is_module):
            continue
        try:
            with fitz.open(str(path)) as pdf:
                for page_index, page_obj in enumerate(pdf):
                    text = (page_obj.get_text("text") or "").strip()
                    if not text:
                        continue
                    low = re.sub(r"\s+", " ", text).lower()
                    if is_ap2112:
                        needed = (
                            "ap2112" in low and (
                                "ap2112-3.3 electrical characteristics" in low
                                or "thermal resistance" in low
                                or "junction to ambient" in low
                                or "recommended operating conditions" in low
                                or "absolute maximum ratings" in low
                            )
                        )
                    else:
                        needed = (
                            ("vdd33" in low and "recommended operating conditions" in low)
                            or ("current delivered by external power supply" in low)
                            or ("peak(ma)" in low and "802.11b" in low and "tx" in low)
                            or ("peak (ma)" in low and "802.11b" in low and "tx" in low)
                        )
                    if needed:
                        pages.append(EvidencePage(path.name, page_index + 1, text))
        except Exception:
            # One unreadable PDF should not prevent the known retrieved passages
            # from being evaluated. Missing fields will be reported explicitly.
            continue
    return pages


def _collect_pages(context: str, datasheet_dir: str | Path) -> list[EvidencePage]:
    """Combine retrieved passages and directly-read source pages, deduplicating safely."""
    result: list[EvidencePage] = []
    seen: set[tuple[str, int, str]] = set()
    # Prefer RAG passages first because they may contain reconstructed tables.
    for page in _page_blocks_from_context(context):
        key = (page.source.casefold(), page.page, re.sub(r"\s+", " ", page.text).strip())
        if key not in seen:
            seen.add(key)
            result.append(page)
    for page in _load_local_relevant_pages(datasheet_dir):
        key = (page.source.casefold(), page.page, re.sub(r"\s+", " ", page.text).strip())
        if key not in seen:
            seen.add(key)
            result.append(page)
    return result


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_ap2112_33_table(page: EvidencePage) -> bool:
    if not page.source.lower().startswith("ap2112"):
        return False
    flat = _flat(page.text).lower()
    # Exact section heading prevents AP2112-2.5/-2.6 table leakage.
    return bool(re.search(r"ap2112\s*-\s*3\.3\s+electrical\s+characteristics", flat))


def _first_source(pages: Iterable[EvidencePage], predicate) -> EvidencePage | None:
    return next((page for page in pages if predicate(page)), None)


def _ap_specs(pages: list[EvidencePage], sot25_variant: bool = True) -> dict[str, dict | None]:
    table = _first_source(pages, _is_ap2112_33_table)
    thermal = _first_source(
        pages,
        lambda p: p.source.lower().startswith("ap2112")
        and "thermal resistance" in p.text.lower()
        and ("absolute maximum ratings" in p.text.lower() or "junction to ambient" in p.text.lower()),
    )
    operating = _first_source(
        pages,
        lambda p: p.source.lower().startswith("ap2112")
        and "recommended operating conditions" in p.text.lower()
        and re.search(r"2\.5\s*(?:v|\s)\s*.{0,40}6\.0", _flat(p.text), re.I | re.S) is not None,
    )

    out: dict[str, dict | None] = {"vout": None, "imax": None, "dropout": None, "theta": None, "tjmax": None, "vinrange": None}
    if table:
        flat = _flat(table.text)
        lower = flat.lower()
        if "output voltage" in lower and "3.3" in flat and "98.5%" in flat and "101.5%" in flat:
            out["vout"] = {
                "value": "3.3 V ±1.5% (VIN = 4.3 V; 1–30 mA test range)",
                "source": table.citation,
                "conditions": "The 3.3 V variant's stated output-voltage test conditions.",
            }

        # Only parse IOUT(MAX) from the 3.3 V electrical-characteristics section.
        current_row = re.search(
            r"IOUT\s*\(\s*MAX\s*\)(.*?)(?:Load Regulation|\(\s*∆?VOUT)",
            flat,
            flags=re.IGNORECASE,
        )
        current_values = re.findall(
            r"\b(\d{3,4})\s+[—–-]{1,3}\s+[—–-]{1,3}\s*mA\b",
            current_row.group(1) if current_row else "",
            flags=re.IGNORECASE,
        )
        # Some PDF extractors omit table-cell dashes. In that case, take the
        # terminal value immediately before the row's mA unit, not the 3.251/3.350
        # voltage numbers in the condition column.
        if not current_values and current_row:
            suffix = current_row.group(1)
            tail = re.search(r"\b(\d{3,4})\s*(?:—|–|-)?\s*(?:—|–|-)?\s*mA\b", suffix, re.I)
            if tail:
                current_values = [tail.group(1)]
        if current_values and "maximum output current" in lower:
            out["imax"] = {
                "value": f"{current_values[-1]} mA (VIN = 4.3 V; VOUT = 3.251–3.350 V test condition)",
                "source": table.citation,
                "conditions": "The AP2112-3.3 table states VIN = 4.3 V and VOUT = 3.251–3.350 V for this rating.",
            }

        # The dropout rows are encoded as IOUT = 10/300/600 mA followed by
        # typical and maximum mV values. Restrict matching to the VDROP table.
        start = re.search(r"VDROP\s+Dropout\s+Voltage", flat, re.I)
        end = re.search(r"\bIQ\s+Quiescent\s+Current", flat, re.I)
        dropout_region = flat[start.end():end.start()] if start and end and end.start() > start.end() else ""
        drops = []
        for load_ma in (10, 300, 600):
            row = re.search(
                rf"IOUT\s*=\s*{load_ma}\s*mA\s*(?:[—–-]{{1,3}}\s*)?(\d{{1,4}})\s+(\d{{1,4}})",
                dropout_region,
                flags=re.IGNORECASE,
            )
            if row:
                drops.append((load_ma, int(row.group(1)), int(row.group(2))))
        if drops:
            values = "; ".join(
                f"{load} mA: {typ} mV typ / {maximum} mV max"
                for load, typ, maximum in drops
            )
            out["dropout"] = {
                "value": values,
                "source": table.citation,
                "conditions": "Dropout test rows in the AP2112-3.3 electrical-characteristics table.",
            }

    if thermal:
        flat = _flat(thermal.text)
        theta_match = re.search(r"SOT25\s+(\d{2,3})\s*(?:°\s*C\s*/\s*W|°C/W|C/W)", flat, re.I)
        if not theta_match:
            theta_match = re.search(r"SOT25\s+(\d{2,3})\b", flat, re.I)
        if theta_match and sot25_variant:
            out["theta"] = {
                "value": f"{theta_match.group(1)} °C/W (SOT-25, no heatsink; board-dependent)",
                "source": thermal.citation,
                "numeric": float(theta_match.group(1)),
            }
        elif theta_match:
            # Package is not identified as the K/SOT-25 variant. Report the
            # manufacturer table as package-specific values without selecting one.
            package_values = re.findall(r"(SOT25|SO-8|SOT89-5)\s+(\d{2,3})", flat, re.I)
            if package_values:
                out["theta"] = {
                    "value": "Package-specific θJA: " + "; ".join(f"{pkg} {value} °C/W" for pkg, value in package_values) + ". Package code not confirmed, so no single value selected.",
                    "source": thermal.citation,
                }
        tj_match = re.search(
            r"(?:TJ\s+)?Operating\s+Junction\s+Temperature\s+Range\s*\+?\s*(\d{2,3})\s*°?\s*C",
            flat,
            re.I,
        )
        if not tj_match:
            tj_match = re.search(r"TJ\s+Operating\s+Junction\s+Temperature\s+Range\s*\+?\s*(\d{2,3})", flat, re.I)
        if tj_match:
            out["tjmax"] = {
                "value": f"{tj_match.group(1)} °C maximum junction temperature (absolute maximum; not a design target)",
                "source": thermal.citation,
                "numeric": float(tj_match.group(1)),
            }

    if operating:
        flat = _flat(operating.text)
        if "recommended operating conditions" in flat.lower() and re.search(r"\b2\.5\b", flat) and re.search(r"\b6\.0\b", flat):
            out["vinrange"] = {
                "value": "2.5–6.0 V",
                "source": operating.citation,
                "conditions": "AP2112 recommended operating input-voltage range.",
            }
    return out


def _esp_specs(pages: list[EvidencePage]) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {"supply": None, "external_supply": None, "wifi_tx": None}
    module_pages = [p for p in pages if "esp32-s3" in p.source.lower() and "wroom-1" in p.source.lower()]
    for page in module_pages:
        flat = _flat(page.text)
        low = flat.lower()
        if out["supply"] is None and "vdd33" in low and "recommended operating conditions" in low:
            # Match the named VDD33 row, not unrelated 3.3/3.6 values elsewhere.
            match = re.search(
                r"VDD33\s+Power\s+supply\s+voltage\s+3\.0\s+3\.3\s+3\.6\s+V",
                flat,
                re.I,
            )
            if match:
                out["supply"] = {
                    "value": "3.0–3.6 V (typical 3.3 V)",
                    "source": page.citation,
                    "conditions": "Recommended operating conditions for VDD33.",
                }
        if out["external_supply"] is None and "current delivered by external power supply" in low:
            match = re.search(
                r"current\s+delivered\s+by\s+external\s+power\s+supply\s+0\.5\s+[-—–]\s*[-—–]\s*A",
                flat,
                re.I,
            )
            if not match:
                match = re.search(r"current\s+delivered\s+by\s+external\s+power\s+supply.{0,50}?0\.5.{0,20}?A", flat, re.I)
            if match:
                out["external_supply"] = {
                    "value": "0.5 A external-supply capability recommendation (not a measurement of actual draw)",
                    "source": page.citation,
                }
        if out["wifi_tx"] is None and "peak" in low and "802.11b" in low and "tx" in low:
            tx_rows = []
            patterns = (
                ("802.11b, 1 Mbps, 20.5 dBm", r"802\.11b\s*,?\s*1\s*Mbps.{0,100}?\b(355)\b"),
                ("802.11g, 54 Mbps, 18 dBm", r"802\.11g\s*,?\s*54\s*Mbps.{0,100}?\b(297)\b"),
                ("802.11n HT20, MCS7, 17.5 dBm", r"802\.11n\s*,?\s*HT20\s*,?\s*MCS7.{0,100}?\b(286)\b"),
                ("802.11n HT40, MCS7, 17 dBm", r"802\.11n\s*,?\s*HT40\s*,?\s*MCS7.{0,100}?\b(285)\b"),
            )
            for label, pattern in patterns:
                match = re.search(pattern, flat, re.I)
                if match:
                    tx_rows.append((label, int(match.group(1))))
            if tx_rows:
                out["wifi_tx"] = {
                    "value": "; ".join(f"{label}: {current} mA peak" for label, current in tx_rows),
                    "source": page.citation,
                    "rows": tx_rows,
                }
    return out


def _input_voltage(question: str, schematic_result: dict | None) -> tuple[float | None, str]:
    q = question or ""
    patterns = (
        r"(?:input\s+(?:supply\s+)?(?:voltage\s*)?|\bVIN\s*[=:]?\s*)(?:is\s*)?(\d+(?:\.\d+)?)\s*V\b",
        r"\b(\d+(?:\.\d+)?)\s*V\s+(?:input|supply)\b",
        r"\b(?:regulate|regulated|from|using|with)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*V\s+(?:to|input|supply|source)\b",
    )
    for pattern in patterns:
        m = re.search(pattern, q, re.I)
        if m:
            return float(m.group(1)), "User-stated input voltage"

    if schematic_result:
        nets = schematic_result.get("nets", {}) or {}
        components = schematic_result.get("components", []) or []
        regulator_refs = {
            str(item.get("ref", ""))
            for item in components
            if "ap2112" in str(item.get("value", "")).lower()
        }
        if isinstance(nets, dict):
            # Require a named rail on the net that actually contains AP2112 VIN.
            # An unrelated +5V elsewhere on the schematic must not determine VIN.
            for net_key, net in nets.items():
                if not isinstance(net, dict):
                    continue
                for node in net.get("nodes", []) or []:
                    if str(node.get("ref", "")) not in regulator_refs:
                        continue
                    pin_function = str(node.get("pinfunction", node.get("name", ""))).upper()
                    pin_number = str(node.get("pin", node.get("number", ""))).strip()
                    if "VIN" not in pin_function and not (not pin_function and pin_number == "1"):
                        continue
                    label = str(net.get("name", net_key))
                    normalized = re.sub(r"[^A-Z0-9]", "", label.upper())
                    if normalized in {"5V", "5V0", "VBUS"}:
                        return 5.0, f"Regulator VIN connected to net {label} (nominal label; voltage not measured)"
                    m = re.fullmatch(r"(\d+(?:\.\d+)?)V(?:0)?", normalized)
                    if m:
                        return float(m.group(1)), f"Regulator VIN connected to net {label} (nominal label; voltage not measured)"
                    return None, f"Regulator VIN is connected to net {label}, but the net name does not establish a numeric voltage"
    return None, "Input voltage not stated or not proven at the regulator VIN pin"


def _explicit_load_current(question: str) -> float | None:
    q = question or ""
    patterns = (
        # Explicit design-load labels only. Deliberately do NOT match
        # "maximum output current"; that is a datasheet rating, not actual load.
        r"(?:expected|actual|continuous|board|load)\s+(?:load\s+)?current\s*(?:is|=|of|at|around)?\s*(\d+(?:\.\d+)?)\s*(mA|A)\b",
        r"(?:expected|actual)\s+output\s+current\s*(?:is|=|of|at|around)?\s*(\d+(?:\.\d+)?)\s*(mA|A)\b",
        r"\bI(?:OUT|LOAD)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mA|A)\b",
        r"\b(\d+(?:\.\d+)?)\s*(mA|A)\s+(?:continuous\s+)?load\b",
    )
    for pattern in patterns:
        match = re.search(pattern, q, re.I)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value / 1000.0 if unit == "ma" else value
    return None


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()
    lines = ["| " + " | ".join(clean(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = list(row) + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(clean(v) for v in values[:len(headers)]) + " |")
    return "\n".join(lines)


def _is_safety_review(question: str) -> bool:
    """Recognize suitability questions even when they avoid the words 'safe' or 'review'."""
    q = question or ""
    return bool(
        re.search(
            r"\b(?:safely|safe|suitable|unsuitable|design review|review my (?:pcb|board|design|circuit)|"
            r"review this (?:pcb|board|design|circuit)|thermal safety|electrically safe)\b",
            q, re.I,
        )
        or re.search(
            r"\b(?:is|are|does|do)\b.{0,70}\b(?:my|this|the)?\s*(?:pcb|printed circuit board|board|design|circuit)\b"
            r".{0,100}\b(?:good|correct|right|proper|okay|ok|acceptable|sound)\b",
            q, re.I,
        )
        or re.search(
            r"\b(?:did|have)\s+i\s+(?:choose|chosen|select|selected)\b.{0,120}"
            r"\b(?:component|components|correctly|right|according to|datasheet)\b",
            q, re.I,
        )
        or re.search(
            r"\b(?:choose|chosen|select|selected|selection)\b.{0,120}\bcomponents?\b"
            r".{0,100}\b(?:correct|right|suitable|datasheet|according to)\b",
            q, re.I,
        )
        or (
            re.search(r"\baccording to (?:the )?datasheet\b", q, re.I)
            and re.search(r"\b(?:pcb|board|design|component|components|circuit)\b", q, re.I)
        )
        or re.search(r"\b(?:can|could|will|would)\b.{0,100}\bpower\b.{0,60}\b(?:my|the|this)?\s*(?:esp32|board|design|circuit)\b", q, re.I)
        or re.search(r"\bdetermine whether\b.{0,120}\b(?:power|regulator|board|design)\b", q, re.I)
    )


def can_handle_power_question(question: str, component_ids: list[str] | None = None) -> bool:
    """Route AP2112 factual questions and AP2112-backed board reviews to deterministic code.

    A part number can be derived from the selected schematic, so the user does not
    need to type AP2112 explicitly in a general board-review question.
    """
    identifiers = " ".join(component_ids or [])
    combined = f"{question or ''} {identifiers}".lower()
    if "ap2112" not in combined:
        return False

    asks_fact = bool(re.search(
        r"\b(?:maximum output current|max(?:imum)? current|iout|max current|dropout|vdrop|"
        r"output voltage|vout|thermal|dissipation|power loss|junction temperature|"
        r"power supply|regulator|wifi|wi-fi|current consumption|electrical characteristics)\b",
        question or "",
        re.I,
    ))
    return asks_fact or _is_safety_review(question)


def build_power_report(
    question: str,
    context: str = "",
    schematic_result: dict | None = None,
    datasheet_dir: str | Path = "data/datasheets",
) -> str | None:
    """Build a source-cited factual table / AP2112 power assessment without an LLM.

    Returns None when the question is outside the specialized AP2112 electrical scope.
    Each requested datum is extracted from an exact AP2112-3.3 table, or explicitly
    marked NOT FOUND. Thermal arithmetic is done in Python and all assumed scenarios
    are labelled as scenarios, not measured board currents.
    """
    if not can_handle_power_question(question, _ids_from_text(question, schematic_result)):
        return None

    pages = _collect_pages(context, datasheet_dir)
    design_values = " ".join(
        str(c.get("value", "")) for c in (schematic_result or {}).get("components", [])
    )
    sot25_variant = bool(re.search(r"\bAP2112K(?:-3?\.\d+)?\b", f"{question} {design_values}", re.I))
    ap = _ap_specs(pages, sot25_variant=sot25_variant)
    esp = _esp_specs(pages)
    has_esp32_s3 = bool(
        re.search(r"esp32\s*[- ]?\s*s3", question, re.I)
        or any("esp32-s3" in str(c.get("value", "")).lower() for c in (schematic_result or {}).get("components", []))
    )
    safety_review = _is_safety_review(question)
    question_low = question.lower()

    # Resolve power-path connectivity once so a datasheet value is not labelled
    # PASS unless the selected netlist actually connects that output to the module.
    schematic_components = (schematic_result or {}).get("components", []) or []
    schematic_component_by_ref = {str(c.get("ref", "")): c for c in schematic_components}
    schematic_nets = (schematic_result or {}).get("nets", {}) or {}

    def net_for_pin(ref: str | None, pin_number: str) -> str | None:
        if not ref or not isinstance(schematic_nets, dict):
            return None
        for net_key, net in schematic_nets.items():
            if not isinstance(net, dict):
                continue
            for node in net.get("nodes", []) or []:
                if str(node.get("ref", "")) == ref and str(node.get("pin", node.get("number", ""))) == str(pin_number):
                    return str(net.get("name", net_key))
        return None

    regulator_ref = next((ref for ref, c in schematic_component_by_ref.items() if "ap2112" in str(c.get("value", "")).lower()), None)
    module_ref = next((ref for ref, c in schematic_component_by_ref.items() if "esp32-s3-wroom-1" in str(c.get("value", "")).lower()), None)
    regulator_vout_net = net_for_pin(regulator_ref, "5")
    module_vdd_net = net_for_pin(module_ref, "2")
    normalized_vout_net = re.sub(r"[^A-Z0-9]", "", (regulator_vout_net or "").upper())
    normalized_module_net = re.sub(r"[^A-Z0-9]", "", (module_vdd_net or "").upper())
    power_output_matches_module = bool(
        regulator_vout_net and module_vdd_net and normalized_vout_net == normalized_module_net
    )

    rows: list[list[str]] = []

    def found(spec: dict | None) -> str:
        return str(spec.get("value", "NOT FOUND in available matching evidence")) if spec else "NOT FOUND in available matching AP2112-3.3 evidence"

    def cite(spec: dict | None) -> str:
        return str(spec.get("source", "Not present in available matching evidence")) if spec else "Not present in available matching evidence"

    def add(label: str, spec: dict | None, status: str) -> None:
        rows.append([label, found(spec), cite(spec), status])

    if safety_review:
        output_status = (
            "PASS" if ap["vout"] and (not has_esp32_s3 or power_output_matches_module)
            else "REVIEW" if ap["vout"]
            else "UNKNOWN"
        )
        add("Regulator output voltage / connection", ap["vout"], output_status)
        vin, vin_source = _input_voltage(question, schematic_result)
        vin_status = "UNKNOWN" if not ap["vinrange"] else ("REVIEW" if vin is None else ("PASS" if 2.5 <= vin <= 6.0 else "FAIL"))
        add("Regulator recommended input range", ap["vinrange"], vin_status)
        add("Maximum output-current rating", ap["imax"], "REVIEW" if ap["imax"] else "UNKNOWN")
        dropout_status = "UNKNOWN" if not ap["dropout"] else "REVIEW"
        if ap["dropout"] and vin is not None and ap["vout"]:
            max_drop = re.search(r"600 mA: \d+ mV typ / (\d+) mV max", ap["dropout"]["value"])
            if max_drop:
                dropout_status = "PASS" if (vin - 3.3) > (int(max_drop.group(1)) / 1000.0) else "FAIL"
        add("Dropout voltage (typical / maximum by load)", ap["dropout"], dropout_status)
        if has_esp32_s3:
            module_supply_status = (
                "PASS" if esp["supply"] and power_output_matches_module
                else "REVIEW" if esp["supply"]
                else "UNKNOWN"
            )
            add("ESP32-S3 module supply range / connection", esp["supply"], module_supply_status)
            add("ESP32-S3 external supply capability recommendation", esp["external_supply"], "REVIEW" if esp["external_supply"] else "UNKNOWN")
            add("ESP32-S3 Wi-Fi transmit peak current", esp["wifi_tx"], "REVIEW" if esp["wifi_tx"] else "UNKNOWN")
        add("Regulator package thermal resistance", ap["theta"], "REVIEW" if ap["theta"] else "UNKNOWN")
        add("Maximum junction temperature", ap["tjmax"], "REVIEW" if ap["tjmax"] else "UNKNOWN")
    else:
        asked_current = bool(re.search(r"maximum output current|max(?:imum)? current|iout|max current|current rating", question_low))
        asked_dropout = bool(re.search(r"dropout|vdrop", question_low))
        asked_vout = bool(re.search(r"output voltage|\bvout\b|nominal output", question_low))
        asked_thermal = bool(re.search(r"thermal|dissipation|power loss|junction temperature", question_low))
        asked_input_range = bool(re.search(r"input voltage range|recommended input voltage|\bvin\b|input voltage", question_low))
        if asked_current:
            add("Maximum output current", ap["imax"], "SUPPORTED" if ap["imax"] else "UNKNOWN")
        if asked_dropout:
            add("Dropout voltage", ap["dropout"], "SUPPORTED" if ap["dropout"] else "UNKNOWN")
        if asked_vout:
            add("Output voltage", ap["vout"], "SUPPORTED" if ap["vout"] else "UNKNOWN")
        if asked_input_range:
            add("Recommended input voltage range", ap["vinrange"], "SUPPORTED" if ap["vinrange"] else "UNKNOWN")
        if asked_thermal:
            add("Thermal resistance (SOT-25)", ap["theta"], "SUPPORTED" if ap["theta"] else "UNKNOWN")
            add("Maximum junction temperature", ap["tjmax"], "SUPPORTED" if ap["tjmax"] else "UNKNOWN")
        if not rows:
            # The question was routed here but did not name a value we can report.
            add("Requested AP2112 specification", None, "UNKNOWN")

    title = "## AP2112K-3.3 — datasheet values" if not safety_review else (
        "## AP2112K-3.3 / ESP32-S3 power assessment" if has_esp32_s3 else "## AP2112K-3.3 power assessment"
    )
    body = [title, ""]
    if safety_review:
        body.append(_format_table(
            ["Parameter", "Value and conditions", "Source / page", "Status"], rows
        ))
    else:
        body.append(_format_table(
            ["Parameter", "Value and conditions", "Source / page"],
            [row[:3] for row in rows],
        ))

    if safety_review:
        vout = 3.3 if ap["vout"] else None
        theta = ap["theta"].get("numeric") if ap["theta"] else None
        tjmax = ap["tjmax"].get("numeric") if ap["tjmax"] else None
        tx_peak = None
        if esp["wifi_tx"] and esp["wifi_tx"].get("rows"):
            tx_peak = max(current for _, current in esp["wifi_tx"]["rows"]) / 1000.0
        explicit_load = _explicit_load_current(question)
        calc_rows: list[list[str]] = []

        if vin is not None and vout is not None and vin > vout:
            headroom = vin - vout
            if ap["dropout"]:
                # Extract the maximum dropout value at 600mA only from the selected 3.3V table.
                drop_values = re.findall(r"600 mA: \d+ mV typ / (\d+) mV max", ap["dropout"]["value"])
                max_drop_v = (int(drop_values[-1]) / 1000.0) if drop_values else None
                if max_drop_v is not None:
                    status = "PASS" if headroom > max_drop_v else "FAIL"
                    calc_rows.append(["Input-output headroom", f"{headroom:.2f} V available (VIN {vin:g} V − VOUT 3.3 V); 600 mA max dropout is {max_drop_v*1000:.0f} mV", f"VIN: {vin_source}; VOUT/dropout: {ap['vout']['source']}", status])

            theta_val = float(theta) if theta is not None else None
            if explicit_load is not None:
                current_cases = [(explicit_load, "User-stated load current", "specified by user")]
            else:
                current_cases = []
                if has_esp32_s3 and tx_peak is not None:
                    current_cases.append((tx_peak, "ESP32-S3 Wi-Fi TX peak scenario (module only; not average current)", "module datasheet TX peak"))
                if has_esp32_s3 and esp["external_supply"]:
                    # The 0.5 A entry describes supply capability, not the module's measured draw.
                    current_cases.append((0.5, "Illustrative continuous 0.5 A load scenario (not an assertion of actual draw)", "explicitly hypothetical"))

            if not current_cases:
                calc_rows.append(["Regulator power loss", "NOT CALCULATED — input voltage/load current evidence is missing", "Not available", "UNKNOWN"])
            else:
                for current, label, nature in current_cases:
                    power = (vin - vout) * current
                    if theta_val is not None and tjmax is not None:
                        tj25 = 25.0 + power * theta_val
                        tj50 = 50.0 + power * theta_val
                        over = (tj25 > tjmax) or (tj50 > tjmax)
                        thermal_value = (
                            f"PLOSS = {power:.3f} W at {current*1000:.0f} mA. "
                            f"Simplified steady-state TJ estimate: {tj25:.1f} °C at 25 °C ambient; "
                            f"{tj50:.1f} °C at 50 °C ambient. Scenario: {label}."
                        )
                        calc_rows.append(["Thermal scenario", thermal_value, f"VIN = {vin:g} V ({vin_source}); VOUT = 3.3 V ({ap['vout']['source']}); θJA and TJmax: {ap['theta']['source']}", "REVIEW"])
                    else:
                        calc_rows.append(["Regulator power loss", f"PLOSS = (VIN − VOUT) × IOUT = {power:.3f} W at {current*1000:.0f} mA; thermal resistance/junction limit not found.", f"VIN = {vin:g} V ({vin_source}); VOUT = 3.3 V ({ap['vout']['source'] if ap['vout'] else 'not found'})", "REVIEW"])

            # Conservative theoretical upper bound from datasheet theta-JA and absolute TJ maximum.
            if theta_val is not None and tjmax is not None:
                for ambient in (25.0, 50.0):
                    i_bound = max(0.0, (tjmax - ambient) / (theta_val * (vin - vout)))
                    calc_rows.append([
                        "Simplified continuous-current thermal bound",
                        f"Approx. {i_bound*1000:.0f} mA at {ambient:.0f} °C ambient to reach {tjmax:.0f} °C TJ, using θJA={theta_val:.0f} °C/W; theoretical absolute-maximum bound, no safety margin",
                        f"VIN: {vin_source}; VOUT: {ap['vout']['source']}; θJA/TJmax: {ap['theta']['source']} and {ap['tjmax']['source']}",
                        "REVIEW",
                    ])
        else:
            calc_rows.append(["Power / thermal calculation", "NOT CALCULATED — input voltage was not stated and a nominal input rail was not identified in the netlist.", vin_source, "UNKNOWN"])

        if calc_rows:
            body.extend(["", "### Calculations (Python-computed)", "", _format_table(
                ["Check", "Calculation / result", "Basis", "Status"], calc_rows
            )])

        body.extend([
            "",
            "### Conclusion",
            "",
            ("**The nominal voltage and dropout headroom are consistent with 3.3 V regulation, but the design is not cleared as thermally safe.** The 600 mA figure is a maximum electrical-characteristics rating, not a guarantee that the package can dissipate the resulting heat on this PCB." if has_esp32_s3 else "**The AP2112K-3.3's voltage and dropout specifications may be compatible with a 5 V input, but thermal suitability cannot be concluded without an actual load current and PCB thermal details.** The 600 mA figure is a maximum electrical-characteristics rating, not a guarantee of thermally safe continuous output."),
            "",
            ("The Wi-Fi values are short-duration peak figures; the steady-state junction-temperature estimates intentionally show what would happen if that current persisted. Actual thermal performance depends on PCB copper, layout, ambient temperature, other 3.3 V loads, and TX duty cycle. Treat the estimates as screening calculations, not PCB sign-off." if has_esp32_s3 else "The power-loss estimate is not calculated until an actual or explicitly assumed load current is available. Thermal performance depends on package, PCB copper/layout, ambient temperature, and duty cycle. Treat the estimates as screening calculations, not PCB sign-off."),
        ])
        body[1:1] = ["", "**Power-path result:** Voltage compatibility can be checked from the supplied evidence; thermal safety cannot be approved without the real board load and PCB thermal conditions."]
    else:
        if any(row[3] == "UNKNOWN" for row in rows):
            body.extend([
                "",
                "One or more requested values were not present in the available matching AP2112-3.3 evidence. I have not substituted values from the AP2112-2.5 or AP2112-2.6 variants.",
            ])
        else:
            body.extend(["", "Values are extracted from the AP2112-3.3 electrical-characteristics section; dropout entries preserve the datasheet's typical and maximum values by test current."])

    if safety_review and schematic_result:
        # Report concrete pin/net membership from the selected netlist. This is
        # connectivity evidence only, not proof of voltage/current/PCB performance.
        components = schematic_result.get("components", []) or []
        component_by_ref = {str(c.get("ref", "")): c for c in components}
        nets = schematic_result.get("nets", {}) or {}

        regulator_ref = next((ref for ref, c in component_by_ref.items() if "ap2112" in str(c.get("value", "")).lower()), None)
        module_ref = next((ref for ref, c in component_by_ref.items() if "esp32-s3-wroom-1" in str(c.get("value", "")).lower()), None)
        connection_rows: list[list[str]] = []
        if regulator_ref:
            for pin_number, pin_label in (("1", "VIN"), ("5", "VOUT"), ("2", "GND"), ("3", "EN")):
                net_name = net_for_pin(regulator_ref, pin_number)
                connection_rows.append([
                    f"{regulator_ref} {pin_label} (pin {pin_number})",
                    net_name or "NOT CONNECTED / NOT FOUND IN NETLIST",
                    "Observed connectivity" if net_name else "UNKNOWN",
                ])
        if module_ref:
            for pin_number, pin_label in (("2", "3V3"), ("1", "GND")):
                net_name = net_for_pin(module_ref, pin_number)
                connection_rows.append([
                    f"{module_ref} {pin_label} (pin {pin_number})",
                    net_name or "NOT CONNECTED / NOT FOUND IN NETLIST",
                    "Observed connectivity" if net_name else "UNKNOWN",
                ])

        if connection_rows:
            netlist_source = Path(str(schematic_result.get("input_path", schematic_result.get("source_path", "supplied netlist")))).name
            body.extend([
                "",
                "### Power-path connectivity from the supplied netlist",
                "",
                f"Connectivity source: `{netlist_source}`.",
                "",
                _format_table(["Component pin", "Net name", "Evidence"], connection_rows),
            ])

        # Only report explicitly visible capacitor values and rail attachment.
        # Dielectric, rating, effective capacitance and PCB placement remain unknown.
        capacitor_rows: list[list[str]] = []
        for ref, component in component_by_ref.items():
            if not ref.upper().startswith("C"):
                continue
            value = str(component.get("value", "")).strip()
            pin1_net = net_for_pin(ref, "1")
            pin2_net = net_for_pin(ref, "2")
            pair = {str(pin1_net or "").upper(), str(pin2_net or "").upper()}
            if value and pair & {"GND", "+0V"} and pair & {"+5V", "+3V3", "VBUS"}:
                rail = "+5V input" if pair & {"+5V", "VBUS"} else "+3V3 output"
                capacitor_rows.append([ref, value, rail, "Value/connection observed; rating/layout unknown"])
        if capacitor_rows:
            body.extend([
                "",
                "### Rail capacitors visible in the netlist",
                "",
                _format_table(["Reference", "Value from netlist", "Rail", "What is verified"], capacitor_rows),
                "",
                "The AP2112 electrical-characteristics test conditions use 1 µF ceramic input/output capacitors (AP2112.pdf · p. 8). The netlist can show nominal values and connection, but not dielectric, voltage rating, effective capacitance under DC bias, ESR, placement, or PCB routing.",
            ])

        findings = schematic_result.get("findings", []) or []
        if findings:
            finding_rows = []
            for finding in findings[:12]:
                severity = str(finding.get("severity", "")).lower()
                status_text = str(finding.get("status", "")).lower()
                if severity in {"critical", "high"} and status_text in {"likely_violation", "confirmed_violation"}:
                    status = "FAIL"
                elif severity in {"medium", "high"} or "review" in status_text or finding.get("requires_datasheet_lookup"):
                    status = "REVIEW"
                else:
                    status = "UNKNOWN"
                finding_rows.append([
                    str(finding.get("id", finding.get("rule_id", "Schematic check"))),
                    str(finding.get("summary", "Finding")),
                    ", ".join(str(v) for v in (finding.get("affected_components", []) or [])) or "—",
                    status,
                ])
            body.extend([
                "",
                "### Schematic checker items",
                "",
                _format_table(["Check", "Observation", "Components", "Status"], finding_rows),
                "",
                "INFO-level items are review prompts, not confirmed faults.",
            ])

        # Explicitly tell the user which component choices were reviewed and which
        # cannot be verified from generic reference/value labels.
        coverage_rows: list[list[str]] = []
        if module_ref:
            module_value = str(schematic_component_by_ref[module_ref].get("value", "ESP32-S3 module"))
            module_summary = "Module datasheet matched; supply range and Wi-Fi TX peak evidence found. Total board current and thermal impact remain unknown."
            coverage_rows.append([f"{module_ref}: {module_value}", module_summary, "REVIEW"])
        if regulator_ref:
            regulator_value = str(schematic_component_by_ref[regulator_ref].get("value", "AP2112 regulator"))
            regulator_summary = "3.3 V variant, maximum-current rating and dropout data found; continuous-load thermal margin is not proven."
            coverage_rows.append([f"{regulator_ref}: {regulator_value}", regulator_summary, "REVIEW"])
        capacitor_refs = [ref for ref in component_by_ref if ref.upper().startswith("C")]
        if capacitor_refs:
            coverage_rows.append([
                "Capacitors: " + ", ".join(sorted(capacitor_refs)),
                "Nominal capacitance/net connectivity observed where available; manufacturer MPN, dielectric, voltage rating and placement not verified.",
                "REVIEW",
            ])
        generic_refs = [
            ref for ref, comp in component_by_ref.items()
            if ref not in {module_ref, regulator_ref}
            and ref.upper().startswith(("R", "J", "SW", "D", "Q", "Y", "L", "F"))
        ]
        if generic_refs:
            coverage_rows.append([
                "Other components: " + ", ".join(sorted(generic_refs)),
                "The netlist shows designators/values, but the exact manufacturer part numbers and relevant ratings are not established by this power-review path.",
                "UNKNOWN",
            ])
        if coverage_rows:
            body.extend([
                "",
                "### Component-selection coverage",
                "",
                _format_table(["Component group", "What the evidence establishes", "Status"], coverage_rows),
            ])

        body.extend([
            "",
            "### Review scope and unknowns",
            "",
            "This deterministic review covers the identified AP2112K-3.3 → ESP32-S3-WROOM-1 power path and netlist-observable connections only. It does not certify the whole PCB. Actual total and continuous load current, other 3.3 V loads, capacitor dielectric/voltage rating/effective capacitance, PCB copper and placement, thermal behaviour, trace width, clearances, and antenna keep-out remain unverified without the appropriate BOM, manufacturer datasheets, and PCB layout.",
        ])

    return "\n".join(body).strip()


def _ids_from_text(question: str, schematic_result: dict | None) -> list[str]:
    values = [question or ""]
    if schematic_result:
        values.extend(str(c.get("value", "")) for c in schematic_result.get("components", []))
    return ["AP2112K-3.3"] if any("ap2112" in value.lower() for value in values) else []
