"""Content-detected multi-format netlist parsing and conservative electrical checks.

Adapters support KiCad XML/S-expression and native .kicad_sch, EAGLE XML,
EasyEDA Pro JSON, EasyEDA/Protel grouped text, selected Altium/Allegro parenthesized
net/node text, PADS-style sectioned ASCII, SPICE-compatible netlists, and CSV/TSV.
Specific vendor export templates vary; unsupported files are rejected explicitly.

The checker reports uncertain cases as review items rather than claiming a circuit
is faulty without evidence. Parsing uses the standard library; KiCad CLI is needed
only when input is a native .kicad_sch file.
"""
from __future__ import annotations

import os
import re
import glob
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from netlist_adapters import parse_netlist_file


# Conservative aliases. Only explicit, recognizable rail names are considered.
_RAIL_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^(?:\+?0?\.0?V|GND|AGND|DGND|PGND|VSS|VEE)$", re.I), 0.0),
    (re.compile(r"^\+?1V0$", re.I), 1.0),
    (re.compile(r"^\+?1V2$", re.I), 1.2),
    (re.compile(r"^\+?1V8$", re.I), 1.8),
    (re.compile(r"^\+?2V5$", re.I), 2.5),
    (re.compile(r"^\+?3V3$", re.I), 3.3),
    (re.compile(r"^\+?5V(?:0)?$", re.I), 5.0),
    (re.compile(r"^VBUS$", re.I), 5.0),
    (re.compile(r"^\+?9V$", re.I), 9.0),
    (re.compile(r"^\+?12V$", re.I), 12.0),
    (re.compile(r"^-5V$", re.I), -5.0),
    (re.compile(r"^-12V$", re.I), -12.0),
)
_POWER_PIN_TOKENS = ("vcc", "vdd", "vin", "vbat", "vbus", "vee", "vss", "gnd", "avdd", "dvdd", "pvdd")


def _text(parent: ET.Element, path: str, default: str = "") -> str:
    node = parent.find(path)
    return (node.text or "").strip() if node is not None else default


def _find_kicad_cli() -> str | None:
    configured = os.environ.get("KICAD_CLI")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("kicad-cli") or shutil.which("kicad-cli.exe")
    if found:
        return found

    # Search common Windows install roots and sort numeric versions naturally
    # (lexicographic sorting incorrectly places 9.0 above 10.0).
    candidates: set[str] = set()
    for pattern in (
        "C:/Program Files/KiCad/*/bin/kicad-cli.exe",
        "C:/KiCad/*/bin/kicad-cli.exe",
    ):
        candidates.update(glob.glob(pattern))
    if not candidates:
        return None

    def version_key(path: str) -> tuple[int, ...]:
        match = re.search(r"KiCad[/\\](\d+(?:\.\d+)*)[/\\]bin[/\\]kicad-cli", path, re.I)
        if not match:
            match = re.search(r"KiCad[/\\](\d+(?:\.\d+)*)", path, re.I)
        return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)

    return max(candidates, key=version_key)


def _schematic_to_netlist(schematic_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    cli = _find_kicad_cli()
    if not cli:
        raise RuntimeError(
            "A .kicad_sch file needs KiCad's command-line tool to export a reliable netlist. "
            "Install KiCad and make kicad-cli available on PATH, or set KICAD_CLI to the "
            "full path of kicad-cli.exe. Or export a KiCad XML netlist from the KiCad GUI and supply the resulting file."
        )
    temp_dir = tempfile.TemporaryDirectory(prefix="schematic_checker_")
    output = Path(temp_dir.name) / "schematic.net"
    command = [cli, "sch", "export", "netlist", "--format", "kicadxml", "-o", str(output), str(schematic_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        temp_dir.cleanup()
        raise RuntimeError(f"Could not export KiCad schematic to a netlist: {exc}") from exc
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout or "No details returned by KiCad CLI.").strip()
        temp_dir.cleanup()
        raise RuntimeError(f"KiCad netlist export failed: {detail}")
    return output, temp_dir


def parse_kicad_netlist(netlist_path: str | Path) -> dict[str, Any]:
    """Parse supported netlists using content detection instead of trusting the extension."""
    return parse_netlist_file(netlist_path)


def _rail_voltage(label: str) -> float | None:
    normalized = re.sub(r"[\s_]", "", label or "")
    for pattern, voltage in _RAIL_PATTERNS:
        if pattern.fullmatch(normalized):
            return voltage
    return None


def _is_power_pin(pin_name: str, pin_type: str) -> bool:
    name = (pin_name or "").lower().replace("/", "")
    return pin_type in {"power_in", "power_out", "power_in_out", "open_collector"} or any(
        token in name for token in _POWER_PIN_TOKENS
    )


def check_schematic(schematic_path: str | Path, max_findings: int = 100) -> dict[str, Any]:
    """Detect a supported EDA netlist by content or export a native KiCad schematic to netlist."""
    original = Path(schematic_path).expanduser().resolve()
    if not original.is_file():
        raise FileNotFoundError(f"Schematic input does not exist: {original}")
    suffix = original.suffix.lower()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    parse_path = original
    input_format = "kicad_xml_netlist"
    if suffix == ".kicad_sch":
        parse_path, temp_dir = _schematic_to_netlist(original)
        input_format = "kicad_schematic_exported_to_netlist"
    else:
        # Format adapters detect content rather than relying on a particular extension.
        input_format = "auto_detected_netlist"

    try:
        design = parse_kicad_netlist(parse_path)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    findings: list[dict[str, Any]] = []

    def add_finding(rule_id: str, severity: str, status: str, summary: str,
                    affected_components: list[str], net: str | None = None,
                    evidence: str = "", suggestion: str = "", confidence: str = "medium") -> None:
        if len(findings) >= max_findings:
            return
        findings.append({
            "id": f"SCH-{len(findings) + 1:03d}",
            "rule_id": rule_id,
            "severity": severity,
            "status": status,
            "summary": summary,
            "affected_components": affected_components,
            "net": net,
            "design_evidence": evidence,
            "datasheet_evidence": [],
            "suggestion": suggestion,
            "confidence": confidence,
            "requires_datasheet_lookup": True,
        })

    # Check whether recognized supply rails with different nominal voltages have
    # accidentally been merged into the same electrical net. This is a strong
    # warning only when both explicit rail power symbols are present in that net.
    for net_name, net in design["nets"].items():
        rail_symbols: list[tuple[str, float, str]] = []
        for node in net["nodes"]:
            value = node.get("component_value", "")
            voltage = _rail_voltage(value)
            if voltage is not None and (node["ref"].startswith("#PWR") or node["ref"].startswith("#FLG") or value):
                rail_symbols.append((node["ref"], voltage, value))
        distinct = {voltage for _, voltage, _ in rail_symbols}
        if len(distinct) > 1:
            details = ", ".join(f"{ref} ({value})" for ref, _, value in rail_symbols)
            add_finding(
                "POWER_RAIL_CONFLICT", "high", "likely_violation",
                "One net contains power symbols with conflicting recognized nominal voltages.",
                sorted({ref for ref, _, _ in rail_symbols}), net_name,
                f"Net '{net_name}' contains {details}.",
                "Inspect the net labels and power symbols. Confirm whether these rails were unintentionally shorted.",
                "high",
            )

    # Unconnected pins are review items: a netlist alone cannot always distinguish
    # intentional NC pins from accidental omissions. Skip pins explicitly named NC.
    for ref, comp in design["components"].items():
        for pin_num, pin in comp["pins"].items():
            if pin.get("connected"):
                continue
            pin_name = pin.get("name", "")
            if re.fullmatch(r"(?:NC|N\.C\.|NOT.CONNECTED|DNC|NO.CONNECT)", pin_name.strip(), re.I):
                continue
            power_like = _is_power_pin(pin_name, pin.get("type", ""))
            severity = "high" if power_like else "medium"
            label = f"{ref} pin {pin_num}" + (f" ({pin_name})" if pin_name else "")
            add_finding(
                "UNCONNECTED_PIN", severity, "requires_review",
                f"{label} is not present on any exported net.", [ref], None,
                f"Symbol '{comp.get('value') or comp.get('libpart')}' defines pin {pin_num} ({pin_name or 'unnamed'}; type={pin.get('type') or 'unknown'}), but the netlist has no net node for it.",
                "Check whether the pin is intentionally left unconnected. For power, ground, enable, reset, or boot-strapping pins, verify the exact device datasheet and reference design.",
                "medium" if not power_like else "medium",
            )

    # Datasheet-driven review prompts for connected power pins. The checker does
    # not assume a pin's safe voltage from its name; RAG/Reasoning must establish it.
    for net_name, net in design["nets"].items():
        rail_voltage = _rail_voltage(net_name)
        if rail_voltage is None:
            rail_voltage = next((_rail_voltage(n.get("component_value", "")) for n in net["nodes"]
                                 if _rail_voltage(n.get("component_value", "")) is not None), None)
        if rail_voltage is None:
            continue
        for node in net["nodes"]:
            comp = design["components"].get(node["ref"])
            pin = comp.get("pins", {}).get(node["pin"], {}) if comp else {}
            pin_name = node.get("pinfunction") or pin.get("name", "")
            pin_type = node.get("pintype") or pin.get("type", "")
            if node["ref"].startswith(("#PWR", "#FLG")) or not _is_power_pin(pin_name, pin_type):
                continue
            # A ground pin connected to a recognized 0 V rail is expected and
            # should not trigger a generic voltage-compatibility check.
            if rail_voltage == 0.0 and re.search(r"(?:^|[^A-Z])(GND|AGND|DGND|PGND|VSS)(?:$|[^A-Z])", pin_name, re.I):
                continue
            add_finding(
                "POWER_PIN_DATASHEET_CHECK", "info", "datasheet_check_required",
                f"Verify that {node['ref']} pin {node['pin']} ({pin_name or 'power pin'}) is compatible with the nominal {rail_voltage:g} V rail.",
                [node["ref"]], net_name,
                f"Net '{net_name}' is associated with a recognized nominal {rail_voltage:g} V rail and connects to {node['ref']}.{node['pin']} ({pin_name or 'unnamed pin'}).",
                "Retrieve the exact component datasheet and compare recommended operating conditions, absolute maximum ratings, and any pin-specific constraints. Do not use absolute maximum ratings as normal operating targets.",
                "low",
            )

    # Conditional thermal review for common linear regulators. A netlist cannot
    # reveal current draw, copper area, ambient temperature, or thermal resistance
    # of the assembled board, so this is a request for calculation—not a failure.
    for ref, comp in design["components"].items():
        value = (comp.get("value") or "").upper()
        if not any(token in value for token in ("AP2112", "AMS1117", "LD1117", "LM1117", "REG_LINEAR", "LDO")):
            continue
        pin_nets: dict[str, str] = {}
        for net_name, net in design["nets"].items():
            for node in net["nodes"]:
                if node.get("ref") == ref:
                    pin_meta = comp.get("pins", {}).get(node.get("pin", ""), {})
                    pin_name = (node.get("pinfunction") or pin_meta.get("name") or "").upper()
                    pin_nets[pin_name] = net_name
        input_net = next((net for pin, net in pin_nets.items() if re.search(r"(?:VIN|V_IN|IN)(?:_|$|\d)", pin)), None)
        output_net = next((net for pin, net in pin_nets.items() if re.search(r"(?:VOUT|V_OUT|OUT)(?:_|$|\d)", pin)), None)
        vin = _rail_voltage(input_net or "")
        vout = _rail_voltage(output_net or "")
        if vin is not None and vout is not None and vin > vout >= 0:
            add_finding(
                "LINEAR_REGULATOR_THERMAL_REVIEW", "medium", "requires_load_and_thermal_review",
                f"{ref} ({comp.get('value')}) appears to regulate {vin:g} V to {vout:g} V; calculate regulator dissipation at the board's maximum load.",
                [ref], output_net,
                f"Netlist associates '{input_net}' with nominal {vin:g} V and '{output_net}' with nominal {vout:g} V. Actual output current, assembled-board thermal resistance, ambient temperature, and duty cycle are unknown.",
                "Calculate P_loss = (Vin - Vout) × Iout using the expected maximum load, then estimate junction temperature from the exact regulator datasheet and the actual PCB copper/thermal design. Do not assume MCU current from the part name alone.",
                "medium",
            )

    if len(findings) >= max_findings:
        findings = findings[:max_findings]
    return {
        "input_path": str(original),
        "input_format": (
            "kicad_schematic_exported_to_netlist"
            if suffix == ".kicad_sch"
            else design.get("input_format", input_format)
        ),
        "component_count": design["component_count"],
        "net_count": design["net_count"],
        "components": [
            {"ref": c["ref"], "value": c["value"], "footprint": c["footprint"]}
            for c in design["components"].values()
        ],
        "nets": design["nets"],
        "findings": findings,
        "finding_count": len(findings),
        "notes": [
            "Unconnected pins are review items because exported netlists do not reliably distinguish intentional no-connects from accidental omissions; some adapters cannot infer all library pins.",
            "Rail names are treated as nominal labels only; actual voltage compatibility must be confirmed from the exact datasheet and circuit context.",
            "Input formats are detected by content. Parser support is format-specific; an unknown vendor format is rejected rather than guessed as another format.",
            "This checker does not replace the source EDA tool's ERC/DRC, SPICE simulation, PCB layout review, current-capacity calculations, or thermal verification.",
        ] + design.get("adapter_notes", []),
    }


def format_findings_for_llm(result: dict[str, Any]) -> str:
    """Compact, readable evidence block to pass to the reasoning LLM."""
    lines = [
        f"SCHEMATIC FILE: {result.get('input_path')}",
        f"INPUT FORMAT: {result.get('input_format')}",
        f"COMPONENTS: {result.get('component_count')} | NETS: {result.get('net_count')}",
    ]
    components = result.get("components", [])
    if components:
        lines.append("COMPONENT INVENTORY:")
        lines.extend(f"- {c['ref']}: {c.get('value') or '(value unspecified)'}" for c in components[:80])
    findings = result.get("findings", [])
    lines.append(f"CHECKER FINDINGS ({len(findings)}):")
    if not findings:
        lines.append("- No findings from the implemented checks. This is not proof that the design is electrically correct.")
    for finding in findings:
        lines.append(
            f"- [{finding['id']}] {finding['severity'].upper()} / {finding['status']}: {finding['summary']} "
            f"Components={finding['affected_components']}; net={finding.get('net') or 'N/A'}. "
            f"Design evidence: {finding.get('design_evidence') or 'N/A'} "
            f"Suggested next step: {finding.get('suggestion') or 'N/A'}"
        )
    for note in result.get("notes", []):
        lines.append(f"NOTE: {note}")
    return "\n".join(lines)
