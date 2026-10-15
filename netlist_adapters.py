"""Content-detected netlist adapters for common EDA and simulation formats.

Normalized schema: components/ref/value/footprint/pins and nets/nodes.
This module is intentionally dependency-light so parsing can be tested offline.
"""
from __future__ import annotations

import csv
import json
import re
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _txt(parent: ET.Element, path: str, default: str = "") -> str:
    node = parent.find(path)
    return (node.text or "").strip() if node is not None else default


def _new_component(ref: str, value: str = "", footprint: str = "", **extra) -> dict[str, Any]:
    return {
        "ref": str(ref), "value": str(value or ""), "footprint": str(footprint or ""),
        "lib": extra.get("lib", ""), "libpart": extra.get("libpart", ""),
        "fields": extra.get("fields", {}), "pins": {},
    }


def _normalize_design(source: str, components: dict[str, dict], nets: dict[str, dict], fmt: str,
                      notes: list[str] | None = None) -> dict[str, Any]:
    """Attach component values, infer observed pins, and mark pin/net membership."""
    connected: set[tuple[str, str]] = set()
    for net_name, net in nets.items():
        net["name"] = str(net.get("name", net_name))
        net["nodes"] = net.get("nodes", [])
        for node in net["nodes"]:
            ref, pin = str(node.get("ref", "")).strip(), str(node.get("pin", "")).strip()
            node["ref"], node["pin"] = ref, pin
            node.setdefault("component_value", components.get(ref, {}).get("value", ""))
            node.setdefault("pinfunction", "")
            node.setdefault("pintype", "")
            if ref and pin:
                connected.add((ref, pin))
                comp = components.setdefault(ref, _new_component(ref))
                comp["pins"].setdefault(pin, {"number": pin, "name": node.get("pinfunction", ""), "type": node.get("pintype", "")})
                comp["pins"][pin]["connected"] = True
                comp["pins"][pin]["net"] = net_name
    for ref, comp in components.items():
        for pin_number, pin in comp.setdefault("pins", {}).items():
            pin["number"] = str(pin.get("number", pin_number))
            pin["connected"] = (ref, str(pin_number)) in connected
            if not pin["connected"]:
                pin.setdefault("net", None)
    return {
        "source_path": str(source), "input_format": fmt,
        "components": components, "nets": nets,
        "component_count": len(components), "net_count": len(nets),
        "adapter_notes": notes or [],
    }


def parse_kicad_xml_root(root: ET.Element, source: str) -> dict[str, Any]:
    if root.tag != "export":
        raise ValueError("Not a KiCad XML netlist: expected <export> root element.")
    components: dict[str, dict] = {}
    for comp in root.findall("./components/comp"):
        ref = (comp.get("ref") or "").strip()
        if not ref:
            continue
        libsource = comp.find("libsource")
        fields = {f.get("name", ""): (f.text or "").strip() for f in comp.findall("./fields/field")}
        components[ref] = _new_component(
            ref, _txt(comp, "value", fields.get("Value", "")),
            _txt(comp, "footprint", fields.get("Footprint", "")),
            lib=libsource.get("lib", "") if libsource is not None else "",
            libpart=libsource.get("part", "") if libsource is not None else "", fields=fields,
        )
    libpart_pins: dict[tuple[str, str], dict] = {}
    for part in root.findall("./libparts/libpart"):
        key = (part.get("lib", ""), part.get("part", ""))
        pins = {}
        for pin in part.findall("./pins/pin"):
            number = (pin.get("num") or "").strip()
            if number:
                pins[number] = {"number": number, "name": (pin.get("name") or "").strip(), "type": (pin.get("type") or "").strip().lower(), "connected": False, "net": None}
        libpart_pins[key] = pins
    for comp in components.values():
        comp["pins"] = {k: dict(v) for k, v in libpart_pins.get((comp["lib"], comp["libpart"]), {}).items()}
    nets: dict[str, dict] = {}
    for net in root.findall("./nets/net"):
        name = (net.get("name") or f"NET_{net.get('code', '?')}").strip()
        nodes = []
        for node in net.findall("./node"):
            ref, pin = (node.get("ref") or "").strip(), (node.get("pin") or "").strip()
            if ref and pin:
                nodes.append({"ref": ref, "pin": pin, "pinfunction": (node.get("pinfunction") or "").strip(), "pintype": (node.get("pintype") or "").strip().lower()})
        nets[name] = {"name": name, "code": net.get("code", ""), "nodes": nodes}
    return _normalize_design(source, components, nets, "kicad_xml_netlist")


def _sexpr_tokens(text: str) -> list[str]:
    return re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', text, flags=re.S)


def _sexpr_parse(text: str):
    tokens = _sexpr_tokens(text)
    stack: list[list] = []
    root = None
    for tok in tokens:
        if tok == '(':
            node: list = []
            if stack:
                stack[-1].append(node)
            else:
                root = node
            stack.append(node)
        elif tok == ')':
            if not stack:
                raise ValueError("Malformed S-expression netlist: unexpected ')' .")
            stack.pop()
        else:
            value = tok
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = re.sub(r'\\([\\"nrt])', lambda m: {"n":"\n", "r":"\r", "t":"\t"}.get(m.group(1), m.group(1)), value[1:-1])
            if not stack:
                raise ValueError("Malformed S-expression netlist: text outside expression.")
            stack[-1].append(value)
    if stack or not isinstance(root, list):
        raise ValueError("Malformed S-expression netlist: unmatched parentheses.")
    return root


def _sexpr_children(node, tag: str) -> list:
    return [x for x in node if isinstance(x, list) and x and x[0] == tag]


def _sexpr_child(node, tag: str):
    items = _sexpr_children(node, tag)
    return items[0] if items else None


def _sexpr_value(node, tag: str, default: str = "") -> str:
    child = _sexpr_child(node, tag)
    return str(child[1]) if child and len(child) > 1 else default


def parse_kicad_sexpr(text: str, source: str) -> dict[str, Any]:
    tree = _sexpr_parse(text)
    if not tree or tree[0] != "export":
        raise ValueError("Not a KiCad S-expression netlist (expected '(export ... )').")
    components: dict[str, dict] = {}
    libparts: dict[tuple[str, str], dict] = {}
    for lp in _sexpr_children(_sexpr_child(tree, "libparts") or [], "libpart"):
        key = (_sexpr_value(lp, "lib"), _sexpr_value(lp, "part"))
        pins = {}
        pins_node = _sexpr_child(lp, "pins")
        for p in _sexpr_children(pins_node or [], "pin"):
            num = _sexpr_value(p, "num")
            if num:
                pins[num] = {"number": num, "name": _sexpr_value(p, "name"), "type": _sexpr_value(p, "type").lower(), "connected": False, "net": None}
        libparts[key] = pins
    for c in _sexpr_children(_sexpr_child(tree, "components") or [], "comp"):
        ref = _sexpr_value(c, "ref")
        if not ref:
            continue
        lib = _sexpr_child(c, "libsource")
        fields = {}
        for f in _sexpr_children(_sexpr_child(c, "fields") or [], "field"):
            label = _sexpr_value(f, "name")
            if label:
                # KiCad S-expression field values may be a bare scalar after the name form.
                name_node = _sexpr_child(f, "name")
                raw = next((x for x in f[1:] if isinstance(x, str)), "")
                fields[label] = raw
        components[ref] = _new_component(ref, _sexpr_value(c, "value", fields.get("Value", "")), _sexpr_value(c, "footprint", fields.get("Footprint", "")), lib=_sexpr_value(lib, "lib") if lib else "", libpart=_sexpr_value(lib, "part") if lib else "", fields=fields)
        components[ref]["pins"] = {k: dict(v) for k, v in libparts.get((components[ref]["lib"], components[ref]["libpart"]), {}).items()}
    nets = {}
    for n in _sexpr_children(_sexpr_child(tree, "nets") or [], "net"):
        name = _sexpr_value(n, "name") or f"NET_{_sexpr_value(n, 'code', '?')}"
        nodes = []
        for nd in _sexpr_children(n, "node"):
            ref, pin = _sexpr_value(nd, "ref"), _sexpr_value(nd, "pin")
            if ref and pin:
                nodes.append({"ref": ref, "pin": pin, "pinfunction": _sexpr_value(nd, "pinfunction"), "pintype": _sexpr_value(nd, "pintype").lower()})
        nets[name] = {"name": name, "code": _sexpr_value(n, "code"), "nodes": nodes}
    return _normalize_design(source, components, nets, "kicad_sexpr_netlist")


def parse_eagle_xml(root: ET.Element, source: str) -> dict[str, Any]:
    if root.tag != "eagle":
        raise ValueError("Not an EAGLE XML document.")
    drawing = root.find("drawing")
    if drawing is None:
        raise ValueError("EAGLE XML is missing <drawing>.")
    board = drawing.find("board")
    schematic = drawing.find("schematic")
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    if board is not None:
        part_parent = board.find("elements")
        if part_parent is not None:
            for part in part_parent.findall("element"):
                ref = part.get("name", "").strip()
                if ref:
                    components[ref] = _new_component(ref, part.get("value", ""), part.get("package", ""))
        for signal in board.findall("./signals/signal"):
            name = signal.get("name", "")
            nodes = []
            for cr in signal.findall("contactref"):
                ref, pin = cr.get("element", ""), cr.get("pad", "")
                if ref and pin:
                    nodes.append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
            nets[name] = {"name": name, "nodes": nodes}
        return _normalize_design(source, components, nets, "eagle_xml_board", ["EAGLE board XML: component pin definitions may be incomplete; connectivity is based on signal contactrefs."])
    if schematic is not None:
        for part in schematic.findall("./parts/part"):
            ref = part.get("name", "").strip()
            if ref:
                components[ref] = _new_component(ref, part.get("value", ""), part.get("device", ""), lib=part.get("library", ""), libpart=part.get("deviceset", ""))
        for net in schematic.findall("./sheets/sheet/nets/net"):
            name = net.get("name", "")
            nodes = []
            for pinref in net.findall("./segment/pinref"):
                ref, pin = pinref.get("part", ""), pinref.get("pin", "")
                if ref and pin:
                    nodes.append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
            nets[name] = {"name": name, "nodes": nodes}
        return _normalize_design(source, components, nets, "eagle_xml_schematic", ["EAGLE schematic XML: library-level electrical pin types are not inferred; run the source EDA tool's ERC for electrical-type checks."])
    raise ValueError("EAGLE XML contains neither a board nor a schematic.")


def _props_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def parse_easyeda_pro_json(payload: dict, source: str) -> dict[str, Any]:
    raw_components = payload.get("components")
    if not isinstance(raw_components, dict):
        raise ValueError("JSON is not a recognized EasyEDA Pro .enet netlist: missing components object.")
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    for _, item in raw_components.items():
        if not isinstance(item, dict):
            continue
        props = _props_map(item.get("props"))
        ref = str(props.get("Designator", props.get("designator", ""))).strip()
        if not ref:
            continue
        value = props.get("Manufacturer Part") or props.get("Value") or props.get("Name") or props.get("DeviceName") or ""
        comp = _new_component(ref, value, props.get("FootprintName") or props.get("Footprint") or "", fields=props)
        for pin_key, pin in _props_map(item.get("pinInfoMap")).items():
            if not isinstance(pin, dict):
                continue
            number = str(pin.get("number") or _props_map(pin.get("props")).get("Pin Number") or pin_key).strip()
            net_name = str(pin.get("net") or "").strip()
            pin_name = str(pin.get("name") or "").strip()
            comp["pins"][number] = {"number": number, "name": pin_name, "type": "", "connected": bool(net_name), "net": net_name or None}
            if net_name:
                nets.setdefault(net_name, {"name": net_name, "nodes": []})["nodes"].append({"ref": ref, "pin": number, "pinfunction": pin_name, "pintype": ""})
        components[ref] = comp
    if not components:
        raise ValueError("EasyEDA Pro JSON was recognized, but no component designators were found.")
    return _normalize_design(source, components, nets, "easyeda_pro_enet_json", ["Parsed EasyEDA Pro .enet JSON using Designator, Value, and pinInfoMap.net fields."])


def parse_spice(text: str, source: str) -> dict[str, Any]:
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    pin_labels = {"R": ("1", "2"), "C": ("1", "2"), "L": ("1", "2"), "V": ("+", "-"), "I": ("+", "-"), "D": ("A", "K"), "Q": ("C", "B", "E"), "M": ("D", "G", "S", "B")}
    logical_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ";", "#", ".")):
            continue
        if stripped.startswith("+") and logical_lines:
            logical_lines[-1] += " " + stripped[1:].strip()
        else:
            logical_lines.append(stripped)
    for line in logical_lines:
        try:
            parts = shlex.split(line, comments=False, posix=True)
        except ValueError:
            continue
        if len(parts) < 4:
            continue
        ref = parts[0]
        if not re.match(r"^[A-Za-z][A-Za-z0-9_$.-]*$", ref):
            continue
        kind = ref[0].upper()
        if kind in {"R", "C", "L", "V", "I"}:
            pin_count, value = 2, " ".join(parts[3:])
        elif kind == "D":
            pin_count, value = 2, parts[3]
        elif kind == "Q":
            pin_count, value = 3, parts[4] if len(parts) > 4 else ""
        elif kind == "M":
            pin_count, value = 4, parts[5] if len(parts) > 5 else ""
        elif kind in {"X", "U"}:
            pin_count, value = max(1, len(parts) - 2), parts[-1]
        else:
            # Conservative default for common two-terminal SPICE parts.
            pin_count, value = 2, " ".join(parts[3:])
        nodes = parts[1:1 + pin_count]
        if len(nodes) < pin_count:
            continue
        pins = {}
        labels = pin_labels.get(kind, tuple(str(i + 1) for i in range(pin_count)))
        for i, pin in enumerate(nodes):
            label = labels[i] if i < len(labels) else str(i + 1)
            pins[label] = {"number": label, "name": label, "type": "", "connected": True, "net": pin}
            nets.setdefault(pin, {"name": pin, "nodes": []})["nodes"].append({"ref": ref, "pin": label, "pinfunction": label, "pintype": ""})
        comp = _new_component(ref, value, fields={"spice_model_or_value": value})
        comp["pins"] = pins
        components[ref] = comp
    if not components:
        raise ValueError("No supported SPICE component lines were recognized.")
    return _normalize_design(source, components, nets, "spice_compatible_netlist", ["SPICE topology was parsed. Node/pin naming is inferred from conventional SPICE element order; this is simulation connectivity, not necessarily a complete PCB netlist."])


def _extract_balanced_groups(text: str) -> list[str]:
    groups, depth, start, quote, escaped = [], 0, None, False, False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quote = False
            continue
        if ch == '"':
            quote = True
        elif ch == '(':
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed grouped netlist: unexpected ')'.")
            if depth == 0 and start is not None:
                groups.append(text[start:i])
                start = None
    return groups


def parse_grouped_netlist(text: str, source: str) -> dict[str, Any]:
    """Parse EasyEDA Std / Protel-style parenthesized net groups conservatively."""
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    node_re = re.compile(r"^([A-Za-z#][A-Za-z0-9_#$.:]*?)-([A-Za-z0-9_.$]+)$")
    for group in _extract_balanced_groups(text):
        lines = [ln.strip().strip(',') for ln in group.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        first = lines[0].strip('"')
        # Tolerate '(Net NAME ...)' forms as well as EasyEDA's '(\nNAME\n Ref-Pin ...)'.
        mname = re.match(r"^(?:NET|SIGNAL)\s+[\"']?(.+?)[\"']?$", first, re.I)
        net_name = (mname.group(1) if mname else first).strip().strip('"')
        nodes = []
        for line in lines[1:]:
            tokens = line.replace('"', '').split()
            for token in tokens[:3]:
                match = node_re.match(token)
                if not match:
                    continue
                ref, pin = match.group(1), match.group(2)
                if ref.upper() in {"NET", "NODE", "CONNECT", "SIGNAL"}:
                    continue
                components.setdefault(ref, _new_component(ref))
                nodes.append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
                break
        if nodes:
            nets[net_name or f"NET_{len(nets)+1}"] = {"name": net_name, "nodes": nodes}
    if not nets:
        raise ValueError("Grouped netlist syntax was not recognized. Try an explicit KiCad XML, KiCad S-expression, EasyEDA Pro .enet, or SPICE netlist export.")
    return _normalize_design(source, components, nets, "easyeda_or_protel_grouped_text", ["Connectivity groups parsed from parenthesized text. Component values, electrical pin types, and unconnected pins may not be available in this export format."])


def parse_tabular_netlist(path: Path, text: str) -> dict[str, Any]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows or not rows[0]:
        raise ValueError("Tabular netlist needs a header row with net, reference, and pin columns.")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    headers = {norm(k): k for k in rows[0].keys() if k}
    net_col = next((headers[k] for k in (
        "net", "netname", "signal", "signalname", "netid", "signalid"
    ) if k in headers), None)
    ref_col = next((headers[k] for k in (
        "ref", "reference", "designator", "component", "refdes",
        "componentdesignator", "partreference", "partref"
    ) if k in headers), None)
    pin_col = next((headers[k] for k in (
        "pin", "pad", "pinnumber", "padnumber", "componentpin", "pinno"
    ) if k in headers), None)
    if not (net_col and ref_col and pin_col):
        raise ValueError("Unrecognized CSV/TSV netlist headers. Expected columns identifying net/signal, reference/designator, and pin/pad; column order does not matter.")
    value_col = next((headers[k] for k in (
        "value", "part", "partnumber", "componentvalue", "manufacturerpart"
    ) if k in headers), None)
    components, nets = {}, {}
    for row in rows:
        ref, pin, name = (row.get(ref_col) or "").strip(), (row.get(pin_col) or "").strip(), (row.get(net_col) or "").strip()
        if not (ref and pin and name):
            continue
        if ref not in components:
            components[ref] = _new_component(ref, row.get(value_col, "") if value_col else "")
        nodes = nets.setdefault(name, {"name": name, "nodes": []})["nodes"]
        nodes.append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
    if not nets:
        raise ValueError("No valid net/reference/pin rows were found in the table.")
    return _normalize_design(str(path), components, nets, "csv_or_tsv_connectivity_table", ["Tabular parser supports connectivity fields only; unknown component pins cannot be inferred."])


def parse_structured_parenthesized(text: str, source: str) -> dict[str, Any]:
    """Best-effort parser for common parenthesized Altium/Protel/Allegro-style net/node exports."""
    tree = _sexpr_parse(text)
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}

    def walk(node):
        if not isinstance(node, list):
            return
        if node and isinstance(node[0], str) and node[0].lower() in {"net", "signal"}:
            name = ""
            if len(node) > 1 and isinstance(node[1], str):
                name = node[1].strip('"')
            if not name:
                name = _sexpr_value(node, "name")
            nodes = []
            def node_walk(child):
                if not isinstance(child, list):
                    return
                if child and isinstance(child[0], str) and child[0].lower() in {"node", "connect", "pinref", "contactref"}:
                    ref = pin = ""
                    if len(child) > 2 and all(isinstance(x, str) for x in child[1:3]):
                        ref, pin = child[1].strip('"'), child[2].strip('"')
                    else:
                        ref = _sexpr_value(child, "ref") or _sexpr_value(child, "part") or _sexpr_value(child, "component")
                        pin = _sexpr_value(child, "pin") or _sexpr_value(child, "pad") or _sexpr_value(child, "number")
                    if ref and pin:
                        nodes.append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
                    return
                for item in child:
                    node_walk(item)
            node_walk(node)
            if nodes:
                nets[name or f"NET_{len(nets)+1}"] = {"name": name, "nodes": nodes}
                for item in nodes:
                    components.setdefault(item["ref"], _new_component(item["ref"]))
            return
        for child in node:
            walk(child)

    walk(tree)
    if not nets:
        raise ValueError("Parenthesized export was not recognized as KiCad, Altium/Protel, or Allegro-style net/node data.")
    return _normalize_design(source, components, nets, "parenthesized_eda_netlist", ["Generic parenthesized netlist parsed by net/node records. Pin direction/types and full component properties may be missing; source-EDA ERC is still required."])


def parse_sectioned_netlist(text: str, source: str) -> dict[str, Any]:
    """Best-effort support for sectioned PADS-style ASCII netlists."""
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    section = ""
    current_net = None
    node_re = re.compile(r"^([A-Za-z#][A-Za-z0-9_#$:.]*?)[.-]([A-Za-z0-9_.$]+)$")
    saw_net_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("*"):
            section = line.strip("* ").upper()
            if section in {"NET", "SIGNAL", "NETS"}:
                saw_net_section = True
                current_net = None
            continue
        if line.startswith((";", "#")):
            continue
        if section in {"PART", "PARTS", "COMPONENT", "COMPONENTS"}:
            fields = re.split(r"\s+", line)
            if len(fields) >= 2 and re.match(r"^[A-Za-z][A-Za-z0-9_$.-]*$", fields[0]):
                components.setdefault(fields[0], _new_component(fields[0], fields[1]))
            continue
        # Some variants emit explicit 'NET <name>' records.
        explicit = re.match(r"^(?:NET|SIGNAL)\s+[\"']?([^\"'\s]+)", line, re.I)
        if explicit:
            current_net = explicit.group(1)
            nets.setdefault(current_net, {"name": current_net, "nodes": []})
            saw_net_section = True
            continue
        # Find a component-pin token in the line, allowing columns after it.
        tokens = line.replace('"', '').split()
        parsed_node = None
        for token in tokens[:4]:
            match = node_re.match(token.rstrip(",;"))
            if match:
                parsed_node = (match.group(1), match.group(2))
                break
        if parsed_node and (saw_net_section or section in {"NET", "SIGNAL", "NETS"}):
            if not current_net:
                current_net = "NET_1"
            ref, pin = parsed_node
            components.setdefault(ref, _new_component(ref))
            nets.setdefault(current_net, {"name": current_net, "nodes": []})["nodes"].append({"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""})
            continue
        # In a net section, a plain token between net groups is a net label.
        if saw_net_section and not parsed_node and not line.startswith(("*", "(", ")")):
            cleaned = line.strip('"')
            if not re.match(r"^(?:END|ENDNET|END_NET|BEGIN|NET|SIGNAL)$", cleaned, re.I):
                current_net = cleaned
                nets.setdefault(current_net, {"name": current_net, "nodes": []})
    nets = {name: net for name, net in nets.items() if net.get("nodes")}
    if not nets:
        raise ValueError("Sectioned PADS/ASCII netlist syntax was not recognized.")
    return _normalize_design(source, components, nets, "pads_or_sectioned_ascii_netlist", ["Sectioned ASCII adapter focuses on net/pin connectivity; vendor-specific properties and electrical types may be omitted."])




def parse_edif(text: str, source: str) -> dict[str, Any]:
    """Read the connectivity subset of EDIF netlist interchange files."""
    # EDIF allows semicolon comments. This simple removal applies outside no
    # quoted strings; quoted semicolons are uncommon, so avoid broad line stripping
    # unless a semicolon appears in a comment position.
    cleaned_lines = []
    for line in text.splitlines():
        quote = False
        escaped = False
        cut = len(line)
        for index, char in enumerate(line):
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quote = False
            else:
                if char == '"':
                    quote = True
                elif char == ";":
                    cut = index
                    break
        cleaned_lines.append(line[:cut])
    tree = _sexpr_parse("\\n".join(cleaned_lines))
    if not tree or str(tree[0]).lower() != "edif":
        raise ValueError("Not an EDIF netlist (expected an (edif ...) root).")

    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}

    def walk(node):
        if not isinstance(node, list):
            return
        if node and isinstance(node[0], str) and node[0].lower() == "instance" and len(node) > 1:
            ref = str(node[1]).strip('"')
            cell_ref = _sexpr_child(node, "cellRef")
            value = str(cell_ref[1]).strip('"') if cell_ref and len(cell_ref) > 1 else ""
            if ref and ref not in components:
                components[ref] = _new_component(ref, value)
        if node and isinstance(node[0], str) and node[0].lower() == "net" and len(node) > 1:
            name = str(node[1]).strip('"')
            nodes = []

            def visit_portrefs(child):
                if not isinstance(child, list):
                    return
                if child and isinstance(child[0], str) and child[0].lower() == "portref":
                    pin = str(child[1]).strip('"') if len(child) > 1 and isinstance(child[1], str) else ""
                    instance_ref = _sexpr_child(child, "instanceRef")
                    ref = str(instance_ref[1]).strip('"') if instance_ref and len(instance_ref) > 1 else ""
                    if ref and pin:
                        nodes.append({"ref": ref, "pin": pin, "pinfunction": pin, "pintype": ""})
                        components.setdefault(ref, _new_component(ref))
                    return
                for nested in child:
                    visit_portrefs(nested)

            visit_portrefs(node)
            if nodes:
                nets[name] = {"name": name, "nodes": nodes}
            # Continue walking so instances in nested contents are also discovered.
        for child in node:
            if isinstance(child, list):
                walk(child)

    walk(tree)
    if not nets:
        raise ValueError("EDIF parsed, but no net/portRef connectivity was recognized.")
    return _normalize_design(
        source, components, nets, "edif_netlist",
        ["EDIF connectivity subset parsed. Hierarchy, bus expansion, port direction, and electrical pin types may not be fully resolved; compare against source EDA netlist/ERC."],
    )

def parse_ipc_d356(text: str, source: str) -> dict[str, Any]:
    """Parse IPC-D-356 bare-board test records (317/327/367) into net/pad connectivity.

    IPC-D-356 is a physical board test netlist. It usually lacks component values,
    symbol pin names/types, and unconnected logical pins, so those facts are not inferred.
    """
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    rows_seen = 0
    for line in text.splitlines():
        if len(line) < 3 or line[:3] not in {"317", "327", "367"}:
            continue
        # Standard IPC-D-356 positions: record code, net name, refdes, dash + pin.
        net_name = line[3:17].strip() if len(line) >= 17 else ""
        ref = line[17:26].strip() if len(line) >= 26 else ""
        pin_field = line[26:32].strip() if len(line) >= 27 else ""
        pin = pin_field[1:].strip() if pin_field.startswith("-") else pin_field.strip()
        # Some exporter variants shift columns or add separators; use a constrained fallback.
        if not (net_name and ref and pin):
            match = re.match(
                r"^3(?:17|27|67)(?P<net>\S+)\s+(?P<ref>[A-Za-z#][A-Za-z0-9_.$-]*)\s+-\s*(?P<pin>[A-Za-z0-9_.$+-]+)",
                line,
            )
            if match:
                net_name, ref, pin = match.group("net"), match.group("ref"), match.group("pin")
        if not (net_name and ref and pin):
            continue
        # The reference field may contain whitespace padding only; normalize the
        # special dash used for anonymous/NC pins while preserving named pads.
        if pin in {"-", "?", "NC"}:
            continue
        components.setdefault(ref, _new_component(ref))
        nets.setdefault(net_name, {"name": net_name, "nodes": []})["nodes"].append(
            {"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""}
        )
        rows_seen += 1
    if rows_seen == 0:
        raise ValueError("No IPC-D-356 test records (317/327/367) were recognized.")
    return _normalize_design(
        source, components, nets, "ipc_d356_board_test_netlist",
        ["IPC-D-356 is a PCB bare-board test netlist: coordinates/access metadata are not used in this logical summary, and missing pads are not proof of unconnected schematic pins."],
    )


def parse_orcad_pcb2(text: str, source: str) -> dict[str, Any]:
    """Parse the classic OrCAD/PCB II component-centric parenthesized export."""
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    ref_re = re.compile(r"^[A-Za-z#][A-Za-z0-9_.$-]*\d+[A-Za-z0-9_.$-]*$")
    for group in _extract_balanced_groups(text):
        # The outer record begins with: checksum/package/refdes/value; nested records
        # then contain (pin_number net_name). This format often has a text preamble.
        header = group.split("(", 1)[0].strip()
        try:
            fields = shlex.split(header, comments=False, posix=True)
        except ValueError:
            continue
        if len(fields) < 3:
            continue
        ref_index = next((i for i, token in enumerate(fields[:5]) if ref_re.fullmatch(token)), None)
        if ref_index is None or ref_index == 0:
            continue
        ref = fields[ref_index]
        package = fields[ref_index - 1]
        value = fields[ref_index + 1] if len(fields) > ref_index + 1 else ""
        comp = _new_component(ref, value, package)
        components[ref] = comp
        for child in re.finditer(r"\(([^()]*)\)", group, flags=re.S):
            try:
                pin_fields = shlex.split(child.group(1), comments=False, posix=True)
            except ValueError:
                continue
            if len(pin_fields) < 2:
                continue
            pin = pin_fields[0]
            for net_name in pin_fields[1:]:
                if net_name in {"?", "?1", "NC", "N.C."}:
                    continue
                comp["pins"].setdefault(pin, {"number": pin, "name": "", "type": "", "connected": True, "net": net_name})
                nets.setdefault(net_name, {"name": net_name, "nodes": []})["nodes"].append(
                    {"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""}
                )
    if not components or not nets:
        raise ValueError("OrCAD/PCB II component-centric netlist syntax was not recognized.")
    return _normalize_design(
        source, components, nets, "orcad_pcb2_netlist",
        ["OrCAD/PCB II component-centric export parsed. Electrical pin types and schematic symbol definitions are not included in this format."],
    )


def parse_net_declaration_text(text: str, source: str) -> dict[str, Any]:
    """Parse common vendor ASCII rows: NET name ref-pin ... or net; ref.pin."""
    components: dict[str, dict] = {}
    nets: dict[str, dict] = {}
    node_re = re.compile(r"^([A-Za-z#][A-Za-z0-9_#$:.]*?)[.-]([A-Za-z0-9_.$+]+)$")
    saw = 0

    def add_node(net_name: str, token: str) -> bool:
        match = node_re.match(token.strip().strip(',;'))
        if not match:
            return False
        ref, pin = match.group(1), match.group(2)
        if ref.upper() in {"NET", "SIGNAL", "NODE", "PIN", "REF", "REFERENCE"}:
            return False
        components.setdefault(ref, _new_component(ref))
        nets.setdefault(net_name, {"name": net_name, "nodes": []})["nodes"].append(
            {"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""}
        )
        return True

    # OrCAD/Mentor-style net declarations: NET 'GND' U1-7 U2-7.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("*", ";", "#", "//")):
            continue
        try:
            tokens = shlex.split(line, comments=False, posix=True)
        except ValueError:
            tokens = line.replace(";", " ").replace(",", " ").split()
        if len(tokens) >= 3 and tokens[0].upper() in {"NET", "SIGNAL"}:
            net_name = tokens[1].strip("'\"")
            added = sum(add_node(net_name, token) for token in tokens[2:])
            saw += added
            continue
        # Common Allegro/net-report form: GND ; U1.1 (or GND, U1-1).
        fields = re.split(r"\s*[;,\t|]\s*", line)
        if len(fields) >= 2 and fields[0] and fields[0].upper() not in {"NET", "SIGNAL"}:
            net_name = fields[0].strip("'\" ")
            added = sum(add_node(net_name, token) for field in fields[1:] for token in field.split())
            if added:
                saw += added
                continue
        # Mentor/Multiwire-style rows: GND U1 7.
        if len(tokens) == 3 and re.fullmatch(r"[A-Za-z#][A-Za-z0-9_.$-]*\d+[A-Za-z0-9_.$-]*", tokens[1]):
            net_name, ref, pin = tokens
            if re.match(r"^[A-Za-z0-9_.$+-]+$", pin) and net_name.upper() not in {"END", "BEGIN", "PART", "COMPONENT"}:
                components.setdefault(ref, _new_component(ref))
                nets.setdefault(net_name.strip("'\""), {"name": net_name.strip("'\""), "nodes": []})["nodes"].append(
                    {"ref": ref, "pin": pin, "pinfunction": "", "pintype": ""}
                )
                saw += 1
    if saw == 0 or not nets:
        raise ValueError("Common vendor ASCII net declarations were not recognized.")
    return _normalize_design(
        source, components, nets, "vendor_ascii_netlist",
        ["Vendor ASCII net declaration parsed by node pattern. Values, unused pins, and pin electrical types may be omitted; confirm against the source EDA tool."],
    )

def _looks_like_tabular_connectivity_header(sample: str) -> bool:
    lines = [line for line in sample.splitlines() if line.strip() and not line.lstrip().startswith(("#", ";", "*"))]
    if not lines:
        return False
    for delimiter in (",", ";", "\t", "|"):
        try:
            fields = next(csv.reader([lines[0]], delimiter=delimiter))
        except (csv.Error, StopIteration):
            continue
        normalized = {re.sub(r"[^a-z0-9]", "", field.lower()) for field in fields}
        has_net = bool(normalized & {"net", "netname", "signal", "signalname", "netid", "signalid"})
        has_ref = bool(normalized & {"ref", "reference", "designator", "component", "refdes", "componentdesignator", "partreference", "partref"})
        has_pin = bool(normalized & {"pin", "pad", "pinnumber", "padnumber", "componentpin", "pinno"})
        if has_net and has_ref and has_pin:
            return True
    return False


def parse_netlist_file(path: str | Path) -> dict[str, Any]:
    """Detect format by content first, then by extension; avoid trusting extensions alone."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Netlist input does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ValueError(f"Could not read netlist '{path}': {exc}") from exc
    stripped = text.lstrip()
    ext = path.suffix.lower()
    if not stripped:
        raise ValueError(f"Netlist '{path.name}' is empty.")

    # IPC-D-356 physical-board test netlists use fixed-width 317/327/367 records.
    if re.search(r"(?m)^3(?:17|27|67)", text) and (
        re.search(r"(?m)^P\s+UNITS\b", text)
        or re.search(r"(?m)^C.*IPC-D-356", text, re.I)
        or ext in {".ipc", ".d356", ".356"}
        or len(re.findall(r"(?m)^3(?:17|27|67)", text)) >= 2
    ):
        return parse_ipc_d356(text, str(path))

    # Classic OrCAD/PCB II is component-centric and often has a preamble before
    # its top-level parenthesized component records.
    if re.search(r"(?im)OrCAD[- /]?0?1|EESchema Netlist", text[:4096]) and re.search(r"(?m)^\s*\(\s*[A-Fa-f0-9]{6,}\s+\S+\s+[A-Za-z#][A-Za-z0-9_.$-]*\d+", text):
        return parse_orcad_pcb2(text, str(path))

    if stripped.startswith("<"):
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError(f"XML-like input is malformed: {exc}") from exc
        if root.tag == "export":
            return parse_kicad_xml_root(root, str(path))
        if root.tag == "eagle":
            return parse_eagle_xml(root, str(path))
        raise ValueError(f"XML root '{root.tag}' is not a supported netlist export. Supported XML roots: KiCad <export>, EAGLE <eagle>.")

    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON-like input is malformed: {exc}") from exc
        return parse_easyeda_pro_json(payload, str(path))

    if stripped.startswith("("):
        if re.match(r"^\(\s*edif\b", stripped, re.I):
            return parse_edif(text, str(path))
        # KiCad S-expression has an explicit export root; EasyEDA/Protel exports commonly group nets.
        if re.match(r"^\(\s*export\b", stripped):
            return parse_kicad_sexpr(text, str(path))
        try:
            return parse_kicad_sexpr(text, str(path))
        except ValueError:
            try:
                return parse_structured_parenthesized(text, str(path))
            except ValueError:
                return parse_grouped_netlist(text, str(path))

    # Sectioned PADS/Allegro/OrCAD exports before SPICE; their headers may begin with '*'.
    if re.search(r"(?im)^\s*\*(?:PADS|NET|SIGNAL|PART|COMPONENT)", text[:8192]):
        try:
            return parse_sectioned_netlist(text, str(path))
        except ValueError:
            pass

    # CSV/TSV header detection before SPICE. Detect field identities independently
    # of their order, because exports differ in whether net/ref/pin are first.
    if ext in {".csv", ".tsv"} or _looks_like_tabular_connectivity_header(text[:4096]):
        return parse_tabular_netlist(path, text)

    spice_candidate = ext in {".cir", ".sp", ".spice", ".ckt", ".spi"} or bool(re.search(r"(?im)^\s*[RCLDQMVIX][A-Za-z0-9_$.-]+\s+\S+\s+\S+\s+\S+", text))
    if spice_candidate:
        return parse_spice(text, str(path))

    # Vendor ASCII formats include OrCAD/Mentor `NET GND U1-7 U2-7`,
    # Multiwire `GND U1 7`, and tabular `GND ; U1.1` rows.
    try:
        return parse_net_declaration_text(text, str(path))
    except ValueError:
        pass

    # EasyEDA Standard's Protel/PADS/FreePCB style exports often use one
    # parenthesized group per net, including .net files.
    if "(" in text and ")" in text:
        return parse_grouped_netlist(text, str(path))

    if any(marker in text.upper() for marker in ("*PADS-PCB*", "*SIGNAL*", "*NET*")):
        return parse_sectioned_netlist(text, str(path))

    raise ValueError(
        f"Unsupported or unrecognized netlist format for '{path.name}'. Detected text format could not be parsed safely. "
        "Supported formats: KiCad XML and S-expression, KiCad schematic (.kicad_sch via KiCad CLI), EAGLE XML, "
        "EasyEDA Pro .enet JSON and selected EasyEDA/Protel grouped exports, EDIF netlist subset, OrCAD/PCB II, "
        "PADS/sectioned ASCII, IPC-D-356 board-test netlists, common Allegro/Altium-style ASCII/net-node reports, "
        "SPICE/PSpice topology, and CSV/TSV connectivity tables. Export a supported netlist or add an adapter for this format."
    )
