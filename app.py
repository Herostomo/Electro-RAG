from __future__ import annotations

import contextlib
import hashlib
import html
import io
import os
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure all existing backend paths (Chroma, datasheets, KG) resolve from the project root.
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASHEET_DIR = ROOT / "data" / "datasheets"
UPLOAD_DIR = ROOT / "user_uploads"
NETLIST_UPLOAD_DIR = UPLOAD_DIR / "netlists"
DATASHEET_DIR.mkdir(parents=True, exist_ok=True)
NETLIST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Imported after setting the working directory so the existing project modules are found.
from document_processor import process_documents  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
import reasoning_llm  # noqa: E402
from models import REASONING_MODEL  # noqa: E402


st.set_page_config(
    page_title="Design Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = r"""
<style>
:root {
  --eda-bg: #0a1220;
  --eda-panel: #101c2c;
  --eda-panel-2: #15243a;
  --eda-border: #263b55;
  --eda-text: #edf4ff;
  --eda-muted: #9fb3cc;
  --eda-blue: #367df4;
  --eda-cyan: #5ab8ff;
}
.stApp { background: radial-gradient(ellipse at 12% 0%, #14253d 0, #0a1220 47%, #080f1a 100%); color: var(--eda-text); }
[data-testid="stHeader"] { background: rgba(8, 15, 26, 0.92); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1800px; padding-top: 1.2rem; padding-bottom: 1rem; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #e4edf9; line-height: 1.48; }
.eda-header { display:flex; align-items:center; gap:14px; padding: 2px 0 18px 0; border-bottom:1px solid #23364e; margin-bottom:18px; }
.eda-logo { width:46px; height:46px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:linear-gradient(145deg,#1a69ef,#40b7ff); color:white; font-size:26px; box-shadow:0 8px 25px rgba(42,126,248,.25); }
.eda-title { font-size:27px; font-weight:750; letter-spacing:-.7px; color:#f5f8ff; line-height:1.15; }
.eda-subtitle { color:#9cb4d1; font-size:13px; margin-top:4px; }
.eda-pill { display:inline-block; border:1px solid #315477; color:#a9d7ff; background:#112940; padding:5px 10px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:.5px; }
.panel-title { font-size:16px; font-weight:700; color:#f3f7ff; margin-bottom:3px; }
.panel-help { color:#9cb0c8; font-size:12px; margin-bottom:13px; }
.section-kicker { font-size:11px; font-weight:800; letter-spacing:1px; text-transform:uppercase; color:#83a4cb; margin:2px 0 8px 0; }
.file-chip { display:flex; align-items:center; gap:9px; padding:9px 10px; border:1px solid #29405b; border-radius:10px; background:#0f1d2e; margin:6px 0; color:#dce8f8; font-size:12px; overflow-wrap:anywhere; }
.file-chip-icon { width:29px; height:32px; border-radius:6px; background:#273e59; color:#a8ceff; display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
.file-chip-name { font-weight:600; }
.file-chip-meta { color:#94a8c0; font-size:10px; margin-top:2px; }
.empty-state { border:1px dashed #2e4968; border-radius:17px; background:rgba(17,31,49,.64); padding:38px 24px; text-align:center; margin:12px 0 22px 0; }
.empty-icon { font-size:39px; margin-bottom:10px; }
.empty-title { color:#edf4ff; font-size:20px; font-weight:750; }
.empty-sub { color:#9db1ca; font-size:13px; margin:7px auto 0 auto; max-width:500px; }
.verdict-card { border:1px solid #886c2b; border-radius:10px; background:linear-gradient(110deg,rgba(116,79,14,.22),rgba(39,36,32,.28)); padding:9px 12px; margin:6px 0 10px 0; display:flex; gap:9px; align-items:center; }
.verdict-card.review { border-color:#946c23; background:linear-gradient(110deg,rgba(125,83,8,.28),rgba(34,35,40,.28)); }
.verdict-card.fail { border-color:#8c3945; background:linear-gradient(110deg,rgba(129,33,46,.28),rgba(34,35,40,.28)); }
.verdict-card.pass { border-color:#216a54; background:linear-gradient(110deg,rgba(19,112,78,.22),rgba(26,39,48,.28)); }
.verdict-icon { font-size:18px; }
.verdict-label { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#a8bad0; font-weight:800; }
.verdict-text { font-size:16px; line-height:1.25; color:#ffc84c; font-weight:800; }
.verdict-card.fail .verdict-text { color:#ff818b; }
.verdict-card.pass .verdict-text { color:#56d8a4; }
.table-wrap { width:100%; overflow-x:auto; border:1px solid #2a405b; border-radius:11px; margin:8px 0 16px 0; }
table.eda-table { width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.42; background:rgba(11,22,36,.72); color:#e6eef9; }
table.eda-table th { text-align:left; background:#1a2a41; color:#cfe3ff; padding:11px 12px; border-bottom:1px solid #35506d; font-weight:750; white-space:normal; }
table.eda-table td { padding:10px 12px; vertical-align:top; border-bottom:1px solid #21344a; overflow-wrap:anywhere; }
table.eda-table tr:nth-child(even) td { background:rgba(28,46,67,.27); }
table.eda-table tr:last-child td { border-bottom:none; }
.status-pill { display:inline-block; min-width:60px; text-align:center; border-radius:7px; padding:4px 8px; font-size:10px; font-weight:850; letter-spacing:.4px; white-space:nowrap; }
.status-pass { color:#c5ffe4; background:#08784f; border:1px solid #149669; }
.status-fail { color:#ffe0e2; background:#962f3e; border:1px solid #c54b5b; }
.status-review { color:#372500; background:#ffc746; border:1px solid #e7a923; }
.status-unknown { color:#e2e9f4; background:#52657b; border:1px solid #7488a0; }
.citation { color:#8fcaff; font-size:11.5px; }
.assistant-note { border-left:3px solid #368af2; background:rgba(24,48,76,.42); border-radius:0 9px 9px 0; padding:10px 13px; margin:9px 0; }
[data-testid="stChatMessage"] { background:rgba(19,32,49,.72); border:1px solid #243a53; border-radius:14px; padding:12px 15px; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { overflow-wrap:anywhere; }
[data-testid="stChatInput"] { border-color:#35506e; background:#101d2e; }
[data-testid="stChatInput"] textarea { color:#eef5ff; }
[data-testid="stFileUploader"] { background:rgba(15,29,46,.55); border:1px dashed #3a5c80; border-radius:11px; padding:7px; }
[data-testid="stFileUploaderDropzone"] { background:#122238; border:1px dashed #355c83; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:#b6cce5; }
.stButton > button { border-radius:10px; min-height:42px; font-weight:700; border:1px solid #355376; transition:all .15s ease; }
.stButton > button[kind="primary"] { background:linear-gradient(120deg,#286cf0,#3188fa); border:1px solid #4e96ff; color:white; }
.stButton > button:hover { border-color:#66b7ff; transform:translateY(-1px); }
[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(16,29,45,.78); border-color:#263e59; border-radius:15px; }
[data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label, [data-testid="stFileUploader"] label { color:#d4e2f4; font-weight:650; }
div[data-testid="stAlert"] { border-radius:10px; }
hr { border-color:#263b54; }
.small-muted { color:#94abc5; font-size:11px; }
@media (max-width: 900px) { .eda-title{font-size:22px;} .block-container{padding-left: .8rem; padding-right: .8rem;} table.eda-table{font-size:11px;} table.eda-table th,table.eda-table td{padding:8px;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._ ()-]+", "_", name).strip(" .")
    return name or "uploaded_file"


def _save_upload(uploaded_file: Any, target_dir: Path) -> tuple[Path, bool, str]:
    """Persist uploaded bytes without overwriting a different file with the same name."""
    target_dir.mkdir(parents=True, exist_ok=True)
    raw = uploaded_file.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    cache = st.session_state.setdefault("_saved_upload_hashes", {})
    if digest in cache and Path(cache[digest]).exists():
        return Path(cache[digest]), False, digest

    base = _safe_filename(uploaded_file.name)
    destination = target_dir / base
    if destination.exists():
        try:
            if destination.read_bytes() == raw:
                cache[digest] = str(destination)
                return destination, False, digest
        except OSError:
            pass
        stem, suffix = destination.stem, destination.suffix
        counter = 2
        while destination.exists():
            destination = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    destination.write_bytes(raw)
    cache[digest] = str(destination)
    return destination, True, digest


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def _list_datasheets() -> list[Path]:
    if not DATASHEET_DIR.exists():
        return []
    valid = {".pdf", ".md"}
    return sorted(
        (p for p in DATASHEET_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid),
        key=lambda p: p.name.lower(),
    )


def _list_netlists() -> list[Path]:
    valid = {".net", ".xml", ".kicad_sch", ".sch", ".json", ".txt"}
    candidates: dict[str, Path] = {}
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix.lower() in valid:
            candidates[str(p.resolve())] = p.resolve()
    for p in NETLIST_UPLOAD_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in valid:
            candidates[str(p.resolve())] = p.resolve()
    return sorted(candidates.values(), key=lambda p: p.name.lower())


def _ollama_models() -> list[str]:
    installed: list[str] = []
    try:
        import ollama
        response = ollama.list()
        items = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        for item in items or []:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            else:
                name = getattr(item, "model", None) or getattr(item, "name", None)
            if name and str(name) not in installed:
                installed.append(str(name))
    except Exception:
        pass
    if not installed:
        installed = [REASONING_MODEL]
    return installed


def _normalise_citations(text: str) -> str:
    # Turn verbose internal source headers into reader-friendly source/page citations.
    def replace_source(match: re.Match[str]) -> str:
        file_name = Path(match.group(1).strip().replace("\\", "/")).name
        page = match.group(2)
        return f"{file_name} · p. {page}"

    text = re.sub(
        r"\[SOURCE:\s*([^;\]]+);\s*page\s*(\d+)[^\]]*\]",
        replace_source,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?i)(?:data[\\/])?datasheets[\\/]", "", text)
    # Relevance scores aren't useful in the engineer-facing report.
    text = re.sub(r"\s*\|?\s*relevance(?:_score)?\s*[=:]\s*[0-9.]+", "", text, flags=re.I)
    return text


def _inline_html(value: str) -> str:
    value = html.escape(_normalise_citations(value), quote=False)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(
        r"\b(PASS|FAIL|REVIEW|UNKNOWN)\b",
        lambda m: m.group(0),
        value,
    )
    return value


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    cells = [c.strip() for c in stripped.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells)


def _render_html_table(rows: list[list[str]]) -> None:
    if not rows:
        return
    headers = rows[0]
    body_rows = rows[1:]
    head_html = "".join(f"<th>{_inline_html(cell)}</th>" for cell in headers)
    body_html: list[str] = []
    for row in body_rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        cells_html = []
        for index, cell in enumerate(padded[: len(headers)]):
            plain = re.sub(r"[*`_]", "", cell).strip().upper()
            if plain in {"PASS", "FAIL", "REVIEW", "UNKNOWN"}:
                css = {
                    "PASS": "status-pass",
                    "FAIL": "status-fail",
                    "REVIEW": "status-review",
                    "UNKNOWN": "status-unknown",
                }[plain]
                content = f'<span class="status-pill {css}">{plain}</span>'
            else:
                content = _inline_html(cell)
                # Source/page citations are visually distinct but compact.
                content = re.sub(
                    r"([A-Za-z0-9_. -]+\.pdf\s*·\s*p\.\s*\d+)",
                    r'<span class="citation">\1</span>',
                    content,
                    flags=re.I,
                )
            cells_html.append(f"<td>{content}</td>")
        body_html.append("<tr>" + "".join(cells_html) + "</tr>")
    table_html = (
        '<div class="table-wrap"><table class="eda-table"><thead><tr>'
        + head_html
        + "</tr></thead><tbody>"
        + "".join(body_html)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_assistant_answer(answer: str) -> None:
    """Render only the final answer with styled verdict, tables, and status labels."""
    answer = _normalise_citations(answer.strip())
    if not answer:
        st.markdown("*The assistant returned an empty response. Please try again.*")
        return

    verdict_match = re.search(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:\*\*)?Verdict\s*:\s*(.+?)(?:\*\*)?\s*$",
        answer,
    )
    if verdict_match:
        verdict = verdict_match.group(1).strip().strip("* ")
        folded = verdict.upper()
        style = "pass" if "SUITABLE" in folded and "NOT SUITABLE" not in folded else (
            "fail" if "NOT SUITABLE" in folded or "FAIL" in folded else "review"
        )
        icon = "✓" if style == "pass" else ("×" if style == "fail" else "⚠")
        st.markdown(
            f'<div class="verdict-card {style}"><div class="verdict-icon">{icon}</div>'
            f'<div><div class="verdict-label">Design assessment</div>'
            f'<div class="verdict-text">{html.escape(verdict)}</div></div></div>',
            unsafe_allow_html=True,
        )
        answer = answer[: verdict_match.start()] + answer[verdict_match.end() :]

    lines = answer.splitlines()
    prose: list[str] = []
    index = 0
    while index < len(lines):
        if (
            lines[index].strip().startswith("|")
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        ):
            if prose:
                block = "\n".join(prose).strip()
                if block:
                    st.markdown(block)
                prose = []
            rows = [_split_table_row(lines[index])]
            index += 2  # Skip the separator row.
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_split_table_row(lines[index]))
                index += 1
            _render_html_table(rows)
            continue
        prose.append(lines[index])
        index += 1

    if prose:
        block = "\n".join(prose).strip()
        if block:
            st.markdown(block)


def _render_file_chip(path: Path, kind: str) -> None:
    try:
        size = _format_size(path.stat().st_size)
    except OSError:
        size = ""
    icon = "PDF" if kind == "pdf" else "NET"
    st.markdown(
        '<div class="file-chip">'
        f'<div class="file-chip-icon">{icon}</div>'
        f'<div><div class="file-chip-name">{html.escape(path.name)}</div>'
        f'<div class="file-chip-meta">{html.escape(size)}</div></div></div>',
        unsafe_allow_html=True,
    )


def _ingest_datasheets() -> tuple[dict | None, str | None, str]:
    """Call the existing ingestion pipeline while suppressing console chatter."""
    logs = io.StringIO()
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            status = process_documents(str(DATASHEET_DIR), visualize=False)
        return status, None, logs.getvalue()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", logs.getvalue()


def _answer_question(question: str, netlist_path: str | None, model_name: str) -> tuple[str, str]:
    """Call the normal project pipeline but return only its final answer to the chat."""
    logs = io.StringIO()
    reasoning_llm.REASONING_MODEL = model_name
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            answer = run_pipeline(question, netlist_path)
        return str(answer or "The assistant did not return an answer."), logs.getvalue()
    except Exception as exc:
        return (
            "**Analysis could not be completed.**\n\n"
            f"The backend raised `{type(exc).__name__}: {exc}`. Check the local Ollama service and project dependencies, then try again.",
            logs.getvalue(),
        )


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_ingestion_status" not in st.session_state:
    st.session_state.last_ingestion_status = None
if "latest_diagnostics" not in st.session_state:
    st.session_state.latest_diagnostics = ""
if "selected_netlist_path" not in st.session_state:
    st.session_state.selected_netlist_path = ""
if "reasoning_model" not in st.session_state:
    st.session_state.reasoning_model = REASONING_MODEL

# App header
st.markdown(
    '<div class="eda-header">'
    '<div class="eda-logo">▦</div>'
    '<div style="flex:1"><div class="eda-title">Design Assistant</div>'
    '<div class="eda-subtitle">Ask questions about your electronics design using datasheets and schematics</div></div>'
    '<div class="eda-pill">LOCAL · DATASHEET-GROUNDED</div></div>',
    unsafe_allow_html=True,
)

chat_col, config_col = st.columns([2.75, 1.0], gap="large")
selected_netlist: str | None = None
selected_model = REASONING_MODEL

with config_col:
    # Panel 1: Datasheets + ingestion
    with st.container(border=True):
        st.markdown('<div class="panel-title">① Upload Datasheets</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-help">Add manufacturer PDF or Markdown datasheets.</div>', unsafe_allow_html=True)
        uploaded_pdfs = st.file_uploader(
            "Choose datasheet files",
            type=["pdf", "md"],
            accept_multiple_files=True,
            key="datasheet_uploader",
            label_visibility="collapsed",
            help="Files are saved to data/datasheets. Ingestion rebuilds the shared knowledge base from all supported files in that folder.",
        )
        newly_added: list[Path] = []
        if uploaded_pdfs:
            for item in uploaded_pdfs:
                saved_path, created, _ = _save_upload(item, DATASHEET_DIR)
                if created:
                    newly_added.append(saved_path)
        if newly_added:
            st.success(f"Added {len(newly_added)} new file(s) to the datasheet library.")

        datasheets = _list_datasheets()
        st.markdown('<div class="section-kicker">Library files</div>', unsafe_allow_html=True)
        if datasheets:
            for path in datasheets[:7]:
                _render_file_chip(path, "pdf" if path.suffix.lower() == ".pdf" else "doc")
            if len(datasheets) > 7:
                st.caption(f"+ {len(datasheets) - 7} more file(s)")
        else:
            st.info("No datasheets added yet.")

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        if st.button("⚡  Ingest Datasheets", type="primary", use_container_width=True, key="ingest_button"):
            current_files = _list_datasheets()
            if not current_files:
                st.error("Upload at least one supported datasheet before ingestion.")
            else:
                with st.status("Building datasheet knowledge base…", expanded=False) as status_box:
                    result, error, log_text = _ingest_datasheets()
                    st.session_state.latest_diagnostics = log_text
                    if error:
                        status_box.update(label="Datasheet ingestion failed", state="error")
                        st.error(error)
                        with st.expander("Ingestion error details"):
                            st.code(log_text[-8000:] or "No additional log output.")
                    else:
                        st.session_state.last_ingestion_status = result
                        status_box.update(label="Datasheet ingestion complete", state="complete")
                if not error and result:
                    st.success(
                        f"Indexed {result.get('document_count', 0)} page record(s), "
                        f"{result.get('text_chunk_count', 0)} text chunks and "
                        f"{result.get('table_count', 0)} tables."
                    )
                    st.caption("The shared index now includes the whole datasheet library.")

        # The status file may be present after a previous CLI ingestion.
        status_file = ROOT / "knowledge_base_status.json"
        if status_file.exists():
            st.markdown('<span class="small-muted">Knowledge base status file found</span>', unsafe_allow_html=True)
        elif not (ROOT / "chroma").exists():
            st.warning("Knowledge base not found yet. Ingest datasheets before asking questions.")

    # Panel 2: Optional netlist / schematic
    with st.container(border=True):
        st.markdown('<div class="panel-title">② Netlist / Schematic</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-help">Optional · attach a design for connectivity-aware review.</div>', unsafe_allow_html=True)
        uploaded_netlist = st.file_uploader(
            "Upload netlist or schematic",
            type=["net", "xml", "kicad_sch", "sch", "json", "txt"],
            accept_multiple_files=False,
            key="netlist_uploader",
            label_visibility="collapsed",
            help="Use a supported exported netlist or schematic. Arbitrary vendor formats may need to be exported to a supported format first.",
        )
        if uploaded_netlist is not None:
            uploaded_hash = hashlib.sha256(uploaded_netlist.getvalue()).hexdigest()
            last_hash = st.session_state.get("_last_netlist_upload_hash")
            saved_path, created, _ = _save_upload(uploaded_netlist, NETLIST_UPLOAD_DIR)
            if uploaded_hash != last_hash:
                st.session_state.selected_netlist_path = str(saved_path.resolve())
                st.session_state["_last_netlist_upload_hash"] = uploaded_hash
            st.caption("Uploaded design file saved locally.")

        netlists = _list_netlists()
        options = [""] + [str(path.resolve()) for path in netlists]
        if st.session_state.selected_netlist_path not in options:
            st.session_state.selected_netlist_path = ""

        selected_netlist = st.selectbox(
            "Select a netlist/schematic",
            options=options,
            format_func=lambda value: "None — datasheet-only chat" if not value else Path(value).name,
            key="selected_netlist_path",
            help="The selected design will be used for new questions. Existing chat replies are not recomputed when you change it.",
        ) or None

        if selected_netlist:
            selected_path = Path(selected_netlist)
            if selected_path.exists():
                _render_file_chip(selected_path, "net")
        else:
            st.caption("No design file selected. You can still ask datasheet questions.")

    # Panel 3: Configuration
    with st.container(border=True):
        st.markdown('<div class="panel-title">⚙ Configuration</div>', unsafe_allow_html=True)
        available_models = _ollama_models()
        current = st.session_state.get("reasoning_model", REASONING_MODEL)
        if current not in available_models:
            available_models = [current] + available_models
        selected_model = st.selectbox(
            "Reasoning model",
            options=available_models,
            index=available_models.index(current),
            key="reasoning_model",
            help="Choose a model already installed in your local Ollama instance.",
        )
        st.markdown('<div class="panel-help">Runs locally through your existing Python and Ollama backend.</div>', unsafe_allow_html=True)

        if st.button("＋  Start New Analysis", use_container_width=True, key="new_analysis"):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<hr>', unsafe_allow_html=True)
        if st.button("Clear chat history", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

with chat_col:
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">⚡</div>'
            '<div class="empty-title">What are we reviewing today?</div>'
            '<div class="empty-sub">Ask a question about voltage limits, power supply sizing, component selection, datasheet specifications, or the connectivity in your selected schematic.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Suggested first question**")
        st.markdown(
            '<div class="assistant-note">Can the selected regulator safely power this design? Compare its output voltage, current capability, relevant peak loads, and thermal dissipation against the component datasheets. Cite source filenames and pages, and separate confirmed issues from risks and unknowns.</div>',
            unsafe_allow_html=True,
        )

    # Prior conversation messages: display only user prompts and final assistant answers.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_answer(message["content"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about your design, components, or datasheets…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing design evidence and preparing the report…"):
                answer, diagnostic_log = _answer_question(
                    prompt,
                    selected_netlist,
                    selected_model,
                )
            st.session_state.latest_diagnostics = diagnostic_log
            st.session_state.messages.append({"role": "assistant", "content": answer})
            render_assistant_answer(answer)
