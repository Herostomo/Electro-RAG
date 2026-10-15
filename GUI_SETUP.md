# Design Assistant GUI

This adds a Streamlit UI layer to the existing Design Assistant Python project. It does not replace the current backend modules.

## What it provides

- Chat panel with prior conversation and an engineer-friendly final-answer renderer.
- Right-side datasheet library uploader and an explicit **Ingest Datasheets** button.
- Ingestion runs against the whole `data/datasheets/` library, preserving the combined knowledge base.
- Optional netlist/schematic upload and selector.
- Local Ollama reasoning-model selector, populated from installed models where the Ollama API is available.
- Backend processing logs are captured and not dumped into the chat; only the returned Design Assistant response is displayed.
- Markdown tables are rendered as styled HTML tables, citations are shortened to filename/page, and PASS/FAIL/REVIEW/UNKNOWN cells are colour-coded.

## Install and launch (Windows / Git Bash)

From the root folder of the existing project, with its virtual environment active:

```bash
pip install -r requirements-gui.txt
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser. Keep Ollama running locally.

## Installation location

Copy `app.py` and `requirements-gui.txt` into the same root folder as the existing `pipeline.py`, `document_processor.py`, `rag.py`, `reasoning_llm.py`, `verification.py`, `schematic_checker.py`, `models.py`, and the rest of your project files. `GUI_SETUP.md` can stay in the project root too.

The app writes uploaded datasheets into `data/datasheets/` and uploaded design files into `user_uploads/netlists/`. It uses the existing `process_documents()` and `run_pipeline()` functions. It changes the reasoning model variable for the current app process when a different installed model is selected.

## Notes

- The first ingestion can take time and rebuilds the shared vector database from every supported file in `data/datasheets/`; it is not limited to only the most recently uploaded PDF.
- Use supported exported netlist/schematic formats. The GUI accepts common extensions, but the existing parser still decides whether each file's contents are supported.
- A successful retrieval is not an electrical sign-off. Treat REVIEW/UNKNOWN results as requiring engineering assessment.
