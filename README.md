# Design Assistant Pipeline (Ollama version)

Implements the architecture diagram up to the **Reasoning LLM** stage.
The `Schematic input -> Schematic checker model -> Reasoning LLM` branch
is intentionally left out — add it later as its own module and just feed
its output into `reasoning_llm.reason()` alongside the RAG context.

## 1. Install Ollama + pull models

```bash
# install Ollama itself: https://ollama.com/download
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct     # optional, only for the Reasoning LLM stage
```

If your GPU/RAM can't handle the 14b model, just set
`REASONING_MODEL = "qwen2.5:7b-instruct"` in `models.py` — one line, no
other code changes needed.

## 2. Install Python deps

```bash
pip install -r requirements.txt
```

## 3. Try it standalone

```bash
mkdir -p data/datasheets   # drop your .pdf / .md datasheets here
python pipeline.py
```

First run builds the knowledge graph + vector store (slower — every
chunk gets an LLM call for triple extraction), then prompts you for
requirements text and prints the Reasoning LLM's output.

## 4. Wiring it into your existing GUI

Your `ManageFile` / `IO` classes currently call `generate_data_store`
and `generate_response` from the old HuggingFace/Groq scripts. Swap
those two call sites:

```python
# top of io.py / wherever these were imported
from pipeline import ingest_datasheets, run_pipeline

# in ManageFile.handle_folder_upload():
ingest_datasheets(folder_path)      # was: generate_data_store(folder_path)

# in IO.generate_response_text():
self.bot_reply = run_pipeline(self.entry_text)   # was: generate_response(self.entry_text)
```

Everything else in your `customtkinter` GUI (file tiles, chat history,
mic input) works unchanged — only the two functions it calls under the
hood are different now.

## File map -> diagram blocks

| File | Diagram block |
|---|---|
| `requirements_collector.py` | Requirements collector model |
| `kg_builder.py` | Knowledge graph Builder + Knowledge graph |
| `data_ingest.py` | Embedded dataset |
| `rag.py` | RAG + Related RAG retrieval |
| `components_selector.py` | components selector model |
| `reasoning_llm.py` | Reasoning LLM |
| `pipeline.py` | wiring / orchestration |
| `models.py` | which Ollama model each stage uses |
| `ollama_utils.py` | shared Ollama call helpers |

## Notes

- `kg_builder.py` runs one LLM call per chunk to extract triples — on a
  large datasheet set this is the slow part. If it's too slow, raise
  `chunk_size` in `data_ingest.split_text` so there are fewer, bigger
  chunks per LLM call.
- `rag.retrieve()` uses the same `score_threshold` cutoff idea as your
  original `query_data.py` — tune `k` and `score_threshold` in `rag.py`
  if retrieval feels too sparse or too noisy.
- Everything runs fully local once the models are pulled — no API keys,
  no `.env` needed (you can delete the Groq dependency entirely).
