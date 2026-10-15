# Electronics Design Assistant

A local, evidence-grounded assistant for electronics engineering. It combines technical-document ingestion, semantic retrieval, source and component filtering, EDA netlist parsing, schematic connectivity checks, deterministic engineering calculations, answer verification, and a Streamlit chat interface.

The project is intended to support engineering investigation and preliminary design review. It is not a replacement for the source EDA tool's electrical/design rule checks, circuit simulation, PCB layout analysis, laboratory measurements, or qualified engineering sign-off. A retrieved passage or a `PASS` result for one check does not prove that an entire design is correct.

## Contents

- [Capabilities](#capabilities)
- [Architecture and data flow](#architecture-and-data-flow)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation on Windows / Git Bash](#installation-on-windows--git-bash)
- [Optional OCR setup](#optional-ocr-setup)
- [Ingest technical documents](#ingest-technical-documents)
- [Run the application](#run-the-application)
- [Ask effective engineering questions](#ask-effective-engineering-questions)
- [Netlist and schematic support](#netlist-and-schematic-support)
- [Retrieval and answer verification](#retrieval-and-answer-verification)
- [Deterministic engineering checks](#deterministic-engineering-checks)
- [Tests](#tests)
- [Generated files and storage](#generated-files-and-storage)
- [Troubleshooting](#troubleshooting)
- [Limitations and future extensions](#limitations-and-future-extensions)

## Capabilities

### Technical-document ingestion

- Reads PDF and Markdown documents from `data/datasheets/`.
- Extracts PDF page text and splits it into overlapping chunks for retrieval.
- Optionally applies OCR to scanned or text-sparse PDF pages and standalone images when the required tools are installed.
- Can optionally create locally generated vision descriptions of selected pages or images when configured.
- Extracts PDF tables into structured records and indexes table text separately from ordinary page text.
- Adds document identity metadata to help distinguish relevant documents and component variants from incidental mentions.
- Creates a local Chroma vector database and a NetworkX knowledge graph.

### Evidence-grounded question answering

- Routes questions through a question-first pipeline.
- Can identify component references from a question and/or a supplied design file.
- Can restrict retrieval to documents matching the requested component or source identity.
- Carries source and page metadata through retrieval so the answer can identify its supporting evidence.
- Uses a language model for questions not covered by implemented deterministic checks.
- Applies semantic similarity and a Natural Language Inference (NLI) model to help identify unsupported or contradictory claims on the general answer path.

### Schematic and netlist review

- Parses supported netlist formats and can use `kicad-cli` to export a native KiCad schematic to a netlist when available.
- Reports component inventories and observed nets, and raises review items for issues such as unconnected pins, recognizable power-rail conflicts, and nominal power-pin connections.
- Treats net names as labels in the design data, not as measurements of physical voltages.

### Deterministic engineering calculations

The application is designed as a hybrid analysis system. Document ingestion, evidence retrieval, source attribution, and connectivity checks are shared capabilities. Engineering calculations should be implemented as deterministic Python rules whenever the required values, units, component variant, and operating conditions can be established reliably.

A deterministic check should:

1. Identify the target component or design element and its exact variant, where relevant.
2. Retrieve matching source evidence and preserve the source, page, parameter, unit, Min/Typ/Max qualifier, and test conditions.
3. Convert extracted evidence into structured fields instead of asking a language model to transcribe numbers from long passages at answer time.
4. Run a calculation only when its required inputs and conditions are known.
5. Build result tables and citations from structured data.
6. Report missing, ambiguous, or mismatched evidence as `UNKNOWN` or `NOT FOUND`, rather than filling gaps with model-generated values.

The codebase contains general netlist-level checks and an initial, limited set of deterministic power-analysis rules. The deterministic component-specific coverage is not universal: a component must not be considered supported merely because it resembles one handled by an existing rule. The architecture is intended to be extensible, with new rule modules added and tested as engineering domains are implemented.

## Architecture and data flow

```text
Technical PDFs / Markdown / optional images
                    |
                    v
        data_ingest.py / table_extractor.py
          |                      |
          |                      +--> extracted PDF tables
          v
    page text + optional OCR / vision records
                    |
                    v
          Local embedding model
                    |
                    v
             Chroma vector DB <---- source/component filtering (rag.py)
                    |
                    +----------------------------+
                    |                            |
                    v                            v
        General question path          Deterministic rule path
        reasoning_llm.py               power_review.py + Python calculations
                    |                            |
                    v                            v
          verification.py / NLI          Structured analysis report
                    |                            |
                    +-------------+--------------+
                                  v
                         Final answer to GUI/CLI

Optional design input --> netlist_adapters.py --> schematic_checker.py
                                                   |
                                                   +--> connectivity evidence

Ingestion also builds --> knowledge_graph.graphml --> kg_view.html
```

The Streamlit GUI calls the existing `run_pipeline()` function. It captures console diagnostics so the chat displays the final answer instead of raw backend debugging output.

**Implementation note:** `requirements_collector.py` and `components_selector.py` are helper modules in the codebase, but the current main question-answering flow does not call them. The knowledge graph is built during ingestion; the current question retrieval path uses Chroma through `rag.py`.

## Project structure

The repository should contain the files below at its root. The names shown here are the canonical project filenames.

```text
project-root/
├── app.py                         # Streamlit GUI
├── pipeline.py                    # Main question-routing and answer pipeline
├── ingest.py                      # CLI entry point for full ingestion
├── diagnose.py                    # Text-extraction and chunk diagnostic
├── data_ingest.py                 # Document loading, OCR, chunking, indexing
├── document_processor.py          # Coordinates ingestion stages
├── table_extractor.py             # PDF table extraction
├── rag.py                         # Chroma retrieval and source/component filtering
├── reasoning_llm.py               # General LLM reasoning
├── verification.py                # Citation, similarity, and NLI checks
├── power_review.py                # Initial deterministic power-analysis rules
├── schematic_checker.py           # Netlist connectivity checks and schematic export
├── netlist_adapters.py            # Content-detected netlist parser adapters
├── ollama_utils.py                # Ollama chat and embedding wrappers
├── models.py                      # Central model configuration
├── requirements_collector.py      # Conservative requirement parsing helper
├── components_selector.py         # Component-selection helper
├── kg_builder.py                  # Knowledge graph construction and queries
├── kg_visualize.py                # Interactive knowledge-graph visualization
├── requirements.txt               # Backend/runtime Python dependencies
├── requirements-gui.txt           # Streamlit dependency
├── test_document_enrichment.py
├── test_netlist_adapters.py
├── test_power_review.py
├── test_rag_source_filter.py
├── test_requirements_collector.py
├── test_schematic_checker.py
├── test_verification.py
├── data/
│   └── datasheets/                # Technical PDFs and Markdown documents
├── user_uploads/
│   └── netlists/                  # Design files uploaded through the GUI
├── chroma/                        # Local persisted vector database (generated)
├── extracted_tables.json          # Extracted table records (generated)
├── knowledge_graph.graphml        # Saved graph (generated)
├── kg_view.html                   # Interactive graph page (generated)
└── knowledge_base_status.json     # Ingestion summary (generated)
```

The vector database, extracted tables, graph, and ingestion-status JSON are generated artifacts. They can be recreated by running ingestion over the current document library.

## Requirements

### Software

- Python 3.x compatible with the installed dependency versions. The repository does not currently pin a Python version.
- [Ollama](https://ollama.com/) installed and running locally.
- The embedding and reasoning models configured in `models.py` pulled into Ollama.
- Tesseract OCR is optional for scanned or image-only content.
- KiCad's `kicad-cli` is optional and is needed only when exporting a native `.kicad_sch` file to a netlist. Providing an already-exported supported netlist avoids that dependency.

### Python packages

`requirements.txt` includes packages for:

- LangChain integrations, ChromaDB, and the Ollama client
- NetworkX and Plotly
- PDF text/table processing
- NumPy and sentence-transformers
- OCR and image handling

The GUI dependency is maintained separately in `requirements-gui.txt`.

## Installation on Windows / Git Bash

Run these commands from the project root—the directory containing `pipeline.py` and `app.py`.

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
```

In PowerShell, activate it with:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-gui.txt
```

### 3. Configure local models

Open `models.py` and note the configured embedding, extraction, and reasoning model identifiers. Make sure Ollama is running and pull each required model by its configured identifier:

```bash
ollama pull <embedding-model-name>
ollama pull <reasoning-model-name>
ollama list
```

If the extraction model is configured separately, pull that model as well. Replace each placeholder with the identifier from `models.py`; the placeholder commands are examples, not literal shell commands.

You can test the embedding model with the Ollama client, using the configured model identifier:

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="<embedding-model-name>")
vector = embeddings.embed_query("technical specification retrieval test")
print("Embedding OK; dimensions:", len(vector))
```

Replace the placeholder in the Python example before running it. If the local connection is refused, confirm Ollama is running and that `ollama list` works.

### 4. Optional: use a different reasoning model

Pull the chosen model into Ollama and update `models.py` or select it through the GUI if it is available in the model selector. Larger models may be slower and may require CPU offload on a machine with limited GPU memory. Changing the reasoning model does not necessarily change the embedding model or other fixed model references in the code.

## Optional OCR setup

Native PDF text extraction is attempted first. OCR adds searchable text to pages with sparse native text, but OCR can misread symbols, units, or numbers. Verify critical engineering values against the original page or table.

### Windows Tesseract setup

Install the Tesseract OCR executable, then set its path for the current Git Bash session. If installed at the path below, use:

```bash
export TESSERACT_CMD='C:/Program Files/Tesseract-OCR/tesseract.exe'
export PATH="/c/Program Files/Tesseract-OCR:$PATH"
tesseract --version
python -c "from data_ingest import _configure_tesseract; print('Tesseract ready:', _configure_tesseract())"
```

These `export` values last only for the current terminal session. Put them in your shell startup file to configure them persistently.

Useful optional environment variables:

| Variable | Purpose | Default in current code |
|---|---|---|
| `TESSERACT_CMD` | Explicit path to the Tesseract executable | Not set; code checks `PATH` and pytesseract configuration |
| `EDA_OCR_ENABLED` | Enables/disables OCR (`0`, `false`, or `no` disables it) | Enabled |
| `EDA_OCR_MIN_NATIVE_CHARS` | Native-text character threshold below which a page is considered sparse | `120` |
| `EDA_VISION_MODEL` | Optional Ollama vision model for bounded image/page descriptions | Empty; disabled |
| `EDA_VISION_MAX_PAGES` | Maximum pages considered by the optional vision path | `40` |
| `EDA_VISION_MIN_DRAWINGS` | Drawing-count threshold that may select pages for vision review | `12` |

Vision descriptions are model-generated context, not authoritative engineering specifications. A generated caption should not replace a value extracted from a source table or verified against the original document.

## Ingest technical documents

### 1. Add documents

Place supported source documents in:

```text
data/datasheets/
```

The ingestion loader supports PDF and Markdown files and can also index raster images when the required image/OCR tools are available. The GUI uploader currently accepts PDF and Markdown files.

### 2. Run ingestion

```bash
python ingest.py
```

The ingestion pipeline:

1. Loads page text from PDFs and content from Markdown files.
2. Optionally creates OCR and vision-description records.
3. Adds document identity metadata using available title/front-matter content and extracted identifiers.
4. Splits text into overlapping chunks for retrieval.
5. Extracts tables from PDFs using `pdfplumber` and stores them as separate searchable documents.
6. Embeds text/table documents using the configured embedding model and persists them in Chroma under `chroma/`.
7. Builds and saves `knowledge_graph.graphml` and `knowledge_base_status.json`; the CLI also creates `kg_view.html` by default.

Review the ingestion summary, including per-document page, chunk, OCR, and table counts. A document that produces little extracted text may not be usable for reliable answers.

**Data warning:** the current indexing implementation clears existing records from the active Chroma collection before inserting the rebuilt index. Back up `chroma/` first if you need to preserve the previous database. If insertion fails partway through, check that Ollama is healthy and rerun ingestion only after reviewing the state of the index.

Re-ingest after adding or changing source documents, or after changing ingestion/index metadata. Editing only GUI presentation or answer prompts does not normally require re-ingestion.

### Optional diagnostic

```bash
python diagnose.py
```

This prints page/text/chunk counts by source and helps identify files that produced little or no extracted content. It does not build the vector database or knowledge graph.

## Run the application

### Option A: Streamlit GUI

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. If Streamlit emits repeated file-watcher or `torchvision` warnings while the interface otherwise works, try:

```bash
streamlit run app.py --server.fileWatcherType none
```

The GUI includes:

- **Chat:** ask technical-document or design questions and see the final answer.
- **Upload documents:** add PDF/Markdown sources to the configured document directory.
- **Ingest:** explicitly rebuild the shared document index.
- **Netlist / schematic:** upload or select a design file for connectivity-aware questions. This is optional for document-only questions.
- **Configuration:** select an installed reasoning model when supported by the GUI, and manage the current chat session.
- **Report rendering:** display citations, Markdown tables, and review statuses.

The selected netlist is applied to new questions; changing it does not recompute earlier answers.

### Option B: Command line

```bash
python pipeline.py
```

Enter the question at the design-requirements prompt. At the separate prompt, enter the path to a supported netlist/schematic or press Enter to skip it. The CLI accepts the question on one line. Type `exit` or `quit` to stop.

Example questions:

```text
For <PART_NUMBER>, what are the specified input range and maximum output current? Cite the source and page, including relevant conditions.
```

```text
Using the selected netlist, trace the target component's supply, output, enable, and ground connections. Distinguish net labels from measured voltages.
```

```text
Review this design against the provided requirements. Separate source facts, calculations, confirmed connectivity, risks, and unknowns. Do not invent missing specifications.
```

Replace placeholders with the component or design details relevant to your project. Provide the design-file path at the separate prompt, or upload/select the file in the GUI.

## Ask effective engineering questions

Questions that specify the component, parameter, unit, operating condition, and desired evidence are more likely to retrieve the right source passage. Examples:

- `For <PART_NUMBER> and the exact variant in the supplied document, find the recommended operating range. Distinguish recommended conditions from absolute maximum ratings and test conditions.`
- `Extract the minimum, typical, and maximum values of <PARAMETER>. Include units, test conditions, source filename, and page; report missing fields as unknown.`
- `For the stated input voltage, output voltage, and load current, calculate power dissipation. Show the formula and clearly label any assumed inputs.`
- `Using the supplied netlist, trace the relevant power and signal connections. Report only observed connectivity and identify missing or ambiguous connections.`
- `Review the design against the supplied requirements. Separate manufacturer facts, derived calculations, netlist observations, risks, and unknowns.`

For a board-specific review, provide a schematic or netlist. Without one, the application cannot establish actual component connectivity. A nominal net name does not prove a physical voltage, current, or operating state. If a requested value is absent or the applicable variant is ambiguous, the intended behavior is to report it as unknown instead of borrowing a value from an unrelated source.

## Netlist and schematic support

`netlist_adapters.py` detects input formats primarily from file content and then attempts supported parsers. Current adapters include:

- KiCad XML netlist and KiCad S-expression export
- Native KiCad schematic input through `kicad-cli` export
- EAGLE XML
- EasyEDA Pro JSON and selected EasyEDA/Protel grouped exports
- A supported EDIF netlist subset
- OrCAD/PCB II component-centric text
- PADS/sectioned ASCII exports
- IPC-D-356 board-test netlists
- Selected Allegro/Altium-style ASCII net/node reports
- SPICE/PSpice topology
- CSV/TSV connectivity tables
- Common vendor ASCII `NET`/`SIGNAL` declarations

This is adapter-specific support, not a guarantee that every version or export from every EDA tool will work. The GUI upload picker exposes a subset of supported formats; other formats may be parsed when made available to the backend. For unsupported or proprietary formats, export a supported netlist from the source EDA application.

The schematic checker identifies observed connections and raises conservative review prompts. It is not equivalent to electrical rule checking (ERC), design rule checking (DRC), simulation, or PCB layout analysis. Some export formats omit library pin metadata, so pin-level checks may be incomplete.

## Retrieval and answer verification

### Strict retrieval (`rag.py`)

- Uses the configured embedding model with the local Chroma store.
- In strict mode, candidate chunks must pass document/component identity checks and a relevance threshold configured in the retrieval code.
- Document front matter can take precedence over incidental mentions later in a file. This helps prevent an unrelated source from being treated as authoritative for the requested component.
- Retrieved results carry source, page, content type, table index, matched identity, and relevance metadata where available.
- If strict matching cannot find suitable evidence, the system should avoid substituting an unrelated high-scoring passage for a component-specific answer.

### General answer path and NLI (`verification.py`)

The general generated-answer path includes checks for:

1. **Citation availability:** whether evidence sources were provided. This does not prove that a citation supports a particular sentence.
2. **Semantic similarity:** comparison between answer claims and available evidence using embeddings.
3. **Natural Language Inference:** a cross-encoder classifies claim/evidence pairs as contradiction, entailment, or neutral.
4. **Source/component match:** checks whether evidence for the requested identity is available in strict mode.

NLI is probabilistic. It can help flag unsupported or contradictory prose, but it is not an electrical rules engine and does not independently validate arithmetic, OCR quality, table-column alignment, units, or whether a test condition applies to the design. Numeric and unit checks should be deterministic where possible.

The deterministic report path returns its own structured result before the general LLM/NLI path. The current pipeline does not independently NLI-verify every such report.

## Deterministic engineering checks

`power_review.py` is the initial specialist rule module in the deterministic layer. It uses source-labelled retrieved evidence and can read relevant local PDF pages with PyMuPDF, parse structured facts, use available connectivity evidence, and calculate supported power/thermal scenarios. The implemented specialist coverage is limited; it should not be interpreted as a universal rule engine for all components or engineering domains.

### Checks shared across designs

- Supported netlist parsers report component and net information from the supplied file.
- The schematic checker raises review items for recognizable rail conflicts, unconnected pins, and selected nominal power-pin connections.
- Strict retrieval is designed to avoid using evidence from an unrelated source or component identity.
- Source/page metadata is preserved when available so factual claims can be traced to the originating material.
- Missing values and incomplete connectivity should remain explicit unknowns instead of being silently inferred.
- Calculations should use explicit inputs and labelled assumptions; example scenarios must not be presented as measured values.
- Thermal estimates based on simplified package parameters are screening calculations, not PCB thermal sign-off. Actual behavior depends on board layout, copper, placement, ambient conditions, loads, and duty cycle.

### Extension model for new engineering domains

To add reliable deterministic support for a new component family or analysis topic, implement a small rule module with the following responsibilities:

1. **Identity matcher:** exact part number, variant, package, and any safe aliases. Do not silently substitute between variants.
2. **Fact extractor:** structured parameters with units, Min/Typ/Max qualifiers, conditions, source, and page. Reject ambiguous table columns.
3. **Calculation functions:** pure Python functions that accept explicit inputs and return values, units, and assumptions. Do not delegate arithmetic to an LLM.
4. **Rule evaluator:** compare extracted facts with schematic/BOM/user requirements and return a status for each individual check.
5. **Report builder:** construct tables and citations from structured results, not from free-form generated prose.
6. **Regression tests:** cover correct evidence, mismatched variants, missing values, malformed/OCR text, unit conversion, boundary values, and known calculations.

Possible future domains include power conversion and thermal checks, processor supply/start-up constraints, amplifier operating ranges, interface voltage compatibility, passive-component derating, and connector/current constraints. These are examples of extension opportunities, not claims that all those specialist checks are currently implemented. Until a specific rule set has been added and tested, the general retrieval/LLM path remains an evidence assistant for that topic—not a deterministic sign-off engine.

### Interpreting statuses

- `PASS`: the specific implemented check is supported by available evidence. It does not mean the whole board passes all checks.
- `FAIL`: the rule found evidence of a concrete incompatibility or violation. Missing evidence alone should not create a failure.
- `REVIEW`: evidence indicates a risk or dependency that needs further engineering review.
- `UNKNOWN`: the required component identity, specification, condition, or design detail was not established.
- `SUPPORTED`: a source passage supports an extracted fact. This is not equivalent to a design passing a check.

## Tests

Run tests from the project root with the virtual environment activated:

```bash
python -m unittest discover -v
```

The test modules cover these areas:

| Test module | Coverage |
|---|---|
| `test_document_enrichment.py` | Document identity metadata and identifier extraction |
| `test_netlist_adapters.py` | Netlist parser adapters and representative fixtures |
| `test_power_review.py` | Deterministic report formatting, calculations, evidence selection, and missing-data behavior |
| `test_rag_source_filter.py` | Document identity and strict retrieval filtering |
| `test_requirements_collector.py` | Explicitly labelled user constraints; unknown values remain unknown |
| `test_schematic_checker.py` | Parsing, power-rail review, unconnected-pin prompts, and nominal power-pin checks |
| `test_verification.py` | Claim/evidence verification behavior for supported and contradictory examples |

Run a single test module when diagnosing a particular area:

```bash
python -m unittest -v test_power_review.py
python -m unittest -v test_rag_source_filter.py
python -m unittest -v test_netlist_adapters.py
python -m unittest -v test_verification.py
```

The first use of sentence-transformers may require downloading the configured NLI model if it is not cached locally. Tests that load a full runtime model may take longer than pure-Python parser or calculation tests.

## Generated files and storage

| Path | Purpose |
|---|---|
| `data/datasheets/` | Source technical documents used for ingestion |
| `chroma/` | Persistent embedding/vector database used by retrieval |
| `extracted_tables.json` | Extracted table records with source/page and identity metadata |
| `knowledge_graph.graphml` | Saved knowledge graph |
| `kg_view.html` | Optional interactive graph visualization |
| `knowledge_base_status.json` | Last ingestion summary |
| `user_uploads/netlists/` | Design files uploaded through the GUI |

Back up source documents and any generated outputs you need before deleting files or rebuilding the index.

## Troubleshooting

### Ollama connection refused or a local runner error

The Ollama service or one of its local model runners may have stopped. Check:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

Run the embedding smoke test described above. If Ollama is not running, start or restart it and retry. If ingestion failed after clearing the active Chroma collection, the previous index may no longer be intact; ensure the next ingestion completes before asking questions.

### Scanned documents produce no OCR records

Check:

```bash
tesseract --version
python -c "from data_ingest import _configure_tesseract; print(_configure_tesseract())"
```

Ensure the Python OCR/image packages and Tesseract executable are installed. Set `TESSERACT_CMD` to the executable path if Tesseract is not available through `PATH`.

### No matching source found

- Confirm the relevant source document is present in `data/datasheets/`.
- Check that its title/front matter identifies the intended component or document variant clearly enough for matching.
- Re-run `python ingest.py` after adding, removing, or replacing a source so the index and identity metadata are refreshed.
- Do not weaken strict source filtering solely to force an answer from an unrelated document.

### Streamlit reports a file-watcher or `torchvision` warning

If the warning is emitted by the file watcher and the application otherwise starts, stop it with `Ctrl+C` and launch with:

```bash
streamlit run app.py --server.fileWatcherType none
```

This disables file watching; it does not fix unrelated backend errors. If the application fails to open, capture the final exception from the terminal.

### Output is generic, truncated, or marked unverified

- Check the retrieved source/page evidence in the CLI diagnostics. The GUI normally shows only the final answer.
- Confirm the appropriate source document was indexed and that its tables and text extracted correctly.
- Use deterministic rules only for checks that have actually been implemented and tested. Other domains rely more heavily on the general evidence/LLM path.
- `UNVERIFIED` does not automatically mean the design is wrong. It means the current checks did not establish the answer with sufficient evidence.

## Limitations and future extensions

1. **Limited specialist rule coverage:** the deterministic analysis framework is extensible, but only a subset of checks currently has dedicated rule implementations. Other component families and design topics rely more heavily on retrieval and LLM reasoning until their rules and tests exist.
2. **PDF extraction quality:** table extraction may lose merged cells, columns, symbols, or units. OCR may misread values. Check the original source for critical specifications.
3. **NLI limits:** the NLI model can misclassify technical claims and must be combined with exact source attribution and deterministic numeric/unit checks.
4. **Netlist is not PCB layout:** connectivity data does not contain all copper geometry, component placement, physical measurements, or thermal-spreading information.
5. **Operating values must be established:** a recommended source capability, test condition, typical value, or nominal net label must not automatically be treated as the actual design load or measured operating state.
6. **Knowledge-graph scope:** a graph is generated and visualized, but Chroma/RAG remains the primary retrieval mechanism in the current pipeline.
7. **Upload restrictions:** the parser can support more formats than the GUI upload picker exposes.
8. **No complete design certification:** validate physical designs using the appropriate EDA checks, simulation, layout/thermal analysis, and bench measurements.

## Development guidance

Extend deterministic analysis one component family or rule at a time. Preserve source-cited fields, operating conditions, units, assumptions, and regression tests for both positive and negative cases. When a fact cannot be extracted confidently, keep it unknown. Accuracy is more valuable than a confident-looking but unsupported answer.
