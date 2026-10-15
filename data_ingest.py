"""
Document loading, text chunking, and Chroma vector-store construction.

The module keeps ordinary text and extracted tables as separate content types
so table semantics are not lost during PDF ingestion.
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from models import EMBEDDING_MODEL

# Optional PDF/image tooling. Native text extraction still works when these are
# absent, but scanned/image-only pages will remain unavailable without OCR.
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - depends on local installation
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - depends on local installation
    pytesseract = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on local installation
    Image = None

try:
    import ollama
except ImportError:  # pragma: no cover - vision is optional
    ollama = None

CHROMA_PATH = "chroma"
OCR_MIN_NATIVE_CHARS = int(os.getenv("EDA_OCR_MIN_NATIVE_CHARS", "120"))
VISION_MAX_PAGES = int(os.getenv("EDA_VISION_MAX_PAGES", "40"))
VISION_MIN_DRAWINGS = int(os.getenv("EDA_VISION_MIN_DRAWINGS", "12"))


_PART_ID_RE = re.compile(
    r"\b(?:ESP32(?:-[A-Z0-9]+){1,5}|"
    r"(?:AP|LM|LT|LTC|TL|OPA|MCP|INA|ADS|MAX|TPS|AMS|NCP|SN|AD|NE|MC|MIC|STM32|ATmega|BQ|DRV|TUSB|CH340|CP210|"
    r"BME|BMP|MPU|ICM|LSM|LIS|BMI|NRF|RP|GD32|CH32|W25Q|MX25|GD25|PCF|ADXL|SHT|DS18B|USBLC|XC6|XC62)[A-Z0-9]*"
    r"(?:-[A-Z0-9.]+)*)\b",
    re.IGNORECASE,
)


def extract_document_part_numbers(text: str) -> list[str]:
    """Extract conservative electronics part identifiers for document-level RAG filtering."""
    found, seen = [], set()
    for match in _PART_ID_RE.finditer(text or ""):
        value = match.group(0).strip(".,:;()[]")
        key = re.sub(r"[^A-Z0-9]", "", value.upper())
        if key and key not in seen:
            seen.add(key)
            found.append(value)
    return found[:100]


def _tag_document_group(docs: list[Document], source_path: Path) -> list[Document]:
    """Attach document identity to native text and OCR records.

    Vision descriptions are intentionally excluded from identity detection so
    a model-generated caption cannot determine which part a PDF claims to cover.
    """
    identity_docs = [
        doc for doc in docs
        if doc.metadata.get("extraction_method") != "vision"
    ]
    combined = "\n".join(d.page_content or "" for d in identity_docs)
    parts = extract_document_part_numbers(combined)

    # Build the first-page identity from native/OCR text for the first three
    # pages, including OCR records for scanned cover pages.
    by_page: dict[int, list[str]] = {}
    for doc in identity_docs:
        page = int(doc.metadata.get("page", 0))
        by_page.setdefault(page, []).append(doc.page_content or "")
    first_pages = sorted(by_page)[:3]
    front_matter = "\n".join(
        "\n".join(by_page[page]) for page in first_pages
    )[:18000]
    primary_parts = extract_document_part_numbers(front_matter)

    serialized = "|".join(parts)
    primary_serialized = "|".join(primary_parts)
    for doc in docs:
        doc.metadata.setdefault("source", str(source_path))
        doc.metadata["document_name"] = source_path.name
        doc.metadata["document_part_numbers"] = serialized
        doc.metadata["document_primary_part_numbers"] = primary_serialized
    return docs


def _configure_tesseract() -> bool:
    """Return whether Tesseract is callable; supports explicit Windows paths."""
    if pytesseract is None or Image is None:
        return False
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path
    elif not shutil.which("tesseract"):
        # pytesseract itself may still have been configured by the user.
        candidate = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        if not Path(candidate).is_file():
            return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _render_page_png(page, scale: float = 2.5) -> bytes:
    """Render one PyMuPDF page to PNG bytes for OCR or optional vision."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pixmap.tobytes("png")


def _vision_model_name() -> str:
    """A vision-capable Ollama model is opt-in because page-by-page analysis is expensive."""
    return os.getenv("EDA_VISION_MODEL", "").strip()


def _describe_page_image(image_bytes: bytes, model: str, page_number: int, source_name: str) -> str:
    """Extract grounded descriptions of diagrams/charts from a rendered PDF page."""
    if ollama is None:
        raise RuntimeError("The Python 'ollama' package is unavailable; install project requirements first.")
    prompt = (
        "You are extracting evidence from an electronics datasheet page image. "
        "Describe only visible information useful for engineering retrieval: circuit/block diagrams, "
        "pinout labels, arrows and connections, graph axes and units, table headings, timing sequences, "
        "and values that are clearly legible. Do not guess unreadable labels or infer unstated ratings. "
        "If the page has no meaningful diagram/graph or the text is unreadable, say that explicitly. "
        "Keep the result concise, and distinguish observed labels from interpretation."
    )
    result = ollama.generate(
        model=model,
        prompt=f"Source: {source_name}, page {page_number}. {prompt}",
        images=[image_bytes],
        options={"temperature": 0.1, "num_ctx": 4096},
    )
    return str(result.get("response", "")).strip()


def _enrich_pdf_pages(path: Path, page_docs: list[Document]) -> list[Document]:
    """Add OCR text for sparse pages and optional descriptions of diagrams/plots.

    OCR is automatically applied to pages with little native text when Tesseract
    is installed. Vision analysis is optional via EDA_VISION_MODEL; only likely
    visual pages are considered and EDA_VISION_MAX_PAGES bounds the number of
    calls per PDF (set to 0 for no cap).
    """
    ocr_enabled = os.getenv("EDA_OCR_ENABLED", "1").lower() not in {"0", "false", "no"}
    ocr_available = ocr_enabled and _configure_tesseract()
    vision_model = _vision_model_name()
    visual_enabled = bool(vision_model and ollama is not None and fitz is not None and Image is not None)
    if fitz is None or (not ocr_available and not visual_enabled):
        sparse = sum(len((doc.page_content or "").strip()) < OCR_MIN_NATIVE_CHARS for doc in page_docs)
        if sparse:
            if not ocr_enabled:
                print(f"PDF enrichment: OCR disabled; {sparse} sparse-text pages in {path.name} were not OCR-processed.")
            elif not ocr_available:
                print(
                    f"PDF enrichment: OCR unavailable for {path.name} ({sparse} sparse-text pages). "
                    "Install PyMuPDF, Pillow, pytesseract and the Tesseract OCR executable."
                )
        if vision_model and not visual_enabled:
            print(f"PDF enrichment: vision model '{vision_model}' requested but dependencies are unavailable.")
        return page_docs

    extras: list[Document] = []
    ocr_count = 0
    vision_candidates: list[tuple[int, int, int, object, bool]] = []
    try:
        pdf = fitz.open(path)
    except Exception as exc:
        print(f"PDF enrichment: cannot open {path.name} with PyMuPDF: {exc}")
        return page_docs

    with pdf:
        for index, page in enumerate(pdf):
            native_text = (page_docs[index].page_content or "") if index < len(page_docs) else ""
            native_len = len(native_text.strip())
            needs_ocr = ocr_available and native_len < OCR_MIN_NATIVE_CHARS
            image_count = len(page.get_images(full=True))
            try:
                drawing_count = len(page.get_drawings())
            except Exception:
                drawing_count = 0

            if needs_ocr or (visual_enabled and (image_count > 0 or drawing_count >= VISION_MIN_DRAWINGS or native_len < OCR_MIN_NATIVE_CHARS)):
                # Rank likely diagram pages first; sparse-text pages are included too.
                vision_candidates.append((index, image_count, drawing_count, page, needs_ocr))

        # OCR every sparse page; vision calls are separately bounded.
        selected_for_vision: set[int] = set()
        if visual_enabled:
            candidates = sorted(
                vision_candidates,
                key=lambda item: (item[1] > 0, item[2], len((page_docs[item[0]].page_content or "").strip()) < OCR_MIN_NATIVE_CHARS),
                reverse=True,
            )
            if VISION_MAX_PAGES > 0:
                candidates = candidates[:VISION_MAX_PAGES]
            selected_for_vision = {item[0] for item in candidates}

        for index, image_count, drawing_count, page, needs_ocr in vision_candidates:
            page_num = index + 1
            image_bytes = None
            native_text = (page_docs[index].page_content or "") if index < len(page_docs) else ""

            if needs_ocr:
                try:
                    image_bytes = _render_page_png(page, scale=2.5)
                    ocr_text = pytesseract.image_to_string(
                        Image.open(BytesIO(image_bytes)),
                        config="--psm 6",
                    ).strip()
                    if len(ocr_text) >= 20:
                        extras.append(Document(
                            page_content=(
                                f"OCR TEXT FROM {path.name}, PAGE {page_num}\n"
                                f"Extraction method: Tesseract OCR. OCR text may contain recognition errors.\n"
                                f"{ocr_text}"
                            ),
                            metadata={
                                "source": str(path), "page": index,
                                "content_type": "ocr", "extraction_method": "ocr",
                                "document_name": path.name,
                            },
                        ))
                        ocr_count += 1
                except Exception as exc:
                    print(f"OCR failed for {path.name}, page {page_num}: {exc}")

            if visual_enabled and index in selected_for_vision:
                try:
                    if image_bytes is None:
                        image_bytes = _render_page_png(page, scale=1.5)
                    description = _describe_page_image(image_bytes, vision_model, page_num, path.name)
                    if len(description) >= 25 and not description.lower().startswith("this page has no meaningful"):
                        extras.append(Document(
                            page_content=(
                                f"VISUAL DESCRIPTION FROM {path.name}, PAGE {page_num}\n"
                                f"Generated by vision model {vision_model}; verify all critical values against the source page.\n"
                                f"{description}"
                            ),
                            metadata={
                                "source": str(path), "page": index,
                                "content_type": "image_description", "extraction_method": "vision",
                                "vision_model": vision_model, "document_name": path.name,
                            },
                        ))
                except Exception as exc:
                    print(f"Vision analysis failed for {path.name}, page {page_num}: {exc}")

    if ocr_count:
        print(f"PDF enrichment: OCR extracted text from {ocr_count} page(s) of {path.name}.")
    if visual_enabled:
        print(
            f"PDF enrichment: visual analysis enabled for {path.name}; "
            f"considered {len(selected_for_vision)} page(s) using '{vision_model}'."
        )
    return page_docs + extras


def _load_standalone_image(path: Path) -> list[Document]:
    """Turn standalone raster images into searchable OCR and optional vision records."""
    if Image is None:
        print(f"Image skipped: Pillow is unavailable for {path.name}.")
        return []

    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    except Exception as exc:
        print(f"Image skipped: could not open {path.name}: {exc}")
        return []

    records: list[Document] = []
    if os.getenv("EDA_OCR_ENABLED", "1").lower() not in {"0", "false", "no"}:
        if _configure_tesseract():
            try:
                text = pytesseract.image_to_string(image, config="--psm 6").strip()
                if len(text) >= 20:
                    records.append(Document(
                        page_content=(
                            f"OCR TEXT FROM IMAGE {path.name}\n"
                            "Extraction method: Tesseract OCR. OCR text may contain recognition errors.\n"
                            f"{text}"
                        ),
                        metadata={
                            "source": str(path), "page": 0,
                            "content_type": "ocr", "extraction_method": "ocr",
                            "document_name": path.name,
                        },
                    ))
            except Exception as exc:
                print(f"OCR failed for image {path.name}: {exc}")

    vision_model = _vision_model_name()
    if vision_model:
        try:
            description = _describe_page_image(image_bytes, vision_model, 1, path.name)
            if len(description) >= 25:
                records.append(Document(
                    page_content=(
                        f"VISUAL DESCRIPTION FROM IMAGE {path.name}\n"
                        f"Generated by vision model {vision_model}; verify critical values against the image.\n"
                        f"{description}"
                    ),
                    metadata={
                        "source": str(path), "page": 0,
                        "content_type": "image_description", "extraction_method": "vision",
                        "vision_model": vision_model, "document_name": path.name,
                    },
                ))
        except Exception as exc:
            print(f"Vision analysis failed for image {path.name}: {exc}")

    if not records:
        print(
            f"Image {path.name} produced no searchable text. Install Tesseract for OCR, "
            "or configure EDA_VISION_MODEL for diagram/image descriptions."
        )
    return _tag_document_group(records, path)


def load_documents(data_path: str):
    """Load Markdown/PDF pages and tag each page with document-level part identifiers.

    These tags allow retrieval to reject passages from unrelated datasheets even
    when the vector similarity score for an unrelated file happens to be high.
    """
    documents: list[Document] = []
    root = Path(data_path)

    for path in root.rglob("*.md"):
        docs = TextLoader(str(path), encoding="utf-8").load()
        for doc in docs:
            doc.metadata["content_type"] = "text"
            doc.metadata["page"] = 0
        documents.extend(_tag_document_group(docs, path))

    for path in root.rglob("*.pdf"):
        docs = PyPDFLoader(str(path)).load()
        for page_index, doc in enumerate(docs):
            doc.metadata["page"] = int(doc.metadata.get("page", page_index))
            doc.metadata["content_type"] = "text"
            doc.metadata["extraction_method"] = "native_text"
        enriched_docs = _enrich_pdf_pages(path, docs)
        documents.extend(_tag_document_group(enriched_docs, path))

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            documents.extend(_load_standalone_image(path))

    return documents


def split_text(documents):
    """Split ordinary text; tables are handled separately."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        # Preserve text/OCR/vision record types so retrieval and citations can
        # distinguish native selectable text from machine-extracted evidence.
        chunk.metadata.setdefault("content_type", "text")
        chunk.metadata["chunk_index"] = i

    print(
        f"Split {len(documents)} native/OCR/visual document record(s) into "
        f"{len(chunks)} searchable text chunk(s)."
    )
    return chunks


def _table_as_text(rows: list[list[object]]) -> str:
    cleaned = []
    for row in rows:
        values = ["" if v is None else str(v).strip() for v in row]
        while values and not values[-1]:
            values.pop()
        if any(values):
            cleaned.append(" | ".join(values))
    return "\n".join(cleaned)


def build_embedded_dataset(
    data_path: str,
    *,
    documents=None,
    chunks=None,
    table_documents=None,
) -> Chroma:
    """
    Build Chroma from precomputed content when supplied.

    Existing callers can still pass only data_path, preserving the original
    interface.
    """
    if chunks is None:
        documents = documents if documents is not None else load_documents(data_path)
        chunks = split_text(documents)

    vector_documents = list(chunks)

    if table_documents:
        vector_documents.extend(table_documents)

    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    existing = db.get()
    if existing.get("ids"):
        db.delete(ids=existing["ids"])

    if vector_documents:
        batch_size = 16
        total = len(vector_documents)

        for start in range(0, total, batch_size):
            batch = vector_documents[start:start + batch_size]
            db.add_documents(batch)

            processed = min(start + len(batch), total)
            print(f"Embedded {processed}/{total} searchable items")

    print(
        f"Saved {len(vector_documents)} searchable items to {CHROMA_PATH} "
        f"({len(chunks)} text chunks, {len(table_documents or [])} table records)."
    )
    return db


# Kept public for compatibility with any existing imports.
__all__ = [
    "CHROMA_PATH",
    "load_documents",
    "split_text",
    "build_embedded_dataset",
]
