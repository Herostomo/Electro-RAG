"""Offline tests for PDF OCR/visual-record metadata handling."""
import sys
import types
import unittest
from pathlib import Path

# The project normally depends on LangChain/Ollama. These tests cover local
# metadata helpers without requiring those runtime packages or a model server.
if "langchain_core.documents" not in sys.modules:
    core = types.ModuleType("langchain_core")
    docs_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = dict(metadata or {})

    docs_mod.Document = Document
    sys.modules["langchain_core"] = core
    sys.modules["langchain_core.documents"] = docs_mod

if "langchain_community.document_loaders" not in sys.modules:
    community = types.ModuleType("langchain_community")
    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = object
    loaders.TextLoader = object
    sys.modules["langchain_community"] = community
    sys.modules["langchain_community.document_loaders"] = loaders

if "langchain_text_splitters" not in sys.modules:
    splitter = types.ModuleType("langchain_text_splitters")
    splitter.RecursiveCharacterTextSplitter = object
    sys.modules["langchain_text_splitters"] = splitter

if "langchain_ollama" not in sys.modules:
    module = types.ModuleType("langchain_ollama")
    module.OllamaEmbeddings = object
    sys.modules["langchain_ollama"] = module

if "langchain_chroma" not in sys.modules:
    module = types.ModuleType("langchain_chroma")
    module.Chroma = object
    sys.modules["langchain_chroma"] = module

from data_ingest import _tag_document_group, extract_document_part_numbers
from langchain_core.documents import Document


class PdfEnrichmentMetadataTests(unittest.TestCase):
    def test_ocr_from_cover_page_can_set_document_primary_identity(self):
        native_page = Document("", {
            "source": "ESP32-S3-module.pdf",
            "page": 0,
            "content_type": "text",
            "extraction_method": "native_text",
        })
        ocr_page = Document(
            "OCR TEXT FROM ESP32-S3-module.pdf, PAGE 1\\nESP32-S3-WROOM-1 Datasheet",
            {
                "source": "ESP32-S3-module.pdf",
                "page": 0,
                "content_type": "ocr",
                "extraction_method": "ocr",
            },
        )
        docs = _tag_document_group([native_page, ocr_page], Path("ESP32-S3-module.pdf"))
        self.assertIn("ESP32-S3-WROOM-1", docs[0].metadata["document_primary_part_numbers"])
        self.assertEqual(docs[1].metadata["content_type"], "ocr")
        self.assertEqual(docs[1].metadata["page"], 0)

    def test_part_identifier_extraction_from_ocr_text(self):
        identifiers = extract_document_part_numbers(
            "Cover page OCR: ESP32-S3-WROOM-1 datasheet AP2112K-3.3 regulator"
        )
        normalized = {item.upper() for item in identifiers}
        self.assertTrue(any(item.startswith("ESP32-S3-WROOM-1") for item in normalized))
        self.assertTrue(any(item.startswith("AP2112K") for item in normalized))


if __name__ == "__main__":
    unittest.main()
