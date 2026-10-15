"""Offline regression tests for strict part-number/source matching in RAG."""
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace

# Keep this module's tests offline: only the RAG's pure-Python filtering helpers
# are under test here; actual vector retrieval requires the installed LangChain/Ollama stack.
if "langchain_ollama" not in sys.modules:
    lm = types.ModuleType("langchain_ollama")
    lm.OllamaEmbeddings = object
    sys.modules["langchain_ollama"] = lm
if "langchain_chroma" not in sys.modules:
    cm = types.ModuleType("langchain_chroma")
    cm.Chroma = object
    sys.modules["langchain_chroma"] = cm
if "data_ingest" not in sys.modules:
    dm = types.ModuleType("data_ingest")
    dm.CHROMA_PATH = "chroma"
    sys.modules["data_ingest"] = dm

import rag


class FakeChroma:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def similarity_search_with_relevance_scores(self, query, k=5):
        self.queries.append(query)
        return self.rows[:k]


class StrictRagSourceFilterTests(unittest.TestCase):
    def test_generic_esp32_datasheet_does_not_match_s3_module(self):
        doc = SimpleNamespace(
            metadata={"source": "esp32.pdf", "document_part_numbers": "ESP32-D0WD-V3|ESP32-U4WDH"},
            page_content="ESP32 Series datasheet passage",
        )
        matched, _ = rag._document_matches_part(doc, "ESP32-S3-WROOM-1")
        self.assertFalse(matched)

    def test_exact_module_datasheet_matches(self):
        doc = SimpleNamespace(
            metadata={"source": "esp32-s3-wroom-1.pdf", "document_part_numbers": "ESP32-S3-WROOM-1|ESP32-S3-WROOM-1U"},
            page_content="module power requirements",
        )
        matched, _ = rag._document_matches_part(doc, "ESP32-S3-WROOM-1")
        self.assertTrue(matched)

    def test_lm2904_does_not_match_ap2112(self):
        doc = SimpleNamespace(
            metadata={"source": "lm2904.pdf", "document_part_numbers": "LM2904|LM2904A|LM2904V"},
            page_content="operational amplifier electrical characteristics",
        )
        matched, _ = rag._document_matches_part(doc, "AP2112K-3.3")
        self.assertFalse(matched)

    def test_wroom_1_does_not_match_wroom_1u_only_datasheet(self):
        doc = SimpleNamespace(
            metadata={
                "source": "esp32-s3-wroom-1u.pdf",
                "document_primary_part_numbers": "ESP32-S3-WROOM-1U",
                "document_part_numbers": "ESP32-S3-WROOM-1U",
            },
            page_content="ESP32-S3-WROOM-1U module specifications",
        )
        matched, _ = rag._document_matches_part(doc, "ESP32-S3-WROOM-1")
        self.assertFalse(matched)

    def test_regulator_datasheet_matches_base_part_family(self):
        doc = SimpleNamespace(
            metadata={"source": "ap2112.pdf", "document_part_numbers": "AP2112K-3.3|AP2112K"},
            page_content="low-dropout regulator electrical characteristics",
        )
        matched, _ = rag._document_matches_part(doc, "AP2112K-3.3")
        self.assertTrue(matched)

    def test_title_metadata_overrides_later_irrelevant_part_mention(self):
        doc = SimpleNamespace(
            metadata={
                "source": "esp32-series.pdf",
                "document_primary_part_numbers": "ESP32-D0WD-V3|ESP32-U4WDH",
                "document_part_numbers": "ESP32-D0WD-V3|ESP32-U4WDH|ESP32-S3-WROOM-1",
            },
            page_content="Reference note mentions ESP32-S3-WROOM-1 but this is an ESP32 series datasheet.",
        )
        matched, reason = rag._document_matches_part(doc, "ESP32-S3-WROOM-1")
        self.assertFalse(matched)
        self.assertIn("front-matter", reason)

    def test_legacy_chunk_mention_does_not_override_wrong_filename(self):
        doc = SimpleNamespace(
            metadata={"source": "lm2904.pdf"},
            page_content="This comparison text mentions AP2112K-3.3, but describes LM2904.",
        )
        matched, _ = rag._document_matches_part(doc, "AP2112K-3.3")
        self.assertFalse(matched)

    def test_strict_retrieve_excludes_unrelated_datasheet_even_when_score_is_high(self):
        right = SimpleNamespace(
            metadata={
                "source": "esp32-s3-wroom-1.pdf",
                "document_primary_part_numbers": "ESP32-S3-WROOM-1|ESP32-S3-WROOM-1U",
                "document_part_numbers": "ESP32-S3-WROOM-1|ESP32-S3-WROOM-1U",
                "page": 17,
            },
            page_content="ESP32-S3-WROOM-1 supply and operating voltage passage",
        )
        wrong = SimpleNamespace(
            metadata={
                "source": "lm2904.pdf",
                "document_primary_part_numbers": "LM2904|LM2904A|LM2904V",
                "document_part_numbers": "LM2904|LM2904A|LM2904V",
                "page": 3,
            },
            page_content="LM2904 operational amplifier absolute maximum ratings",
        )
        db = FakeChroma([(wrong, 0.99), (right, 0.72)])
        with mock.patch.object(rag, "get_retriever", return_value=db):
            result = rag.retrieve(
                "Review ESP32-S3-WROOM-1 supply voltage",
                component_ids=["ESP32-S3-WROOM-1"],
                strict_component_filter=True,
            )
        self.assertTrue(result["found"])
        self.assertIn("esp32-s3-wroom-1.pdf", result["context"])
        self.assertNotIn("LM2904", result["context"])
        self.assertEqual(result["match_audit"][0]["matched"], 1)

    def test_strict_retrieve_rejects_exact_part_but_irrelevant_low_score_passage(self):
        doc = SimpleNamespace(
            metadata={
                "source": "ap2112k.pdf",
                "document_primary_part_numbers": "AP2112K-3.3|AP2112K",
                "document_part_numbers": "AP2112K-3.3|AP2112K",
                "page": 0,
            },
            page_content="Table of contents with little electrical detail",
        )
        db = FakeChroma([(doc, 0.15)])
        with mock.patch.object(rag, "get_retriever", return_value=db):
            result = rag.retrieve(
                "AP2112K-3.3 maximum output current and thermal characteristics",
                component_ids=["AP2112K-3.3"],
                strict_component_filter=True,
            )
        self.assertFalse(result["found"])
        self.assertEqual(result["match_audit"][0]["source_identity_matches"], 1)
        self.assertEqual(result["match_audit"][0]["matched"], 0)

    def test_extracts_only_explicit_part_identifiers(self):
        ids = rag.extract_component_ids("Review ESP32-S3-WROOM-1 with AP2112K-3.3")
        self.assertEqual(ids, ["ESP32-S3-WROOM-1", "AP2112K-3.3"])


if __name__ == "__main__":
    unittest.main()
