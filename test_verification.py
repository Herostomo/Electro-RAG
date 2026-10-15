"""Offline tests for conservative, passage-aware NLI verification."""
import sys
import types
import unittest
from unittest import mock

import numpy as np

# Supply minimal dependency stubs so these tests do not download or load models.
if "ollama_utils" not in sys.modules:
    ollama_stub = types.ModuleType("ollama_utils")
    ollama_stub.embed_text = lambda model, text: [1.0, 0.0]
    sys.modules["ollama_utils"] = ollama_stub
elif not hasattr(sys.modules["ollama_utils"], "embed_text"):
    sys.modules["ollama_utils"].embed_text = lambda model, text: [1.0, 0.0]

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return np.tile(np.array([[0.1, 0.7, 0.2]]), (len(pairs), 1))

    st_stub.CrossEncoder = _FakeCrossEncoder
    sys.modules["sentence_transformers"] = st_stub

import verification


class PassageAwareVerificationTests(unittest.TestCase):
    def test_irrelevant_evidence_is_neutral_not_contradiction(self):
        result = verification.check_entailment(
            ["GPIO17 must be configured for a 1.8 V input."],
            "LM2904 amplifier output short-circuit behavior is continuous.",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["neutral_count"], 1)
        self.assertEqual(result["contradiction_count"], 0)
        self.assertEqual(result["unsupported_technical_count"], 1)

    def test_only_high_confidence_contradiction_fails(self):
        claim = "U2 pin 1 operating voltage is 1.8V."
        context = "U2 pin 1 VIN supports 5V and is not rated for 1.8V."
        with mock.patch.object(
            verification._nli_model,
            "predict",
            return_value=np.array([[0.93, 0.02, 0.05]]),
        ):
            result = verification.check_entailment([claim], context)
        self.assertFalse(result["passed"])
        self.assertEqual(result["contradiction_count"], 1)

    def test_weak_contradiction_prediction_is_not_overcalled(self):
        claim = "U2 pin 1 operating voltage is 1.8V."
        context = "U2 pin 1 VIN supports 5V and is not rated for 1.8V."
        with mock.patch.object(
            verification._nli_model,
            "predict",
            return_value=np.array([[0.48, 0.28, 0.24]]),
        ):
            result = verification.check_entailment([claim], context)
        self.assertFalse(result["passed"])
        self.assertEqual(result["contradiction_count"], 0)
        self.assertEqual(result["neutral_count"], 1)
        self.assertEqual(result["unsupported_technical_count"], 1)


if __name__ == "__main__":
    unittest.main()
