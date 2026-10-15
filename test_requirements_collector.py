"""Tests that electrical requirements are never fabricated from unlabelled numbers."""
import sys
import types
import unittest

# The extraction function is tested directly, without requiring Ollama at test time.
if "ollama_utils" not in sys.modules:
    ollama_stub = types.ModuleType("ollama_utils")
    ollama_stub.chat_json = lambda *args, **kwargs: {}
    sys.modules["ollama_utils"] = ollama_stub

import requirements_collector as rc


class RequirementsCollectorElectricalTests(unittest.TestCase):
    def test_unspecified_electrical_values_remain_unknown(self):
        result = rc._extract_explicit_electrical_fields(
            "Review my ESP32 PCB design for power supply and connectivity issues."
        )
        self.assertEqual(result, {
            "input_voltage": "unknown",
            "output_voltage": "unknown",
            "current_limit": "unknown",
            "frequency": "unknown",
        })

    def test_explicit_labeled_values_are_preserved(self):
        result = rc._extract_explicit_electrical_fields(
            "Input voltage is 5V, output voltage is 3.3V, maximum current is 500mA, clock frequency is 24MHz."
        )
        self.assertEqual(result["input_voltage"], "5V")
        self.assertEqual(result["output_voltage"], "3.3V")
        self.assertEqual(result["current_limit"], "500mA")
        self.assertEqual(result["frequency"], "24MHz")

    def test_wrong_units_are_not_assigned_to_electrical_fields(self):
        result = rc._extract_explicit_electrical_fields(
            "Input voltage is 500mA and current limit is 3.3V; frequency is 100mA."
        )
        self.assertEqual(result["input_voltage"], "unknown")
        self.assertEqual(result["current_limit"], "unknown")
        self.assertEqual(result["frequency"], "unknown")

    def test_unlabeled_numbers_are_not_assigned_to_electrical_fields(self):
        result = rc._extract_explicit_electrical_fields(
            "The note says 12V, 1A and 50Hz, review my design."
        )
        self.assertTrue(all(value == "unknown" for value in result.values()))


if __name__ == "__main__":
    unittest.main()
