"""Offline tests for deterministic AP2112/ESP32-S3 power review."""
import unittest

from power_review import build_power_report, can_handle_power_question, _ap_specs, _esp_specs, _page_blocks_from_context

CONTEXT = r"""
[SOURCE: AP2112.pdf; page 8; component filter: AP2112K-3.3]
AP2112 Electrical Characteristics (Cont.) AP2112-3.3 Electrical Characteristics (@VIN = 4.3V, CIN = 1.0uF, COUT = 1.0uF, Typical TA = +25 C)
Symbol Parameter Conditions Min Typ Max Unit
VOUT Output Voltage VIN = 4.3V, 1mA <= IOUT <= 30mA VOUT *98.5% 3.3 VOUT *101.5% V
IOUT(MAX) Maximum Output Current VIN = 4.3V, VOUT = 3.251V to 3.350V 600 --- --- mA
Load Regulation VOUT = 3.3V, VIN = VOUT+1V
VDROP Dropout Voltage
IOUT = 10mA --- 5 8
mV IOUT = 300mA --- 125 200
mV IOUT = 600mA --- 250 400
IQ Quiescent Current VIN = 4.3V

--- RELEVANT EVIDENCE ---

[SOURCE: esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf; page 27; component filter: ESP32-S3-WROOM-1]
6.2 Recommended Operating Conditions Table 6-2
Symbol Parameter Min Typ Max Unit
VDD33 Power supply voltage 3.0 3.3 3.6 V
IVDD Current delivered by external power supply 0.5 --- --- A

--- RELEVANT EVIDENCE ---

[SOURCE: esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf; page 28; component filter: ESP32-S3-WROOM-1]
TABLE 6-3 Current Consumption
WorkMode RFCondition Description Peak(mA)
Active(RFworking) TX 802.11b,1Mbps,@20.5dBm 355
802.11g,54Mbps,@18dBm 297
802.11n,HT20,MCS7,@17.5dBm 286
802.11n,HT40,MCS7,@17dBm 285

--- RELEVANT EVIDENCE ---

[SOURCE: AP2112.pdf; page 3; component filter: AP2112K-3.3]
Absolute Maximum Ratings
VCC Power Supply Voltage 6.5 V
TJ Operating Junction Temperature Range +150 C
ThetaJA Thermal Resistance (Junction to Ambient)(No Heatsink)
SOT25 184 C/W
SO-8 114 C/W
SOT89-5 120 C/W
Recommended Operating Conditions VIN Supply Voltage 2.5 6.0 V
"""


class PowerReviewTests(unittest.TestCase):
    def test_component_only_electrical_question_is_routed(self):
        self.assertTrue(can_handle_power_question(
            "What is the maximum output current and dropout voltage of AP2112K-3.3?"
        ))

    def test_ap2112_variant_specific_data_parses(self):
        pages = _page_blocks_from_context(CONTEXT)
        values = _ap_specs(pages)
        self.assertIn("600 mA", values["imax"]["value"])
        self.assertIn("300 mA: 125 mV typ / 200 mV max", values["dropout"]["value"])
        self.assertIn("600 mA: 250 mV typ / 400 mV max", values["dropout"]["value"])
        self.assertEqual(values["theta"]["numeric"], 184.0)
        self.assertEqual(values["tjmax"]["numeric"], 150.0)

    def test_wrong_variants_do_not_supply_ap2112_33_table(self):
        only_wrong_variant = """[SOURCE: AP2112.pdf; page 6]\nAP2112-2.5 Electrical Characteristics VOUT 2.5 V IOUT(MAX) 600 mA VDROP IOUT = 600mA 250 400 mV"""
        values = _ap_specs(_page_blocks_from_context(only_wrong_variant))
        self.assertIsNone(values["imax"])
        self.assertIsNone(values["dropout"])

    def test_fact_answer_is_deterministic_and_has_clean_table(self):
        answer = build_power_report(
            "What is the maximum output current and dropout voltage of AP2112K-3.3?",
            context=CONTEXT,
            datasheet_dir="/this/path/does/not/exist",
        )
        self.assertIsNotNone(answer)
        self.assertNotIn("Verdict:", answer)
        self.assertIn("| Parameter | Value and conditions | Source / page |", answer)
        self.assertIn("600 mA", answer)
        self.assertIn("AP2112.pdf · p. 8", answer)
        self.assertNotIn("AP2112-2.5", answer)

    def test_board_review_math_is_generated_in_code(self):
        answer = build_power_report(
            "Using the ESP32-S3-WROOM-1 and AP2112K-3.3, can it safely power my board from a 5 V input? Check Wi-Fi current peaks and thermal dissipation.",
            context=CONTEXT,
            datasheet_dir="/this/path/does/not/exist",
        )
        self.assertIsNotNone(answer)
        self.assertIn("Power-path result:", answer)
        self.assertIn("0.604 W", answer)  # (5.0 - 3.3) * 0.355 A
        self.assertIn("136.0 °C", answer) # 25 + (0.604 * 184), within rounding
        self.assertIn("181.4 °C", answer) # hypothetical continuous 0.5 A
        self.assertIn("---", answer)

    def test_missing_data_is_not_fabricated(self):
        answer = build_power_report(
            "What are the maximum output current and dropout voltage of AP2112K-3.3?",
            context="",
            datasheet_dir="/this/path/does/not/exist",
        )
        self.assertIn("NOT FOUND", answer)
        self.assertNotIn("600 mA", answer)


    def test_general_pcb_review_routes_from_netlist_parts_without_llm(self):
        schematic = {
            "components": [
                {"ref": "U1", "value": "ESP32-S3-WROOM-1"},
                {"ref": "U2", "value": "AP2112K-3.3"},
                {"ref": "C1", "value": "1uF"},
                {"ref": "C2", "value": "1uF"},
            ],
            "nets": {
                "+5V": {"name": "+5V", "nodes": [
                    {"ref": "U2", "pin": "1", "pinfunction": "VIN_1"},
                    {"ref": "U2", "pin": "3", "pinfunction": "EN_3"},
                    {"ref": "C1", "pin": "1"},
                ]},
                "+3V3": {"name": "+3V3", "nodes": [
                    {"ref": "U2", "pin": "5", "pinfunction": "VOUT_5"},
                    {"ref": "U1", "pin": "2", "pinfunction": "3V3_2"},
                    {"ref": "C2", "pin": "1"},
                ]},
                "GND": {"name": "GND", "nodes": [
                    {"ref": "U2", "pin": "2", "pinfunction": "GND_2"},
                    {"ref": "U1", "pin": "1", "pinfunction": "GND_1"},
                    {"ref": "C1", "pin": "2"},
                    {"ref": "C2", "pin": "2"},
                ]},
            },
            "findings": [],
        }
        question = (
            "Is my pcb design good for ESP32-S3? Did I choose the components "
            "correctly according to the datasheet?"
        )
        answer = build_power_report(
            question,
            context=CONTEXT,
            schematic_result=schematic,
            datasheet_dir="/this/path/does/not/exist",
        )
        self.assertIsNotNone(answer)
        self.assertIn("AP2112K-3.3 / ESP32-S3 power assessment", answer)
        self.assertIn("U2 VIN (pin 1) | +5V", answer)
        self.assertIn("ESP32-S3", answer)
        self.assertIn("does not certify the whole PCB", answer)
        self.assertIn("0.604 W", answer)
        self.assertNotIn("QUALITY GATE", answer)

    def test_board_review_intent_is_detected_without_safe_keyword(self):
        from power_review import _is_safety_review, can_handle_power_question
        question = "Is my pcb design good for ESP32-S3? Did I choose the components correctly according to the datasheet?"
        self.assertTrue(_is_safety_review(question))
        self.assertTrue(can_handle_power_question(
            question,
            component_ids=["ESP32-S3-WROOM-1", "AP2112K-3.3"],
        ))


if __name__ == "__main__":
    unittest.main()
