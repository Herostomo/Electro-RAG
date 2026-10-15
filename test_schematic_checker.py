"""Small stdlib-only tests for schematic_checker.py."""
from pathlib import Path
import tempfile
import unittest

from schematic_checker import check_schematic, parse_kicad_netlist, format_findings_for_llm

FIXTURE = '''<?xml version="1.0" encoding="UTF-8"?>
<export version="E">
  <components>
    <comp ref="U1"><value>Example MCU</value><footprint>Package_SO:SOIC-8</footprint><libsource lib="Device" part="MCU"/></comp>
    <comp ref="#PWR01"><value>+5V</value><libsource lib="power" part="+5V"/></comp>
    <comp ref="#PWR02"><value>+3V3</value><libsource lib="power" part="+3V3"/></comp>
    <comp ref="R1"><value>10k</value><libsource lib="Device" part="R"/></comp>
  </components>
  <libparts>
    <libpart lib="Device" part="MCU"><pins>
      <pin num="1" name="VDD" type="power_in"/>
      <pin num="2" name="GND" type="power_in"/>
      <pin num="3" name="GPIO1" type="input"/>
    </pins></libpart>
    <libpart lib="Device" part="R"><pins>
      <pin num="1" name="~" type="passive"/>
      <pin num="2" name="~" type="passive"/>
    </pins></libpart>
  </libparts>
  <nets>
    <net code="1" name="POWER_MERGE"><node ref="U1" pin="1" pinfunction="VDD" pintype="power_in"/><node ref="#PWR01" pin="1" pinfunction="+5V" pintype="power_out"/><node ref="#PWR02" pin="1" pinfunction="+3V3" pintype="power_out"/></net>
    <net code="2" name="GND"><node ref="U1" pin="2" pinfunction="GND" pintype="power_in"/></net>
  </nets>
</export>
'''


class SchematicCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "fixture.net"
        self.path.write_text(FIXTURE, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_parse_counts_components_and_nets(self):
        parsed = parse_kicad_netlist(self.path)
        self.assertEqual(parsed["component_count"], 4)
        self.assertEqual(parsed["net_count"], 2)
        self.assertEqual(parsed["components"]["U1"]["value"], "Example MCU")

    def test_detects_conflicting_power_symbols_and_unconnected_pins(self):
        result = check_schematic(self.path)
        rule_ids = {finding["rule_id"] for finding in result["findings"]}
        self.assertIn("POWER_RAIL_CONFLICT", rule_ids)
        self.assertIn("UNCONNECTED_PIN", rule_ids)
        self.assertIn("POWER_PIN_DATASHEET_CHECK", rule_ids)
        self.assertIn("CHECKER FINDINGS", format_findings_for_llm(result))


if __name__ == "__main__":
    unittest.main()
