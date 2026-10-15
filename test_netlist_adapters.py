"""Offline tests for content-detected netlist adapters."""
from pathlib import Path
import tempfile
import unittest

from netlist_adapters import parse_netlist_file


FIXTURES = {
    "board.net": '''<?xml version="1.0"?>
<export version="E"><components><comp ref="R1"><value>10k</value></comp></components>
<nets><net code="1" name="GND"><node ref="R1" pin="1"/><node ref="C1" pin="2"/></net></nets></export>''',
    "board_sexpr.any": '''(export (version "E") (components (comp (ref "R1") (value "10k") (libsource (lib "Device") (part "R"))))
(nets (net (code "1") (name "GND") (node (ref "R1") (pin "1")) (node (ref "C1") (pin "2")))) )''',
    "eagle.sch": '''<?xml version="1.0"?><eagle version="9"><drawing><schematic><parts>
<part name="U1" value="MCU"/><part name="R1" value="10k"/></parts><sheets><sheet><nets><net name="GND"><segment>
<pinref part="U1" pin="GND"/><pinref part="R1" pin="2"/></segment></net></nets></sheet></sheets></schematic></drawing></eagle>''',
    "board.enet": '''{"version":"2.0.0","components":{"gge1":{"props":{"Designator":"R1","Value":"10K"},"pinInfoMap":{"1":{"number":"1","net":"GND"},"2":{"number":"2","net":"VCC"}}}}}''',
    "test.cir": '''* tiny SPICE network\nV1 VCC 0 5\nR1 VCC OUT 10k\nC1 OUT 0 1uF\n.end''',
    "easyeda.net": '''(\nGND\nR1-2 0603\nC1-2 0603\n)\n(\nVCC\nR1-1 0603\nU1-1 MCU\n)''',
    "connectivity.csv": "net,reference,pin,value\nGND,U1,1,MCU\nGND,C1,2,100nF\n",
    "board.asc": '''*PADS-PCB*\n*PART*\nU1 MCU\nR1 10k\n*SIGNAL*\nGND\nU1.1\nR1.2\nVCC\nU1.2\nR1.1\n*END*''',
    "allegro.txt": '''((net GND (node U1 1) (node C1 2)) (net VCC (node U1 2) (node R1 1)))''',
    "orcad.net": '''OrCAD-01
OrCAD PCB II netlist
( 6CB84CBA 14DIP300 U1 74LS00
( 1 Q_1 )
( 2 CLOCK )
( 7 GND )
)
( 8A22B5C0 RES0805 R1 10k
( 1 GND )
( 2 VCC )
)''',
    "ipc.custom": '''C IPC-D-356A bare-board test netlist
P UNITS CUST 0
317GND           U1       -1    D0300A01X+004500Y+003200X0500Y0000R000
327GND           C1       -2    D0300A01X+004800Y+003200X0500Y0000R000
317VCC           U1       -2    D0300A01X+004500Y+003500X0500Y0000R000
''',
    "mentor.txt": '''NET 'GND' U1-7 U2-7
NET 'VCC' U1-14 U2-14
''',
    "reordered.rpt": "Pin Number,Component Designator,Net Name,Value\n1,U1,GND,MCU\n2,C1,GND,100nF\n",
    "board.edf": '''(edif Demo
 (library Lib (cell MCU (view NETLIST (contents (port A (direction INPUT)) (port B (direction OUTPUT))))))
 (design Demo (cellRef Top (libraryRef Lib)))
 (instance U1 (cellRef MCU (libraryRef Lib)))
 (instance U2 (cellRef MCU (libraryRef Lib)))
 (net GND (joined (portRef A (instanceRef U1)) (portRef B (instanceRef U2))))
 (net DATA (joined (portRef B (instanceRef U1)) (portRef A (instanceRef U2))))
)''',
}


class NetlistAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def parse(self, name):
        path = self.root / name
        path.write_text(FIXTURES[name], encoding="utf-8")
        return parse_netlist_file(path)

    def test_kicad_xml_content_detection(self):
        result = self.parse("board.net")
        self.assertEqual(result["input_format"], "kicad_xml_netlist")
        self.assertEqual(result["nets"]["GND"]["nodes"][0]["ref"], "R1")

    def test_kicad_sexpr_content_detection(self):
        result = self.parse("board_sexpr.any")
        self.assertEqual(result["input_format"], "kicad_sexpr_netlist")
        self.assertIn("GND", result["nets"])

    def test_eagle_xml_schematic(self):
        result = self.parse("eagle.sch")
        self.assertEqual(result["input_format"], "eagle_xml_schematic")
        self.assertEqual(result["net_count"], 1)

    def test_easyeda_pro_enet_json(self):
        result = self.parse("board.enet")
        self.assertEqual(result["input_format"], "easyeda_pro_enet_json")
        self.assertEqual(set(result["nets"]), {"GND", "VCC"})

    def test_spice_compatible_netlist(self):
        result = self.parse("test.cir")
        self.assertEqual(result["input_format"], "spice_compatible_netlist")
        self.assertIn("OUT", result["nets"])

    def test_easyeda_grouped_text(self):
        result = self.parse("easyeda.net")
        self.assertEqual(result["input_format"], "easyeda_or_protel_grouped_text")
        self.assertEqual(len(result["nets"]["GND"]["nodes"]), 2)

    def test_csv_connectivity(self):
        result = self.parse("connectivity.csv")
        self.assertEqual(result["input_format"], "csv_or_tsv_connectivity_table")
        self.assertEqual(result["net_count"], 1)

    def test_sectioned_pads_ascii(self):
        result = self.parse("board.asc")
        self.assertEqual(result["input_format"], "pads_or_sectioned_ascii_netlist")
        self.assertEqual(len(result["nets"]["GND"]["nodes"]), 2)

    def test_parenthesized_allegro_style(self):
        result = self.parse("allegro.txt")
        self.assertEqual(result["input_format"], "parenthesized_eda_netlist")
        self.assertEqual(set(result["nets"]), {"GND", "VCC"})

    def test_orcad_pcb2_component_centric_export(self):
        result = self.parse("orcad.net")
        self.assertEqual(result["input_format"], "orcad_pcb2_netlist")
        self.assertEqual(result["components"]["U1"]["value"], "74LS00")
        self.assertIn("GND", result["nets"])

    def test_ipc_d356_board_test_netlist(self):
        result = self.parse("ipc.custom")
        self.assertEqual(result["input_format"], "ipc_d356_board_test_netlist")
        self.assertIn("U1", result["components"])
        self.assertIn("VCC", result["nets"])

    def test_vendor_net_declaration_text(self):
        result = self.parse("mentor.txt")
        self.assertEqual(result["input_format"], "vendor_ascii_netlist")
        self.assertEqual(len(result["nets"]["GND"]["nodes"]), 2)

    def test_tabular_headers_are_order_independent_and_content_detected(self):
        result = self.parse("reordered.rpt")
        self.assertEqual(result["input_format"], "csv_or_tsv_connectivity_table")
        self.assertEqual(result["components"]["U1"]["value"], "MCU")

    def test_edif_netlist_subset(self):
        result = self.parse("board.edf")
        self.assertEqual(result["input_format"], "edif_netlist")
        self.assertEqual(result["nets"]["GND"]["nodes"][0]["ref"], "U1")


if __name__ == "__main__":
    unittest.main()
