"""PDF table extraction plus safe deterministic table normalization."""

from __future__ import annotations

from pathlib import Path
import json
import re

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

from langchain_core.documents import Document

TABLE_STORE_PATH = "extracted_tables.json"

# Conservative part-number patterns. These are intentionally limited to common
# electronics naming families rather than treating every uppercase token as a part.
_PART_PATTERNS = (
    r"\b(?:ESP32(?:-[A-Z0-9]+){1,5}|"
    r"(?:AP|LM|LT|LTC|TL|OPA|MCP|INA|ADS|MAX|TPS|AMS|NCP|SN|AD|NE|MC|MIC|STM32|ATmega|BQ|DRV|TUSB|CH340|CP210|"
    r"BME|BMP|MPU|ICM|LSM|LIS|BMI|NRF|RP|GD32|CH32|W25Q|MX25|GD25|PCF|ADXL|SHT|DS18B|USBLC|XC6|XC62)[A-Z0-9]*"
    r"(?:-[A-Z0-9.]+)*)\b",
)


def _clean_rows(rows):
    cleaned = []
    for row in rows or []:
        if not row:
            continue
        cleaned.append([
            "" if value is None else " ".join(str(value).split())
            for value in row
        ])
    return cleaned


def _find_part_numbers(text: str) -> list[str]:
    found = []
    for pattern in _PART_PATTERNS:
        found.extend(re.findall(pattern, text or "", flags=re.IGNORECASE))
    result = []
    seen = set()
    for item in found:
        key = item.upper()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_pdf_tables(data_path: str) -> list[dict]:
    if pdfplumber is None:
        print(
            "pdfplumber is not installed; skipping structured PDF table extraction. "
            "Tables will still remain available through normal page text/RAG."
        )
        return []

    records: list[dict] = []
    for path in Path(data_path).rglob("*.pdf"):
        try:
            with pdfplumber.open(path) as pdf:
                # Tag every extracted table with identifiers found across the source
                # PDF, because tables on later pages often omit the part number.
                page_texts = [page.extract_text() or "" for page in pdf.pages]
                whole_doc_text = "\n".join(page_texts)
                document_parts = _find_part_numbers(whole_doc_text)
                primary_parts = _find_part_numbers("\n".join(page_texts[:3])[:18000])
                for page_number, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_parts = _find_part_numbers(page_text)
                    try:
                        tables = page.extract_tables() or []
                    except Exception as exc:
                        print(
                            f"Table extraction failed for {path} page {page_number + 1}: {exc}"
                        )
                        continue

                    for table_index, rows in enumerate(tables):
                        rows = _clean_rows(rows)
                        nonempty = [r for r in rows if any(cell for cell in r)]
                        if not nonempty:
                            continue

                        records.append(
                            {
                                "source": str(path),
                                "page": page_number,
                                "table_index": table_index,
                                "rows": nonempty,
                                "page_text": page_text,
                                "component_hints": page_parts,
                                "document_part_numbers": "|".join(document_parts),
                                "document_primary_part_numbers": "|".join(primary_parts),
                            }
                        )
        except Exception as exc:
            print(f"Could not open {path} for table extraction: {exc}")

    Path(TABLE_STORE_PATH).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Extracted {len(records)} table(s) into structured table storage.")
    return records


def table_records_to_documents(records: list[dict]) -> list[Document]:
    docs = []
    for record in records:
        lines = [" | ".join(row) for row in record["rows"]]
        content = (
            f"TABLE FROM {record['source']} | "
            f"PAGE {record['page'] + 1} | "
            f"TABLE {record['table_index'] + 1}\n"
            + "\n".join(lines)
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": record["source"],
                    "page": record["page"],
                    "table_index": record["table_index"],
                    "content_type": "table",
                    "component_hints": ", ".join(record.get("component_hints", [])),
                    "document_part_numbers": record.get("document_part_numbers", ""),
                    "document_primary_part_numbers": record.get("document_primary_part_numbers", ""),
                    "document_name": Path(record["source"]).name,
                },
            )
        )
    return docs
