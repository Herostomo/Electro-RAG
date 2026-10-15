"""
Document-processing pipeline.

This is intentionally separate from question answering:
load -> table extraction -> chunk -> embeddings -> KG -> visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

from data_ingest import load_documents, split_text, build_embedded_dataset
from table_extractor import extract_pdf_tables, table_records_to_documents
from kg_builder import build_knowledge_graph
from kg_visualize import plot_knowledge_graph

STATUS_PATH = "knowledge_base_status.json"


def process_documents(
    data_path: str,
    *,
    visualize: bool = True,
    max_kg_workers: int = 2,
    kg_batch_size: int = 4,
) -> dict:
    print("\n========================================")
    print("       DOCUMENT PROCESSING")
    print("========================================\n")

    documents = load_documents(data_path)
    if not documents:
        raise FileNotFoundError(
            f"No supported documents found in '{data_path}'. "
            "Add PDF/MD files before processing."
        )

    unique_pages = {
        (str(doc.metadata.get("source", "")), int(doc.metadata.get("page", 0)))
        for doc in documents
    }
    ocr_records = sum(doc.metadata.get("extraction_method") == "ocr" for doc in documents)
    vision_records = sum(doc.metadata.get("extraction_method") == "vision" for doc in documents)
    print(
        f"Loaded {len(unique_pages)} unique document page(s), plus "
        f"{ocr_records} OCR record(s) and {vision_records} visual-description record(s)."
    )

    chunks = split_text(documents)
    pages_by_source = defaultdict(set)
    native_pages_by_source = defaultdict(set)
    enrichment_by_source = Counter()
    chunks_by_source = Counter()
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        pages_by_source[source].add(int(doc.metadata.get("page", 0)))
        method = doc.metadata.get("extraction_method", "native_text")
        if method == "native_text" and (doc.page_content or "").strip():
            native_pages_by_source[source].add(int(doc.metadata.get("page", 0)))
        elif method in {"ocr", "vision"}:
            enrichment_by_source[(source, method)] += 1
    for chunk in chunks:
        chunks_by_source[str(chunk.metadata.get("source", "unknown"))] += 1

    print("\nPer-PDF extraction summary:")
    for source in sorted(pages_by_source):
        print(
            f"  {Path(source).name}: pages={len(pages_by_source[source])}, "
            f"native_text_pages={len(native_pages_by_source[source])}, "
            f"ocr_records={enrichment_by_source[(source, 'ocr')]}, "
            f"visual_records={enrichment_by_source[(source, 'vision')]}, "
            f"chunks={chunks_by_source[source]}"
        )
    table_records = extract_pdf_tables(data_path)
    table_documents = table_records_to_documents(table_records)

    print("\n========================================")
    print("       BUILDING VECTOR DATABASE")
    print("========================================\n")
    build_embedded_dataset(
        data_path,
        documents=documents,
        chunks=chunks,
        table_documents=table_documents,
    )

    print("\n========================================")
    print("       BUILDING KNOWLEDGE GRAPH")
    print("========================================\n")
    graph = build_knowledge_graph(
        data_path,
        chunks=chunks,
        table_records=table_records,
        max_workers=max_kg_workers,
        batch_size=kg_batch_size,
    )

    if visualize:
        plot_knowledge_graph(graph)

    status = {
        "processed": True,
        "data_path": str(data_path),
        "document_count": len(documents),
        "unique_page_count": len(unique_pages),
        "ocr_record_count": ocr_records,
        "image_description_record_count": vision_records,
        "text_chunk_count": len(chunks),
        "table_count": len(table_records),
        "kg_nodes": graph.number_of_nodes(),
        "kg_edges": graph.number_of_edges(),
    }
    Path(STATUS_PATH).write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    print("\n========================================")
    print("       DOCUMENT PROCESSING COMPLETE")
    print("========================================\n")

    return status
