"""
Explicit document-ingestion entry point.

Run this only when documents are added or changed.
"""

from document_processor import process_documents

DATASHEETS_FOLDER = "data/datasheets"


if __name__ == "__main__":
    print("\n========================================")
    print("       DATASHEET INGESTION")
    print("========================================")
    print(f"\nDatasheet folder: {DATASHEETS_FOLDER}\n")

    try:
        process_documents(
            DATASHEETS_FOLDER,
            visualize=True,
        )
        print("\nYou can now run:")
        print("    python pipeline.py")
    except Exception as exc:
        print("\n========================================")
        print("       INGESTION FAILED")
        print("========================================")
        print(f"\nError: {exc}")
