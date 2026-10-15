
from collections import defaultdict

from data_ingest import load_documents, split_text


FOLDER = "data/datasheets"

print("Loading datasheets...")
documents = load_documents(FOLDER)

stats = defaultdict(
    lambda: {
        "pages": 0,
        "nonempty_pages": 0,
        "characters": 0,
    }
)

for doc in documents:
    source = doc.metadata.get("source", "unknown")
    content = doc.page_content or ""

    stats[source]["pages"] += 1
    stats[source]["characters"] += len(content)

    if content.strip():
        stats[source]["nonempty_pages"] += 1

print("\nSplitting extracted text into chunks...")
chunks = split_text(documents)

chunk_counts = defaultdict(int)

for chunk in chunks:
    source = chunk.metadata.get("source", "unknown")
    chunk_counts[source] += 1

print("\n" + "=" * 75)
print("DATASHEET EXTRACTION DIAGNOSTIC")
print("=" * 75)

for source, item in sorted(stats.items()):
    print(f"\nPDF: {source}")
    print(f"  Pages loaded:          {item['pages']}")
    print(f"  Pages with text:       {item['nonempty_pages']}")
    print(f"  Extracted characters:  {item['characters']}")
    print(f"  Text chunks:           {chunk_counts[source]}")

print("\n" + "=" * 75)
print("TOTALS")
print("=" * 75)
print("Total pages loaded:", len(documents))
print("Total text chunks:", len(chunks))
print("Total searchable table records: check extracted_tables.json")
