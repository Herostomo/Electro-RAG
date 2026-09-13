"""RAG retrieval over text chunks and structured table records."""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from models import EMBEDDING_MODEL
from data_ingest import CHROMA_PATH

RAG_SCORE_THRESHOLD = 0.4
RAG_K = 5
MAX_CONTEXT_CHARS = 12000
MAX_DOC_CHARS = 3500


def get_retriever() -> Chroma:
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )


def _search(db, query: str, k: int):
    if not query or not query.strip():
        return []
    return db.similarity_search_with_relevance_scores(query.strip(), k=k)


def _dedupe(results):
    dedup = {}
    for doc, score in results:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("table_index"),
            doc.metadata.get("content_type", "text"),
            doc.page_content,
        )
        if key not in dedup or score > dedup[key][1]:
            dedup[key] = (doc, score)
    return sorted(dedup.values(), key=lambda item: item[1], reverse=True)


def _build_bounded_context(ranked):
    context_parts = []
    sources = []
    scores = []
    remaining = MAX_CONTEXT_CHARS

    for doc, score in ranked:
        if remaining <= 0:
            break

        content = doc.page_content.strip()
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "\n[table/text truncated for model context]"

        # Do not add a document that would leave almost no useful context.
        if not content:
            continue

        if len(content) > remaining:
            content = content[:remaining] + "\n[context truncated]"

        context_parts.append(content)
        remaining -= len(content)
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content_type": doc.metadata.get("content_type", "text"),
            "table_index": doc.metadata.get("table_index"),
        })
        scores.append(float(score))

    return context_parts, sources, scores


def retrieve(
    query: str,
    k: int = RAG_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
    secondary_query: str | None = None,
) -> dict:
    """
    Use the original user question as the primary retrieval query.

    Only fall back to a selector-generated query when the primary search is
    too weak. This prevents noisy generated table identifiers from expanding
    the retrieval context for an otherwise good factual question.
    """
    db = get_retriever()
    primary = _dedupe(_search(db, query, k))[:k]

    # Prefer the user's exact question. Use the selector query only when the
    # primary retrieval is weak.
    ranked = primary
    if (not ranked or ranked[0][1] < score_threshold) and secondary_query:
        secondary = _dedupe(_search(db, secondary_query, max(2, k // 2)))
        ranked = _dedupe(primary + secondary)[:k]

    if not ranked or ranked[0][1] < score_threshold:
        return {
            "context": "",
            "sources": [],
            "scores": [],
            "found": False,
        }

    context_parts, sources, scores = _build_bounded_context(ranked)
    if not context_parts:
        return {
            "context": "",
            "sources": [],
            "scores": [],
            "found": False,
        }

    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
        "scores": scores,
        "found": True,
    }
