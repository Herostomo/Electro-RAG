"""
Question-answering pipeline plus a compatibility wrapper for document
ingestion.

Document ingestion now belongs to document_processor.py and is only run when
the user explicitly processes documents.
"""

from __future__ import annotations

import os

from requirements_collector import collect_requirements
from kg_builder import load_knowledge_graph, query_knowledge_graph
from rag import retrieve
from components_selector import select_components
from reasoning_llm import reason
from verification import verify

MAX_REGENERATIONS = 2


def _verification_feedback(verdict: dict) -> str:
    problems = []

    if not verdict["citation_existence"]["passed"]:
        problems.append(
            f"Citation existence: {verdict['citation_existence']['detail']}"
        )
    if not verdict["semantic_similarity"]["passed"]:
        problems.append(
            f"Semantic similarity: {verdict['semantic_similarity']['detail']}"
        )
    if not verdict["entailment_nli"]["passed"]:
        problems.append(
            f"Entailment/NLI: {verdict['entailment_nli']['detail']}"
        )

    return " | ".join(problems)


def ingest_datasheets(datasheets_path: str, visualize: bool = True) -> dict:
    """
    Backward-compatible wrapper. The actual implementation lives in
    document_processor.py and is NOT called by run_pipeline().
    """
    from document_processor import process_documents

    return process_documents(
        datasheets_path,
        visualize=visualize,
    )


def _knowledge_base_ready() -> bool:
    graph_ready = os.path.exists("knowledge_graph.graphml")
    chroma_ready = os.path.exists("chroma")
    return graph_ready and chroma_ready


def run_pipeline(raw_requirements: str) -> str:
    if not _knowledge_base_ready():
        return (
            "Please process a datasheet before asking questions. "
            "Run the document-processing step first."
        )

    requirements = collect_requirements(raw_requirements)
    print("Requirements collected:", requirements)

    graph = load_knowledge_graph()
    kg_context = query_knowledge_graph(
        graph,
        requirements.get("component_hints", []),
    )
    print("Knowledge graph context:\n", kg_context)

    selection = select_components(requirements, kg_context)
    print("Component selection:", selection)

    # The original user question is the primary retrieval query.
    rag_result = retrieve(
        raw_requirements,
        secondary_query=selection.get("rag_query"),
    )
    print("RAG retrieval found relevant chunks:", rag_result["found"])

    feedback = None

    for attempt in range(1, MAX_REGENERATIONS + 2):
        answer = reason(
            selection,
            rag_result,
            feedback=feedback,
            original_query=raw_requirements,
        )

        verdict = verify(answer, rag_result)

        print(
            f"Verification attempt {attempt}: "
            f"passed={verdict['passed']}"
        )

        if verdict["passed"] or attempt == MAX_REGENERATIONS + 1:
            return answer

        feedback = _verification_feedback(verdict)
        print("Regenerating because:", feedback)

    return answer


if __name__ == "__main__":
    print("\n========================================")
    print("          DESIGN ASSISTANT")
    print("========================================")

    if not _knowledge_base_ready():
        print(
            "\nNo processed knowledge base was found."
            "\nProcess a datasheet first, then run this assistant again."
        )
    else:
        print("\nKnowledge graph and vector database are loaded from disk.")
        print("Type 'exit' or 'quit' to stop.")
        print("Press Ctrl+C to stop at any time.\n")

        while True:
            try:
                user_requirements = input(
                    "Describe your design requirements: "
                ).strip()

                if user_requirements.lower() in {"exit", "quit"}:
                    print("\nExiting Design Assistant...")
                    break

                if not user_requirements:
                    continue

                print("\n" + "=" * 70)
                print("Processing query...")
                print("=" * 70 + "\n")

                try:
                    answer = run_pipeline(user_requirements)
                    print("\n=== Reasoning LLM output ===\n")
                    print(answer)
                except Exception as exc:
                    print(f"\nError while processing query: {exc}")
                    print("You can continue asking another question.")

                print("\n" + "=" * 70 + "\n")

            except KeyboardInterrupt:
                print("\n\nExiting Design Assistant...")
                break
