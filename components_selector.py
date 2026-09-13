"""'components selector model' block.
Takes structured requirements + knowledge-graph context, proposes
candidate components, and formulates the query to send to RAG.
"""
from ollama_utils import chat_json
from models import EXTRACTION_MODEL

SYSTEM_PROMPT = """You are a Component Selector for an electronics/PCB design assistant.
You receive:
1. Structured requirements (JSON)
2. Facts pulled from a knowledge graph built from datasheets/manuals

Propose candidate components and generate a focused search query to pull more
supporting detail from a datasheet vector database (RAG).

For factual datasheet questions, keep the RAG query concise and based on the
user-relevant parameter/component. Do NOT copy long lists of graph node/table
identifiers into the query. The original user question remains the primary
retrieval query downstream. Return STRICT JSON:
{
  "candidate_components": [{"component": "...", "reason": "..."}],
  "open_questions": ["..."],
  "rag_query": "a single focused search string for the RAG system"
}"""


def select_components(requirements: dict, kg_context: str) -> dict:
    user_content = (
        f"REQUIREMENTS:\n{requirements}\n\n"
        f"KNOWLEDGE GRAPH FACTS:\n{kg_context}"
    )
    return chat_json(EXTRACTION_MODEL, SYSTEM_PROMPT, user_content)
