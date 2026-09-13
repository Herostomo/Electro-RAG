"""
Central helper for talking to local Ollama models.
Uses the `ollama` python package directly for chat + embeddings so every
pipeline stage goes through one place.
"""
import json
import ollama


def chat_json(model: str, system: str, user: str, temperature: float = 0.2) -> dict:
    """
    Calls an Ollama chat model and forces JSON-formatted output.
    Returns a parsed Python dict. Raises ValueError if the model
    returns invalid JSON (rare, but weak/small models can do it).
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": temperature},
    )
    content = response["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model '{model}' returned invalid JSON:\n{content}") from e


def chat_text(model: str, system: str, user: str, temperature: float = 0.3) -> str:
    """Calls an Ollama chat model and returns plain text output."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def embed_text(model: str, text: str) -> list[float]:
    """Returns an embedding vector for a piece of text using an Ollama embedding model."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
