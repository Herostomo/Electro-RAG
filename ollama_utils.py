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
        options={"temperature": temperature,
                 "num_ctx": 8192,
                },
    )
    content = response["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model '{model}' returned invalid JSON:\n{content}") from e



def chat_text(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
) -> str:
    """Stream Ollama's response so generation progress is visible."""

    print(f"\nGenerating answer with {model}...\n", flush=True)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        options={
            "temperature": min(temperature, 0.2),
            "num_ctx": 8192,
            # 500 tokens truncated legitimate engineering tables/conclusions.
            # A moderate cap plus repetition penalty reduces the common small-model loop.
            "num_predict": 1200,
            "repeat_penalty": 1.12,
            "top_p": 0.9,
        },
    )

    chunks = []

    for part in response:
        text = part.get("message", {}).get("content", "")
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print("\n\nAnswer generation complete.", flush=True)
    return "".join(chunks)



def embed_text(model: str, text: str) -> list[float]:
    """Returns an embedding vector for a piece of text using an Ollama embedding model."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
