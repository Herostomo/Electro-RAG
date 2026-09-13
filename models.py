"""
Central place to pick which locally-pulled Ollama models each pipeline
stage uses. Change these once here and everything else picks it up.

Tuned for a 4GB VRAM card (e.g. RTX 3050 Ti) — qwen2.5:3b-instruct stays
fully on GPU, which matters a lot for kg_builder.py since it fires one
LLM call per document chunk.

Pull them first:
    ollama pull nomic-embed-text
    ollama pull qwen2.5:3b-instruct
    # optional, only if you want a stronger reasoning stage and can accept
    # slower generation from partial CPU offload on a 4GB card:
    # ollama pull qwen2.5:7b-instruct
"""

EMBEDDING_MODEL = "nomic-embed-text"       # ~274MB, trivial on any GPU
EXTRACTION_MODEL = "qwen2.5:3b-instruct"   # requirements parsing, KG extraction, component selection - stays on GPU
REASONING_MODEL = "qwen2.5:3b-instruct"    # final reasoning stage - swap to "qwen2.5:7b-instruct" for more quality at the cost of speed
