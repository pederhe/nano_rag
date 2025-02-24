import os

# Query engine configuration
QUERY_ENGINE = "ollama"  # Optional values: "ollama" or "vllm"
TEMPERATURE = 0.7  # Default temperature value

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:1.5b"

# vLLM configuration
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Embedding configuration
EMBEDDING_TYPE = "ollama"  # Optional values: "ollama" or "nomic"