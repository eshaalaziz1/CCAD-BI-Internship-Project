"""Shared config constants (no Ollama import)."""

MODEL = "medgemma1.5"
PROMPT_VERSION = "v5"
TEMPERATURE = 0.2
NUM_PREDICT = 700
KEEP_ALIVE = "30m"
MAX_RETRIES = 1

CHAT_OPTIONS = {
    "temperature": TEMPERATURE,
    "num_predict": NUM_PREDICT,
}
