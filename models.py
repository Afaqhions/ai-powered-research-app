from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
import logging

log = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    {"id": "groq:llama-3.3-70b-versatile",  "label": "Llama 3.3 70B",         "provider": "Groq",   "free": True},
    {"id": "groq:llama-3.1-8b-instant",      "label": "Llama 3.1 8B",          "provider": "Groq",   "free": True},
    {"id": "groq:mixtral-8x7b-32768",        "label": "Mixtral 8x7B",          "provider": "Groq",   "free": True},
    {"id": "groq:gemma2-9b-it",              "label": "Gemma 2 9B",            "provider": "Groq",   "free": True},
    {"id": "openai:gpt-4o",                  "label": "GPT-4o",                "provider": "OpenAI", "free": False},
    {"id": "openai:gpt-4o-mini",             "label": "GPT-4o Mini",           "provider": "OpenAI", "free": False},
    {"id": "openai:gpt-4.1",                 "label": "GPT-4.1",               "provider": "OpenAI", "free": False},
    {"id": "openai:gpt-4.1-mini",            "label": "GPT-4.1 Mini",          "provider": "OpenAI", "free": False},
    {"id": "openai:gpt-4.1-nano",            "label": "GPT-4.1 Nano",          "provider": "OpenAI", "free": False},
    {"id": "openai:o3-mini",                 "label": "o3-mini",               "provider": "OpenAI", "free": False},
    {"id": "google_genai:gemini-2.5-pro-exp-03-25", "label": "Gemini 2.5 Pro",         "provider": "Google", "free": True},
    {"id": "google_genai:gemini-2.0-flash",  "label": "Gemini 2.0 Flash",       "provider": "Google", "free": True},
    {"id": "google_genai:gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite", "provider": "Google", "free": True},
    {"id": "google_genai:gemini-1.5-flash",  "label": "Gemini 1.5 Flash",       "provider": "Google", "free": True},
    {"id": "google_genai:gemini-1.5-pro",    "label": "Gemini 1.5 Pro",         "provider": "Google", "free": True},
]

def resolve_model(model_id: str, temperature: float = 0) -> BaseChatModel:
    return init_chat_model(model_id, temperature=temperature)

def lookup_provider(model_id: str) -> str:
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m["provider"]
    return model_id.split(":")[0] if ":" in model_id else "unknown"

def lookup_label(model_id: str) -> str:
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m["label"]
    return model_id
