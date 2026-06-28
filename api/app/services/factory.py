from .ai_service import AIService
from .openai_service import OpenAIService
from .gemini_service import GeminiService
from .groq_service import GroqService

def get_ai_service (provider: str) -> str:

    providers = {

        "openai" : OpenAIService,
        "gemini" : GeminiService,
        "groq" : GroqService

    }

    provider_lower = provider.lower()

    if provider_lower not in providers:
        raise ValueError(f'Provedor {provider} não listado no sistema.')
    
    return providers[provider_lower]()