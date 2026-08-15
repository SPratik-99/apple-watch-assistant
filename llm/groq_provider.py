from typing import Dict, List

from groq import Groq

from config import GROQ_MODEL, get_groq_api_key
from llm.base import LLMProvider


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self):
        key = get_groq_api_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.client = Groq(api_key=key)
        self.model = GROQ_MODEL

    def generate(self, system_prompt: str, messages: List[Dict], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=temperature,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()

    def status(self) -> Dict:
        return {"provider": "Groq", "model": self.model, "online": True}
