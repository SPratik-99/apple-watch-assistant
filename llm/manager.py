from typing import Dict, List

from llm.groq_provider import GroqProvider
from llm.huggingface_provider import HuggingFaceLocalProvider


class LLMManager:
    """Exactly two provider options: Groq or offline Hugging Face."""

    def __init__(self, selected: str = "groq"):
        self.selected = selected
        self.provider = None
        self.error = None
        self._initialize()

    def _initialize(self):
        try:
            if self.selected == "groq":
                self.provider = GroqProvider()
            elif self.selected == "huggingface":
                self.provider = HuggingFaceLocalProvider()
            else:
                raise ValueError("Provider must be 'groq' or 'huggingface'")
        except Exception as exc:
            self.error = str(exc)

    def generate(self, system_prompt: str, messages: List[Dict]) -> str:
        if not self.provider:
            raise RuntimeError(self.error or "LLM provider unavailable")
        return self.provider.generate(system_prompt, messages)

    def status(self) -> Dict:
        if self.provider:
            return self.provider.status()
        return {"provider": self.selected, "available": False, "error": self.error}
