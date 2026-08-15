from abc import ABC, abstractmethod
from typing import Dict, List


class LLMProvider(ABC):
    name = "unknown"

    @abstractmethod
    def generate(self, system_prompt: str, messages: List[Dict], temperature: float = 0.2) -> str:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> Dict:
        raise NotImplementedError
