from typing import Dict, List

from conversation import SYSTEM_PROMPT, build_messages
from retrieval.vector_store import StaticPDFRetriever
from router import route_query
from web.apple_client import AppleClient
from llm.manager import LLMManager


class AppleWatchAssistant:
    def __init__(self, provider: str = "groq"):
        self.provider_name = provider
        self.llm = LLMManager(provider)
        self.retriever = StaticPDFRetriever()
        self.apple = AppleClient()

    def answer(self, question: str, history: List[Dict] = None) -> Dict:
        route = route_query(question, history)
        pdf_results = self.retriever.search(question) if route.needs_pdf else []

        web_result = (
            self.apple.fetch(question)
            if route.needs_web
            else {"available": False, "evidence": "", "sources": []}
        )

        messages = build_messages(question, history or [], pdf_results, web_result)
        response = self.llm.generate(SYSTEM_PROMPT, messages)

        return {
            "response": response,
            "route": route.__dict__,
            "pdf_sources": [r["metadata"] for r in pdf_results],
            "web_sources": web_result.get("sources", []),
            "provider": self.llm.status(),
        }

    def status(self):
        return {
            "provider": self.llm.status(),
            "retrieval": self.retriever.status(),
            "apple_reachable": self.apple.health(),
        }
