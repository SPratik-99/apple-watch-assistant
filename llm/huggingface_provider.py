from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LOCAL_LLM_MODEL, LOCAL_MODEL_DIR
from llm.base import LLMProvider


class HuggingFaceLocalProvider(LLMProvider):
    name = "huggingface"

    def __init__(self):
        model_source = str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "config.json").exists() else LOCAL_LLM_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForCausalLM.from_pretrained(model_source)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.model_source = model_source

    def generate(self, system_prompt: str, messages: List[Dict], temperature: float = 0.3) -> str:
        chat = [{"role": "system", "content": system_prompt}] + messages
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=max(temperature, 0.2),
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def status(self) -> Dict:
        return {"provider": "Offline Hugging Face", "model": self.model_source, "device": self.device, "online": False}
