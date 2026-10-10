"""Training-time API and local Qwen text generation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .config import LLMConfig
from .utils import fingerprint

class BaseLLM:
    def generate(self, prompt):
        raise NotImplementedError

    def generate_json(self, prompt, default=None):
        text = self.generate(prompt).strip()
        # Local reasoning models may emit a think block before the JSON.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char in "[{":
                try:
                    return decoder.raw_decode(text[index:])[0]
                except json.JSONDecodeError:
                    continue
        return default

class CachedLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, prompt):
        identity = {
            "backend": self.config.backend, "model": self.config.model,
            "temperature": self.config.temperature, "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens, "prompt": prompt,
            "endpoint": os.environ.get("OPENAI_BASE_URL", os.environ.get("BASE_URL", "")),
        }
        path = Path(self.config.cache_dir) / (fingerprint(identity) + ".txt")
        if path.exists():
            return path.read_text()
        text = self._generate(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return text

class RemoteOpenAILLM(CachedLLM):
    def __init__(self, config):
        super().__init__(config)
        self.client = None

    def _generate(self, prompt):
        from openai import OpenAI
        if self.client is None:
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
            endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
            if not key:
                raise ValueError("Set OPENAI_API_KEY before using the API backend")
            options = {"api_key": key, "max_retries": 3, "timeout": 120}
            if endpoint:
                options["base_url"] = endpoint
            self.client = OpenAI(**options)
        request = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature, "top_p": self.config.top_p,
            "max_completion_tokens": self.config.max_new_tokens,
        }
        if self.config.model.startswith("gpt-5"):
            request["reasoning_effort"] = "none"
        response = self.client.chat.completions.create(**request)
        if not response.choices or response.choices[0].finish_reason == "length":
            raise RuntimeError("LLM output missing or truncated; increase max-new-tokens")
        return response.choices[0].message.content or ""

class LocalQwenLLM(CachedLLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is not None:
            return
        device = self.config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.float32
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=dtype,
        ).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate(self, prompt):
        self._load()
        import torch
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context is None and hasattr(self.model.config, "text_config"):
            context = getattr(self.model.config.text_config, "max_position_embeddings", None)
        if context and encoded["input_ids"].shape[1] + self.config.max_new_tokens > context:
            raise ValueError("Prompt and output budget exceed the model context window")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, encoded["input_ids"].shape[1]:]
        if len(tokens) >= self.config.max_new_tokens and int(tokens[-1]) != self.tokenizer.eos_token_id:
            raise RuntimeError("Qwen output truncated; increase max-new-tokens")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

def build_llm(config):
    if config.backend == "api":
        return RemoteOpenAILLM(config)
    if config.backend == "local":
        return LocalQwenLLM(config)
    raise ValueError("backend must be api or local")
