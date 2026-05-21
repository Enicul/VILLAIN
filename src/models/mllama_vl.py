"""Mllama vision-language wrapper for the multi-agent pipeline."""

import os
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


@dataclass
class MllamaGenerationConfig:
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    do_sample: bool = True


class MllamaVLModel:
    """Wrapper matching the Qwen3VLModel interface used by VILLAIN agents."""

    def __init__(
        self,
        model_name: str = "/home/aied_test/models/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda:0",
        generation_config: Optional[MllamaGenerationConfig] = None,
        suppression_strength: float = 1.0,
        suppression_layers: Optional[List[int]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.generation_config = generation_config or MllamaGenerationConfig()
        self.suppression_strength = suppression_strength
        self.suppression_layers = suppression_layers or []

        self._model = None
        self._processor = None

    def _apply_ffn_suppression(self):
        if self.suppression_strength >= 1.0:
            print("[MllamaVLModel] suppression_strength=1.0, no suppression applied")
            return

        patched = []
        target_layers = set(self.suppression_layers)
        for name, module in self._model.named_modules():
            parts = name.split(".")
            if parts[-1] != "mlp" or len(parts) < 2 or not parts[-2].isdigit():
                continue
            if "vision_model" in name or "vision" in name:
                continue
            layer_idx = int(parts[-2])
            if layer_idx not in target_layers:
                continue

            orig_forward = module.forward

            def make_patched(orig, strength):
                def patched_forward(*args, **kwargs):
                    return orig(*args, **kwargs) * strength
                return patched_forward

            module.forward = types.MethodType(
                lambda self, *args, _f=make_patched(orig_forward, self.suppression_strength), **kwargs: _f(*args, **kwargs),
                module,
            )
            patched.append(f"{name} (layer {layer_idx})")

        if not patched:
            raise RuntimeError(f"No Mllama language MLP modules found for layers {self.suppression_layers}")
        print(f"[MllamaVLModel] Suppressed strength={self.suppression_strength}: {patched}")

    def load(self):
        if self._model is None:
            print(f"[MllamaVLModel] Loading model: {self.model_name}")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            self._apply_ffn_suppression()
            self._model.eval()
            print(f"[MllamaVLModel] Model loaded on {self.device}")
        return self._model, self._processor

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.load()
        return self._processor

    def _convert_messages(self, messages: List[Dict]):
        converted = []
        images = []
        for message in messages:
            content = []
            for item in message.get("content", []):
                if item.get("type") == "text":
                    content.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    path = item.get("image") or item.get("image_path") or item.get("path")
                    if path and os.path.exists(path):
                        images.append(Image.open(path).convert("RGB"))
                        content.append({"type": "image"})
            converted.append({"role": message.get("role", "user"), "content": content})
        return converted, images

    def generate(
        self,
        messages: List[Dict],
        generation_config: Optional[MllamaGenerationConfig] = None,
        **kwargs,
    ) -> str:
        model, processor = self.load()
        config = generation_config or self.generation_config
        converted, images = self._convert_messages(messages)

        text = processor.apply_chat_template(
            converted,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {"text": text, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = processor(**processor_kwargs).to(model.device)

        do_sample = kwargs.get("do_sample", config.do_sample)
        requested_max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        gen_kwargs = {
            "max_new_tokens": min(requested_max_new_tokens, config.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = kwargs.get("temperature", config.temperature)
            gen_kwargs["top_p"] = kwargs.get("top_p", config.top_p)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = generated_ids[0, inputs["input_ids"].shape[-1]:]
        return processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def extract_response_after_think(self, output_text: str) -> str:
        return output_text

    def generate_and_extract(
        self,
        messages: List[Dict],
        generation_config: Optional[MllamaGenerationConfig] = None,
        **kwargs,
    ) -> str:
        return self.generate(messages, generation_config, **kwargs)
