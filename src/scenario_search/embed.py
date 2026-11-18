"""Thin wrappers around the two local encoders (no network access at query time).

* CLIP ViT-B/16 (OpenAI weights, Hugging Face format) for text -> camera-frame search.
* all-MiniLM-L6-v2 (sentence-transformers weights) for text -> scenario-caption search.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class ClipEncoder:
    def __init__(self, model_dir: Path = config.CLIP_DIR, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(str(model_dir)).eval().to(device)
        proc_dir = model_dir / "processor" if (model_dir / "processor").exists() else model_dir
        self.processor = CLIPProcessor.from_pretrained(str(proc_dir))
        self.device = device

    def encode_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            out = self.model.get_text_features(**{k: v.to(self.device) for k, v in inp.items()})
            out = getattr(out, "pooler_output", out)
        return _l2(out.cpu().numpy().astype(np.float32))

    def encode_images(self, images: list) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.processor(images=images, return_tensors="pt")
            out = self.model.get_image_features(pixel_values=inp["pixel_values"].to(self.device))
            out = getattr(out, "pooler_output", out)
        return _l2(out.cpu().numpy().astype(np.float32))


class TextEncoder:
    """Mean-pooled MiniLM sentence embeddings (same recipe as sentence-transformers)."""

    def __init__(self, model_dir: Path = config.TEXT_DIR, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval().to(device)
        self.device = device

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inp = self.tok(
                    texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                inp = {k: v.to(self.device) for k, v in inp.items()}
                h = self.model(**inp).last_hidden_state
                m = inp["attention_mask"].unsqueeze(-1).float()
                outs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        return _l2(np.concatenate(outs).astype(np.float32))


@lru_cache(maxsize=1)
def clip_encoder() -> ClipEncoder:
    return ClipEncoder()


@lru_cache(maxsize=1)
def text_encoder() -> TextEncoder:
    return TextEncoder()
