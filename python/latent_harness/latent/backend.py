"""Eval-harness backend for a trained latent reasoner.

It takes the same pre-tokenized `latent`-condition prompts as every other
backend, inserts the K latent positions before </think>, and samples the answer
after the forced prefix, so latent models are graded by the identical runner.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional
import os

import torch

from ..evals.backends import Completion, SamplingConfig
from ..evals.grading import THINK_CLOSE
from .model import LatentReasoner, resolve_device


class LatentBackend:
    name = "latent"

    def __init__(self, model: LatentReasoner, tokenizer: Any, batch_size: int = 16, checkpoint: Optional[str] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto", batch_size: int = 16) -> "LatentBackend":
        from transformers import AutoTokenizer

        model = LatentReasoner.load(path, resolve_device(device))
        tokenizer = AutoTokenizer.from_pretrained(model.config.model, revision=model.config.revision)
        return cls(model, tokenizer, batch_size=batch_size, checkpoint=os.path.abspath(path))

    def generate(self, prompt_ids: List[List[int]], config: SamplingConfig) -> List[List[Completion]]:
        generator = torch.Generator().manual_seed(config.seed)
        flat = [ids for ids in prompt_ids for _ in range(config.n)]
        results = []
        for start in range(0, len(flat), self.batch_size):
            results.extend(
                self.model.generate(
                    flat[start : start + self.batch_size],
                    think_close_id=self.think_close_id,
                    eos_id=self.tokenizer.eos_token_id,
                    pad_id=self.pad_id,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    generator=generator,
                )
            )
        completions = [
            Completion(
                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                token_ids=ids,
                finish_reason="stop" if hit_eos else "length",
                thinking_tokens=self.model.config.num_latents,
            )
            for ids, hit_eos in results
        ]
        return [completions[i * config.n : (i + 1) * config.n] for i in range(len(prompt_ids))]

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "latent_config": asdict(self.model.config),
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "batch_size": self.batch_size,
            "torch": torch.__version__,
        }
