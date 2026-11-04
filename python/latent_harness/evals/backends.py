"""Generation backends: MLX on Apple Silicon, vLLM on CUDA GPUs.

Both take pre-tokenized prompts, so the chat template is applied once, in
prompts.py, identically for every backend.
"""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional


@dataclass
class SamplingConfig:
    n: int  # samples per prompt
    temperature: float
    top_p: float
    max_tokens: int
    seed: int


@dataclass
class Completion:
    text: str
    token_ids: List[int]
    finish_reason: str  # "stop" (end of sequence) or "length" (hit max_tokens)
    thinking_tokens: Optional[int] = None  # set by backends that think outside the text (latent positions)


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class MLXBackend:
    """mlx-lm batched generation on Apple Silicon.

    All prompts and samples in one call share a sampler and a seed. mlx-lm does
    not report why a sequence stopped, so a completion that used its whole token
    budget is recorded as "length".
    """

    name = "mlx"

    def __init__(self, model: str, revision: Optional[str] = None, batch_size: int = 8):
        from mlx_lm import load

        self.model, self._wrapper = load(model, revision=revision)
        self.tokenizer = self._wrapper._tokenizer
        self.batch_size = batch_size

    def generate(self, prompt_ids: List[List[int]], config: SamplingConfig) -> List[List[Completion]]:
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(config.seed)
        flat = [ids for ids in prompt_ids for _ in range(config.n)]
        response = batch_generate(
            self.model,
            self._wrapper,
            flat,
            max_tokens=config.max_tokens,
            return_token_ids=True,
            sampler=make_sampler(temp=config.temperature, top_p=config.top_p),
            completion_batch_size=self.batch_size,
            prefill_batch_size=min(self.batch_size, 8),
        )
        assert response.token_ids is not None
        completions = [
            Completion(
                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                token_ids=list(ids),
                finish_reason="length" if len(ids) >= config.max_tokens else "stop",
            )
            for ids in response.token_ids
        ]
        return [completions[i * config.n : (i + 1) * config.n] for i in range(len(prompt_ids))]

    def describe(self) -> Dict[str, Any]:
        import mlx.core as mx

        return {
            "backend": self.name,
            "batch_size": self.batch_size,
            "device": str(mx.default_device()),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
        }


class VLLMBackend:
    """vLLM on a CUDA GPU, for scaling the same evaluation beyond a laptop."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.92,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
    ):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt_ids: List[List[int]], config: SamplingConfig) -> List[List[Completion]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params)
        return [
            [
                Completion(
                    text=self.tokenizer.decode(o.token_ids, skip_special_tokens=True),
                    token_ids=list(o.token_ids),
                    finish_reason="length" if o.finish_reason == "length" else "stop",
                )
                for o in out.outputs
            ]
            for out in outputs
        ]

    def describe(self) -> Dict[str, Any]:
        import torch

        return {
            "backend": self.name,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vllm": _package_version("vllm"),
        }


def load_backend(name: str, model: str, revision: Optional[str] = None, **kwargs: Any):
    """`model` is a Hub id or local path; for the latent backend, a training checkpoint directory."""
    if name == "mlx":
        return MLXBackend(model, revision=revision, **kwargs)
    if name == "vllm":
        return VLLMBackend(model, revision=revision, **kwargs)
    if name == "latent":
        from ..latent.backend import LatentBackend

        return LatentBackend.from_checkpoint(model, **kwargs)
    raise ValueError(f"Unknown backend {name!r}; expected 'mlx', 'vllm' or 'latent'")
