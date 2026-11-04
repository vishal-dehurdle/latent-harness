"""The latent reasoner: R1-Distill with K pause tokens refined by a LOTUS loop.

Sequence layout, identical to the `direct` eval condition, so K=0 is exactly the
no-reasoning floor:

    <prompt><｜Assistant｜><think>\\n\\n [z_1 .. z_K] </think>\\n\\nThe final answer is $\\boxed{ answer}$.<eos>

LOTUS recurrence over R iterations, with every weight shared across iterations:

    h^(0) = 0
    h^(t) = f_theta(E + g(h^(t-1)) | C_pre)

- f_theta is the whole backbone (frozen, plus LoRA) run on the K latent
  positions while attending to the prompt's KV cache C_pre. It returns the
  final post-norm hidden states, the ones the LM head reads.
- E (K x d) are learnable pause embeddings.
- g maps hidden states back to input-embedding scale:
  g(h) = gamma * s_emb * W (h / rms(h)), with W spectrally normalized
  (||W||_2 <= 1) and gamma < 1 damping the feedback. That alone does not make
  the whole loop contractive, which depends on the backbone too; training logs
  the step residuals ||h^(t) - h^(t-1)|| (not the Jacobian) to show how it settles.

Z = h^(R) is what training aligns to the teacher's reasoning (Sinkhorn) and
decodes through the frozen LM head (keyword loss). The answer is decoded with
the final iteration's latent KV entries in context.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .lora import DEFAULT_TARGETS, apply_lora


CONFIG_FILE = "latent_config.json"
WEIGHTS_FILE = "latent.safetensors"


@dataclass
class LatentConfig:
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    revision: str = "main"  # training resolves this to a commit SHA
    num_latents: int = 4  # K; 0 trains the same model with no latent positions (the fine-tuned floor)
    num_iterations: int = 4  # R
    spectral_norm: bool = True  # constrain ||W||_2 <= 1 in the feedback map
    feedback_scale: float = 0.9  # gamma
    lora_rank: int = 16  # 0 freezes the backbone entirely
    lora_alpha: float = 32.0
    lora_targets: Tuple[str, ...] = DEFAULT_TARGETS

    def __post_init__(self) -> None:
        self.lora_targets = tuple(self.lora_targets)
        if self.num_latents < 0 or self.num_iterations < 1:
            raise ValueError("num_latents must be >= 0 and num_iterations >= 1")


@dataclass
class LatentPass:
    cache: Any  # KV cache: prompt plus the final iteration's latent positions
    attention_mask: torch.Tensor  # (B, prompt + K)
    lengths: torch.Tensor  # (B,) real tokens and latents per row, i.e. each row's next position id
    trajectory: List[torch.Tensor]  # h^(1) .. h^(R), each (B, K, d); empty when K = 0


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def split_at_think_close(ids: Sequence[int], think_close_id: int) -> Tuple[List[int], List[int]]:
    """Split a direct-layout prompt at its last </think>: latents go between the two parts."""
    ids = list(ids)
    if think_close_id not in ids:
        raise ValueError("Prompt has no </think>; latent prompts use the direct layout")
    i = len(ids) - 1 - ids[::-1].index(think_close_id)
    return ids[:i], ids[i:]


def _embedding_stats(weight: torch.Tensor, chunk: int = 16384) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Per-dimension mean and std of the input embeddings, and their mean RMS, without a full fp32 copy."""
    n, d = weight.shape
    total = torch.zeros(d, device=weight.device)
    squares = torch.zeros(d, device=weight.device)
    with torch.no_grad():
        for start in range(0, n, chunk):
            rows = weight[start : start + chunk].float()
            total += rows.sum(0)
            squares += rows.pow(2).sum(0)
    mean = total / n
    std = (squares / n - mean.pow(2)).clamp(min=0).sqrt()
    rms = float((squares.sum() / (n * d)).sqrt())
    return mean.cpu(), std.cpu(), rms


class LatentReasoner(nn.Module):
    def __init__(self, config: LatentConfig, base: nn.Module):
        super().__init__()
        self.config = config
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        if config.lora_rank > 0:
            apply_lora(base, config.lora_rank, config.lora_alpha, config.lora_targets)

        embeddings = base.get_input_embeddings().weight
        device, d = embeddings.device, embeddings.shape[1]
        mean, std, rms = _embedding_stats(embeddings)
        # Each pause embedding starts as a random vector with the statistics of a real token embedding.
        init = mean + std * torch.randn(config.num_latents, d)
        self.latent_embeddings = nn.Parameter(init.to(device))
        self.register_buffer("embed_rms", torch.tensor(rms, device=device))

        feedback = nn.Linear(d, d, bias=False, device=device, dtype=torch.float32)
        with torch.no_grad():
            feedback.weight.copy_(torch.eye(d))
        self.feedback = spectral_norm(feedback) if config.spectral_norm else feedback

    # ------------------------------------------------------------------ construction and checkpoints

    @classmethod
    def from_pretrained(
        cls, config: LatentConfig, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "LatentReasoner":
        from transformers import AutoModelForCausalLM

        device = device or resolve_device()
        base = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.revision, dtype=dtype or default_dtype(device)
        )
        return cls(config, base.to(device))

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """LoRA adapters, pause embeddings and the feedback map (with its spectral-norm buffers)."""
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {
            k: v
            for k, v in self.state_dict().items()
            if k in trainable or k.startswith("feedback.") or k == "embed_rms"
        }

    def save(self, path: str) -> None:
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.trainable_state_dict().items()}
        save_file(tensors, os.path.join(path, WEIGHTS_FILE))
        with open(os.path.join(path, CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load_trainable(self, path: str) -> None:
        from safetensors.torch import load_file

        state = load_file(os.path.join(path, WEIGHTS_FILE), device=str(self.device))
        missing = set(self.trainable_state_dict()) - set(state)
        _, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint {path} does not match the model: missing {sorted(missing)[:5]}, "
                               f"unexpected {sorted(unexpected)[:5]}")

    @classmethod
    def load(
        cls, path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "LatentReasoner":
        with open(os.path.join(path, CONFIG_FILE), encoding="utf-8") as f:
            config = LatentConfig(**json.load(f))
        model = cls.from_pretrained(config, device, dtype)
        model.load_trainable(path)
        return model

    # ------------------------------------------------------------------ forward pieces

    @property
    def decoder(self) -> nn.Module:
        return self.base.model

    @property
    def device(self) -> torch.device:
        return self.latent_embeddings.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.get_input_embeddings().weight.dtype

    def feed_back(self, h: torch.Tensor) -> torch.Tensor:
        """g(h) = gamma * s_emb * W (h / rms(h)): hidden states back to input-embedding scale."""
        h = h.float()
        unit = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.config.feedback_scale * self.embed_rms * self.feedback(unit)

    def prefill(self, prefix_ids: Sequence[Sequence[int]], pad_id: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """Left-padded prompt pass; returns its KV cache, attention mask and true lengths."""
        width = max(len(ids) for ids in prefix_ids)
        ids = torch.full((len(prefix_ids), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(prefix_ids), width), dtype=torch.long)
        for i, row in enumerate(prefix_ids):
            ids[i, width - len(row) :] = torch.tensor(row, dtype=torch.long)
            mask[i, width - len(row) :] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.decoder(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        return out.past_key_values, mask, mask.sum(-1)

    def think(self, cache: Any, attention_mask: torch.Tensor, lengths: torch.Tensor) -> LatentPass:
        """Run the LOTUS loop on K latent positions after the prompt."""
        from transformers import DynamicCache

        k = self.config.num_latents
        if k == 0:
            return LatentPass(cache, attention_mask, lengths, [])

        # Every iteration re-runs the latent positions against the same prompt cache.
        prompt_kv = [(layer.keys, layer.values) for layer in cache.layers]
        batch = attention_mask.shape[0]
        positions = lengths[:, None] + torch.arange(k, device=self.device)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch, k)], dim=-1)
        pause = self.latent_embeddings.unsqueeze(0).expand(batch, -1, -1)

        h: Optional[torch.Tensor] = None
        trajectory = []
        for _ in range(self.config.num_iterations):
            x = pause if h is None else pause + self.feed_back(h)
            step_cache = DynamicCache(ddp_cache_data=prompt_kv, config=self.base.config)
            out = self.decoder(
                inputs_embeds=x.to(self.dtype),
                attention_mask=attention_mask,
                position_ids=positions,
                past_key_values=step_cache,
                use_cache=True,
            )
            h = out.last_hidden_state
            trajectory.append(h)
        return LatentPass(step_cache, attention_mask, lengths + k, trajectory)

    def extend(
        self, cache: Any, attention_mask: torch.Tensor, lengths: torch.Tensor, ids: torch.Tensor, ids_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        """Append right-padded tokens after the cache; returns their logits and the extended state."""
        positions = lengths[:, None] + torch.arange(ids.shape[1], device=self.device)
        attention_mask = torch.cat([attention_mask, ids_mask], dim=-1)
        out = self.base(
            input_ids=ids, attention_mask=attention_mask, position_ids=positions, past_key_values=cache, use_cache=True
        )
        return out.logits, out.past_key_values, attention_mask, lengths + ids_mask.sum(-1)

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        think_close_id: int,
        eos_id: int,
        pad_id: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        generator: torch.Generator,
    ) -> List[Tuple[List[int], bool]]:
        """Prompt -> K latents -> forced answer prefix -> sampled answer. Returns (token ids, hit eos)."""
        parts = [split_at_think_close(p, think_close_id) for p in prompts]
        suffix = parts[0][1]
        if any(s != suffix for _, s in parts):
            raise ValueError("All prompts in a batch must share the same answer prefix")

        cache, mask, lengths = self.prefill([p for p, _ in parts], pad_id)
        state = self.think(cache, mask, lengths)
        batch = len(prompts)
        suffix_ids = torch.tensor([suffix] * batch, device=self.device)
        logits, cache, mask, lengths = self.extend(
            state.cache, state.attention_mask, state.lengths, suffix_ids, torch.ones_like(suffix_ids)
        )

        outputs: List[List[int]] = [[] for _ in range(batch)]
        done = torch.zeros(batch, dtype=torch.bool)
        for _ in range(max_tokens):
            next_ids = sample_next(logits[:, -1], temperature, top_p, generator)
            for i in range(batch):
                if not done[i]:
                    outputs[i].append(int(next_ids[i]))
            done |= next_ids.cpu() == eos_id
            if bool(done.all()):
                break
            step = next_ids[:, None].to(self.device)
            logits, cache, mask, lengths = self.extend(cache, mask, lengths, step, torch.ones_like(step))
        return [(ids, bool(ids and ids[-1] == eos_id)) for ids in outputs]


def sample_next(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> torch.Tensor:
    """Nucleus sampling on the CPU (seeded, device-independent); temperature 0 is greedy."""
    logits = logits.float().cpu()
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    keep = sorted_probs.cumsum(-1) - sorted_probs < top_p  # smallest prefix reaching top_p; never empty
    choice = torch.multinomial(sorted_probs * keep, 1, generator=generator)
    return order.gather(-1, choice).squeeze(-1)
