"""Two-stage training of the latent reasoner.

Stage 1, Sinkhorn alignment:  loss = OT(Z, H_teacher)
Stage 2, answer fine-tuning:  loss = CE(answer) + ot_weight * OT(Z, H_teacher)
                                     + frozen_head_weight * keyword CE(lm_head(Z))

Z = h^(R), the final LOTUS iteration's latent states, and H_teacher the frozen
backbone's pooled hidden states over the trace's reasoning tokens. Both are
divided by the batch's mean teacher-state norm before the Sinkhorn solve, so
epsilon is relative to unit scale.

Ablations are settings, not separate code paths: no Sinkhorn is
`stage1_steps=0, ot_weight=0`; no frozen head is `frozen_head_weight=0`; no
spectral norm and K=0 live in LatentConfig.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import os
import shutil
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..frozen_head import FrozenHeadConstraint
from ..sinkhorn import sinkhorn_optimal_transport
from .data import LatentTrainingSet, collate
from .model import LatentPass, LatentReasoner


CHECKPOINT_DIR = "checkpoint"
FINAL_DIR = "final"
TRAIN_STATE_FILE = "train_state.pt"
TRAIN_CONFIG_FILE = "train_config.json"
LOG_FILE = "train_log.jsonl"


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    stage1_steps: int = 100  # optimizer steps of Sinkhorn alignment alone
    stage2_steps: int = 1000
    batch_size: int = 4  # micro-batch
    grad_accum: int = 4  # micro-batches per optimizer step
    lr: float = 1e-4
    warmup_steps: int = 30
    min_lr_ratio: float = 0.1  # cosine decay floor, as a fraction of lr
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ot_weight: float = 1.0  # stage 2 only; stage 1 is OT alone
    frozen_head_weight: float = 0.1
    ot_epsilon: float = 0.1
    ot_iters: int = 50
    # OT cost on per-channel standardised states (the teacher store's mean and std), so the component every
    # state shares and a few massive channels don't dominate it; a blurred coupling there collapses the K latents.
    ot_standardize: bool = False
    val_fraction: float = 0.02  # held out by a hash of the problem id
    max_examples: Optional[int] = None  # train on a prefix of the shuffled data (pilot runs)
    seed: int = 0
    save_every: int = 100
    val_every: int = 100

    @property
    def total_steps(self) -> int:
        return self.stage1_steps + self.stage2_steps


# ---------------------------------------------------------------------- losses

class _FloatHead(nn.Module):
    """The frozen LM head, returning float32 logits for stable log-softmax."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.to(self.head.weight.dtype)).float()


def ot_loss(
    z: torch.Tensor,
    states: torch.Tensor,
    states_mask: torch.Tensor,
    epsilon: float,
    iters: int,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    z, h = z.float(), states.float()
    if stats is not None:
        mean, std = stats
        z, h = (z - mean) / std, (h - mean) / std
    scale = h.norm(dim=-1)[states_mask].mean().detach().clamp(min=1e-6)
    cost, _ = sinkhorn_optimal_transport(
        z / scale, h / scale, epsilon=epsilon, max_iters=iters, log_domain=True, mask_h=states_mask
    )
    return cost.mean()


def answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Next-token cross-entropy on the answer tokens, and teacher-forced token accuracy."""
    logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    scored = targets != -100
    accuracy = (logits.argmax(-1)[scored] == targets[scored]).float().mean()
    return {"answer": loss, "answer_token_acc": accuracy}


def trajectory_stats(trajectory: Sequence[torch.Tensor]) -> Dict[str, Any]:
    """Step residuals ||h^(t) - h^(t-1)|| (absolute and relative), averaged over latents and batch."""
    absolute, relative = [], []
    for prev, curr in zip(trajectory, trajectory[1:]):
        delta = (curr.float() - prev.float()).norm(dim=-1)
        absolute.append(round(float(delta.mean()), 4))
        relative.append(round(float((delta / prev.float().norm(dim=-1).clamp(min=1e-6)).mean()), 5))
    return {"residual": absolute, "relative_residual": relative}


def batch_losses(
    model: LatentReasoner,
    batch: Dict[str, Any],
    stage: int,
    config: TrainConfig,
    head: FrozenHeadConstraint,
    pad_id: int,
    ot_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, Any]:
    cache, mask, lengths = model.prefill(batch["prefix"], pad_id)
    state: LatentPass = model.think(cache, mask, lengths)
    parts: Dict[str, torch.Tensor] = {}

    if state.trajectory:
        z = state.trajectory[-1]
        states = batch["states"].to(model.device)
        states_mask = batch["states_mask"].to(model.device)
        parts["ot"] = ot_loss(z, states, states_mask, config.ot_epsilon, config.ot_iters, ot_stats)
        if stage == 2 and config.frozen_head_weight > 0:
            parts["frozen_head"], _, rank = head(z, keyword_ids=batch["keyword_ids"])
            parts["effective_rank"] = torch.tensor(rank)

    if stage == 2:
        suffix = batch["suffix"].to(model.device)
        logits, *_ = model.extend(state.cache, state.attention_mask, state.lengths, suffix, batch["suffix_mask"].to(model.device))
        parts.update(answer_loss(logits, batch["labels"].to(model.device)))

    if stage == 1:
        total = parts["ot"]
    else:
        total = parts["answer"]
        if "ot" in parts and config.ot_weight > 0:
            total = total + config.ot_weight * parts["ot"]
        if "frozen_head" in parts:
            total = total + config.frozen_head_weight * parts["frozen_head"]
    parts["loss"] = total
    return {"parts": parts, "trajectory": state.trajectory}


# ---------------------------------------------------------------------- data order

def split_indices(ids: Sequence[str], val_fraction: float) -> Dict[str, List[int]]:
    """A deterministic train/validation split by problem id, stable as the dataset grows."""
    val = [i for i, pid in enumerate(ids) if zlib.crc32(pid.encode()) % 10_000 < val_fraction * 10_000]
    held = set(val)
    return {"train": [i for i in range(len(ids)) if i not in held], "val": val}


class EpochSampler:
    """Reshuffles every epoch; its position is checkpointed so a resumed run sees the same order."""

    def __init__(self, indices: Sequence[int], seed: int):
        self.indices = list(indices)
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.indices)

    def take(self, n: int) -> List[int]:
        out = []
        while len(out) < n:
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self._shuffle()
            out.append(int(self.order[self.position]))
            self.position += 1
        return out

    def state(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load(self, state: Dict[str, int]) -> None:
        self.epoch, self.position = state["epoch"], state["position"]
        self._shuffle()


# ---------------------------------------------------------------------- loop

def _lr_factor(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _memory_gb(device: torch.device) -> Dict[str, float]:
    """Accelerator memory: the driver's total (with the allocator's cached blocks) on MPS or the peak on CUDA,
    and the live tensors. A total that grows while live stays flat is caching, not a leak."""
    if device.type == "mps":
        total, live = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
    elif device.type == "cuda":
        total, live = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    else:
        return {}
    return {"memory_gb": round(total / 2**30, 2), "live_gb": round(live / 2**30, 2)}


def _truncate_log(path: str, last_step: int) -> None:
    """Drop log records past the checkpoint: a resumed run redoes those steps."""
    if not os.path.exists(path):
        return
    kept = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line)["step"] <= last_step:
                    kept.append(line)
            except json.JSONDecodeError:
                break  # torn final write
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(kept)


def _save_checkpoint(model, optimizer, scheduler, sampler, step: int, out_dir: str) -> None:
    final = os.path.join(out_dir, CHECKPOINT_DIR)
    tmp, old = final + ".tmp", final + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save(tmp)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler.state(),
            "torch_rng": torch.get_rng_state(),
        },
        os.path.join(tmp, TRAIN_STATE_FILE),
    )
    if os.path.exists(final):
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


@torch.no_grad()
def validate(model, dataset, indices, config, head, pad_id, ot_stats=None) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    for start in range(0, len(indices), config.batch_size):
        batch = collate([dataset[i] for i in indices[start : start + config.batch_size]], pad_id)
        parts = batch_losses(model, batch, 2, config, head, pad_id, ot_stats)["parts"]
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(batch["ids"])
    model.train()
    return {f"val_{k}": round(v / len(indices), 5) for k, v in totals.items()}


def train(model: LatentReasoner, tokenizer: Any, dataset: LatentTrainingSet, config: TrainConfig) -> str:
    """Train, resuming from out_dir/checkpoint if present. Returns the final checkpoint directory."""
    if model.config.num_latents == 0 and (config.stage1_steps or config.ot_weight or config.frozen_head_weight):
        raise ValueError("With K=0 there are no latent states: set stage1_steps=0, ot_weight=0, frozen_head_weight=0")
    os.makedirs(config.out_dir, exist_ok=True)
    config_path = os.path.join(config.out_dir, TRAIN_CONFIG_FILE)
    settings = {"train": asdict(config), "latent": asdict(model.config)}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(settings)):
                raise RuntimeError(f"{config.out_dir} was trained with different settings; use a new --out-dir")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

    torch.manual_seed(config.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    head = FrozenHeadConstraint(lm_head=_FloatHead(model.base.lm_head), hidden_size=model.latent_embeddings.shape[1])
    splits = split_indices([e.id for e in dataset.store.examples], config.val_fraction)
    train_indices = splits["train"]
    if config.max_examples is not None:
        rng = np.random.default_rng(config.seed)
        train_indices = sorted(rng.permutation(train_indices)[: config.max_examples].tolist())
    sampler = EpochSampler(train_indices, config.seed)
    ot_stats = None
    if config.ot_standardize:
        mean, std = dataset.store.channel_stats()
        ot_stats = (torch.from_numpy(mean).to(model.device), torch.from_numpy(std).to(model.device))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, config))

    start_step = 0
    checkpoint = os.path.join(config.out_dir, CHECKPOINT_DIR)
    if os.path.exists(os.path.join(checkpoint, TRAIN_STATE_FILE)):
        model.load_trainable(checkpoint)
        state = torch.load(os.path.join(checkpoint, TRAIN_STATE_FILE), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        start_step = state["step"]
        print(f"[train] resumed at step {start_step}", flush=True)
    log_path = os.path.join(config.out_dir, LOG_FILE)
    _truncate_log(log_path, start_step)

    print(
        f"[train] {len(train_indices)} train / {len(splits['val'])} val examples, "
        f"{sum(p.numel() for p in params):,} trainable parameters, steps {start_step}->{config.total_steps}",
        flush=True,
    )
    model.train()
    for step in range(start_step, config.total_steps):
        stage = 1 if step < config.stage1_steps else 2
        started = time.time()
        sums: Dict[str, float] = {}
        trajectory: List[torch.Tensor] = []
        for _ in range(config.grad_accum):
            batch = collate([dataset[i] for i in sampler.take(config.batch_size)], pad_id)
            out = batch_losses(model, batch, stage, config, head, pad_id, ot_stats)
            (out["parts"]["loss"] / config.grad_accum).backward()
            for name, value in out["parts"].items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) / config.grad_accum
            trajectory = out["trajectory"]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        record = {
            "step": step + 1,
            "stage": stage,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": round(float(grad_norm), 4),
            **{k: round(v, 5) for k, v in sums.items()},
            **trajectory_stats([h.detach() for h in trajectory]),
            "epoch": sampler.epoch,
            "seconds": round(time.time() - started, 2),
            **_memory_gb(model.device),
        }
        last = step + 1 == config.total_steps
        if splits["val"] and ((step + 1) % config.val_every == 0 or last) and stage == 2:
            record.update(validate(model, dataset, splits["val"], config, head, pad_id, ot_stats))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        shown = {k: record[k] for k in ("loss", "answer", "ot", "answer_token_acc", "val_answer") if k in record}
        print(f"[train] step {step + 1}/{config.total_steps} stage {stage} {shown} {record['seconds']}s "
              f"{record.get('memory_gb')} GB ({record.get('live_gb')} live)", flush=True)
        if (step + 1) % config.save_every == 0 or last:
            _save_checkpoint(model, optimizer, scheduler, sampler, step + 1, config.out_dir)
        if model.device.type == "mps":
            # The MPS allocator keeps every freed block; with prompt lengths varying per batch the pool
            # otherwise grows step after step until a 16 GB Mac swaps.
            torch.mps.empty_cache()

    final = os.path.join(config.out_dir, FINAL_DIR)
    model.save(final)
    return final
