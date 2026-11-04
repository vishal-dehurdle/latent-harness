"""The real latent reasoner (Phase 1): R1-Distill with K pause tokens, a LOTUS loop,
Sinkhorn alignment to the model's own reasoning traces and a frozen-head keyword loss.
"""

from .lora import DEFAULT_TARGETS, LoRALinear, apply_lora
from .model import LatentConfig, LatentPass, LatentReasoner, resolve_device, split_at_think_close
from .data import (
    LatentTrainingSet,
    TeacherStore,
    TraceExample,
    build_meta,
    collate,
    keyword_token_ids,
    pool_states,
    select_traces,
    teacher_state_fn,
    write_teacher_store,
)
from .train import TrainConfig, train

__all__ = [
    "DEFAULT_TARGETS",
    "LoRALinear",
    "apply_lora",
    "LatentConfig",
    "LatentPass",
    "LatentReasoner",
    "resolve_device",
    "split_at_think_close",
    "LatentTrainingSet",
    "TeacherStore",
    "TraceExample",
    "build_meta",
    "collate",
    "keyword_token_ids",
    "pool_states",
    "select_traces",
    "teacher_state_fn",
    "write_teacher_store",
    "TrainConfig",
    "train",
]
