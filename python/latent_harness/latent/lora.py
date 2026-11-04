"""Minimal LoRA adapters for the frozen backbone's linear layers.

Kept in-tree (instead of depending on peft) so the trainable parameters, their
names in checkpoints and their dtype handling are explicit and testable.
"""

from typing import List, Sequence
import math

import torch
import torch.nn as nn


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen and A, B trained in float32."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B = 0, so the adapter starts as a no-op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return out + (self.scaling * delta).to(out.dtype)


def apply_lora(model: nn.Module, rank: int, alpha: float, targets: Sequence[str] = DEFAULT_TARGETS) -> List[LoRALinear]:
    """Wrap every nn.Linear whose attribute name is in `targets`; returns the new adapters."""
    adapters = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                adapter = LoRALinear(child, rank, alpha)
                setattr(module, name, adapter)
                adapters.append(adapter)
    if not adapters:
        raise ValueError(f"No linear layers named {list(targets)} found to adapt")
    return adapters
