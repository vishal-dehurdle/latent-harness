"""How much does the OT loss reward splitting the K latents apart?

  python scripts/analysis/ot_gap.py outputs/latent_data/gsm8k_train

Compares the loss with all K slots at the trace mean (collapsed) against the loss at the split attractor that
ot_attractor.py finds, on 64 random training traces, for raw and standardised targets at two epsilons.
"""

import json
import sys

import numpy as np
import torch

from latent_harness.sinkhorn import sinkhorn_optimal_transport

DATA, K = sys.argv[1], 4
rows = [json.loads(line) for line in open(f"{DATA}/examples.jsonl")]
states = np.memmap(f"{DATA}/states.f16", dtype=np.float16, mode="r")
d = states.size // sum(r["state_count"] for r in rows)
states = states.reshape(-1, d)

# Per-channel statistics over every stored state (chunked; float64 accumulators).
total, sq, n = np.zeros(d), np.zeros(d), 0
for start in range(0, len(states), 1 << 16):
    x = states[start : start + (1 << 16)].astype(np.float64)
    total += x.sum(0)
    sq += (x * x).sum(0)
    n += len(x)
mean = torch.from_numpy(total / n).float()
std = torch.from_numpy(np.sqrt(np.maximum(sq / n - (total / n) ** 2, 1e-12))).float()

rng = np.random.default_rng(0)
picks = rng.choice(len(rows), 64, replace=False)


def ot(z, h, eps):
    return float(sinkhorn_optimal_transport(z[None], h[None], epsilon=eps, max_iters=50, log_domain=True)[0])


for name, eps in (("raw", 0.1), ("raw", 0.03), ("standardized", 0.1), ("standardized", 0.03)):
    collapsed, split = [], []
    for i in picks:
        r = rows[i]
        h = torch.from_numpy(states[r["state_offset"] : r["state_offset"] + r["state_count"]].astype(np.float32))
        if name == "standardized":
            h = (h - mean) / std
        h = h / h.norm(dim=1).mean()
        collapsed.append(ot(h.mean(0, keepdim=True).repeat(K, 1), h, eps))
        # With the coupling fixed, each slot's optimum is the coupling-weighted mean of its targets: iterate to the attractor.
        z = h[rng.choice(len(h), K, replace=False)].clone()
        for _ in range(30):
            _, pi = sinkhorn_optimal_transport(z[None], h[None], epsilon=eps, max_iters=50, log_domain=True)
            z = (pi[0] @ h) / pi[0].sum(1, keepdim=True)
        split.append(ot(z, h, eps))
    c, s = np.mean(collapsed), np.mean(split)
    print(f"{name:13s} eps {eps:4.2f}: OT collapsed {c:.4f}, split {s:.4f}, splitting saves {100 * (c - s) / c:.1f}%")
