"""Where does the OT loss pull K latents, for a given target normalisation and epsilon?

With the coupling detached, the loss <Pi, C> is minimised at z_k = sum_j Pi_kj h_j / sum_j Pi_kj, so iterating
that update from K distinct starting points finds the attractor the latents are trained towards. A blurred
coupling sends every slot to the trace mean (collapse); a sharp one splits the trace into K parts.

  python scripts/analysis/ot_attractor.py outputs/latent_data/gsm8k_train
"""

import json
import sys

import numpy as np
import torch

from latent_harness.sinkhorn import sinkhorn_optimal_transport

DATA, K, N = sys.argv[1], 4, 64
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
mean = total / n
std = np.sqrt(np.maximum(sq / n - mean**2, 1e-12))
print(f"{n} states; top-5 channels hold {np.sort(sq / n)[-5:].sum() / (sq / n).sum():.0%} of raw energy; "
      f"std ratio max/median {std.max() / np.median(std):.0f}")

rng = np.random.default_rng(0)
picks = rng.choice(len(rows), N, replace=False)


def attractor(h, epsilon, iters, rounds=30):
    """h: (S, d) already normalised. Returns the K slot positions after `rounds` updates and the coupling."""
    start = rng.choice(len(h), K, replace=False)
    z = h[start].clone()
    for _ in range(rounds):
        _, pi = sinkhorn_optimal_transport(z[None], h[None], epsilon=epsilon, max_iters=iters, log_domain=True)
        pi = pi[0]
        z = (pi @ h) / pi.sum(1, keepdim=True)
    return z, pi


def slot_cosine(z):
    zn = z / z.norm(dim=1, keepdim=True)
    g = zn @ zn.T
    return float((g.sum() - K) / (K * (K - 1)))


print(f"\n{'targets':13s} {'eps':>6s} {'iters':>5s} {'slot cos (raw)':>15s} {'slot cos (norm)':>16s} {'position spread':>16s}")
for name in ("raw", "standardized"):
    for epsilon in (0.1, 0.03, 0.01):
        for iters in (50, 200):
            raw_cos, norm_cos, spreads = [], [], []
            for i in picks:
                r = rows[i]
                h_raw = torch.from_numpy(states[r["state_offset"] : r["state_offset"] + r["state_count"]].astype(np.float32))
                h = (h_raw - torch.from_numpy(mean).float()) / torch.from_numpy(std).float() if name == "standardized" else h_raw
                scale = h.norm(dim=1).mean()  # as in train.ot_loss
                z, pi = attractor(h / scale, epsilon, iters)
                norm_cos.append(slot_cosine(z))
                # the same slots mapped back to raw space, where the earlier diagnostic measured 0.999
                z_raw = (pi @ h_raw) / pi.sum(1, keepdim=True)
                raw_cos.append(slot_cosine(z_raw))
                position = torch.linspace(0, 1, len(h))
                centres = (pi @ position) / pi.sum(1)
                spreads.append(float(centres.std()))
            print(f"{name:13s} {epsilon:6.2f} {iters:5d} {np.mean(raw_cos):15.4f} {np.mean(norm_cos):16.4f} {np.mean(spreads):16.3f}")
