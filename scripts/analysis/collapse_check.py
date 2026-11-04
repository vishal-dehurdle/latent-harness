"""Do the trained latents differ across problems and across slots, and what do they carry?

  python scripts/analysis/collapse_check.py outputs/latent/k4/final outputs/latent_data/gsm8k_train

Runs a latent checkpoint on 256 training-set problems (the 39 held-out ones plus a random sample) and
compares the final latent states Z with the pooled teacher states they were aligned to, and with the
question-end hidden state of the same backbone (no latents) as a control.
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from latent_harness.latent import LatentReasoner, LatentTrainingSet, TeacherStore, resolve_device
from latent_harness.latent.data import collate
from latent_harness.latent.train import split_indices
from latent_harness.sinkhorn import sinkhorn_optimal_transport

CHECKPOINT, DATA, N = sys.argv[1], sys.argv[2], 256

model = LatentReasoner.load(CHECKPOINT, resolve_device("auto")).eval()
tok = AutoTokenizer.from_pretrained(model.config.model, revision=model.config.revision)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
store = TeacherStore(DATA)
dataset = LatentTrainingSet(store, tok)
splits = split_indices([e.id for e in store.examples], 0.02)
rng = np.random.default_rng(0)
idx = splits["val"] + sorted(rng.choice(splits["train"], N - len(splits["val"]), replace=False).tolist())

latents, teacher_means, teacher_states, prompt_states, trace_states = [], [], [], [], []
with torch.no_grad():
    for start in range(0, len(idx), 8):
        batch = collate([dataset[i] for i in idx[start : start + 8]], pad)
        cache, mask, lengths = model.prefill(batch["prefix"], pad)
        latents.append(model.think(cache, mask, lengths).trajectory[-1].float().cpu())
        for prefix in batch["prefix"]:  # what the model holds at the end of the question, without latents
            ids = torch.tensor([prefix], device=model.device)
            prompt_states.append(model.base.model(input_ids=ids).last_hidden_state[0, -1].float().cpu())
        for b in range(len(batch["ids"])):
            states = batch["states"][b][batch["states_mask"][b]]
            trace_states.append(states)
            teacher_means.append(states.mean(0))
            teacher_states.append(states[rng.integers(len(states))])  # one random state per problem
Z = torch.cat(latents).numpy()  # (N, K, d)
T = torch.stack(teacher_means).numpy()  # (N, d)
S = torch.stack(teacher_states).numpy()  # (N, d)


def shared_fraction(X):
    """Share of the vectors' energy in their common mean: 1 = identical, 0 = nothing in common."""
    m = X.mean(0)
    return float(m @ m / (X * X).sum(1).mean())


def mean_cosine(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = Xn @ Xn.T
    return float((G.sum() - len(X)) / (len(X) * (len(X) - 1)))


def effective_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def top_channel_share(X, k=5):
    energy = (X * X).mean(0)
    return float(np.sort(energy)[-k:].sum() / energy.sum())


def match_accuracy(A, B):
    """How often a problem's centred A is most similar to its own centred B among all N problems."""
    A, B = A - A.mean(0), B - B.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return float(((A @ B.T).argmax(1) == np.arange(len(A))).mean())


rows = {f"latent slot {k + 1}": Z[:, k] for k in range(Z.shape[1])}
rows["teacher state (random one per problem)"] = S
rows["teacher mean per problem"] = T
print(f"N = {len(idx)} problems, d = {Z.shape[2]}, chance match = {100 / len(idx):.1f}%\n")
print(f"{'':42s} {'shared':>7s} {'cos':>6s} {'erank':>6s} {'top5ch':>7s}")
for name, X in rows.items():
    print(f"{name:42s} {shared_fraction(X):7.3f} {mean_cosine(X):6.3f} {effective_rank(X):6.1f} {top_channel_share(X):7.3f}")
within = np.mean([mean_cosine(Z[i]) for i in range(len(Z))])
print(f"\ncosine between the 4 latent slots of the same problem: {within:.3f}")
print(f"match latents (mean of slots) to own teacher mean: {100 * match_accuracy(Z.mean(1), T):.1f}%")
print(f"match one teacher state to own teacher mean:       {100 * match_accuracy(S, T):.1f}%  (reference)")
P = torch.stack(prompt_states).numpy()
nval = len(splits["val"])
print(f"match question-end state (no latents) to teacher:  {100 * match_accuracy(P, T):.1f}%  (control)")
print(f"held-out only ({nval}): latents {100 * match_accuracy(Z[:nval].mean(1), T[:nval]):.1f}%, "
      f"question-end {100 * match_accuracy(P[:nval], T[:nval]):.1f}% (chance {100 / nval:.1f}%)")
Zc, Pc = Z.mean(1) - Z.mean(1).mean(0), P - P.mean(0)
Zc /= np.linalg.norm(Zc, axis=1, keepdims=True)
Pc /= np.linalg.norm(Pc, axis=1, keepdims=True)
print(f"cosine of centred latents with centred question-end state, same problem: {(Zc * Pc).sum(1).mean():.3f}")

# The same slots in the standardised space the --ot-standardize loss uses, and which part of its own reasoning
# trace each slot is coupled to (0 = start, 1 = end; spread = std of the 4 slot centres).
mean, std = store.channel_stats()
Zs = (Z - mean) / std
print(f"cosine between the 4 latent slots, standardised space: {np.mean([mean_cosine(Zs[i]) for i in range(len(Zs))]):.3f}")
spreads = []
for i, states in enumerate(trace_states):
    h = (states.numpy() - mean) / std
    scale = np.linalg.norm(h, axis=1).mean()
    _, pi = sinkhorn_optimal_transport(torch.from_numpy(Zs[i] / scale)[None], torch.from_numpy(h / scale)[None],
                                       epsilon=0.03, max_iters=50, log_domain=True)
    pi = pi[0].numpy()
    centres = pi @ np.linspace(0, 1, len(h)) / pi.sum(1)
    spreads.append(centres.std())
print(f"trace position each slot is coupled to: spread across slots {np.mean(spreads):.3f} (simulated split attractor 0.12-0.13)")
print(json.dumps({"n": len(idx), "z_norm": float(np.linalg.norm(Z, axis=2).mean()), "t_norm": float(np.linalg.norm(T, axis=1).mean())}))
