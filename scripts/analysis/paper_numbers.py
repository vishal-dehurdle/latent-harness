"""Every number the write-up cites, computed from the saved runs (outputs/ only; nothing typed by hand).

  python scripts/analysis/paper_numbers.py   # -> outputs/paper_numbers.json, and a readable dump on stdout
"""

import json
import os

import numpy as np
from safetensors import safe_open

from latent_harness.evals.runner import load_records, summarize
from latent_harness.latent.train import split_indices

from compare_runs import paired  # this directory is on sys.path when run as a script

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs"))
BASE = os.path.join(ROOT, "baseline", "DeepSeek-R1-Distill-Qwen-1.5B")
RUNS = {"K0": "latent/k0", "K4": "latent/k4", "K4std": "latent/k4_std"}
EVALS = {"direct": os.path.join(BASE, "gsm8k__direct.jsonl"), "think": os.path.join(BASE, "gsm8k__think.jsonl")}
EVALS.update({name: os.path.join(ROOT, run, "final", "eval", "gsm8k__latent.jsonl") for name, run in RUNS.items()})


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


out = {"evals": {}, "paired": {}, "train": {}, "data": {}}

# GSM8K test evals: the harness's own summary (bootstrap CI over problems), plus token counts.
records = {name: load_records(path) for name, path in EVALS.items()}
for name, recs in records.items():
    s = summarize(recs)
    out["evals"][name] = {k: s[k] for k in ("problems", "pass@1", "ci95", "mean_completion_tokens",
                                             "median_completion_tokens", "mean_thinking_tokens", "hit_max_tokens_pct")}
correct = {name: {r["problem_id"]: bool(r["correct"]) for r in recs} for name, recs in records.items()}
rng = np.random.default_rng(0)
full = sorted(correct["direct"])
for a, b in (("K0", "direct"), ("K4", "direct"), ("K4std", "direct"), ("K0", "K4"), ("K0", "K4std"), ("K4std", "K4")):
    out["paired"][f"{a}-{b}"] = paired(np.array([correct[a][i] for i in full]), np.array([correct[b][i] for i in full]), rng)
# The think run is partial: compare every condition on exactly the problems it has finished.
sub = sorted(correct["think"])
out["think_subset"] = {name: round(100 * float(np.mean([correct[name][i] for i in sub])), 2) for name in correct}
out["think_subset"]["n"] = len(sub)
out["think_subset"]["first_id"], out["think_subset"]["last_id"] = sub[0], sub[-1]
# Share of the direct-to-think gap each condition closes, on those problems.
gap = out["think_subset"]["think"] - out["think_subset"]["direct"]
out["think_subset"]["gap_closed_pct"] = {name: round(100 * (out["think_subset"][name] - out["think_subset"]["direct"]) / gap, 1)
                                         for name in ("K0", "K4", "K4std")}
for b in ("direct", "K0", "K4std"):
    out["paired"][f"think-{b} (think subset)"] = paired(np.array([correct["think"][i] for i in sub]),
                                                       np.array([correct[b][i] for i in sub]), rng)
think_recs = records["think"]
out["evals"]["think"]["mean_thinking_tokens_correct"] = round(float(np.mean(
    [r["thinking_tokens"] for r in think_recs if r["correct"]])), 1)

# Training logs. The train/validation split is the one train.py makes from the teacher store's ids.
SPLIT = split_indices([e["id"] for e in read_jsonl(os.path.join(ROOT, "latent_data", "gsm8k_train", "examples.jsonl"))], 0.02)
out["split"] = {"train": len(SPLIT["train"]), "val": len(SPLIT["val"])}
for name, run in RUNS.items():
    log = read_jsonl(os.path.join(ROOT, run, "train_log.jsonl"))
    cfg = json.load(open(os.path.join(ROOT, run, "train_config.json")))
    s1 = [r for r in log if r["stage"] == 1]
    s2 = [r for r in log if r["stage"] == 2]
    with safe_open(os.path.join(ROOT, run, "final", "latent.safetensors"), "pt") as f:
        params = {k: int(np.prod(f.get_slice(k).get_shape())) for k in f.keys()}
    trainable = sum(v for k, v in params.items() if "lora" in k or k == "latent_embeddings" or k.startswith("feedback.parametrizations.weight.original"))
    t = {
        "config": cfg,
        "steps": len(log),
        "trainable_params": trainable,
        "lora_params": sum(v for k, v in params.items() if "lora" in k),
        "sec_per_step_stage2": round(float(np.median([r["seconds"] for r in s2])), 2),
        "hours": round(sum(r["seconds"] for r in log) / 3600, 2),
        "stage2_epochs": round(len(s2) * cfg["train"]["batch_size"] * cfg["train"]["grad_accum"] / len(SPLIT["train"]), 2),
        "peak_memory_gb": max(r.get("memory_gb", 0) for r in log),
        "val": [{k: r[k] for k in r if k == "step" or k.startswith("val_")} for r in log if "val_answer" in r],
        "train_answer_ce_last20": round(float(np.mean([r["answer"] for r in s2[-20:]])), 3),
        "train_answer_token_acc_last20": round(float(np.mean([r["answer_token_acc"] for r in s2[-20:]])), 3),
    }
    if s1:
        t["stage1_ot_first5"] = round(float(np.mean([r["ot"] for r in s1[:5]])), 3)
        t["stage1_ot_last30"] = round(float(np.mean([r["ot"] for r in s1[-30:]])), 3)
        t["stage1_rel_residual_first"] = s1[0]["relative_residual"]
        t["stage1_rel_residual_last"] = s1[-1]["relative_residual"]
    if s2 and "ot" in s2[0]:
        t["stage2_ot_last20"] = round(float(np.mean([r["ot"] for r in s2[-20:]])), 3)
        t["stage2_rel_residual_last"] = s2[-1]["relative_residual"]
        t["stage2_effective_rank_first5"] = round(float(np.mean([r["effective_rank"] for r in s2[:5]])), 2)
        t["stage2_effective_rank_last20"] = round(float(np.mean([r["effective_rank"] for r in s2[-20:]])), 2)
        t["stage2_frozen_head_first5"] = round(float(np.mean([r["frozen_head"] for r in s2[:5]])), 2)
        t["stage2_frozen_head_last20"] = round(float(np.mean([r["frozen_head"] for r in s2[-20:]])), 2)
    out["train"][name] = t

# Teacher data.
traces = os.path.join(ROOT, "traces", "DeepSeek-R1-Distill-Qwen-1.5B")
out["data"]["trace_summary"] = json.load(open(os.path.join(traces, "summary.json")))["results"]["gsm8k_train"]["think"]
out["data"]["trace_settings"] = json.load(open(os.path.join(traces, "gsm8k_train__think.settings.json")))
store = os.path.join(ROOT, "latent_data", "gsm8k_train")
meta = json.load(open(os.path.join(store, "meta.json")))
examples = read_jsonl(os.path.join(store, "examples.jsonl"))
reasoning = np.array([e["reasoning_tokens"] for e in examples]) if "reasoning_tokens" in examples[0] else None
out["data"]["store"] = {"kept": meta["kept"], "dropped": meta["dropped"], "max_states": meta["max_states"],
                        "max_reasoning_tokens": meta["max_reasoning_tokens"], "hidden_size": meta["hidden_size"],
                        "states": int(sum(e["state_count"] for e in examples)), "revision": meta["revision"]}
if reasoning is not None:
    out["data"]["store"]["reasoning_tokens"] = {"mean": round(float(reasoning.mean()), 1), "median": float(np.median(reasoning)),
                                                "p90": float(np.percentile(reasoning, 90)), "max": int(reasoning.max())}
out["eval_settings"] = {name: json.load(open(path.replace(".jsonl", ".settings.json"))) for name, path in EVALS.items()}

with open(os.path.join(ROOT, "paper_numbers.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
