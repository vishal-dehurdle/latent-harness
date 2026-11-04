# latent-harness

A controlled test bed for latent-token reasoning on DeepSeek-R1-Distill models.

It trains $K$ continuous "pause" tokens, placed where the model's reasoning would go and refined by a looped pass over the backbone (LOTUS), and aligns them to the model's own reasoning with Sinkhorn optimal transport. Every latent model is scored with the same harness as the model's own baselines and compared, problem by problem, with a **$K=0$ control** trained identically but with no latent tokens. That control separates what the latents add from what the fine-tuning adds.

The code accompanies the paper *[Sinkhorn-Aligned Latent Tokens Add Nothing Over Fine-Tuning](https://vishalvermalabs.com/papers/empirical-validation-latent-reasoning-sinkhorn/)*. Its result is negative.

## Result

DeepSeek-R1-Distill-Qwen-1.5B on all 1,319 GSM8K test problems, one sample at temperature 0.6:

| Condition | Accuracy | 95% CI |
|---|---|---|
| Untrained, answering directly | 11.22% | 9.55–12.96 |
| Fine-tuned control, $K=0$ | **19.79%** | 17.66–21.99 |
| $K=4$ latent tokens | 18.42% | 16.38–20.55 |
| $K=4$, standardised OT targets (`--ot-standardize --ot-epsilon 0.03`) | 19.56% | 17.44–21.68 |
| Full reasoning (first 448 problems) | 86.16% | 82.81–89.29 |

The latents add nothing over the control. $K=4$ minus $K=0$ is −1.36 points (exact McNemar $p = 0.12$); with standardised targets it is −0.23 ($p = 0.86$).

A diagnosis (`scripts/analysis/collapse_check.py`) shows why:
- **The four latents collapse into one** (cosine 0.999 within a problem).
- **They mostly re-encode the question.** They identify their own problem 98.8% of the time, but the question's final hidden state alone does so 96.1% of the time.
- **The latent pathway can't produce the targets.** In both runs the latents never reached even the alignment loss of four identical latents.

The paper has the full analysis and limitations.

## What's in it

| Path | What it does |
|---|---|
| `latent_harness.evals`, `scripts/eval_baseline.py` | Benchmark evaluation: GSM8K, MATH-500, AIME 2024 and GPQA-Diamond under `think`, `no_think` and `direct` conditions. It has MLX, vLLM and latent-model backends, bootstrap CIs and a manifest per run, and interrupted runs resume. |
| `latent_harness.latent`, `scripts/build_latent_data.py`, `scripts/train_latent.py` | The latent model and its two-stage training on the model's own reasoning traces. |
| `latent_harness.sinkhorn`, `latent_harness.frozen_head` | The Sinkhorn alignment loss (batched, log-domain, masked) and the frozen-head keyword loss. |
| `scripts/analysis/` | The collapse diagnostic, simulations of the alignment loss, paired comparison of eval runs (bootstrap CI and exact McNemar test), and `paper_numbers.py`, which writes every number the paper cites from saved runs. |

## Install

```bash
git clone https://github.com/vishal-dehurdle/latent-harness.git
cd latent-harness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"    # eval: math-verify, plus mlx-lm on Apple Silicon
```

Requires Python 3.10+ and transformers 5+. Everything in the paper ran on an Apple M4 with 16 GB, using Python 3.13, PyTorch 2.14 and transformers 5.17. The CUDA path (`.[eval-cuda]`, vLLM for baselines) is implemented but untested.

## Evaluate a model

`scripts/eval_baseline.py` measures DeepSeek-R1-Distill models on the official test sets with DeepSeek's published protocol: no system prompt, temperature 0.6, top-p 0.95, and generation starting inside `<think>`.

- **Data:** GSM8K (1,319), MATH-500 (500), AIME 2024 (30) and GPQA-Diamond (198), loaded at pinned revisions. There are no fallbacks: if a dataset can't be loaded, the run stops before the model loads.
- **Grading:** `math-verify` symbolic equivalence for math, and the option letter for GPQA. A response that never closes `</think>` counts as unanswered.
- **Conditions:**
  - `think`: full reasoning.
  - `no_think`: an empty reasoning block.
  - `direct`: the answer is forced immediately, the floor that latent tokens must beat.
- **Outputs:**
  - One JSONL record per sample, with the full response and token counts.
  - `summary.json`, with pass@1 and bootstrap 95% CIs.
  - A `manifest-<timestamp>.json` per invocation, recording the model SHA, dataset revisions, settings, git commit and package versions.
- **Resuming:** interrupted runs resume where they stopped. Run under `caffeinate -i` so a Mac doesn't sleep.

```bash
# GPQA-Diamond is gated: accept the terms at huggingface.co/datasets/Idavidrein/gpqa, then
hf auth login

# Smoke test (marked partial; not reportable)
python scripts/eval_baseline.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --benchmarks gsm8k math500 --limit 8 --max-tokens 8192

# Full baseline
python scripts/eval_baseline.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Rebuild the summary from saved generations
python scripts/eval_baseline.py --summarize-only --out-dir outputs/baseline/DeepSeek-R1-Distill-Qwen-1.5B
```

On Apple Silicon the default backend is MLX, and `--batch-size` sets how many sequences decode at once. On a CUDA GPU the backend is vLLM.

## Train a latent model

- **Layout:** the same as the `direct` condition, with $K$ latent positions inside the empty reasoning block: `<think>\n\n [z_1 .. z_K] </think>\n\nThe final answer is $\boxed{`. With $K=0$ it is exactly the no-reasoning floor.
- **LOTUS loop:** $h^{(t)} = f(E + g(h^{(t-1)}) \mid C_{\text{pre}})$ for $R$ iterations.
  - $f$ is the whole backbone (frozen, plus LoRA) attending to the prompt's KV cache.
  - $E$ are learnable pause embeddings.
  - $g$ is a spectrally normalised feedback map damped by $\gamma = 0.9$.
- **Losses:** stage 1 aligns $Z = h^{(R)}$ to the teacher's pooled reasoning states with Sinkhorn OT. Stage 2 adds answer cross-entropy and the frozen-head keyword loss, computed through the model's own LM head.
- **Controls:** every result needs the $K=0$ run trained the same way. Ablations are flags: `--no-spectral-norm`, `--ot-weight 0 --stage1-steps 0`, and `--frozen-head-weight 0`.

```bash
# 1. Reasoning traces on a training split (disjoint from every test set)
python scripts/eval_baseline.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --benchmarks gsm8k_train --conditions think --max-tokens 4096 --batch-size 16 --limit 2500 \
    --out-dir outputs/traces/DeepSeek-R1-Distill-Qwen-1.5B

# 2. Keep correct traces; store the frozen model's pooled hidden states over their reasoning
python scripts/build_latent_data.py \
    --traces outputs/traces/DeepSeek-R1-Distill-Qwen-1.5B/gsm8k_train__think.jsonl \
    --out-dir outputs/latent_data/gsm8k_train

# 3. Train K=4 and the K=0 control (each resumes from <out-dir>/checkpoint if interrupted)
python scripts/train_latent.py --data-dir outputs/latent_data/gsm8k_train --out-dir outputs/latent/k4 \
    --num-latents 4 --num-iterations 4 --stage1-steps 100 --stage2-steps 400
python scripts/train_latent.py --data-dir outputs/latent_data/gsm8k_train --out-dir outputs/latent/k0 \
    --num-latents 0 --stage1-steps 0 --ot-weight 0 --frozen-head-weight 0 --stage2-steps 400

# 4. Evaluate with the same harness and grader as the baselines
python scripts/eval_baseline.py --backend latent --model outputs/latent/k4/final --benchmarks gsm8k --batch-size 16

# 5. Compare runs problem by problem, and diagnose the latents
python scripts/analysis/compare_runs.py K0=outputs/latent/k0/final/eval/gsm8k__latent.jsonl \
    K4=outputs/latent/k4/final/eval/gsm8k__latent.jsonl
python scripts/analysis/collapse_check.py outputs/latent/k4/final outputs/latent_data/gsm8k_train
```

**On a 16 GB Mac:**
- A $K=4$ run takes about 1.3 hours: stage-2 steps take about 9.5 s, with 3.6 GB of live tensors and 8–10.5 GB in total.
- `train_latent.py` caps the MPS allocator at the GPU's recommended working set, so an overrun fails instead of swapping for hours.
- The trainer releases cached blocks after every step.

## Reproduce the paper

The paper's runs are the commands above. The run with standardised targets is the $K=4$ command plus `--ot-standardize --ot-epsilon 0.03`. The full-reasoning baseline is `eval_baseline.py --benchmarks gsm8k --conditions think --max-tokens 16384`.

With the runs in `outputs/`, these regenerate every number the paper cites, plus the loss simulations:

```bash
python scripts/analysis/paper_numbers.py        # -> outputs/paper_numbers.json
python scripts/analysis/ot_gap.py outputs/latent_data/gsm8k_train
python scripts/analysis/ot_attractor.py outputs/latent_data/gsm8k_train
```

Run outputs are not in the repository. Each run directory records its exact command, git state, package versions and host in a manifest.

## Tests

```bash
pytest
```

The tests cover:
- The Sinkhorn solver: exact marginals, log-domain stability, masking and gradients.
- The frozen-head keyword loss and effective rank.
- The evaluation harness: prompts, grading, strict dataset loading, resumable runs and summaries.
- The latent model on a tiny random Qwen2:
  - The cached, batched LOTUS loop matches a from-scratch recompute, and $K=0$ reproduces the direct condition.
  - Gradients flow, checkpoints save and reload, and data loading and training resume.
  - Standardised OT ignores what every state shares.

## Citation

```bibtex
@misc{verma2026latenttokens,
  author = {Verma, Vishal},
  title  = {Sinkhorn-Aligned Latent Tokens Add Nothing Over Fine-Tuning: A Controlled Negative Result on DeepSeek-R1-Distill-Qwen-1.5B},
  year   = {2026},
  url    = {https://vishalvermalabs.com/papers/empirical-validation-latent-reasoning-sinkhorn/}
}
```

## License

MIT. See [LICENSE](LICENSE).
