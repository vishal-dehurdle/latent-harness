#!/usr/bin/env python3
"""Measure DeepSeek-R1-Distill baselines and trained latent models on GSM8K, MATH-500, AIME 2024 and GPQA-Diamond.

Every number comes from actual generations, graded by math-verify (math) or
option letter (GPQA), and every generation is saved, so results can be audited.
Run on a training split (gsm8k_train, math_train) with the think condition, it
produces the reasoning traces that latent-model training learns from.

Conditions:
  think     full reasoning (DeepSeek's recommended setup)
  no_think  empty reasoning block, free-form answer
  direct    empty reasoning block, answer forced immediately; the floor that
            K latent tokens have to beat
  latent    the direct layout with K latent positions before </think>
            (--backend latent --model <checkpoint dir> only)

Examples:
  # Local smoke test on Apple Silicon
  python scripts/eval_baseline.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
      --benchmarks gsm8k math500 --limit 2 --max-tokens 1024

  # A trained latent model, graded exactly like the direct floor
  python scripts/eval_baseline.py --backend latent --model outputs/latent/k4/final --benchmarks gsm8k

  # Recompute summary.json from saved generations
  python scripts/eval_baseline.py --summarize-only \\
      --out-dir outputs/baseline/DeepSeek-R1-Distill-Qwen-1.5B
"""

from pathlib import Path
import argparse
import glob
import json
import os
import platform
import sys

from latent_harness import provenance
from latent_harness.evals import (
    ALL_CONDITIONS,
    BENCHMARKS,
    CONDITIONS,
    DATASET_SPECS,
    LATENT_CONDITION,
    canonical_benchmark,
    load_backend,
    load_problems,
    load_records,
    results_path,
    run_condition,
    summarize,
)


REPO_ROOT = provenance.REPO_ROOT

# Samples per problem for pass@1; smaller test sets get more samples to reduce noise.
# Training splits need one trace per problem.
DEFAULT_SAMPLES = {"gsm8k": 1, "math500": 4, "aime2024": 16, "gpqa_diamond": 4, "gsm8k_train": 1, "math_train": 1}


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _print_table(summary: dict, partial: bool) -> None:
    title = "BASELINE RESULTS" + ("  (PARTIAL RUN: --limit was set, not reportable)" if partial else "")
    print("\n" + title)
    print(f"{'BENCHMARK':<14}{'CONDITION':<10}{'N':>5}{'PASS@1':>9}{'95% CI':>17}{'MEAN TOK':>10}{'HIT MAX':>9}")
    for benchmark, by_condition in summary.items():
        for condition, s in by_condition.items():
            ci = f"[{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}]"
            print(
                f"{benchmark:<14}{condition:<10}{s['problems']:>5}{s['pass@1']:>8.2f}%{ci:>17}"
                f"{s['mean_completion_tokens']:>10.0f}{s['hit_max_tokens_pct']:>8.1f}%"
            )


def _parse_samples(values: list, benchmarks: list) -> dict:
    """`--samples 1 aime2024=4`: a bare number applies to every benchmark, BENCHMARK=N overrides one."""
    overall = [int(v) for v in values if "=" not in v]
    samples = {b: overall[-1] if overall else DEFAULT_SAMPLES[b] for b in benchmarks}
    for value in values:
        if "=" in value:
            name, n = value.split("=", 1)
            key = canonical_benchmark(name)
            if key in samples:
                samples[key] = int(n)
    return samples


def _summarize_dir(out_dir: str) -> dict:
    summary: dict = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*__*.jsonl"))):
        benchmark, condition = os.path.basename(path)[: -len(".jsonl")].split("__")
        records = load_records(path)
        if records:
            summary.setdefault(benchmark, {})[condition] = summarize(records)
    return summary


def _write_summary(out_dir: str) -> tuple:
    """Summarize every results file in the directory; partial if any invocation used --limit."""
    summary = _summarize_dir(out_dir)
    partial = False
    for path in glob.glob(os.path.join(out_dir, "manifest-*.json")):
        with open(path, encoding="utf-8") as f:
            partial = partial or bool(json.load(f).get("partial"))
    _write_json(os.path.join(out_dir, "summary.json"), {"partial": partial, "results": summary})
    return summary, partial


def main() -> None:
    on_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Hub id or local path; with --backend latent, a training checkpoint directory")
    parser.add_argument("--revision", default="main", help="Model revision on the Hugging Face Hub")
    parser.add_argument("--backend", choices=["mlx", "vllm", "latent"], default="mlx" if on_apple_silicon else "vllm")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS,
                        help=f"Default: {' '.join(CONDITIONS)}; with --backend latent, {LATENT_CONDITION}")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[],
        metavar="N|BENCHMARK=N",
        help=f"Samples per problem: N for every benchmark, BENCHMARK=N for one (default {DEFAULT_SAMPLES})",
    )
    parser.add_argument("--max-tokens", type=int, default=32768, help="Generation budget (DeepSeek used 32768)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Only the first N problems per benchmark (smoke tests; not reportable)")
    parser.add_argument("--chunk-size", type=int, default=64, help="Problems per generate call; results are saved after each")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="MLX and latent: sequences decoded concurrently (bounded by memory)")
    parser.add_argument("--device", default="auto", help="Latent backend only: mps, cuda or cpu")
    parser.add_argument("--max-model-len", type=int, help="vLLM: context length (default max-tokens + 2048)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92, help="vLLM only")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    parser.add_argument("--dtype", default="auto", help="vLLM only; use float16 on GPUs without bfloat16")
    parser.add_argument("--out-dir", help="Default: outputs/baseline/<model name>")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summary.json from saved results")
    args = parser.parse_args()

    latent = args.backend == "latent"
    if args.out_dir:
        out_dir = args.out_dir
    elif latent:
        out_dir = os.path.join(args.model, "eval")
    else:
        out_dir = str(REPO_ROOT / "outputs" / "baseline" / Path(args.model.rstrip("/")).name)
    if args.summarize_only:
        _print_table(*_write_summary(out_dir))
        return

    conditions = args.conditions or ([LATENT_CONDITION] if latent else list(CONDITIONS))
    if latent != (conditions == [LATENT_CONDITION]):
        parser.error(f"The {LATENT_CONDITION} condition needs --backend latent, which supports only that condition")
    benchmarks = [canonical_benchmark(b) for b in args.benchmarks]
    os.makedirs(out_dir, exist_ok=True)

    # Load every dataset before the model, so a missing or gated dataset fails in seconds, not hours in.
    problems = {b: load_problems(b, limit=args.limit) for b in benchmarks}
    samples = _parse_samples(args.samples, benchmarks)

    # One manifest per invocation, so resumed runs keep the history of every session.
    started_at = provenance.now()
    manifest_path = os.path.join(out_dir, f"manifest-{started_at.replace(':', '')}.json")
    manifest = {
        "started_at": started_at,
        "command": sys.argv,
        "partial": args.limit is not None,
        "model": args.model,
        "model_revision": args.revision,
        "model_sha": "see backend.latent_config" if latent else provenance.model_sha(args.model, args.revision),
        "datasets": {b: DATASET_SPECS[b] for b in benchmarks},
        "samples_per_problem": samples,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": args.seed},
        "git": provenance.git_state(),
        "packages": provenance.package_versions(),
        "host": provenance.host(),
    }
    _write_json(manifest_path, manifest)

    if latent:
        backend = load_backend("latent", args.model, device=args.device, batch_size=args.batch_size)
    elif args.backend == "mlx":
        backend = load_backend("mlx", args.model, revision=args.revision, batch_size=args.batch_size)
    else:
        backend = load_backend(
            "vllm",
            args.model,
            revision=args.revision,
            max_model_len=args.max_model_len or args.max_tokens + 2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
        )
    manifest["backend"] = backend.describe()
    _write_json(manifest_path, manifest)

    for benchmark in benchmarks:
        for condition in conditions:
            run_condition(
                backend,
                problems[benchmark],
                condition,
                n_samples=samples[benchmark],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                out_dir=out_dir,
                model=args.model,
                chunk_size=args.chunk_size,
            )
            _write_summary(out_dir)
            print(f"  -> {results_path(out_dir, benchmark, condition)}", flush=True)

    manifest["finished_at"] = provenance.now()
    _write_json(manifest_path, manifest)
    _print_table(*_write_summary(out_dir))


if __name__ == "__main__":
    main()
