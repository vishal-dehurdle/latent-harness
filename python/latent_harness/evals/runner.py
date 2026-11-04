"""Resumable evaluation runs and their summaries.

Each (benchmark, condition) pair writes one JSONL file with a record per
generated sample, including the full response. An interrupted run resumes
where it stopped: problems that already have all their samples are skipped.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List
import json
import os
import zlib

import numpy as np

from .backends import SamplingConfig
from .grading import THINK_CLOSE, grade
from .prompts import build_prompt
from .tasks import EvalProblem


def results_path(out_dir: str, benchmark: str, condition: str) -> str:
    return os.path.join(out_dir, f"{benchmark}__{condition}.jsonl")


def load_records(path: str) -> List[Dict[str, Any]]:
    """All complete lines of a results file. A torn last line from an interrupted write is skipped."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def _resume(path: str, n_samples: int) -> List[Dict[str, Any]]:
    """Keep only problems with all their samples, rewriting the file if anything was dropped."""
    if not os.path.exists(path):
        return []
    records = load_records(path)
    counts = Counter(r["problem_id"] for r in records)
    kept = [r for r in records if counts[r["problem_id"]] == n_samples]
    with open(path, encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    if len(kept) != line_count:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
    return kept


def _check_settings(path: str, settings: Dict[str, Any]) -> None:
    """Refuse to append to results that were produced with different settings."""
    settings_path = path[: -len(".jsonl")] + ".settings.json"
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != settings:
            raise RuntimeError(
                f"{settings_path} was written with different settings:\n  existing {existing}\n"
                f"  requested {settings}\nUse a new --out-dir for a different configuration."
            )
        return
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def run_condition(
    backend: Any,
    problems: List[EvalProblem],
    condition: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    out_dir: str,
    model: str,
    chunk_size: int = 64,
) -> List[Dict[str, Any]]:
    """Generate and grade `n_samples` responses per problem, appending to the results file."""
    benchmark = problems[0].benchmark
    path = results_path(out_dir, benchmark, condition)
    os.makedirs(out_dir, exist_ok=True)
    _check_settings(path, {
        "model": model,
        "backend": backend.name,
        "benchmark": benchmark,
        "condition": condition,
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    })

    tokenizer = backend.tokenizer
    think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    records = _resume(path, n_samples)
    done = {r["problem_id"] for r in records}
    pending = [p for p in problems if p.id not in done]

    with open(path, "a", encoding="utf-8") as f:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            prompts = [build_prompt(p, condition, tokenizer) for p in chunk]
            prompt_ids = [tokenizer.encode(pr.text, add_special_tokens=False) for pr in prompts]
            config = SamplingConfig(
                n=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=prompts[0].max_tokens or max_tokens,
                seed=zlib.crc32(f"{seed}:{benchmark}:{condition}:{chunk[0].id}".encode()),
            )
            outputs = backend.generate(prompt_ids, config)

            for problem, prompt, ids, completions in zip(chunk, prompts, prompt_ids, outputs):
                for sample, completion in enumerate(completions):
                    response = prompt.response_prefix + completion.text
                    result = grade(problem, condition, response)
                    if completion.thinking_tokens is not None:
                        thinking = completion.thinking_tokens
                    elif condition != "think":
                        thinking = 0
                    elif think_close_id in completion.token_ids:
                        thinking = completion.token_ids.index(think_close_id)
                    else:
                        thinking = len(completion.token_ids)
                    record = {
                        "problem_id": problem.id,
                        "benchmark": benchmark,
                        "condition": condition,
                        "sample": sample,
                        "seed": config.seed,
                        "gold": problem.gold,
                        "answer": result.answer,
                        "correct": result.correct,
                        "finished": result.finished,
                        "finish_reason": completion.finish_reason,
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(completion.token_ids),
                        "thinking_tokens": thinking,
                        "response": response,
                    }
                    f.write(json.dumps(record) + "\n")
                    records.append(record)
            f.flush()

            accuracy = 100.0 * np.mean([r["correct"] for r in records])
            print(
                f"[{benchmark}/{condition}] {len(records) // n_samples}/{len(problems)} problems, "
                f"running pass@1 {accuracy:.1f}%",
                flush=True,
            )
    return records


def summarize(records: List[Dict[str, Any]], n_bootstrap: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """pass@1 averaged over samples, with a bootstrap 95% CI over problems."""
    if not records:
        raise ValueError("No records to summarize")
    by_problem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_problem[r["problem_id"]].append(r)

    per_problem = np.array([np.mean([r["correct"] for r in rs]) for rs in by_problem.values()])
    rng = np.random.default_rng(seed)
    boot = per_problem[rng.integers(0, len(per_problem), size=(n_bootstrap, len(per_problem)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    tokens = np.array([r["completion_tokens"] for r in records])

    def rate(flag: str, value: Any) -> float:
        return round(100.0 * float(np.mean([r[flag] == value for r in records])), 2)

    return {
        "problems": len(per_problem),
        "samples_per_problem": sorted({len(rs) for rs in by_problem.values()}),
        "pass@1": round(100.0 * float(per_problem.mean()), 2),
        "ci95": [round(100.0 * float(low), 2), round(100.0 * float(high), 2)],
        "mean_completion_tokens": round(float(tokens.mean()), 1),
        "median_completion_tokens": float(np.median(tokens)),
        "mean_thinking_tokens": round(float(np.mean([r["thinking_tokens"] for r in records])), 1),
        "hit_max_tokens_pct": rate("finish_reason", "length"),
        "unfinished_pct": rate("finished", False),
        "no_answer_pct": rate("answer", None),
    }
