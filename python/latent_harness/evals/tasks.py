"""Benchmark loaders for baseline and latent-model evaluation.

Every loader reads the official dataset at a pinned revision and raises on any
failure. There are deliberately no offline or curated fallbacks: a run that
cannot load the real data must stop instead of producing numbers.

The training splits (TRAIN_SETS) are only used to generate reasoning traces for
latent-model training. They are disjoint from every test set: MATH-500 is drawn
from the MATH test split.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import random


BENCHMARKS = ("gsm8k", "math500", "aime2024", "gpqa_diamond")
TRAIN_SETS = ("gsm8k_train", "math_train")

ALIASES = {
    "aime": "aime2024",
    "math": "math500",
    "math_500": "math500",
    "gpqa": "gpqa_diamond",
}

# Revisions pinned so every run grades against identical data.
DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "size": 1319,
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "size": 500,
    },
    "aime2024": {
        "path": "HuggingFaceH4/aime_2024",
        "name": None,
        "split": "train",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "size": 30,
    },
    # Gated: accept the terms at huggingface.co/datasets/Idavidrein/gpqa, then `hf auth login`.
    "gpqa_diamond": {
        "path": "Idavidrein/gpqa",
        "name": "gpqa_diamond",
        "split": "train",
        "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6",
        "size": 198,
    },
    "gsm8k_train": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "size": 7473,
    },
    "math_train": {
        "path": "DigitalLearningGmbH/MATH-lighteval",
        "name": "default",
        "split": "train",
        "revision": "0530c78699ea5e8eb5530600900e1f328b48acad",
        "size": 7500,
    },
}

# GPQA ships the correct answer first; options are shuffled per question with this seed.
GPQA_SHUFFLE_SEED = 0


@dataclass
class EvalProblem:
    id: str
    benchmark: str
    question: str
    gold: str
    choices: Optional[List[str]] = None  # GPQA only, already shuffled; gold is then the letter
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return self.choices is not None


def canonical_benchmark(name: str) -> str:
    key = name.lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown benchmark {name!r}; expected one of {', '.join(DATASET_SPECS)}")
    return key


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} in `text`, with nested braces handled; None if there is none."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth:
        return None
    content = text[i : j - 1].strip()
    return content or None


def _gsm8k(i: int, row: Dict[str, Any], benchmark: str = "gsm8k") -> EvalProblem:
    solution = row["answer"]
    if "####" not in solution:
        raise ValueError(f"{benchmark} row {i} has no '####' answer marker")
    gold = solution.split("####")[-1].strip().replace(",", "")
    return EvalProblem(id=f"{benchmark}_{i:04d}", benchmark=benchmark, question=row["question"], gold=gold)


def _math500(i: int, row: Dict[str, Any]) -> EvalProblem:
    return EvalProblem(
        id=f"math500_{i:03d}",
        benchmark="math500",
        question=row["problem"],
        gold=row["answer"],
        metadata={"unique_id": row["unique_id"], "subject": row["subject"], "level": str(row["level"])},
    )


def _aime2024(i: int, row: Dict[str, Any]) -> EvalProblem:
    return EvalProblem(
        id=f"aime2024_{i:02d}",
        benchmark="aime2024",
        question=row["problem"],
        gold=str(row["answer"]),
        metadata={"source_id": str(row["id"]), "url": row["url"]},
    )


def _gpqa_diamond(i: int, row: Dict[str, Any]) -> EvalProblem:
    options = [row["Correct Answer"].strip()] + [
        row[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)
    ]
    order = list(range(4))
    random.Random(f"{GPQA_SHUFFLE_SEED}:{row['Record ID']}").shuffle(order)
    return EvalProblem(
        id=f"gpqa_diamond_{i:03d}",
        benchmark="gpqa_diamond",
        question=row["Question"].strip(),
        gold="ABCD"[order.index(0)],
        choices=[options[j] for j in order],
        metadata={"record_id": row["Record ID"]},
    )


def _math_train(i: int, row: Dict[str, Any]) -> Optional[EvalProblem]:
    # A handful of reference solutions write `\boxed 2` or an empty box; those rows are skipped.
    gold = last_boxed(row["solution"])
    if gold is None:
        return None
    return EvalProblem(
        id=f"math_train_{i:04d}",
        benchmark="math_train",
        question=row["problem"],
        gold=gold,
        metadata={"type": row["type"], "level": row["level"]},
    )


_ROW_PARSERS: Dict[str, Callable[[int, Dict[str, Any]], Optional[EvalProblem]]] = {
    "gsm8k": _gsm8k,
    "math500": _math500,
    "aime2024": _aime2024,
    "gpqa_diamond": _gpqa_diamond,
    "gsm8k_train": lambda i, row: _gsm8k(i, row, "gsm8k_train"),
    "math_train": _math_train,
}


def load_problems(benchmark: str, limit: Optional[int] = None) -> List[EvalProblem]:
    """Load a benchmark's full split (or its first `limit` problems, for smoke tests).

    Only training splits may skip rows (those without a usable reference answer);
    a test-set row that cannot be parsed raises.
    """
    from datasets import load_dataset

    key = canonical_benchmark(benchmark)
    spec = DATASET_SPECS[key]
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"], revision=spec["revision"])
    if len(ds) != spec["size"]:
        raise RuntimeError(
            f"{key}: expected {spec['size']} problems at revision {spec['revision'][:8]}, got {len(ds)}"
        )
    count = len(ds) if limit is None else min(limit, len(ds))
    problems = [_ROW_PARSERS[key](i, ds[i]) for i in range(count)]
    if key not in TRAIN_SETS and any(p is None for p in problems):
        raise RuntimeError(f"{key}: a test-set row has no usable answer")
    return [p for p in problems if p is not None]
