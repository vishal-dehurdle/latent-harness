"""Real benchmark evaluation: official datasets, DeepSeek-R1 prompts, math-verify grading.

The same harness scores the DeepSeek baseline (Phase 0) and, later, the latent
model, so the two are always measured identically.
"""

from .tasks import BENCHMARKS, DATASET_SPECS, TRAIN_SETS, EvalProblem, canonical_benchmark, last_boxed, load_problems
from .prompts import ALL_CONDITIONS, CONDITIONS, LATENT_CONDITION, Prompt, build_prompt
from .grading import Grade, answer_region, extract_choice, grade
from .backends import Completion, SamplingConfig, load_backend
from .runner import load_records, results_path, run_condition, summarize

__all__ = [
    "BENCHMARKS",
    "DATASET_SPECS",
    "TRAIN_SETS",
    "EvalProblem",
    "canonical_benchmark",
    "last_boxed",
    "load_problems",
    "ALL_CONDITIONS",
    "CONDITIONS",
    "LATENT_CONDITION",
    "Prompt",
    "build_prompt",
    "Grade",
    "answer_region",
    "extract_choice",
    "grade",
    "Completion",
    "SamplingConfig",
    "load_backend",
    "load_records",
    "results_path",
    "run_condition",
    "summarize",
]
