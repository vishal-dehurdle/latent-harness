"""Paired comparison of eval runs on the same problems.

  python scripts/analysis/compare_runs.py NAME=path/to/gsm8k__latent.jsonl NAME=...

Prints each run's accuracy, then for every pair the paired difference with a bootstrap 95% CI and an exact
McNemar p-value (two-sided binomial test on the problems where the two runs disagree).
"""

import json
import sys
from itertools import combinations
from math import comb
from typing import Any, Dict

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def paired(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    """x - y on the same problems: difference, bootstrap 95% CI, discordant counts, exact McNemar p."""
    d = x.astype(float) - y.astype(float)
    boot = d[rng.integers(len(d), size=(10_000, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b, c = int((x & ~y).sum()), int((y & ~x).sum())
    return {"n": len(d), "diff": round(100 * d.mean(), 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
            "x_only": b, "y_only": c, "p": round(mcnemar_exact(b, c), 4)}


def main() -> None:
    runs = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        with open(path, encoding="utf-8") as f:
            runs[name] = {r["problem_id"]: bool(r["correct"]) for r in map(json.loads, f)}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    X = {name: np.array([r[i] for i in ids]) for name, r in runs.items()}
    print(f"{len(ids)} problems in common\n")
    for name, x in X.items():
        print(f"{name:12s} {100 * x.mean():6.2f}%")

    rng = np.random.default_rng(0)
    print(f"\n{'A - B':26s} {'diff':>7s} {'95% CI':>17s} {'A only':>7s} {'B only':>7s} {'p':>7s}")
    for a, b in combinations(X, 2):
        r = paired(X[a], X[b], rng)
        lo, hi = r["ci95"]
        print(f"{a + ' - ' + b:26s} {r['diff']:+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {r['x_only']:7d} {r['y_only']:7d} {r['p']:7.3f}")


if __name__ == "__main__":
    main()
