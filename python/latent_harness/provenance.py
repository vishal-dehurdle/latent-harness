"""What produced a result: timestamps, git state, model revisions and package versions.

Every script that writes numbers or checkpoints records these in its manifest.
"""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict
import os
import platform
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "transformers", "datasets", "safetensors", "math-verify", "mlx", "mlx-lm", "vllm")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> Dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def model_sha(model: str, revision: str) -> str:
    """The Hub commit a model revision resolves to (so "main" is pinned after the fact)."""
    if os.path.isdir(model):
        return "local directory"
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model, revision=revision).sha or "unknown"
    except Exception as e:
        return f"unresolved: {e}"


def package_versions() -> Dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def host() -> Dict[str, str]:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
