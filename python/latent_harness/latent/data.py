"""Training data for the latent reasoner, built from the model's own reasoning traces.

1. Traces: `scripts/eval_baseline.py --benchmarks gsm8k_train --conditions think`
   writes one record per sampled response on a training split.
2. `select_traces` keeps one correct, finished trace per problem with a boxed
   answer and a bounded reasoning length, and counts why the others were dropped.
3. `write_teacher_store` runs the frozen backbone over each trace and keeps the
   final hidden states of its reasoning tokens, mean-pooled into at most
   `max_states` contiguous windows and stored as float16.

Pooling keeps storage bounded (128 x 1536 x 2 bytes per trace) and cheap to read
every training step. Under squared-Euclidean cost, moving mass uniformly across a
window's tokens costs the same as moving it to the window's mean plus a constant
(the window's variance), so pooling coarsens the transport plan without changing
where the latent states are pulled.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
import json
import os

import numpy as np
import torch

from ..evals.grading import THINK_CLOSE
from ..evals.prompts import LATENT_CONDITION, build_prompt
from ..evals.runner import load_records
from ..evals.tasks import EvalProblem, last_boxed
from ..frozen_head import SemanticKeywordExtractor
from .model import split_at_think_close


MATH_ANSWER_CLOSE = "}$."  # closes the direct prefix "The final answer is $\boxed{"
EXAMPLES_FILE = "examples.jsonl"
STATES_FILE = "states.f16"
META_FILE = "meta.json"


@dataclass
class TraceExample:
    id: str
    benchmark: str
    question: str
    gold: str
    reasoning: str  # text inside the think block
    answer: str  # content of the trace's final \boxed{}
    reasoning_tokens: int
    keyword_ids: List[int] = field(default_factory=list)
    state_offset: int = -1  # first row in states.f16
    state_count: int = 0

    def problem(self) -> EvalProblem:
        return EvalProblem(id=self.id, benchmark=self.benchmark, question=self.question, gold=self.gold)


# ---------------------------------------------------------------------- trace selection

def select_traces(
    records: Iterable[Dict[str, Any]], problems: Dict[str, EvalProblem], max_reasoning_tokens: int
) -> Tuple[List[TraceExample], Dict[str, int]]:
    """One usable trace per problem, plus counts of every dropped record by reason."""
    kept: Dict[str, TraceExample] = {}
    dropped: Counter = Counter()
    for r in records:
        if r["condition"] != "think":
            raise ValueError(f"Traces must come from the think condition, got {r['condition']!r}")
        response = r["response"]
        if r["problem_id"] in kept:
            reason = "duplicate"
        elif not r["correct"]:
            reason = "incorrect"
        elif not r["finished"] or r["finish_reason"] == "length":
            reason = "unfinished"
        elif response.count(THINK_CLOSE) != 1:
            reason = "malformed"
        elif r["thinking_tokens"] > max_reasoning_tokens:
            reason = "too_long"
        else:
            reasoning, answer_region = response.split(THINK_CLOSE)
            answer = last_boxed(answer_region)
            if answer is None:
                reason = "no_boxed_answer"
            else:
                problem = problems[r["problem_id"]]
                kept[problem.id] = TraceExample(
                    id=problem.id,
                    benchmark=problem.benchmark,
                    question=problem.question,
                    gold=problem.gold,
                    reasoning=reasoning.rstrip(),
                    answer=answer,
                    reasoning_tokens=r["thinking_tokens"],
                )
                continue
        dropped[reason] += 1
    return list(kept.values()), dict(dropped)


def keyword_token_ids(text: str, tokenizer: Any, top_k: int = 16, max_ids: int = 32) -> List[int]:
    """Token ids of the reasoning's most frequent content words, encoded as they appear mid-sentence."""
    ids: List[int] = []
    for word in SemanticKeywordExtractor(top_k=top_k).extract_from_text(text):
        for token in tokenizer.encode(" " + word, add_special_tokens=False):
            if token not in ids and tokenizer.decode([token]).strip():
                ids.append(token)
    return ids[:max_ids]


# ---------------------------------------------------------------------- teacher states

def pool_states(h: torch.Tensor, max_states: int) -> torch.Tensor:
    """Mean of each of `max_states` contiguous, near-equal windows (unchanged if already short enough)."""
    n = h.shape[0]
    if n <= max_states:
        return h
    edges = torch.linspace(0, n, max_states + 1).round().long()
    window = torch.bucketize(torch.arange(n), edges[1:-1], right=True).to(h.device)
    sums = torch.zeros(max_states, h.shape[1], dtype=h.dtype, device=h.device).index_add_(0, window, h)
    counts = torch.bincount(window, minlength=max_states).to(h.dtype)
    return sums / counts[:, None]


def teacher_state_fn(model: Any, tokenizer: Any, max_states: int) -> Callable[[List[TraceExample]], List[np.ndarray]]:
    """Frozen-backbone final hidden states over each trace's reasoning tokens, pooled."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @torch.no_grad()
    def compute(batch: List[TraceExample]) -> List[np.ndarray]:
        sequences, starts = [], []
        for example in batch:
            prompt = build_prompt(example.problem(), "think", tokenizer).text  # ends with "<think>\n"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            reasoning_ids = tokenizer.encode(example.reasoning, add_special_tokens=False)
            sequences.append(prompt_ids + reasoning_ids)
            starts.append(len(prompt_ids))
        width = max(len(s) for s in sequences)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, s in enumerate(sequences):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        hidden = model.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        return [
            pool_states(hidden[i, start : len(s)].float(), max_states).cpu().numpy()
            for i, (start, s) in enumerate(zip(starts, sequences))
        ]

    return compute


class TeacherStore:
    """examples.jsonl (one TraceExample per line) + states.f16 (float16 rows of width d) + meta.json."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, META_FILE), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.examples = [TraceExample(**r) for r in load_records(os.path.join(path, EXAMPLES_FILE))]
        rows = sum(e.state_count for e in self.examples)
        self._states = np.memmap(
            os.path.join(path, STATES_FILE), dtype=np.float16, mode="r", shape=(rows, self.meta["hidden_size"])
        )

    def __len__(self) -> int:
        return len(self.examples)

    def states(self, i: int) -> np.ndarray:
        e = self.examples[i]
        return np.asarray(self._states[e.state_offset : e.state_offset + e.state_count])

    def channel_stats(self, chunk: int = 1 << 16) -> Tuple[np.ndarray, np.ndarray]:
        """Per-channel mean and standard deviation over every stored state (float64 accumulation)."""
        total = np.zeros(self._states.shape[1])
        squares = np.zeros_like(total)
        for start in range(0, len(self._states), chunk):
            x = self._states[start : start + chunk].astype(np.float64)
            total += x.sum(0)
            squares += (x * x).sum(0)
        n = max(len(self._states), 1)
        mean = total / n
        std = np.sqrt(np.maximum(squares / n - mean**2, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)


def write_teacher_store(
    out_dir: str,
    examples: Sequence[TraceExample],
    compute_states: Callable[[List[TraceExample]], List[np.ndarray]],
    meta: Dict[str, Any],
    batch_size: int = 4,
) -> None:
    """Append teacher states for every example not yet stored. Interrupted runs resume where they stopped."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != meta:
            raise RuntimeError(f"{out_dir} holds states built with different settings; use a new directory")
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    examples_path = os.path.join(out_dir, EXAMPLES_FILE)
    states_path = os.path.join(out_dir, STATES_FILE)
    stored = load_records(examples_path) if os.path.exists(examples_path) else []
    with open(examples_path, "w", encoding="utf-8") as f:  # drops a torn last line, if any
        f.writelines(json.dumps(r) + "\n" for r in stored)
    rows = sum(r["state_count"] for r in stored)
    row_bytes = meta["hidden_size"] * 2
    with open(states_path, "ab") as f:  # states written after the last indexed example are discarded
        f.truncate(rows * row_bytes)

    done = {r["id"] for r in stored}
    pending = sorted((e for e in examples if e.id not in done), key=lambda e: (e.reasoning_tokens, e.id))
    with open(states_path, "ab") as states_file, open(examples_path, "a", encoding="utf-8") as index_file:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            arrays = compute_states(batch)
            for example, states in zip(batch, arrays):
                if states.shape[1] != meta["hidden_size"]:
                    raise ValueError(f"Teacher states have width {states.shape[1]}, expected {meta['hidden_size']}")
                states_file.write(states.astype(np.float16).tobytes())
                example.state_offset, example.state_count = rows, len(states)
                rows += len(states)
            states_file.flush()
            index_file.writelines(json.dumps(asdict(e)) + "\n" for e in batch)
            index_file.flush()
            print(f"[teacher] {len(done) + start + len(batch)}/{len(done) + len(pending)} traces", flush=True)


# ---------------------------------------------------------------------- training batches

class LatentTrainingSet(torch.utils.data.Dataset):
    """Each item: the latent-layout prompt split around the latents, the answer target and teacher states."""

    def __init__(self, store: TeacherStore, tokenizer: Any):
        self.store = store
        self.tokenizer = tokenizer
        self.think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        example = self.store.examples[i]
        prompt = build_prompt(example.problem(), LATENT_CONDITION, self.tokenizer)
        ids = self.tokenizer.encode(prompt.text, add_special_tokens=False)
        prefix, suffix = split_at_think_close(ids, self.think_close_id)
        target = self.tokenizer.encode(example.answer + MATH_ANSWER_CLOSE, add_special_tokens=False)
        target.append(self.tokenizer.eos_token_id)
        return {
            "id": example.id,
            "prefix": prefix,
            "suffix": suffix + target,
            "labels": [-100] * len(suffix) + target,
            "states": torch.from_numpy(self.store.states(i).astype(np.float32)),
            "keyword_ids": example.keyword_ids,
        }


def collate(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    batch = len(items)
    width = max(len(it["suffix"]) for it in items)
    suffix = torch.full((batch, width), pad_id, dtype=torch.long)
    suffix_mask = torch.zeros((batch, width), dtype=torch.long)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    n_states = max(it["states"].shape[0] for it in items)
    states = torch.zeros((batch, n_states, items[0]["states"].shape[1]))
    states_mask = torch.zeros((batch, n_states), dtype=torch.bool)
    for i, it in enumerate(items):
        n = len(it["suffix"])
        suffix[i, :n] = torch.tensor(it["suffix"])
        suffix_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(it["labels"])
        s = it["states"].shape[0]
        states[i, :s] = it["states"]
        states_mask[i, :s] = True
    return {
        "ids": [it["id"] for it in items],
        "prefix": [it["prefix"] for it in items],
        "suffix": suffix,
        "suffix_mask": suffix_mask,
        "labels": labels,
        "states": states,
        "states_mask": states_mask,
        "keyword_ids": [it["keyword_ids"] for it in items],
    }


def build_meta(model: str, revision: str, hidden_size: int, max_states: int, max_reasoning_tokens: int,
               traces: Sequence[str]) -> Dict[str, Any]:
    return {
        "model": model,
        "revision": revision,
        "hidden_size": hidden_size,
        "max_states": max_states,
        "max_reasoning_tokens": max_reasoning_tokens,
        "traces": sorted(os.path.abspath(t) for t in traces),
        "pooling": "mean over contiguous near-equal windows of final post-norm hidden states",
    }


def load_trace_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(load_records(path))
    return records
