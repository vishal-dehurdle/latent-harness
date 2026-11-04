"""Unit tests for the real-benchmark evaluation harness (no model or network needed)."""

import json

import pytest

from latent_harness.evals import (
    Completion,
    EvalProblem,
    answer_region,
    build_prompt,
    extract_choice,
    grade,
    load_problems,
    results_path,
    run_condition,
    summarize,
)
from latent_harness.evals import tasks


class FakeR1Tokenizer:
    """Mimics the DeepSeek-R1-Distill chat template and the bits of the tokenizer the runner uses."""

    THINK_CLOSE_ID = 999

    def __init__(self, opens_think: bool = True):
        self.opens_think = opens_think

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and add_generation_prompt
        text = "<｜begin▁of▁sentence｜><｜User｜>" + messages[0]["content"] + "<｜Assistant｜>"
        return text + ("<think>\n" if self.opens_think else "")

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text) // 10))

    def convert_tokens_to_ids(self, token):
        assert token == "</think>"
        return self.THINK_CLOSE_ID


class FakeBackend:
    name = "fake"

    def __init__(self, text="Weighing the options.</think>\n\nAnswer: B"):
        self.tokenizer = FakeR1Tokenizer()
        self.text = text
        self.calls = 0

    def generate(self, prompt_ids, config):
        self.calls += 1
        completion = Completion(self.text, [1, 2, FakeR1Tokenizer.THINK_CLOSE_ID, 3], "stop")
        return [[completion] * config.n for _ in prompt_ids]


def math_problem(gold="72"):
    return EvalProblem(id="gsm8k_0000", benchmark="gsm8k", question="How many clips?", gold=gold)


def choice_problem(i=0, gold="B"):
    return EvalProblem(
        id=f"gpqa_diamond_{i:03d}",
        benchmark="gpqa_diamond",
        question="Which is right?",
        gold=gold,
        choices=["w", "x", "y", "z"],
    )


def run(backend, problems, out_dir, **overrides):
    settings = dict(condition="think", n_samples=2, temperature=0.6, top_p=0.95, max_tokens=64,
                    seed=0, out_dir=str(out_dir), model="fake-model", chunk_size=2)
    settings.update(overrides)
    return run_condition(backend, problems, **settings)


# ---------------------------------------------------------------- prompts

def test_think_prompt_opens_reasoning_exactly_once():
    for opens_think in (True, False):
        prompt = build_prompt(math_problem(), "think", FakeR1Tokenizer(opens_think))
        assert prompt.text.endswith("<｜Assistant｜><think>\n")
        assert prompt.text.count("<think>") == 1
        assert "Please reason step by step, and put your final answer within \\boxed{}." in prompt.text
        assert prompt.response_prefix == "" and prompt.max_tokens is None


def test_no_think_prompt_closes_an_empty_reasoning_block():
    prompt = build_prompt(math_problem(), "no_think", FakeR1Tokenizer())
    assert prompt.text.endswith("<｜Assistant｜><think>\n\n</think>\n\n")


def test_direct_prompt_forces_an_immediate_short_answer():
    math = build_prompt(math_problem(), "direct", FakeR1Tokenizer())
    assert math.text.endswith("</think>\n\nThe final answer is $\\boxed{")
    assert math.response_prefix == "The final answer is $\\boxed{" and math.max_tokens == 48

    choice = build_prompt(choice_problem(), "direct", FakeR1Tokenizer())
    assert choice.text.endswith("</think>\n\nAnswer: ") and choice.max_tokens == 4


def test_gpqa_prompt_lists_shuffled_choices():
    prompt = build_prompt(choice_problem(), "think", FakeR1Tokenizer())
    assert "A) w\nB) x\nC) y\nD) z" in prompt.text
    assert "'Answer: $LETTER'" in prompt.text


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        build_prompt(math_problem(), "telepathy", FakeR1Tokenizer())


def test_latent_prompt_uses_the_direct_layout():
    latent = build_prompt(math_problem(), "latent", FakeR1Tokenizer())
    direct = build_prompt(math_problem(), "direct", FakeR1Tokenizer())
    assert latent == direct


# ---------------------------------------------------------------- grading

def test_answer_region_requires_finished_reasoning():
    assert answer_region("still thinking \\boxed{72}", "think") is None
    assert answer_region("work</think>\n\nso \\boxed{72}", "think") == "\n\nso \\boxed{72}"
    assert answer_region("plain answer", "no_think") == "plain answer"


@pytest.mark.parametrize("text,letter", [
    ("Answer: C", "C"),
    ("**Answer:** (B)", "B"),
    ("answer: D.", "D"),
    ("Answer: $\\boxed{A}$", "A"),
    ("so it is \\boxed{\\text{C}}", "C"),
    ("Answer: B\nWait, actually Answer: D", "D"),
    ("Answer: As shown, none", None),
    ("no letter here", None),
])
def test_extract_choice(text, letter):
    assert extract_choice(text) == letter


def test_choice_grading_uses_the_letter():
    assert grade(choice_problem(gold="B"), "think", "hmm</think>Answer: B").correct
    assert not grade(choice_problem(gold="B"), "think", "hmm</think>Answer: C").correct


def test_math_grading_checks_equivalence_not_strings():
    pytest.importorskip("math_verify")
    assert grade(math_problem("0.5"), "think", "x</think>\\boxed{\\dfrac{1}{2}}").correct
    assert grade(math_problem("1000"), "no_think", "It is \\boxed{1,000}.").correct
    assert grade(math_problem("\\left( 3, \\frac{\\pi}{2} \\right)"), "no_think", "\\boxed{(3,\\frac{\\pi}{2})}").correct
    assert not grade(math_problem("72"), "think", "x</think>\\boxed{73}").correct


def test_unfinished_reasoning_scores_wrong_even_with_a_boxed_guess():
    pytest.importorskip("math_verify")
    result = grade(math_problem("72"), "think", "maybe \\boxed{72}? let me check")
    assert not result.correct and not result.finished and result.answer is None


def test_direct_answer_is_graded_with_its_forced_prefix():
    pytest.importorskip("math_verify")
    assert grade(math_problem("72"), "direct", "The final answer is $\\boxed{" + "72}$.").correct


# ---------------------------------------------------------------- datasets

GPQA_ROW = {
    "Question": " What is it? ",
    "Correct Answer": "right",
    "Incorrect Answer 1": "wrong1",
    "Incorrect Answer 2": "wrong2",
    "Incorrect Answer 3": "wrong3",
    "Record ID": "rec123",
}


def test_gpqa_shuffle_is_deterministic_and_gold_tracks_the_correct_answer():
    first = tasks._gpqa_diamond(0, GPQA_ROW)
    second = tasks._gpqa_diamond(0, dict(GPQA_ROW))
    assert first.choices == second.choices and first.gold == second.gold
    assert first.choices["ABCD".index(first.gold)] == "right"
    assert sorted(first.choices) == ["right", "wrong1", "wrong2", "wrong3"]


def test_loaders_never_fall_back_when_the_dataset_is_unavailable(monkeypatch):
    datasets = pytest.importorskip("datasets")

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(datasets, "load_dataset", unavailable)
    with pytest.raises(ConnectionError):
        load_problems("gsm8k")


def test_loaders_reject_an_unexpected_dataset_size(monkeypatch):
    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [GPQA_ROW] * 5)
    with pytest.raises(RuntimeError, match="expected 198"):
        load_problems("gpqa")


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError):
        load_problems("mmlu")


# ---------------------------------------------------------------- runner

def test_run_writes_every_sample_with_token_accounting(tmp_path):
    records = run(FakeBackend(), [choice_problem(0), choice_problem(1)], tmp_path)
    assert len(records) == 4
    assert all(r["correct"] and r["finished"] for r in records)
    assert all(r["thinking_tokens"] == 2 and r["completion_tokens"] == 4 for r in records)
    with open(results_path(str(tmp_path), "gpqa_diamond", "think")) as f:
        assert len(f.readlines()) == 4


def test_run_resumes_without_regenerating_finished_problems(tmp_path):
    problems = [choice_problem(0), choice_problem(1)]
    run(FakeBackend(), problems, tmp_path)
    backend = FakeBackend()
    records = run(backend, problems, tmp_path)
    assert backend.calls == 0 and len(records) == 4


def test_run_drops_partial_problems_and_torn_lines_then_regenerates(tmp_path):
    problems = [choice_problem(0), choice_problem(1), choice_problem(2)]
    run(FakeBackend(), problems[:2], tmp_path)
    path = results_path(str(tmp_path), "gpqa_diamond", "think")
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:3])  # problem 1 now has only one of its two samples
        f.write('{"problem_id": "gpqa_dia')  # torn write

    records = run(FakeBackend(), problems, tmp_path)
    ids = sorted(r["problem_id"] for r in records)
    assert ids == sorted([p.id for p in problems for _ in range(2)])
    with open(path) as f:
        assert [json.loads(line)["problem_id"] for line in f].count("gpqa_diamond_001") == 2


def test_run_refuses_to_mix_settings_in_one_results_file(tmp_path):
    run(FakeBackend(), [choice_problem(0)], tmp_path)
    with pytest.raises(RuntimeError, match="different settings"):
        run(FakeBackend(), [choice_problem(0)], tmp_path, max_tokens=128)


def test_summarize_averages_samples_within_problems():
    def record(pid, correct, tokens):
        return {"problem_id": pid, "correct": correct, "completion_tokens": tokens, "thinking_tokens": tokens - 1,
                "finish_reason": "stop", "finished": True, "answer": "x"}

    records = [record("a", True, 10), record("a", False, 20), record("b", True, 30), record("b", True, 40)]
    s = summarize(records)
    assert s["problems"] == 2 and s["samples_per_problem"] == [2]
    assert s["pass@1"] == 75.0
    assert 50.0 <= s["ci95"][0] <= 75.0 <= s["ci95"][1] <= 100.0
    assert s["mean_completion_tokens"] == 25.0
    assert s["hit_max_tokens_pct"] == 0.0 and s["no_answer_pct"] == 0.0
