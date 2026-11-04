"""Tests for the Phase 1 latent reasoner on a tiny random Qwen2 (CPU, no downloads)."""

import json
import os
import re

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from latent_harness.evals import EvalProblem, build_prompt, last_boxed, run_condition
from latent_harness.evals import tasks
from latent_harness.latent import (
    LatentConfig,
    LatentReasoner,
    LatentTrainingSet,
    TeacherStore,
    TraceExample,
    TrainConfig,
    collate,
    keyword_token_ids,
    pool_states,
    select_traces,
    split_at_think_close,
    train,
    write_teacher_store,
)
from latent_harness.latent.backend import LatentBackend
from latent_harness.latent.model import sample_next
from latent_harness.latent.train import answer_loss, ot_loss, split_indices


class CharTokenizer:
    """ASCII characters as ids 0-127, plus the R1 special tokens the harness relies on."""

    SPECIALS = {"<bos>": 128, "<eos>": 129, "<think>": 130, "</think>": 131, "<|User|>": 132, "<|Assistant|>": 133}
    eos_token_id = 129
    pad_token_id = 129
    _split = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<bos><|User|>" + messages[0]["content"] + "<|Assistant|><think>\n"

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in self._split.split(text):
            if part in self.SPECIALS:
                ids.append(self.SPECIALS[part])
            else:
                ids.extend(ord(c) if ord(c) < 128 else ord("?") for c in part)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        names = {v: k for k, v in self.SPECIALS.items()}
        return "".join(("" if skip_special_tokens else names[i]) if i in names else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.SPECIALS[token]


VOCAB, HIDDEN = 140, 32
TOK = CharTokenizer()


def tiny_base(seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False,
        bos_token_id=128, eos_token_id=129, pad_token_id=129,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def reasoner(k=3, r=3, lora_rank=4, seed=0, perturb_lora=True, **overrides):
    model = LatentReasoner(LatentConfig(model="tiny", num_latents=k, num_iterations=r, lora_rank=lora_rank, **overrides),
                           tiny_base(seed))
    if perturb_lora:  # LoRA B starts at zero; make the adapters do something
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.05)
    return model


def math_problem(i=0, question="What is 2+3?", gold="5"):
    return EvalProblem(id=f"gsm8k_train_{i:04d}", benchmark="gsm8k_train", question=question, gold=gold)


def latent_prompt_ids(problem):
    return TOK.encode(build_prompt(problem, "latent", TOK).text)


# ---------------------------------------------------------------------- the LOTUS loop

def reference_pass(model, prompt_ids):
    """No cache and no padding: recompute the whole sequence at every iteration."""
    prefix, suffix = split_at_think_close(prompt_ids, TOK.SPECIALS["</think>"])
    embed = model.base.get_input_embeddings()
    p = embed(torch.tensor([prefix]))
    k = model.config.num_latents
    h, x, trajectory = None, None, []
    for _ in range(model.config.num_iterations):
        x = model.latent_embeddings[None] if h is None else model.latent_embeddings[None] + model.feed_back(h)
        h = model.decoder(inputs_embeds=torch.cat([p, x], 1)).last_hidden_state[:, -k:]
        trajectory.append(h)
    s = embed(torch.tensor([suffix]))
    logits = model.base(inputs_embeds=torch.cat([p, x, s], 1)).logits[:, -len(suffix):]
    return trajectory, logits


def test_cached_batched_loop_matches_a_from_scratch_recompute():
    model = reasoner()
    prompts = [latent_prompt_ids(math_problem(0)), latent_prompt_ids(math_problem(1, "Add 10 and 32 then halve it."))]
    assert len(prompts[0]) != len(prompts[1])  # exercises left padding

    parts = [split_at_think_close(p, TOK.SPECIALS["</think>"]) for p in prompts]
    cache, mask, lengths = model.prefill([p for p, _ in parts], TOK.pad_token_id)
    state = model.think(cache, mask, lengths)
    suffix = torch.tensor([s for _, s in parts])
    logits, *_ = model.extend(state.cache, state.attention_mask, state.lengths, suffix, torch.ones_like(suffix))

    for i, prompt in enumerate(prompts):
        ref_traj, ref_logits = reference_pass(model, prompt)
        for t, ref_h in enumerate(ref_traj):
            torch.testing.assert_close(state.trajectory[t][i : i + 1], ref_h, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(logits[i : i + 1], ref_logits, atol=1e-4, rtol=1e-4)


def test_gradients_reach_every_trainable_part_and_nothing_else():
    model = reasoner().train()
    prompt = latent_prompt_ids(math_problem())
    prefix, suffix = split_at_think_close(prompt, TOK.SPECIALS["</think>"])
    cache, mask, lengths = model.prefill([prefix], TOK.pad_token_id)
    state = model.think(cache, mask, lengths)
    ids = torch.tensor([suffix + [ord("5"), TOK.eos_token_id]])
    labels = torch.tensor([[-100] * len(suffix) + [ord("5"), TOK.eos_token_id]])
    logits, *_ = model.extend(state.cache, state.attention_mask, state.lengths, ids, torch.ones_like(ids))
    answer_loss(logits, labels)["answer"].backward()

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_A" in n for n in trainable) and "latent_embeddings" in trainable
    assert any(n.startswith("feedback.") for n in trainable)
    for name, p in trainable.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    frozen = [p for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(p.grad is None for p in frozen)


def test_spectral_norm_keeps_the_feedback_map_non_expansive():
    model = reasoner().train()
    optimizer = torch.optim.SGD(model.feedback.parameters(), lr=10.0)
    x = torch.randn(64, HIDDEN)
    for _ in range(20):
        optimizer.zero_grad()
        (-model.feedback(x).pow(2).sum()).backward()  # pushes the weights to grow
        optimizer.step()
    for _ in range(20):  # let the power iteration settle
        model.feedback(x)
    assert torch.linalg.matrix_norm(model.feedback.weight.detach(), ord=2) <= 1.0 + 1e-3


def test_without_latents_the_model_is_exactly_the_direct_condition():
    model = reasoner(k=0, perturb_lora=False)  # LoRA B = 0, so the backbone is untouched
    prompt = TOK.encode(build_prompt(math_problem(), "direct", TOK).text)
    assert prompt == latent_prompt_ids(math_problem())
    generated, _ = model.generate([prompt], TOK.SPECIALS["</think>"], TOK.eos_token_id, TOK.pad_token_id,
                                  max_tokens=6, temperature=0, top_p=1.0, generator=torch.Generator())[0]
    expected = model.base.generate(torch.tensor([prompt]), max_new_tokens=6, do_sample=False,
                                   eos_token_id=TOK.eos_token_id, pad_token_id=TOK.pad_token_id)[0, len(prompt):]
    assert generated == expected.tolist()[: len(generated)]


def test_greedy_generation_starts_from_the_reference_answer_logits():
    model = reasoner()
    prompt = latent_prompt_ids(math_problem())
    _, ref_logits = reference_pass(model, prompt)
    generated, _ = model.generate([prompt], TOK.SPECIALS["</think>"], TOK.eos_token_id, TOK.pad_token_id,
                                  max_tokens=3, temperature=0, top_p=1.0, generator=torch.Generator())[0]
    assert generated[0] == int(ref_logits[0, -1].argmax())


def test_nucleus_sampling_is_seeded_and_respects_top_p():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]] * 3)
    assert sample_next(logits, 0, 1.0, torch.Generator()).tolist() == [0, 0, 0]
    assert sample_next(logits, 1.0, 1e-6, torch.Generator().manual_seed(1)).tolist() == [0, 0, 0]
    a = sample_next(logits, 1.0, 0.95, torch.Generator().manual_seed(7))
    b = sample_next(logits, 1.0, 0.95, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# ---------------------------------------------------------------------- data

def trace_record(pid, correct=True, finished=True, thinking=50, response=None, sample=0):
    return {
        "problem_id": pid, "condition": "think", "sample": sample, "correct": correct, "finished": finished,
        "finish_reason": "stop" if finished else "length", "thinking_tokens": thinking,
        "response": response if response is not None else "2 plus 3 gives 5.\n</think>\n\nSo \\boxed{5}.",
    }


def test_trace_selection_keeps_one_clean_trace_per_problem():
    problems = {math_problem(i).id: math_problem(i) for i in range(7)}
    ids = list(problems)
    records = [
        trace_record(ids[0]),
        trace_record(ids[0], sample=1),
        trace_record(ids[1], correct=False),
        trace_record(ids[2], finished=False),
        trace_record(ids[3], thinking=9999),
        trace_record(ids[4], response="x</think>y</think>\\boxed{5}"),
        trace_record(ids[5], response="reasoning</think>The answer is 5."),
        trace_record(ids[6], response="reasoning  \n</think>\n\n\\boxed{\\frac{1}{2}}"),
    ]
    kept, dropped = select_traces(records, problems, max_reasoning_tokens=4096)
    assert [e.id for e in kept] == [ids[0], ids[6]]
    assert kept[0].reasoning == "2 plus 3 gives 5." and kept[0].answer == "5"
    assert kept[1].answer == "\\frac{1}{2}" and kept[1].reasoning == "reasoning"
    assert dropped == {"duplicate": 1, "incorrect": 1, "unfinished": 1, "too_long": 1, "malformed": 1, "no_boxed_answer": 1}


def test_last_boxed_and_math_train_rows():
    assert last_boxed("a \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{\\{1, 2\\}}") == "\\{1, 2\\}"
    assert last_boxed("\\boxed 2") is None and last_boxed("\\boxed{}") is None and last_boxed("\\boxed{3") is None
    row = {"problem": "P", "solution": "so \\boxed{7}", "type": "Algebra", "level": "Level 2"}
    assert tasks._math_train(3, row).gold == "7"
    assert tasks._math_train(3, dict(row, solution="so $\\boxed 7$")) is None
    assert tasks._ROW_PARSERS["gsm8k_train"](5, {"question": "Q", "answer": "x #### 1,200"}).id == "gsm8k_train_0005"


def test_pooling_averages_contiguous_windows():
    h = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    assert torch.equal(pool_states(h, 16), h)
    pooled = pool_states(h, 3)  # windows [0, 3), [3, 7), [7, 10)
    expected = torch.stack([h[0:3].mean(0), h[3:7].mean(0), h[7:10].mean(0)])
    torch.testing.assert_close(pooled, expected)


def test_keyword_ids_are_content_tokens():
    ids = keyword_token_ids("We multiply 48 by 2 to get 96 apples, so apples total 96.", TOK, top_k=4)
    assert ids and all(TOK.decode([i]).strip() for i in ids) and len(ids) == len(set(ids))


def make_examples(n=6):
    return [
        TraceExample(id=f"gsm8k_train_{i:04d}", benchmark="gsm8k_train", question=f"What is {i}+{i}?", gold=str(2 * i),
                     reasoning=f"{i} plus {i} is {2 * i}.", answer=str(2 * i), reasoning_tokens=10 + i,
                     keyword_ids=[ord("p"), ord("s")])
        for i in range(n)
    ]


def fake_states(batch):
    return [np.full((3 + len(e.reasoning) % 4, HIDDEN), float(e.answer), dtype=np.float32) for e in batch]


META = {"model": "tiny", "revision": "abc", "hidden_size": HIDDEN, "max_states": 8}


def test_teacher_store_resumes_after_a_torn_write(tmp_path):
    examples = make_examples(5)
    write_teacher_store(str(tmp_path), examples[:3], fake_states, META, batch_size=2)
    with open(tmp_path / "examples.jsonl", "a") as f:
        f.write('{"id": "gsm8k_tr')  # interrupted index write
    with open(tmp_path / "states.f16", "ab") as f:
        f.write(b"\0" * 64)  # states written without their index line

    calls = []
    write_teacher_store(str(tmp_path), examples, lambda b: calls.append(len(b)) or fake_states(b), META, batch_size=2)
    assert sum(calls) == 2
    store = TeacherStore(str(tmp_path))
    assert sorted(e.id for e in store.examples) == sorted(e.id for e in examples)
    for i, e in enumerate(store.examples):
        np.testing.assert_array_equal(store.states(i), fake_states([e])[0].astype(np.float16))

    with pytest.raises(RuntimeError, match="different settings"):
        write_teacher_store(str(tmp_path), examples, fake_states, dict(META, max_states=4))


def test_training_batch_targets_only_the_answer(tmp_path):
    write_teacher_store(str(tmp_path), make_examples(2), fake_states, META)
    dataset = LatentTrainingSet(TeacherStore(str(tmp_path)), TOK)
    item = dataset[1]
    assert TOK.decode(item["prefix"]).endswith("<|Assistant|><think>\n\n")
    assert TOK.decode(item["suffix"]).startswith("</think>\n\nThe final answer is $\\boxed{")
    scored = [t for t, label in zip(item["suffix"], item["labels"]) if label != -100]
    assert TOK.decode(scored) == "2}$.<eos>"
    batch = collate([dataset[0], dataset[1]], TOK.pad_token_id)
    assert batch["states"].shape[0] == 2 and batch["states_mask"].sum(-1).tolist() == [
        dataset[0]["states"].shape[0], dataset[1]["states"].shape[0]]


def test_validation_split_is_stable_by_id():
    ids = [f"p{i}" for i in range(1000)]
    split = split_indices(ids, 0.1)
    assert 50 < len(split["val"]) < 150 and not set(split["val"]) & set(split["train"])
    assert split_indices(ids + ["extra"], 0.1)["val"][: len(split["val"])] == split["val"]


# ---------------------------------------------------------------------- training and evaluation end to end

def train_config(tmp_path, **overrides):
    settings = dict(data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "run"), stage1_steps=1, stage2_steps=2,
                    batch_size=2, grad_accum=1, warmup_steps=1, val_fraction=0.3, save_every=2, val_every=2, lr=1e-3)
    settings.update(overrides)
    return TrainConfig(**settings)


def test_training_runs_checkpoints_and_reloads(tmp_path):
    write_teacher_store(str(tmp_path / "data"), make_examples(8), fake_states, META)
    store = TeacherStore(str(tmp_path / "data"))
    config = train_config(tmp_path)
    model = reasoner(perturb_lora=False)
    final = train(model, TOK, LatentTrainingSet(store, TOK), config)

    log = [json.loads(line) for line in open(os.path.join(config.out_dir, "train_log.jsonl"))]
    assert [r["stage"] for r in log] == [1, 2, 2]
    assert "answer" not in log[0] and np.isfinite(log[0]["ot"])
    assert all(np.isfinite(r["loss"]) and "answer" in r and "frozen_head" in r for r in log[1:])
    assert len(log[-1]["residual"]) == model.config.num_iterations - 1 and "val_answer" in log[-1]

    reloaded = reasoner(perturb_lora=False, seed=0)
    reloaded.load_trainable(final)
    for k, v in model.trainable_state_dict().items():
        torch.testing.assert_close(reloaded.trainable_state_dict()[k], v)
    reloaded.eval(), model.eval()
    prompt = latent_prompt_ids(math_problem())
    args = (TOK.SPECIALS["</think>"], TOK.eos_token_id, TOK.pad_token_id, 5, 0, 1.0, torch.Generator())
    assert model.generate([prompt], *args) == reloaded.generate([prompt], *args)

    # Rerunning a finished run trains nothing more and drops log records past the checkpoint (steps a
    # crashed run logged but will redo); changing its settings is refused.
    log_path = os.path.join(config.out_dir, "train_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": len(log) + 1}) + "\n" + '{"step": ')
    train(reasoner(perturb_lora=False), TOK, LatentTrainingSet(store, TOK), config)
    assert [json.loads(line)["step"] for line in open(log_path)] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="different settings"):
        train(reasoner(perturb_lora=False), TOK, LatentTrainingSet(store, TOK), train_config(tmp_path, lr=5e-4))


def test_standardized_ot_ignores_what_every_state_shares(tmp_path):
    write_teacher_store(str(tmp_path / "data"), make_examples(6), fake_states, META)
    store = TeacherStore(str(tmp_path / "data"))
    everything = np.concatenate([store.states(i) for i in range(len(store))]).astype(np.float64)
    mean, std = store.channel_stats(chunk=5)
    np.testing.assert_allclose(mean, everything.mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std, everything.std(0), rtol=1e-4, atol=1e-6)

    # A large offset shared by every state (and the latents) changes the raw cost but not the standardised one.
    gen = torch.Generator().manual_seed(0)
    z, h = torch.randn(2, 3, HIDDEN, generator=gen), torch.randn(2, 5, HIDDEN, generator=gen)
    mask = torch.ones(2, 5, dtype=torch.bool)
    offset = 50 * torch.randn(HIDDEN, generator=gen)

    def standardized(z, h):
        flat = h.reshape(-1, HIDDEN)
        return ot_loss(z, h, mask, 0.1, 50, (flat.mean(0), flat.std(0, unbiased=False)))

    torch.testing.assert_close(standardized(z + offset, h + offset), standardized(z, h))
    assert not torch.allclose(ot_loss(z + offset, h + offset, mask, 0.1, 50), ot_loss(z, h, mask, 0.1, 50))

    config = train_config(tmp_path, ot_standardize=True, ot_epsilon=0.03)
    train(reasoner(), TOK, LatentTrainingSet(store, TOK), config)
    log = [json.loads(line) for line in open(os.path.join(config.out_dir, "train_log.jsonl"))]
    assert all(np.isfinite(r["ot"]) for r in log)


def test_k0_control_trains_on_the_answer_alone(tmp_path):
    write_teacher_store(str(tmp_path / "data"), make_examples(4), fake_states, META)
    dataset = LatentTrainingSet(TeacherStore(str(tmp_path / "data")), TOK)
    with pytest.raises(ValueError, match="K=0"):
        train(reasoner(k=0), TOK, dataset, train_config(tmp_path))
    config = train_config(tmp_path, stage1_steps=0, ot_weight=0.0, frozen_head_weight=0.0, val_fraction=0.0)
    train(reasoner(k=0), TOK, dataset, config)
    log = [json.loads(line) for line in open(os.path.join(config.out_dir, "train_log.jsonl"))]
    assert [r["stage"] for r in log] == [2, 2] and all("ot" not in r and r["residual"] == [] for r in log)


def test_latent_backend_is_graded_by_the_standard_runner(tmp_path):
    pytest.importorskip("math_verify")
    backend = LatentBackend(reasoner(k=2), TOK, batch_size=2)
    problems = [EvalProblem(id=f"gsm8k_{i:04d}", benchmark="gsm8k", question=f"What is {i}+1?", gold=str(i + 1))
                for i in range(3)]
    records = run_condition(backend, problems, "latent", n_samples=2, temperature=0.6, top_p=0.95, max_tokens=8,
                            seed=0, out_dir=str(tmp_path), model="tiny", chunk_size=2)
    assert len(records) == 6
    assert all(r["condition"] == "latent" and r["thinking_tokens"] == 2 for r in records)
    assert all(r["response"].startswith("The final answer is $\\boxed{") for r in records)
    assert all(r["completion_tokens"] <= 48 for r in records)  # the direct condition's budget applies
