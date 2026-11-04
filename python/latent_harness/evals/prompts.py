"""Prompt construction following DeepSeek-R1's published evaluation setup.

- No system prompt; all instructions go in the user turn.
- Math: "Please reason step by step, and put your final answer within \\boxed{}."
- GPQA-Diamond: the simple-evals multiple-choice template.
- The response starts inside a "<think>\\n" block, as DeepSeek recommends.

Conditions:
- think:    full reasoning.
- no_think: an empty reasoning block, then a free-form answer.
- direct:   an empty reasoning block and a forced answer prefix with a tiny
            token budget. This is the no-reasoning floor that K latent tokens
            have to beat.
- latent:   the direct layout, scored for a latent model. Its backend inserts K
            latent positions inside the empty reasoning block, just before
            </think>, so K=0 is exactly the direct condition.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .tasks import EvalProblem


CONDITIONS = ("think", "no_think", "direct")  # baseline conditions
LATENT_CONDITION = "latent"
ALL_CONDITIONS = CONDITIONS + (LATENT_CONDITION,)

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)

THINK_OPEN = "<think>\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"

DIRECT_PREFIX = {"math": "The final answer is $\\boxed{", "choice": "Answer: "}
DIRECT_MAX_TOKENS = {"math": 48, "choice": 4}


@dataclass
class Prompt:
    text: str  # full model input with the chat template applied
    response_prefix: str  # forced start of the response; graded together with the completion
    max_tokens: Optional[int]  # overrides the run's token budget (direct condition only)


def user_message(problem: EvalProblem) -> str:
    if problem.is_multiple_choice:
        assert problem.choices is not None and len(problem.choices) == 4
        a, b, c, d = problem.choices
        return GPQA_TEMPLATE.format(question=problem.question, a=a, b=b, c=c, d=d)
    return f"{problem.question}\n{MATH_INSTRUCTION}"


def build_prompt(problem: EvalProblem, condition: str, tokenizer: Any) -> Prompt:
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {', '.join(ALL_CONDITIONS)}")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message(problem)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Recent R1-Distill templates already open the think block; older ones do not.
    if text.endswith(THINK_OPEN):
        text = text[: -len(THINK_OPEN)]

    if condition == "think":
        return Prompt(text=text + THINK_OPEN, response_prefix="", max_tokens=None)
    if condition == "no_think":
        return Prompt(text=text + EMPTY_THINK, response_prefix="", max_tokens=None)

    kind = "choice" if problem.is_multiple_choice else "math"
    prefix = DIRECT_PREFIX[kind]
    return Prompt(text=text + EMPTY_THINK + prefix, response_prefix=prefix, max_tokens=DIRECT_MAX_TOKENS[kind])
