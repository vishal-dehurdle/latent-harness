"""Answer extraction and scoring shared by every evaluation condition.

Math answers (GSM8K, MATH-500, AIME) are graded with math-verify, which checks
symbolic equivalence, so 1/2, 0.5 and \\dfrac{1}{2} all match. GPQA-Diamond is
graded on the chosen option letter.

A `think` response that never closes its reasoning block is scored as
unanswered, even if an intermediate \\boxed{} appears inside the reasoning.
Those are almost always length cut-offs, and grading partial reasoning would
let mid-trace guesses count as answers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re

from .tasks import EvalProblem


THINK_CLOSE = "</think>"

_ANSWER_LINE = re.compile(r"(?:answer|Answer|ANSWER)\s*[:：]\s*[*_]*\s*\$?\s*\(?\s*([A-D])\b")
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?([A-D])\)?\s*\}")


@dataclass
class Grade:
    answer: Optional[str]
    correct: bool
    finished: bool


def answer_region(response: str, condition: str) -> Optional[str]:
    """The part of a response that holds the final answer, or None if reasoning never finished."""
    if condition != "think":
        return response
    if THINK_CLOSE not in response:
        return None
    return response.rsplit(THINK_CLOSE, 1)[1]


def extract_choice(text: str) -> Optional[str]:
    """The last option letter given as 'Answer: X', or failing that as \\boxed{X}."""
    for pattern in (_ANSWER_LINE, _BOXED_LETTER):
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
    return None


def grade_math(region: str, gold: str) -> Tuple[Optional[str], bool]:
    from math_verify import parse, verify

    gold_parsed = parse(f"${gold}$")
    if not gold_parsed:
        raise ValueError(f"math-verify cannot parse gold answer {gold!r}")
    pred = parse(region)
    if not pred:
        return None, False
    answer = next((p for p in pred if isinstance(p, str)), str(pred[0]))
    return answer, bool(verify(gold_parsed, pred))


def grade(problem: EvalProblem, condition: str, response: str) -> Grade:
    region = answer_region(response, condition)
    if region is None:
        return Grade(answer=None, correct=False, finished=False)
    if problem.is_multiple_choice:
        choice = extract_choice(region)
        return Grade(answer=choice, correct=choice == problem.gold, finished=True)
    answer, correct = grade_math(region, problem.gold)
    return Grade(answer=answer, correct=correct, finished=True)
