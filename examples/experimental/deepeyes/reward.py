"""DeepEyes reward: 0.8 * accuracy + 0.2 * format + 1.2 * (tool used and answer correct).

Follows ``compute_score`` in verl's ``recipe/deepeyes/deepeyes.py``. Accuracy comes from an LLM judge
when ``DEEPEYES_JUDGE_URL`` points at an OpenAI-compatible server (the recipe uses Qwen2.5-72B-Instruct
with the prompt below); otherwise from ``_rule_judge``, a lenient word match against the reference
answer that needs no extra GPUs.
"""

import os
import re

import httpx

_JUDGE_SYSTEM = (
    "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
    "provided standard answer, given a specific question.\n"
    "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
    "standard answer.\n"
    'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
    "any explanation or other text."
)
_JUDGE_USER = (
    "I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
    "answer is correct.\n\n---\n**Example 1:**\n[Question]: Is the countertop tan or blue?\n"
    "[Standard Answer]: The countertop is tan.\n[Model's Answer]: tan\n[Your Judgement]: CORRECT\n---\n"
    "**Example 2:**\n[Question]: Is the man phone both blue and closed?\n"
    "[Standard Answer]: Yes, the man phone is both blue and closed.\n[Model's Answer]: No.\n"
    "[Your Judgement]: INCORRECT\n---\n**Task:**\n[Question]: {question}\n[Standard Answer]: {answer}\n"
    "[Model's Answer]: {prediction}\n[Your Judgement]:"
)
_STOPWORDS = frozenset("a an the is are was were of to in on at it its this that and or with by for as be".split())


async def compute_score(
    assistant_texts: list[str], *, used_tool: bool, question: str, answer: str, options: dict[str, str] | None = None
) -> tuple[float, dict[str, float]]:
    """Score one episode from the text of its assistant turns (tool calls are parsed out of them).

    With ``options`` (multiple choice, e.g. V* Bench) ``answer`` is the correct letter and the prediction
    is judged by the letter it picks; otherwise by the judge below.
    """
    text = "".join(assistant_texts)
    format_error = text.count("<think>") != text.count("</think>")
    final = assistant_texts[-1] if assistant_texts else ""
    final = final.split("</think>")[-1]
    match = re.search(r"<answer>(.*?)</answer>", final, re.DOTALL)
    if final.count("<answer>") != final.count("</answer>") or match is None:
        format_error = True
    prediction = (match.group(1) if match else final).strip()
    if not prediction:
        format_error = True

    if options:
        correct = _choice_judge(answer, prediction, options)
    else:
        correct = bool(prediction) and await _judge(question, answer, prediction)
    if len(prediction) >= 1000:  # long answers are judge hacking in the recipe
        correct, format_error = False, True
    accuracy = 1.0 if correct else 0.0
    reward = 0.8 * accuracy - 0.2 * format_error + 1.2 * (used_tool and correct)
    return reward, {"accuracy": accuracy, "format_error": float(format_error)}


async def _judge(question: str, answer: str, prediction: str) -> bool:
    url = os.environ.get("DEEPEYES_JUDGE_URL")
    if not url:
        return _rule_judge(answer, prediction)
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{url}/chat/completions",
            json={
                "model": os.environ.get("DEEPEYES_JUDGE_MODEL", "judge"),
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": _JUDGE_USER.format(question=question, answer=answer, prediction=prediction),
                    },
                ],
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
    verdict = response.json()["choices"][0]["message"]["content"]
    return (
        re.search(r"\bINCORRECT\b", verdict, re.IGNORECASE) is None
        and re.search(r"\bCORRECT\b", verdict, re.IGNORECASE) is not None
    )


def _rule_judge(answer: str, prediction: str) -> bool:
    """Every content word of the prediction appears in the reference answer.

    Lenient on purpose: "tan" matches "The countertop is tan.", while "blue", or "yes" against
    "No, ...", does not. It cannot judge paraphrases.
    """
    predicted = _words(prediction) - _STOPWORDS
    return bool(predicted) and predicted <= _words(answer)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _choice_judge(label: str, prediction: str, options: dict[str, str]) -> bool:
    """The picked letter ("B", "(B)", "B. red") or the text of the picked option equals the correct option."""
    letter = re.match(r"\s*\(?([A-Z])\)?(?:[\s.:)]|$)", prediction)
    if letter and letter.group(1) in options:
        return letter.group(1) == label
    return prediction.strip().rstrip(".").lower() == options.get(label, "").lower()
