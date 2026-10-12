"""DeepEyes v1 reward, as in seed-verl's scripts/deepeyes_v1/reward.py (the official DeepEyes
``vl_agent.py::compute_score``).

Score = 0.8 * accuracy - 0.2 * format error + 1.2 * (tool used and accuracy), range -0.2 to 2.0.
Accuracy is the verdict of an LLM judge (Qwen2.5-7B-Instruct served by SGLang at
``DEEPEYES_JUDGE_URL``) on the text of the last ``<answer>``, with the official judge prompt.

The input is the episode's response as verl's reward manager sees it: every assistant turn and tool
response, decoded without special tokens. So a tool response counts as tool use, and the vision tags
the official check also counts never appear.
"""

import asyncio
import logging
import os
import random
import time

import httpx

logger = logging.getLogger(__name__)

JUDGE_URL = os.environ.get("DEEPEYES_JUDGE_URL", "http://127.0.0.1:18901/v1")
# The judge starts after the Miles launcher's cleanup (which kills every SGLang process), so early
# requests wait for it.
JUDGE_STARTUP_TIMEOUT = 1800
_JUDGE_CONCURRENCY = asyncio.Semaphore(128)

# Official judge prompt (vl_agent.py get_chat_template / get_gpt4_score_ICE), verbatim.
_JUDGE_INSTRUCTION = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
_JUDGE_EXAMPLES = [
    """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
    """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""",
    """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""",
    """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""",
    """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""",
    """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""",
    """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""",
]


async def compute_score(solution: str, *, ground_truth: str, question: str) -> tuple[float, dict[str, float]]:
    format_error = solution.count("<think>") != solution.count("</think>")
    after_think = solution.split("</think>")[-1].strip()
    if after_think.count("<answer>") != after_think.count("</answer>"):
        format_error = True

    answer = extract_answer(solution)
    accuracy, judge_error = await _judge(question, ground_truth, answer)
    if len(answer) >= 1000:  # judge hacking
        accuracy, format_error = 0.0, True

    tool_used = "<tool_response>" in solution
    score = 0.8 * accuracy - 0.2 * format_error + 1.2 * (tool_used and accuracy > 0.5)
    info = {"accuracy": accuracy, "format_error": float(format_error), "tool_used": float(tool_used)}
    return score, {**info, "judge_error": float(judge_error)}


def extract_answer(solution: str) -> str:
    """Official extraction: the text of the last ``<answer>``, no fallback."""
    return solution.split("<answer>")[-1].split("</answer>")[0].strip()


async def _judge(question: str, ground_truth: str, answer: str) -> tuple[float, bool]:
    """The judge's verdict (1.0 or 0.0), and whether the request failed (scored 0, like the recipe)."""
    prompt = _JUDGE_INSTRUCTION + "".join(example + "\n\n" for example in _JUDGE_EXAMPLES)
    prompt += f"\n[Question]: {question}\n[Standard Answer]: {ground_truth}\n[Model_answer] : {answer}\nJudgement:"
    body = {
        "model": "judge",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "seed": random.randint(0, 1000000),
        "temperature": 0.3,
        "max_tokens": 16,
    }
    try:
        async with _JUDGE_CONCURRENCY:
            verdict = (await _post_judge(body)).strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
        logger.warning("DeepEyes judge request failed: %r", exc)
        return 0.0, True
    if "Judgement:" in verdict:
        verdict = verdict.split("Judgement:")[-1].strip()
    if "1" in verdict:
        return 1.0, False
    if "0" not in verdict:
        logger.warning("DeepEyes judge returned no verdict: %r", verdict)
    return 0.0, False


async def _post_judge(body: dict) -> str:
    deadline = time.monotonic() + JUDGE_STARTUP_TIMEOUT
    async with httpx.AsyncClient(timeout=120) as client:
        while True:
            try:
                response = await client.post(f"{JUDGE_URL}/chat/completions", json=body)
            except httpx.ConnectError:
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(10)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
