"""A small ALFWorld tool agent for Miles' existing agentic_tool_call adapter.

Adapted from Dressage's ALFWorld whitebox recipe (Apache-2.0):
https://github.com/Accio-Lab/Dressage/tree/main/dressage/recipes/alfworld
Only the environment loop lives here. Miles owns sessions, TITO, and samples.
"""

import json
import os
import time

import httpx

from miles.rollout.agentic.session import openai_session_url

SYSTEM_PROMPT = (
    "You are acting in ALFWorld TextWorld. Each turn, choose exactly one "
    "command from the admissible commands provided in the latest tool "
    "response. Reason briefly inside <think></think> if useful, then call "
    "the `env_step` tool with that exact command. Follow ALFWorld TextWorld "
    "command style such as `go to dresser 1`, `take mug 1 from cabinet 3`, "
    "`use desklamp 1`. Do not output a final natural-language answer."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "env_step",
            "description": "Execute one ALFWorld command and return the observation and admissible commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Exactly one admissible command."}},
                "required": ["command"],
            },
        },
    }
]


def make_env(game_file: str, max_steps: int):
    # TextWorld is optional for the rest of Miles; import it only in the agent process.
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    env_id = textworld.gym.register_game(
        game_file,
        request_infos=textworld.EnvInfos(won=True, admissible_commands=True, description=True, inventory=True),
        max_episode_steps=max_steps,
        # Official downloaded games contain internal object IDs. Use ALFWorld's
        # own readable naming, with no random renaming between repeated episodes.
        wrappers=[AlfredDemangler(shuffle=False)],
    )
    return textworld.gym.make(env_id)


def _observation_text(observation: str, commands: list[str]) -> str:
    admissible = "\n".join(f"- {command}" for command in commands if command != "help") or "None"
    return f"### Observation\n{observation.strip()[:2000]}\n\n### Admissible Commands\n{admissible}"


def _command(message: dict, admissible: list[str]) -> str | None:
    # Let Miles/SGLang parse model-specific tool syntax. Do not parse/rewrite the
    # assistant text ourselves: replaying the exact response preserves TITO matching.
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "env_step":
        return None
    arguments = calls[0]["function"]["arguments"]
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return None
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return command if isinstance(command, str) and command in admissible else None


def _feedback(message: dict, text: str) -> list[dict]:
    calls = message.get("tool_calls") or []
    if calls:
        # Reject multiple calls without advancing the environment, but answer each
        # call ID so the next request is still a valid tool conversation.
        return [{"role": "tool", "tool_call_id": call["id"], "content": text} for call in calls]
    return [{"role": "user", "content": text}]


async def run(base_url: str, prompt, request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    """Run one fresh game. Return terminal reward through v2's agent metadata."""
    max_steps = int(os.environ.get("ALFWORLD_MAX_STEPS", "30"))
    if max_steps < 1:
        raise ValueError("ALFWORLD_MAX_STEPS must be positive")
    env = make_env(metadata["game_file"], max_steps)
    try:
        # The standard Miles adapter runs each episode in a subprocess. Blocking
        # TextWorld reset/step calls therefore cannot stall other agents.
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as client:
            return await _run_episode(env, client, base_url, prompt, request_kwargs, max_steps)
    finally:
        env.close()


async def _run_episode(env, client, base_url, prompt, request_kwargs, max_steps) -> dict:
    started = time.monotonic()
    observation, infos = env.reset()
    tool_time = time.monotonic() - started
    admissible = list(infos["admissible_commands"])
    # ALFWorld's reset observation includes the concrete task; the dataset prompt
    # is just a readable label/fallback, never a pre-rendered model prompt.
    task = (
        observation.split("Your task is to:", 1)[1].strip().split("\n", 1)[0]
        if "Your task is to:" in observation
        else prompt
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### Task\n{task}\n\n{_observation_text(observation, admissible)}"},
    ]
    won, num_steps, num_turns, invalid_actions = False, 0, 0, 0
    stop_reason = "turn_limit"
    for _ in range(max_steps):
        num_turns += 1
        response = await client.post(
            f"{openai_session_url(base_url)}/chat/completions",
            json={
                **request_kwargs,
                "model": "default",
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        message = choice["message"]
        # Preserve tool-call IDs, arguments and reasoning_content verbatim.
        messages.append(message)
        command = _command(message, admissible)
        done = False
        if command is None:
            invalid_actions += 1
            feedback = (
                "Invalid action. Call env_step exactly once with a command from the admissible commands. "
                "Environment state did not change.\n\n"
            )
        else:
            started = time.monotonic()
            observation, _, done, infos = env.step(command)
            tool_time += time.monotonic() - started
            num_steps += 1
            admissible = list(infos["admissible_commands"])
            won = bool(infos["won"])
            feedback = ""
        messages.extend(_feedback(message, feedback + _observation_text(observation, admissible)))
        if won or done or choice["finish_reason"] == "length":
            stop_reason = "success" if won else "env_done" if done else "model_length"
            break
    return {
        "reward": float(won),
        "task_success": won,
        "num_turns": num_turns,
        "num_steps": num_steps,
        "invalid_action_count": invalid_actions,
        "stop_reason": stop_reason,
        "agent_metrics": {"total_tool_time": tool_time},
    }


async def require_reward(args, samples, **kwargs):
    """Fail visibly if an agent crashed after making model calls.

    Successful episodes already receive reward from v2's default postprocessor.
    Its adapter also collects partial traces on an exception; those must not be
    silently treated as task failures (reward 0).
    """
    if samples == []:
        return []
    raise RuntimeError("ALFWorld episode has no terminal reward; inspect the preceding agent-function error.")
