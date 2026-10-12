"""τ²-bench harness: one retail or airline customer-service episode as a Miles agent function.

τ²-bench runs the episode, through the async fork AReaL trains with
(``dhh1995/tau2-bench@dhh/async-and-custom-completion``):

- the policy is tau2's ``LLMAgent``, whose completions go to the agent server Miles passes in;
- the customer is tau2's ``UserSimulator``, an LLM served separately (``TAU2_USER_URL``) that knows the
  task's scenario and nothing else;
- tool calls run against the task's database, and the episode ends when either side stops or after
  ``TAU2_MAX_STEPS`` messages.

The reward is tau2's own evaluation (``EvaluationType.ALL``). For these domains that is the final
database state against the one the expected actions produce, times whether the agent said what it
had to.
"""

import asyncio
import functools
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import litellm
import numpy as np
import tau2.utils.llm_utils
from litellm.main import ModelResponse
from loguru import logger as tau2_logger
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)

USER_URL = os.environ.get("TAU2_USER_URL", "http://127.0.0.1:18902/v1")
# tau2's default user arguments (temperature 0), with a cap on a runaway reply.
USER_LLM_ARGS = {"temperature": 0.0, "max_tokens": 1024}
MAX_STEPS = int(os.environ.get("TAU2_MAX_STEPS", "100"))
# The simulator starts after the Miles launcher's cleanup (which kills every SGLang process), so early
# requests wait for it.
USER_STARTUP_TIMEOUT = 1800
# Retries of a request whose connection dropped (ReadError, RemoteProtocolError), on either URL. The agent
# server keeps a retried turn as a newer branch of the same parent and trains on the latest one.
TRANSPORT_RETRIES = 2
EVAL_WORKERS = int(os.environ.get("TAU2_EVAL_WORKERS", "8"))
_ENVIRONMENTS = {"retail": (retail_environment, RetailDB), "airline": (airline_environment, FlightDB)}
_METRICS = ("db_match", "communicate", "agent_stop", "user_stop", "max_steps", "error", "num_turns", "num_tool_calls")

# tau2 prices each completion through litellm, which knows neither model; the cost is unused.
tau2.utils.llm_utils.get_response_cost = lambda response: 0.0
litellm.suppress_debug_info = True
tau2_logger.remove()
tau2_logger.add(sys.stderr, level="WARNING")


async def run(base_url: str, prompt: str, request_kwargs: dict, metadata: dict, **_) -> dict:
    task = Task.model_validate(metadata["task"])
    domain = _domain(metadata["domain"], metadata["db"])
    env = await asyncio.to_thread(registry.get_env_constructor(domain))
    # No idle connection is kept: a turn often waits seconds for the user simulator, and a request sent on a
    # connection the server has meanwhile closed fails with a ReadError.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600), limits=httpx.Limits(max_keepalive_connections=0)
    ) as client:
        agent = LLMAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm="policy",
            llm_args=request_kwargs,
            completion_fn=_completion(client, f"{base_url}/v1/chat/completions", wait_for_server=False),
        )
        user = UserSimulator(
            instructions=str(task.user_scenario),
            llm="user",
            llm_args=USER_LLM_ARGS,
            completion_fn=_completion(client, f"{USER_URL}/chat/completions", wait_for_server=True),
        )
        orchestrator = Orchestrator(
            domain=domain, agent=agent, user=user, environment=env, task=task, max_steps=MAX_STEPS
        )
        try:
            simulation = await orchestrator.arun()
        except Exception as exc:  # tau2 raises on errors it cannot recover from, e.g. an exhausted context
            logger.warning("tau2 episode %s failed: %r", task.id, exc)
            return {"reward": 0.0, **_episode_metrics(orchestrator.get_trajectory(), None, error=True)}

    reward_info = RewardInfo.model_validate(
        await asyncio.get_running_loop().run_in_executor(
            _eval_pool(),
            _evaluate,
            metadata["domain"],
            metadata["db"],
            metadata["task"],
            simulation.model_dump(mode="json"),
        )
    )
    metrics = _episode_metrics(simulation.messages, reward_info, termination=simulation.termination_reason)
    return {"reward": reward_info.reward, **metrics}


@functools.cache
def _domain(domain: str, db: str | None) -> str:
    """The registry name of ``domain`` on database ``db``; ``None`` is the domain's own database.

    tau2's evaluator rebuilds the environment through the registry to replay the expected actions, so a
    task on another database is a domain of its own there. Registered on first use, on the event loop
    (and in each evaluation process).
    """
    if db is None:
        return domain
    get_environment, db_cls = _ENVIRONMENTS[domain]
    name = f"{domain}::{Path(db).stem}"

    def environment(solo_mode: bool = False):
        return get_environment(db=db_cls.model_validate_json(_read(db)), solo_mode=solo_mode)

    registry.register_domain(environment, name)
    return name


@functools.cache
def _eval_pool() -> ProcessPoolExecutor:
    """Processes for tau2's evaluation. It rebuilds two environments and hashes their whole databases
    (tens of MB of JSON) in pure Python: here, that would hold the GIL of every episode's event loop."""
    return ProcessPoolExecutor(EVAL_WORKERS, mp_context=multiprocessing.get_context("forkserver"))


def _evaluate(domain_name: str, db: str | None, task: dict, simulation: dict) -> dict:
    """tau2's evaluation of one episode, in an evaluation process."""
    return evaluate_simulation(
        simulation=SimulationRun.model_validate(simulation),
        task=Task.model_validate(task),
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=_domain(domain_name, db),
    ).model_dump(mode="json")


@functools.cache
def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def _completion(client: httpx.AsyncClient, url: str, *, wait_for_server: bool):
    """tau2's completion function: the request as an OpenAI chat body, the reply as litellm's ModelResponse."""

    async def complete(**kwargs) -> ModelResponse:
        kwargs.pop("num_retries", None)
        # tau2 sends tool_calls=None on plain assistant messages; OpenAI servers reject the null.
        kwargs["messages"] = [
            {k: v for k, v in m.items() if not (k == "tool_calls" and v is None)} for m in kwargs["messages"]
        ]
        deadline = time.monotonic() + USER_STARTUP_TIMEOUT
        retries = 0
        while True:
            try:
                response = await client.post(url, json=kwargs)
                break
            except httpx.ConnectError:
                if not wait_for_server or time.monotonic() > deadline:
                    raise
                await asyncio.sleep(10)
            except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
                retries += 1
                if retries > TRANSPORT_RETRIES:
                    raise
                logger.warning("tau2 request to %s dropped (%r), retry %d", url, exc, retries)
        response.raise_for_status()
        return ModelResponse(**response.json())

    return complete


def _episode_metrics(messages, reward_info, *, termination=None, error: bool = False) -> dict[str, float]:
    turns = [m for m in messages if m.role == "assistant"]
    metrics = {
        "num_turns": float(len(turns)),
        "num_tool_calls": float(sum(len(m.tool_calls or []) for m in turns)),
        "error": float(error),
        "agent_stop": float(termination == TerminationReason.AGENT_STOP),
        "user_stop": float(termination == TerminationReason.USER_STOP),
        "max_steps": float(termination == TerminationReason.MAX_STEPS),
    }
    if reward_info is not None and reward_info.db_check is not None:
        metrics["db_match"] = float(reward_info.db_check.db_match)
    if reward_info is not None and reward_info.communicate_checks:
        metrics["communicate"] = float(all(check.met for check in reward_info.communicate_checks))
    return metrics


async def score_failed_episodes(args, samples, **kwargs) -> list[float]:
    """Reward for samples whose agent function raised (Miles calls the reward model only for those): 0."""
    if samples:
        logger.warning("%d tau2 samples have no reward because their agent function failed", len(samples))
    return [0.0] * len(samples)


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Episode outcomes (database match, how the episode ended, turns) next to Miles' default rollout metrics."""
    metrics = {f"rollout/tau2/{key}": value for key, value in _mean_metrics(samples).items()}
    tracking.log(args, {**metrics, "rollout/step": compute_rollout_step(args, rollout_id)}, step_key="rollout/step")
    return False


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    """The same per eval set, and the reward per domain."""
    metrics = {}
    for name, result in data.items():
        samples = result.get("samples") or []
        metrics |= {f"eval/{name}/tau2/{key}": value for key, value in _mean_metrics(samples).items()}
        for domain in sorted({sample.metadata.get("domain") for sample in samples} - {None}):
            rewards = [s.reward or 0.0 for s in samples if s.metadata.get("domain") == domain]
            metrics[f"eval/{name}/by_domain/{domain}"] = float(np.mean(rewards))
    tracking.log(args, {**metrics, "eval/step": compute_rollout_step(args, rollout_id)}, step_key="eval/step")
    return False


def _mean_metrics(samples) -> dict[str, float]:
    metrics = {}
    for key in _METRICS:
        values = [sample.metadata[key] for sample in samples if key in sample.metadata]
        if values:
            metrics[key] = float(np.mean(values))
    return metrics
