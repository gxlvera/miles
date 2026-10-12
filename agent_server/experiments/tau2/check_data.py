"""Check the τ² data the way the harness uses it, without any model (CPU).

For every task: build its environment through the harness (its own database variant, registered as
a domain of its own), replay the expected actions as a perfect agent, and score that trajectory with
tau2's ``evaluate_simulation``. The database check must pass for every task: it shows that each
variant loads, that evaluation rebuilds the environment on the same database, and that the expected
actions are valid there. The reward can still be 0 when a task requires the agent to communicate
information, which this replay never says. Also reports the time an environment takes to build.

Example:
  TAU2_DATA_DIR=/root/tau2-bench/data python agent_server/experiments/tau2/check_data.py --data /vol/data/tau2
"""

import argparse
import collections
import json
import time
from datetime import datetime

from examples.experimental.tau2.agent import _domain, _eval_pool, _evaluate
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks of each split")
    args = parser.parse_args()

    for split in ("train", "test"):
        rows = [json.loads(line) for line in open(f"{args.data}/{split}.jsonl")][: args.limit]
        db_match, reward, build_time = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        for row in rows:
            meta = row["metadata"]
            task = Task.model_validate(meta["task"])
            domain = _domain(meta["domain"], meta["db"])
            start = time.perf_counter()
            env = registry.get_env_constructor(domain)()
            build_time[domain].append(time.perf_counter() - start)
            info = evaluate_simulation(
                simulation=_replay_expected_actions(task, env),
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
            )
            key = meta["domain"]
            db_match[key, info.db_check is None or info.db_check.db_match] += 1
            reward[key, info.reward] += 1
        print(f"{split}: {len(rows)} tasks")
        print(f"  expected actions replayed, database check passed: {dict(db_match)}")
        print(f"  reward of that replay (0 when the task also needs communicated info): {dict(reward)}")
        for domain, times in sorted(build_time.items()):
            print(f"  environment build time {domain}: mean {sum(times) / len(times):.3f}s over {len(times)}")
        assert all(passed for (_, passed) in db_match), "an expected action does not reproduce its database"
        _check_eval_pool(rows[:6])


def _check_eval_pool(rows: list[dict]) -> None:
    """The harness scores episodes in a process pool: the same replays must score the same there."""
    local, pooled = [], []
    for row in rows:
        meta = row["metadata"]
        task = Task.model_validate(meta["task"])
        domain = _domain(meta["domain"], meta["db"])
        simulation = _replay_expected_actions(task, registry.get_env_constructor(domain)())
        local.append(
            evaluate_simulation(
                simulation=simulation, task=task, evaluation_type=EvaluationType.ALL, solo_mode=False, domain=domain
            ).reward
        )
        future = _eval_pool().submit(
            _evaluate, meta["domain"], meta["db"], meta["task"], simulation.model_dump(mode="json")
        )
        pooled.append(future)
    pooled = [future.result()["reward"] for future in pooled]
    print(f"  evaluation process pool: rewards {pooled} (in process {local})")
    assert pooled == local


def _replay_expected_actions(task: Task, env) -> SimulationRun:
    """A trajectory of an agent that makes exactly the expected tool calls, then the user stops."""
    messages = [AssistantMessage(role="assistant", content="Hi! How can I help you today?")]
    for index, action in enumerate(task.evaluation_criteria.actions or []):
        call = ToolCall(id=f"call_{index}", name=action.name, arguments=action.arguments, requestor=action.requestor)
        messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[call]))
        messages.append(env.get_response(call))
    messages.append(UserMessage(role="user", content="###STOP###"))
    now = datetime.now().isoformat()
    return SimulationRun(
        id=f"replay-{task.id}",
        task_id=task.id,
        start_time=now,
        end_time=now,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
    )


if __name__ == "__main__":
    main()
