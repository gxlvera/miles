import json

import httpx
import pytest

from examples.experimental.alfworld import agent
from examples.experimental.alfworld.prepare_data import prepare_data

from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.utils.types import Sample


class Game:
    def __init__(self, *, win_after=2):
        self.actions = []
        self.closed = False
        self.win_after = win_after

    def reset(self):
        return "Your task is to: put the mug away.", {"admissible_commands": ["look"], "won": False}

    def step(self, command):
        self.actions.append(command)
        won = len(self.actions) >= self.win_after
        return "You see a mug.", 1 if won else 0, won, {"admissible_commands": ["look"], "won": won}

    def close(self):
        self.closed = True


def tool_message(command="look", *, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should inspect the room.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "env_step", "arguments": json.dumps({"command": command})},
            }
        ],
    }


async def run_agent(monkeypatch, game, messages, *, max_steps=30, finish_reason="tool_calls"):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/sessions/test/v1/chat/completions"
        message = messages[min(len(requests) - 1, len(messages) - 1)]
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(agent, "make_env", lambda game_file, max_steps: game)
    monkeypatch.setattr(
        agent.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond))
    )
    monkeypatch.setenv("ALFWORLD_MAX_STEPS", str(max_steps))
    result = await agent.run(
        "http://localhost/sessions/test",
        "put the mug away",
        {"temperature": 0.8, "max_tokens": 123},
        {"game_file": "/games/test.tw-pddl"},
    )
    return result, requests


async def test_full_history_and_sampling_args_reach_the_session(monkeypatch):
    game = Game()
    first, second = tool_message(), tool_message(call_id="call_2")
    result, requests = await run_agent(monkeypatch, game, [first, second])
    assert result["reward"] == 1.0
    assert result["num_turns"] == result["num_steps"] == 2
    assert result["stop_reason"] == "success"
    assert result["agent_metrics"]["total_tool_time"] >= 0
    assert game.closed
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    assert requests[1]["messages"][2] == first  # including reasoning and unmodified tool JSON
    assert requests[1]["messages"][3]["tool_call_id"] == "call_1"
    for request in requests:
        assert request["temperature"] == 0.8
        assert request["max_tokens"] == 123
        assert request["tools"] == agent.TOOLS


@pytest.mark.parametrize(
    "bad_message",
    [
        tool_message("invalid command"),
        {"role": "assistant", "content": "I forgot the tool call."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "bad", "function": {"name": "env_step", "arguments": "not JSON"}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [*tool_message(call_id="a")["tool_calls"], *tool_message(call_id="b")["tool_calls"]],
        },
    ],
)
async def test_invalid_action_does_not_step_the_game(monkeypatch, bad_message):
    game = Game(win_after=1)
    result, requests = await run_agent(monkeypatch, game, [bad_message, tool_message()])
    assert result["reward"] == 1
    assert result["invalid_action_count"] == 1
    assert result["num_turns"] == 2
    assert game.actions == ["look"]
    feedback = requests[1]["messages"][3:]
    ids = [call["id"] for call in bad_message.get("tool_calls", [])]
    assert [message.get("tool_call_id") for message in feedback] == (ids or [None])
    assert all("state did not change" in message["content"] for message in feedback)


async def test_turn_limit_is_a_real_failure_reward(monkeypatch):
    game = Game(win_after=100)
    result, _ = await run_agent(monkeypatch, game, [tool_message()], max_steps=2)
    assert result["reward"] == 0
    assert result["stop_reason"] == "turn_limit"
    assert len(game.actions) == 2
    assert game.closed


async def test_native_v2_postprocessor_applies_the_episode_reward(monkeypatch):
    metadata, _ = await run_agent(monkeypatch, Game(win_after=1), [tool_message()])
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=2,
        loss_mask=[1, 1],
        metadata={"leaf": {"node_id": 0, "path_node_ids": [0]}},
    )
    session = {"agent": metadata, "tree": {"nodes": [{"id": 0, "completion_span": [1, 3]}]}}
    (result,) = default_postprocess([sample], session)
    assert result.reward == 1.0
    assert result.metadata["task_success"] is True
    assert result.loss_mask == [1, 1]


async def test_length_finish_stops_without_another_model_call(monkeypatch):
    result, requests = await run_agent(monkeypatch, Game(win_after=100), [tool_message()], finish_reason="length")
    assert len(requests) == 1
    assert result["stop_reason"] == "model_length"


async def test_environment_exception_is_not_reward_zero(monkeypatch):
    game = Game()

    def broken_step(command):
        raise RuntimeError("environment crashed")

    game.step = broken_step
    with pytest.raises(RuntimeError, match="environment crashed"):
        await run_agent(monkeypatch, game, [tool_message()])
    assert game.closed
    # The native adapter catches the agent exception and collects partial traces;
    # the configured reward fallback makes that case fail visibly at the next stage.
    with pytest.raises(RuntimeError, match="no terminal reward"):
        await agent.require_reward(None, [object()])
    assert await agent.require_reward(None, []) == []


def test_data_preparation_keeps_games_and_splits_distinct(tmp_path):
    root = tmp_path / "games"
    for split in ("train", "valid_unseen"):
        for task in ("task_a", "task_b"):
            directory = root / "json_2.1.1" / split / task / "trial_same_name"
            directory.mkdir(parents=True)
            (directory / "game.tw-pddl").write_text('{"solvable": true}')
            (directory / "traj_data.json").write_text(
                json.dumps({"turk_annotations": {"anns": [{"task_desc": task}]}})
            )
    output = tmp_path / "train.jsonl"
    unsolvable = root / "json_2.1.1" / "train" / "unsolvable"
    unsolvable.mkdir()
    (unsolvable / "game.tw-pddl").write_text('{"solvable": false}')
    assert prepare_data(root, output, "train") == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({row["metadata"]["instance_id"] for row in rows}) == 2
    assert all("/train/" in row["metadata"]["game_file"] for row in rows)
    assert rows[0]["prompt"] == "task_a"
    assert prepare_data(root, output, "valid_unseen", limit=1) == 1
    with pytest.raises(FileNotFoundError, match="No ALFWorld games"):
        prepare_data(root, output, "missing")
