"""Claude Code PreToolUse hook: keep observations within a small policy context."""

import json
import sys


def limit_input(event: dict) -> dict:
    limits = {"Read": ("limit", 150), "Grep": ("head_limit", 40)}
    if event["tool_name"] not in limits:
        return {}
    key, cap = limits[event["tool_name"]]
    inputs = dict(event["tool_input"])
    requested = inputs.get(key)
    if isinstance(requested, int) and 0 < requested <= cap:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"Use a bounded {event['tool_name']} call: set {key} to an integer from 1 to {cap}, then retry. Keep existing offset/pattern fields.",
        }
    }


if __name__ == "__main__":
    print(json.dumps(limit_input(json.load(sys.stdin))))
