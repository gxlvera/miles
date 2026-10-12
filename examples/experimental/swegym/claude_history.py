"""History identity for Claude Code using the original Qwen3.5 HF template."""

from copy import deepcopy


def normalize_message(message):
    """Mirror Qwen's text trimming for matching only; preserve all tool values.

    The recipe fixes the Qwen3.5 template, whose rendering trims these fields.
    No generic model or user request silently opts into this projection.
    """
    result = deepcopy(message)
    content = result.get("content")
    if isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        for block in content
    ):
        content = "".join(block["text"] for block in content)
    if isinstance(content, str):
        result["content"] = content.strip()
    if isinstance(result.get("reasoning_content"), str):
        result["reasoning_content"] = result["reasoning_content"].strip()
    return result
