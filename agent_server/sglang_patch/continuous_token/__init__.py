"""Continuous token: incremental prompt building for agent_server sessions, run inside SGLang.

The source lives here and ``sglang_patch/apply.sh`` copies it into SGLang as ``sglang.srt.continuous_token``;
agent_server's tests import it from this directory.
"""

from .builder import ContinuousTokenBuilder, GLMContinuousTokenBuilder, MergeResult, QwenContinuousTokenBuilder
from .continuation import Continuation, continue_prompt, create_builder
from .media import load_image, message_images

__all__ = [
    "Continuation",
    "ContinuousTokenBuilder",
    "GLMContinuousTokenBuilder",
    "MergeResult",
    "QwenContinuousTokenBuilder",
    "continue_prompt",
    "create_builder",
    "load_image",
    "message_images",
]
