"""Continuous token: incremental prompt building for agent_server sessions, run inside SGLang.

The source lives here and is copied into SGLang as ``sglang.srt.continuous_token``
(see ``agent_server/sglang_patch/README.md``); agent_server's tests import it from this directory.
"""

from .builder import ContinuousTokenBuilder, MergeResult, QwenContinuousTokenBuilder
from .continuation import Continuation, continue_prompt, create_builder
from .media import load_image, message_images

__all__ = [
    "Continuation",
    "ContinuousTokenBuilder",
    "MergeResult",
    "QwenContinuousTokenBuilder",
    "continue_prompt",
    "create_builder",
    "load_image",
    "message_images",
]
