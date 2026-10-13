import os

import pytest


@pytest.fixture(scope="session")
def tokenizer():
    """Qwen3.5-4B tokenizer from the local HF cache (or ``AGENT_SERVER_TEST_TOKENIZER``); skips without it."""
    transformers = pytest.importorskip("transformers")
    name = os.environ.get("AGENT_SERVER_TEST_TOKENIZER", "Qwen/Qwen3.5-4B")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"tokenizer {name} is not available locally")
    if not tokenizer.encode("\n", add_special_tokens=False):
        pytest.skip(f"tokenizer {name} loaded without its vocabulary (incomplete local files)")
    return tokenizer
