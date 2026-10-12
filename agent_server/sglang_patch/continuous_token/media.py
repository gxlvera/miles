"""Images in OpenAI chat messages: which ones a prompt has, and how to load one.

Shared by the agent server (image references it stores and exports) and SGLang (images it encodes),
so both agree on which blocks are images and in which order.
"""

import base64
import io
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


def message_images(messages: list[dict[str, Any]]) -> list[str]:
    """The images of ``messages`` in prompt order: the URL (or data URI) of each ``image_url`` block."""
    images = []
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = block["image_url"]
                images.append(url["url"] if isinstance(url, dict) else url)
    return images


def load_image(ref: str) -> Image.Image:
    """A data URI, an http(s) URL or a local path, decoded with PIL as RGB.

    The trainer decodes the same references the same way (``miles.utils.processing_utils.load_image_ref``),
    so both sides process identical pixels.
    """
    if ref.startswith("data:"):
        data = base64.b64decode(ref.split(",", 1)[1])
    elif ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref, timeout=60) as response:
            data = response.read()
    else:
        data = Path(ref).read_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")
