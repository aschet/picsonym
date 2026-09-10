# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Extracting an image's own embedded ComfyUI generation prompt, if any."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from PIL import Image

__all__ = ["extract_prompt"]


def extract_prompt(image: str | os.PathLike[str] | bytes) -> str | None:
    """Extract the text prompt embedded in a ComfyUI-generated image.

    ComfyUI embeds its full node graph as PNG metadata. This returns the
    longest "text" input found across all nodes, which is reliably the
    main descriptive prompt rather than a short negative prompt or an
    unrelated string field elsewhere in the graph.

    :param image: Path to an image file, or raw image bytes.
    :returns: The longest embedded prompt, or None if `image` carries no
        parseable ComfyUI prompt metadata — including if it isn't a
        ComfyUI-generated PNG at all.
    """
    try:
        source = io.BytesIO(image) if isinstance(image, bytes) else Path(image)
        with Image.open(source) as decoded:
            raw = decoded.info.get("prompt")
        if raw is None:
            return None
        nodes = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(nodes, dict):
        return None

    texts = []
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        text = inputs.get("text")
        if isinstance(text, str):
            texts.append(text)

    return max(texts, key=len) if texts else None
