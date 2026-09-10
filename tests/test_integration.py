# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Optional integration test against a real, locally running Ollama server.

Unlike every other test in this suite, this one makes a real network call.
It is skipped unless a model is actually available on a reachable Ollama
server at `DEFAULT_BASE_URL` -- so it never runs in CI or on a machine
without Ollama, and never *requires* a specific model to be pulled. The
model checked for defaults to a small, vision-capable Gemma model, and can
be overridden via the `PICSONYM_TEST_MODEL` environment variable.
"""

from __future__ import annotations

import io
import json
import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from PIL import Image

from picsonym import Picsonym
from picsonym._client import DEFAULT_BASE_URL

_DEFAULT_TEST_MODEL = "gemma4:e4b"
_TEST_MODEL = os.environ.get("PICSONYM_TEST_MODEL", _DEFAULT_TEST_MODEL)


def _available_models() -> frozenset[str]:
    try:
        with urlopen(f"{DEFAULT_BASE_URL}/models", timeout=1) as response:
            payload = json.load(response)
    except (URLError, OSError):
        return frozenset()
    return frozenset(entry["id"] for entry in payload.get("data", []))


pytestmark = pytest.mark.skipif(
    _TEST_MODEL not in _available_models(),
    reason=(
        f"model {_TEST_MODEL!r} not available on a reachable Ollama server "
        f"at {DEFAULT_BASE_URL} (set PICSONYM_TEST_MODEL to use a different one)"
    ),
)


def test_title_from_prompt_against_real_server() -> None:
    generator = Picsonym(model=_TEST_MODEL)

    title = generator.title_from_prompt(
        "a lone lighthouse at dusk, waves crashing against dark rocks"
    )

    assert isinstance(title, str)
    assert 1 <= len(title.split()) <= 6


def test_title_from_image_against_real_server() -> None:
    generator = Picsonym(model=_TEST_MODEL)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 30, 60)).save(buffer, format="PNG")

    title = generator.title_from_image(buffer.getvalue())

    assert isinstance(title, str)
    assert 1 <= len(title.split()) <= 6
