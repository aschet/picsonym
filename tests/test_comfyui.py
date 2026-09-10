# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for extracting an embedded ComfyUI generation prompt."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from picsonym.comfyui import extract_prompt
from tests.conftest import make_comfyui_png_bytes


def test_returns_the_longest_text_field() -> None:
    """The longest node is reliably the descriptive prompt.

    Not a short negative prompt or another short string field.
    """
    data = make_comfyui_png_bytes("blurry, low quality", "a lone lighthouse at dusk")

    assert extract_prompt(data) == "a lone lighthouse at dusk"


def test_accepts_a_path_as_well_as_bytes(tmp_path: Path) -> None:
    data = make_comfyui_png_bytes("a lone lighthouse at dusk")
    path = tmp_path / "image.png"
    path.write_bytes(data)

    assert extract_prompt(path) == "a lone lighthouse at dusk"


def test_returns_none_when_there_is_no_prompt_metadata() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")

    assert extract_prompt(buffer.getvalue()) is None


def test_returns_none_for_malformed_json() -> None:
    info = PngInfo()
    info.add_text("prompt", "not valid json{")
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG", pnginfo=info)

    assert extract_prompt(buffer.getvalue()) is None


def test_returns_none_when_json_is_not_an_object() -> None:
    info = PngInfo()
    info.add_text("prompt", json.dumps(["not", "an", "object"]))
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG", pnginfo=info)

    assert extract_prompt(buffer.getvalue()) is None


def test_returns_none_when_no_node_has_a_text_field() -> None:
    info = PngInfo()
    info.add_text(
        "prompt", json.dumps({"1": {"inputs": {"seed": 42}, "class_type": "KSampler"}})
    )
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG", pnginfo=info)

    assert extract_prompt(buffer.getvalue()) is None


def test_returns_none_for_undecodable_image_data() -> None:
    assert extract_prompt(b"not an image") is None
