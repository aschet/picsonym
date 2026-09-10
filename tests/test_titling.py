# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for title-generation orchestration: prompts, cleaning, validation.

These use :class:`~tests.conftest.FakeBackend`, not a fake OpenAI client —
:class:`_TitleGenerator` only depends on the :class:`~picsonym.backends.LLMBackend`
interface, so its own logic (which system prompt to use, cleaning the
response, rejecting malformed output) should be verifiable without any
OpenAI-specific plumbing. See tests/test_backends.py for OpenAIBackend itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picsonym.titling import (
    _SYSTEM_PROMPT_FROM_IMAGE,
    _SYSTEM_PROMPT_FROM_IMAGE_AND_PROMPT,
    _SYSTEM_PROMPT_FROM_PROMPT,
    _TitleGenerator,
)
from tests.conftest import FakeBackend, make_comfyui_png_bytes


def test_title_from_prompt_uses_the_prompt_system_prompt(
    fake_backend: FakeBackend,
) -> None:
    generator = _TitleGenerator(backend=fake_backend)

    title = generator.title_from_prompt("a lone lighthouse at dusk")

    assert title == "Quiet Morning Light"
    [call] = fake_backend.calls
    assert call == {
        "system_prompt": _SYSTEM_PROMPT_FROM_PROMPT,
        "user_text": "a lone lighthouse at dusk",
        "image": None,
    }


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_title_from_prompt_rejects_blank_input(
    fake_backend: FakeBackend, prompt: str
) -> None:
    generator = _TitleGenerator(backend=fake_backend)
    with pytest.raises(ValueError, match="empty"):
        generator.title_from_prompt(prompt)
    assert fake_backend.calls == []


def test_title_from_image_reads_a_path_and_uses_the_image_system_prompt(
    fake_backend: FakeBackend, tmp_path: Path
) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"pretend-image-bytes")
    generator = _TitleGenerator(backend=fake_backend)

    title = generator.title_from_image(image_path)

    assert title == "Quiet Morning Light"
    [call] = fake_backend.calls
    assert call["system_prompt"] == _SYSTEM_PROMPT_FROM_IMAGE
    assert call["image"] == b"pretend-image-bytes"


def test_title_from_image_accepts_raw_bytes_unchanged(
    fake_backend: FakeBackend,
) -> None:
    """The backend gets the image as-is; normalization is backend-specific."""
    generator = _TitleGenerator(backend=fake_backend)

    generator.title_from_image(b"pretend-image-bytes")

    [call] = fake_backend.calls
    assert call["image"] == b"pretend-image-bytes"


def test_title_from_image_uses_its_own_embedded_comfyui_prompt(
    fake_backend: FakeBackend, tmp_path: Path
) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(make_comfyui_png_bytes("a lone lighthouse at dusk"))
    generator = _TitleGenerator(backend=fake_backend)

    generator.title_from_image(image_path)

    [call] = fake_backend.calls
    assert call["system_prompt"] == _SYSTEM_PROMPT_FROM_IMAGE_AND_PROMPT
    assert call["user_text"] == (
        "Generation prompt: a lone lighthouse at dusk\n\nTitle this image."
    )


def test_title_from_image_prompt_argument_overrides_embedded_metadata(
    fake_backend: FakeBackend, tmp_path: Path
) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(make_comfyui_png_bytes("embedded prompt"))
    generator = _TitleGenerator(backend=fake_backend)

    generator.title_from_image(image_path, prompt="explicit prompt")

    [call] = fake_backend.calls
    assert call["system_prompt"] == _SYSTEM_PROMPT_FROM_IMAGE_AND_PROMPT
    assert call["user_text"] == (
        "Generation prompt: explicit prompt\n\nTitle this image."
    )


def test_title_from_image_missing_file_raises(
    fake_backend: FakeBackend, tmp_path: Path
) -> None:
    generator = _TitleGenerator(backend=fake_backend)
    with pytest.raises(FileNotFoundError):
        generator.title_from_image(tmp_path / "missing.png")
    assert fake_backend.calls == []


def test_response_is_stripped_of_quotes_and_extra_whitespace() -> None:
    backend = FakeBackend(content='  "Quiet   Morning"  ')
    generator = _TitleGenerator(backend=backend)

    assert generator.title_from_prompt("a scene") == "Quiet Morning"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("quiet morning light", "Quiet Morning Light"),
        ("QUIET MORNING LIGHT", "Quiet Morning Light"),
        ("don't look back", "Don't Look Back"),
    ],
)
def test_all_one_case_response_is_recovered_to_title_case(
    content: str, expected: str
) -> None:
    """Not every model reliably follows the Title Case instruction."""
    backend = FakeBackend(content=content)
    generator = _TitleGenerator(backend=backend)

    assert generator.title_from_prompt("a scene") == expected


def test_already_title_cased_response_with_lowercase_small_words_is_untouched() -> None:
    """A model that already got real Title Case right (small words lowercase).

    Must not be force-capitalized into a worse result.
    """
    backend = FakeBackend(content="Echoes of a Lost Era")
    generator = _TitleGenerator(backend=backend)

    assert generator.title_from_prompt("a scene") == "Echoes of a Lost Era"


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_response_raises_runtime_error(content: str | None) -> None:
    backend = FakeBackend(content=content)
    generator = _TitleGenerator(backend=backend)

    with pytest.raises(RuntimeError, match="no title content"):
        generator.title_from_prompt("a scene")


def test_non_str_response_raises_type_error() -> None:
    """A buggy backend returning e.g. a list must fail clearly.

    Rather than crashing inside title cleaning with an unrelated
    AttributeError.
    """
    backend = FakeBackend()
    backend.content = ["Quiet", "Morning", "Light"]  # type: ignore[assignment]
    generator = _TitleGenerator(backend=backend)

    with pytest.raises(TypeError, match=r"FakeBackend\.generate\(\) must return str"):
        generator.title_from_prompt("a scene")


def test_overlong_response_raises_runtime_error() -> None:
    """Covers a model drafting multiple candidate titles instead of one."""
    garbled = "Window Warmth Amid Snowstorm\nQuiet Window Knitting\nSnowbound Window"
    backend = FakeBackend(content=garbled)
    generator = _TitleGenerator(backend=backend)

    with pytest.raises(RuntimeError, match="malformed title"):
        generator.title_from_prompt("a scene")


class TestBackendSelection:
    def test_model_required_when_no_backend_given(self) -> None:
        with pytest.raises(ValueError, match="model"):
            _TitleGenerator()

    def test_custom_backend_needs_no_model(self, fake_backend: FakeBackend) -> None:
        generator = _TitleGenerator(backend=fake_backend)
        assert generator.title_from_prompt("a scene") == "Quiet Morning Light"

    def test_model_is_ignored_when_backend_given(
        self, fake_backend: FakeBackend
    ) -> None:
        # Should not raise, and should use fake_backend rather than trying
        # to build a real OpenAI client from an unused model name.
        generator = _TitleGenerator(model="unused-model", backend=fake_backend)
        generator.title_from_prompt("a scene")
        assert fake_backend.calls
