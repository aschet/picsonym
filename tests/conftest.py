# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Shared test doubles, no mocking library.

Two layers get their own fake, matching picsonym's own seam:

- :class:`FakeBackend` implements :class:`picsonym.backends.LLMBackend`
  directly, for tests of :class:`~picsonym.titling._TitleGenerator` and
  :class:`~picsonym.renaming.Picsonym` — title cleaning, validation,
  renaming logic. These don't care about OpenAI's wire format at all.
- :class:`FakeOpenAIClient` fakes the one method
  :class:`~picsonym.backends.OpenAIBackend` actually calls on a real
  :class:`openai.OpenAI` instance (`client.chat.completions.create`), for
  tests of :class:`OpenAIBackend` itself.

Neither ever makes a real network call or requires a running server.
"""

from __future__ import annotations

import io
import json
from typing import Any, cast

import httpx2
import openai
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def make_bad_request_error(*, param: str | None) -> openai.BadRequestError:
    """Build a real `BadRequestError` with a given `.param`, like the API sends."""
    response = httpx2.Response(
        status_code=400, request=httpx2.Request("POST", "http://test")
    )
    return openai.BadRequestError(
        "bad request", response=response, body={"param": param}
    )


def make_comfyui_png_bytes(*texts: str) -> bytes:
    """Build a minimal PNG with a ComfyUI-style embedded ``prompt`` chunk.

    Each of `texts` becomes one node's ``inputs.text`` field, mirroring
    how ComfyUI embeds one node per graph step (e.g. a positive and a
    negative prompt).
    """
    nodes = {
        str(index): {"inputs": {"text": text}, "class_type": "CLIPTextEncode"}
        for index, text in enumerate(texts)
    }
    info = PngInfo()
    info.add_text("prompt", json.dumps(nodes))
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


class FakeBackend:
    """A minimal stand-in for :class:`picsonym.backends.LLMBackend`."""

    def __init__(
        self,
        content: str | None = "Quiet Morning Light",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, *, system_prompt: str, user_text: str, image: bytes | None
    ) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_text": user_text, "image": image}
        )
        if self.error is not None:
            raise self.error
        return self.content or ""


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str | None, *, empty_choices: bool = False) -> None:
        self.choices = [] if empty_choices else [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str | None = "Quiet Morning Light") -> None:
        self.content = content
        self.error: Exception | None = None
        self.empty_choices = False
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.calls.append(kwargs)
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return FakeCompletion(self.content, empty_choices=self.empty_choices)


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """A minimal stand-in for :class:`openai.OpenAI`, injectable via `client=`."""

    def __init__(self, content: str | None = "Quiet Morning Light") -> None:
        self.completions = FakeCompletions(content=content)
        self.chat = FakeChat(self.completions)


def as_client(fake: FakeOpenAIClient) -> openai.OpenAI:
    """Cast a fake client to `openai.OpenAI` for use at an injection point.

    `FakeOpenAIClient` only implements the `.chat.completions.create`
    surface that :class:`~picsonym.backends.OpenAIBackend` actually calls,
    so it is not a structural match for the full `openai.OpenAI` type;
    this cast documents that mismatch is intentional and confined to tests.
    """
    return cast("openai.OpenAI", fake)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def fake_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()
