# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for the OpenAI-compatible backend and client construction."""

from __future__ import annotations

import base64
import io

import openai
import pytest
from PIL import Image

from picsonym._client import DEFAULT_BASE_URL, build_client
from picsonym.backends import OpenAIBackend
from picsonym.images import normalize_image
from tests.conftest import FakeOpenAIClient, as_client, make_bad_request_error


def _make_image_bytes(size: tuple[int, int], fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=fmt)
    return buffer.getvalue()


def test_text_only_sends_plain_string_content(fake_client: FakeOpenAIClient) -> None:
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    result = backend.generate(
        system_prompt="system", user_text="a lone lighthouse at dusk", image=None
    )

    assert result == "Quiet Morning Light"
    [call] = fake_client.completions.calls
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a lone lighthouse at dusk"},
    ]


def test_image_is_normalized_and_sent_as_data_url(
    fake_client: FakeOpenAIClient,
) -> None:
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")
    image_bytes = _make_image_bytes((2, 2), "PNG")

    backend.generate(
        system_prompt="system", user_text="Title this image.", image=image_bytes
    )

    [call] = fake_client.completions.calls
    assert call["messages"][0] == {"role": "system", "content": "system"}
    user_content = call["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Title this image."}
    normalized = base64.b64encode(normalize_image(image_bytes)).decode()
    expected_url = f"data:image/webp;base64,{normalized}"
    assert user_content[1] == {"type": "image_url", "image_url": {"url": expected_url}}


def test_undecodable_image_raises(fake_client: FakeOpenAIClient) -> None:
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")
    with pytest.raises(ValueError, match="not a supported image format"):
        backend.generate(system_prompt="s", user_text="u", image=b"not an image")


def test_reasoning_is_always_disabled(fake_client: FakeOpenAIClient) -> None:
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    backend.generate(system_prompt="s", user_text="u", image=None)

    [call] = fake_client.completions.calls
    assert call["reasoning_effort"] == "none"


def test_retries_without_reasoning_effort_if_the_model_rejects_it(
    fake_client: FakeOpenAIClient,
) -> None:
    """Some models (e.g. OpenAI's gpt-6-astra) error instead of ignoring it."""
    fake_client.completions.error = make_bad_request_error(param="reasoning_effort")
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    result = backend.generate(system_prompt="s", user_text="u", image=None)

    assert result == "Quiet Morning Light"
    first_call, retry_call = fake_client.completions.calls
    assert first_call["reasoning_effort"] == "none"
    assert "reasoning_effort" not in retry_call


def test_an_unrelated_bad_request_is_not_retried(
    fake_client: FakeOpenAIClient,
) -> None:
    error = make_bad_request_error(param="model")
    fake_client.completions.error = error
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    with pytest.raises(openai.BadRequestError) as exc_info:
        backend.generate(system_prompt="s", user_text="u", image=None)

    assert exc_info.value is error
    assert len(fake_client.completions.calls) == 1


def test_temperature_none_omits_the_parameter(fake_client: FakeOpenAIClient) -> None:
    """None means "use the backend's own default", not "send 0.0"."""
    backend = OpenAIBackend(
        client=as_client(fake_client), model="test-model", temperature=None
    )

    backend.generate(system_prompt="s", user_text="u", image=None)

    [call] = fake_client.completions.calls
    assert "temperature" not in call


@pytest.mark.parametrize("content", [None, ""])
def test_no_content_returns_empty_string(content: str | None) -> None:
    fake_client = FakeOpenAIClient(content=content)
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    assert backend.generate(system_prompt="s", user_text="u", image=None) == ""


def test_no_choices_raises_runtime_error(fake_client: FakeOpenAIClient) -> None:
    fake_client.completions.empty_choices = True
    backend = OpenAIBackend(client=as_client(fake_client), model="test-model")

    with pytest.raises(RuntimeError, match="no choices"):
        backend.generate(system_prompt="s", user_text="u", image=None)


class TestBuildClient:
    def test_explicit_api_key_wins(self) -> None:
        client = build_client(api_key="secret", base_url=DEFAULT_BASE_URL, client=None)
        assert client.api_key == "secret"

    def test_env_var_used_when_no_explicit_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        client = build_client(api_key=None, base_url=DEFAULT_BASE_URL, client=None)
        assert client.api_key == "from-env"

    def test_falls_back_to_local_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = build_client(api_key=None, base_url=DEFAULT_BASE_URL, client=None)
        assert client.api_key == "ollama"

    def test_injected_client_is_returned_unchanged(
        self, fake_client: FakeOpenAIClient
    ) -> None:
        injected = as_client(fake_client)
        result = build_client(api_key="ignored", base_url="ignored", client=injected)
        assert result is injected
