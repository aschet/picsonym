# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Pluggable backends for talking to a title-generating model.

:class:`Picsonym` (see :mod:`picsonym.titling`) owns the *what* — the
system prompts and the sanity checks on the result. A backend owns the
*how* — turning a system prompt, a user text, and an optional image into
one raw response string. The default, :class:`OpenAIBackend`, does this
over an OpenAI-compatible chat completions API; implement :class:`LLMBackend`
yourself to run a model already loaded in-process instead (e.g. inside a
ComfyUI custom node), with no HTTP call and no dependency on the OpenAI SDK.
"""

from __future__ import annotations

import base64
from typing import Final, Protocol

import openai

from ._client import DEFAULT_BASE_URL, build_client
from .images import normalize_image

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TEMPERATURE",
    "LLMBackend",
    "OpenAIBackend",
]

# None omits the field from the request entirely, rather than sending
# 0.0, falling back to each model's own default.
DEFAULT_TEMPERATURE: Final = None
DEFAULT_MAX_OUTPUT_TOKENS: Final = 20

# Reasoning is always off: it can burn the whole max_output_tokens budget
# on hidden chain-of-thought before emitting any visible title (empty
# content, finish_reason "length"), adding tens of seconds for no benefit
# on a task this short.


class LLMBackend(Protocol):
    """Interface a title-generation backend must implement."""

    def generate(
        self, *, system_prompt: str, user_text: str, image: bytes | None
    ) -> str:
        """Generate one raw title string.

        :param system_prompt: The instruction defining what a good title
            looks like — picsonym owns this text; pass it through as
            whatever "system"/instruction channel the model uses (or fold
            it into the prompt for a model with no such channel), rather
            than replacing it.
        :param user_text: The prompt to title (text-based titling), or a
            short fixed instruction accompanying `image` (image-based
            titling).
        :param image: Raw image bytes to title, or None for text-only
            titling. A backend calling a remote API will typically want to
            shrink this first via :func:`picsonym.normalize_image`; a
            backend running a local model usually should not, and can hand
            it to that model's own preprocessing untouched.
        :returns: The model's raw response text. Cleanup and a sanity
            check happen in the caller — return the output as-is.
        """
        ...


class OpenAIBackend:
    """Default :class:`LLMBackend`: an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        client: openai.OpenAI | None = None,
        model: str,
        temperature: float | None = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Construct a backend against an OpenAI-compatible API.

        :param api_key: Explicit API key; falls back to ``OPENAI_API_KEY``,
            then a local placeholder suitable for Ollama. Ignored if
            `client` is given.
        :param base_url: OpenAI-compatible API base URL. Defaults to a
            local Ollama server. Ignored if `client` is given.
        :param client: Pre-built :class:`openai.OpenAI` instance — a
            dependency-injection point for tests, or for advanced SDK
            configuration such as custom retries or timeouts.
        :param model: Model name to use; must be vision-capable to title
            from images. Required, with no default: no model is
            universally available on a local Ollama install, and guessing
            one would fail confusingly at API-call time instead of clearly
            at construction time.
        :param temperature: Sampling temperature passed to the API.
            Defaults to `None`, which omits the field entirely so the
            backend falls back to its own default (e.g. a model's own
            default temperature on Ollama) — see :data:`DEFAULT_TEMPERATURE`.
            Pass an explicit float to override.
        :param max_output_tokens: Maximum tokens the API may generate;
            titles are short, so the default is deliberately small.
        :raises openai.OpenAIError: see :func:`picsonym._client.build_client`.
        """
        self._client = build_client(api_key=api_key, base_url=base_url, client=client)
        self._model = model
        self._temperature: float | None = temperature
        self._max_output_tokens = max_output_tokens

    def generate(
        self, *, system_prompt: str, user_text: str, image: bytes | None
    ) -> str:
        """See :meth:`LLMBackend.generate`.

        :raises ValueError: if `image` is given and is not a format Pillow
            can decode.
        :raises openai.OpenAIError: on API failure.
        :raises RuntimeError: if the API response contains no choices —
            seen from some providers under content filtering or overload,
            distinct from a normal empty-content response.
        """
        content: str | list[dict[str, object]]
        if image is None:
            content = user_text
        else:
            webp = normalize_image(image)
            data_url = f"data:image/webp;base64,{base64.b64encode(webp).decode()}"
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        completion = self._create_completion(messages)
        if not completion.choices:
            raise RuntimeError(
                f"the API response for model {self._model!r} contained no choices"
            )
        return completion.choices[0].message.content or ""

    def _create_completion(
        self, messages: list[dict[str, object]]
    ) -> openai.types.chat.ChatCompletion:
        """Call the chat completions API.

        Split out from :meth:`generate` as a seam for subclasses that need
        to pass extra, provider-specific request fields (e.g. OpenAI's own
        `response_format`, or Ollama's grammar-constrained `format` via
        `extra_body`) — override this to adjust the call while still
        reusing `generate`'s message/image handling.
        """
        if self._temperature is None:
            return self._client.chat.completions.create(  # type: ignore[call-overload,no-any-return]
                model=self._model,
                max_tokens=self._max_output_tokens,
                messages=messages,
                reasoning_effort="none",
            )
        return self._client.chat.completions.create(  # type: ignore[call-overload,no-any-return]
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            messages=messages,
            reasoning_effort="none",
        )
