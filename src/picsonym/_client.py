# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Private helper for resolving an :class:`openai.OpenAI` client.

Kept separate from :mod:`picsonym.backends` so API-key and base-URL
resolution stays isolated from message construction.
"""

from __future__ import annotations

import os
from typing import Final

import openai

__all__: list[str] = []

#: Default OpenAI-compatible API base URL: a local Ollama server.
DEFAULT_BASE_URL: Final = "http://localhost:11434/v1"

# Ollama's OpenAI-compatible endpoint ignores the API key, but the openai
# SDK requires a non-empty string to construct a client.
_LOCAL_API_KEY_PLACEHOLDER: Final = "ollama"


def build_client(
    *, api_key: str | None, base_url: str, client: openai.OpenAI | None
) -> openai.OpenAI:
    """Return `client` verbatim if given, else construct one.

    The key is resolved as: explicit `api_key` -> the ``OPENAI_API_KEY``
    environment variable -> a local placeholder. This means no key needs to
    be configured to use the local-Ollama default, while a real
    ``OPENAI_API_KEY`` (or explicit `api_key`) is still honored when
    `base_url` points at OpenAI's cloud or another real provider.

    :param api_key: Explicit API key, or None to fall back as described
        above. Ignored if `client` is given.
    :param base_url: OpenAI-compatible API base URL. Ignored if `client`
        is given.
    :param client: A pre-built client to return unchanged, bypassing all
        key/URL resolution.
    :returns: An :class:`openai.OpenAI` client.
    :raises openai.OpenAIError: propagated unmodified from
        :class:`openai.OpenAI` on construction failure.
    """
    if client is not None:
        return client

    resolved_key = (
        api_key or os.environ.get("OPENAI_API_KEY") or _LOCAL_API_KEY_PLACEHOLDER
    )
    return openai.OpenAI(api_key=resolved_key, base_url=base_url)
