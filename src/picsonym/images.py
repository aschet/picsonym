# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Image normalization for compact transfer to a remote API.

This is a standalone utility, not something every title-generation backend
must use: it exists for backends that send an image over a network (like
:class:`picsonym.backends.OpenAIBackend`), where a smaller payload matters.
A backend running a model already loaded in-process has no such transfer to
optimize and should generally pass image bytes through untouched.
"""

from __future__ import annotations

import io
from typing import Final

from PIL import Image

__all__ = ["DEFAULT_MAX_DIMENSION", "DEFAULT_WEBP_QUALITY", "normalize_image"]

#: Default longest-side cap for :func:`normalize_image`. Local and cloud
#: vision models alike encode images internally at well under this
#: resolution, so anything larger only inflates the request.
DEFAULT_MAX_DIMENSION: Final = 1024

#: Default WEBP quality for :func:`normalize_image`.
DEFAULT_WEBP_QUALITY: Final = 90


def normalize_image(
    data: bytes,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
    """Decode, downscale if needed, and re-encode image bytes as WEBP.

    WEBP compresses photographic content far smaller than PNG, and
    downscaling anything larger than `max_dimension` keeps the result
    small — a title-generation API only needs to infer a mood from the
    image, not reproduce it, so neither the lossy encoding nor the resize
    meaningfully affects title quality. Works for any format Pillow can
    decode (PNG, JPEG, GIF, WEBP, BMP, TIFF, ...); HEIC is not supported,
    since decoding it needs the separate, less portable
    ``pillow-heif``/``libheif`` stack.

    :param data: Raw image bytes.
    :param max_dimension: Images larger than this on their longest side
        are downscaled to it, preserving aspect ratio; smaller images are
        left untouched (never upscaled).
    :param quality: WEBP encoder quality, 0-100.
    :returns: The re-encoded image as WEBP bytes.
    :raises ValueError: if `data` is not a format Pillow can decode.
    """
    try:
        decoded = Image.open(io.BytesIO(data))
    except OSError as exc:
        raise ValueError(f"not a supported image format: {exc}") from exc

    with decoded as source:
        image = source.convert("RGB")
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality)
        return buffer.getvalue()
