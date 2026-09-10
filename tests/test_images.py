# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for image normalization."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from picsonym.images import DEFAULT_MAX_DIMENSION, normalize_image


def _make_image_bytes(size: tuple[int, int], fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=fmt)
    return buffer.getvalue()


def test_small_image_is_not_upscaled() -> None:
    data = normalize_image(_make_image_bytes((2, 2), "PNG"))
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (2, 2)
        assert image.format == "WEBP"


def test_large_image_is_downscaled_preserving_aspect_ratio() -> None:
    original_size = (DEFAULT_MAX_DIMENSION * 4, DEFAULT_MAX_DIMENSION * 2)
    data = normalize_image(_make_image_bytes(original_size, "PNG"))
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        assert max(width, height) == DEFAULT_MAX_DIMENSION
        assert width / height == pytest.approx(original_size[0] / original_size[1])


def test_custom_max_dimension_and_quality_are_honored() -> None:
    data = normalize_image(
        _make_image_bytes((500, 100), "PNG"), max_dimension=200, quality=50
    )
    with Image.open(io.BytesIO(data)) as image:
        assert max(image.size) == 200


@pytest.mark.parametrize("pillow_format", ["PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF"])
def test_every_supported_format_normalizes_to_webp(pillow_format: str) -> None:
    data = normalize_image(_make_image_bytes((2, 2), pillow_format))
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "WEBP"


def test_undecodable_data_raises() -> None:
    """Covers both plain garbage and formats Pillow can't decode (e.g. HEIC)."""
    with pytest.raises(ValueError, match="not a supported image format"):
        normalize_image(b"not an image")
