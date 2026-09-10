# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Cross-platform safe filename handling.

This module has no dependency on the OpenAI SDK or network access: it only
turns arbitrary title text into a filename usable on disk and resolves
naming collisions.

:func:`sanitize_filename` follows Windows' filename rules, which are the
strictest of the major platforms (forbidden characters, reserved device
names, no trailing dot or space). A name valid on Windows is therefore also
valid on Unix and macOS, whose only hard requirements are "no ``/``" and "no
NUL byte" — both already excluded here — so one sanitizer covers all three.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

__all__ = [
    "IMAGE_EXTENSIONS",
    "resolve_collision",
    "sanitize_filename",
]

#: Extensions considered images by :func:`picsonym.renaming.Picsonym.rename_folder`.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
)

# Characters Windows forbids anywhere in a filename, plus ASCII control chars.
_FORBIDDEN_CHARS_RE: Final = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN_RE: Final = re.compile(r"\s+")

# Windows reserved device names, case-insensitively, as a full filename stem.
_RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(10)}
    | {f"LPT{digit}" for digit in range(10)}
)

_FALLBACK_STEM: Final = "Untitled"


def _strip_trailing_dots_and_spaces(text: str) -> str:
    return text.rstrip(" .")


def sanitize_filename(
    title: str,
    *,
    extension: str = "",
    replacement: str = "_",
    max_length: int = 255,
) -> str:
    """Sanitize `title` into a filename valid on Windows, Unix, and macOS.

    :param title: Human-readable title, without extension.
    :param extension: Extension to append verbatim, including the leading
        dot (e.g. ``".jpg"``); assumed to already be filesystem-safe.
    :param replacement: Substitute used for forbidden characters and for
        prefixing a reserved device name.
    :param max_length: Maximum length of the returned filename, stem plus
        extension combined.
    :returns: ``<sanitized-stem><extension>``.
    :raises ValueError: if `max_length` cannot fit at least one stem
        character plus `extension`.

    >>> sanitize_filename('A Title: With/Bad*Chars?')
    'A Title_ With_Bad_Chars_'
    >>> sanitize_filename('con', extension='.txt')
    '_con.txt'
    >>> sanitize_filename('...')
    'Untitled'
    >>> sanitize_filename('A' * 10, max_length=5)
    'AAAAA'
    """
    max_stem_length = max_length - len(extension)
    if max_stem_length < 1:
        raise ValueError(
            f"max_length={max_length!r} is too small to fit extension {extension!r}"
        )

    stem = _FORBIDDEN_CHARS_RE.sub(replacement, title)
    stem = _WHITESPACE_RUN_RE.sub(" ", stem).strip()
    stem = _strip_trailing_dots_and_spaces(stem)

    if stem.upper() in _RESERVED_NAMES:
        stem = f"{replacement}{stem}"

    if not stem:
        stem = _FALLBACK_STEM

    if len(stem) > max_stem_length:
        stem = _strip_trailing_dots_and_spaces(stem[:max_stem_length])
        if not stem:
            stem = _FALLBACK_STEM[:max_stem_length]

    return stem + extension


def resolve_collision(path: Path) -> Path:
    """Return `path`, or the first free ``<stem> (2)<suffix>``, ``(3)``, ...

    Not doctested since it depends on filesystem state; see
    ``tests/test_filenames.py`` for coverage using ``tmp_path``.

    :param path: Candidate path.
    :returns: A path guaranteed not to already exist.
    """
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
