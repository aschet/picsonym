# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for cross-platform filename sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest

from picsonym.filenames import resolve_collision, sanitize_filename


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("A Title: With/Bad*Chars?", "A Title_ With_Bad_Chars_"),
        ('Quotes "Here"', "Quotes _Here_"),
        ("Pipe|Question?Star*", "Pipe_Question_Star_"),
        ("\x00Control\x1f", "_Control_"),
    ],
)
def test_forbidden_characters_are_replaced(title: str, expected: str) -> None:
    assert sanitize_filename(title) == expected


def test_custom_replacement_character() -> None:
    assert sanitize_filename("Bad/Char", replacement="-") == "Bad-Char"


@pytest.mark.parametrize("reserved", ["CON", "con", "Aux", "NUL", "COM1", "lpt9"])
def test_reserved_device_names_are_prefixed(reserved: str) -> None:
    assert sanitize_filename(reserved) == f"_{reserved}"


def test_reserved_name_check_is_on_the_whole_stem() -> None:
    """A word merely containing a reserved name must not be flagged."""
    assert sanitize_filename("Console") == "Console"


def test_trailing_dots_and_spaces_are_stripped() -> None:
    assert sanitize_filename("Trailing Space and Dot. ") == "Trailing Space and Dot"


def test_whitespace_runs_are_collapsed() -> None:
    assert sanitize_filename("Too   Many    Spaces") == "Too Many Spaces"


def test_empty_after_sanitization_falls_back() -> None:
    assert sanitize_filename("...") == "Untitled"
    assert sanitize_filename("   ") == "Untitled"
    assert sanitize_filename("") == "Untitled"


def test_extension_is_appended_verbatim() -> None:
    assert sanitize_filename("A Title", extension=".jpg") == "A Title.jpg"


def test_max_length_truncates_the_stem() -> None:
    result = sanitize_filename("A" * 300, extension=".jpg", max_length=10)
    assert result == "AAAAAA.jpg"
    assert len(result) == 10


def test_truncation_can_re_expose_a_trailing_dot_or_space() -> None:
    result = sanitize_filename("AAAAA. B", max_length=6)
    assert not result.endswith(" ") and not result.endswith(".")


def test_max_length_too_small_for_extension_raises() -> None:
    with pytest.raises(ValueError, match="max_length"):
        sanitize_filename("Title", extension=".jpeg", max_length=3)


class TestResolveCollision:
    def test_returns_path_unchanged_when_free(self, tmp_path: Path) -> None:
        candidate = tmp_path / "Title.jpg"
        assert resolve_collision(candidate) == candidate

    def test_appends_counter_on_collision(self, tmp_path: Path) -> None:
        (tmp_path / "Title.jpg").touch()
        assert resolve_collision(tmp_path / "Title.jpg") == tmp_path / "Title (2).jpg"

    def test_skips_past_multiple_collisions(self, tmp_path: Path) -> None:
        (tmp_path / "Title.jpg").touch()
        (tmp_path / "Title (2).jpg").touch()
        (tmp_path / "Title (3).jpg").touch()
        assert resolve_collision(tmp_path / "Title.jpg") == tmp_path / "Title (4).jpg"
