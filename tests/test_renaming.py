# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for renaming image files to their generated titles."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from picsonym.renaming import Picsonym
from tests.conftest import FakeOpenAIClient, as_client


def _make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


_PNG_BYTES = _make_png_bytes()


def make_picsonym(fake_client: FakeOpenAIClient) -> Picsonym:
    return Picsonym(client=as_client(fake_client), model="test-model")


def test_rename_file_uses_generated_title(
    fake_client: FakeOpenAIClient, tmp_path: Path
) -> None:
    image = tmp_path / "IMG_1234.png"
    image.write_bytes(_PNG_BYTES)
    picsonym = make_picsonym(fake_client)

    result = picsonym.rename_file(image)

    expected = tmp_path / "Quiet Morning Light.png"
    assert result.renamed == expected
    assert result.title == "Quiet Morning Light"
    assert result.error is None
    assert expected.exists()
    assert not image.exists()


def test_rename_file_resolves_collisions(
    fake_client: FakeOpenAIClient, tmp_path: Path
) -> None:
    (tmp_path / "Quiet Morning Light.png").touch()
    image = tmp_path / "IMG_1234.png"
    image.write_bytes(_PNG_BYTES)
    picsonym = make_picsonym(fake_client)

    result = picsonym.rename_file(image)

    assert result.renamed == tmp_path / "Quiet Morning Light (2).png"
    assert result.renamed is not None and result.renamed.exists()


def test_rename_file_reports_title_generation_failure(
    fake_client: FakeOpenAIClient, tmp_path: Path
) -> None:
    fake_client.completions.error = RuntimeError("boom")
    image = tmp_path / "IMG_1234.png"
    image.write_bytes(_PNG_BYTES)
    picsonym = make_picsonym(fake_client)

    result = picsonym.rename_file(image)

    assert result.renamed is None
    assert result.error == "boom"
    assert image.exists()


def test_rename_file_retitles_even_if_the_name_already_looks_like_a_title(
    fake_client: FakeOpenAIClient, tmp_path: Path
) -> None:
    """No skip heuristic: every file is always re-titled."""
    image = tmp_path / "Summer Vacation.png"
    image.write_bytes(_PNG_BYTES)
    picsonym = make_picsonym(fake_client)

    result = picsonym.rename_file(image)

    assert result.title == "Quiet Morning Light"
    assert fake_client.completions.calls


class TestRenameFolder:
    def test_only_touches_image_extensions_non_recursively(
        self, fake_client: FakeOpenAIClient, tmp_path: Path
    ) -> None:
        (tmp_path / "IMG_1.png").write_bytes(_PNG_BYTES)
        (tmp_path / "notes.txt").write_text("hello")
        subfolder = tmp_path / "sub"
        subfolder.mkdir()
        (subfolder / "IMG_2.png").write_bytes(_PNG_BYTES)
        picsonym = make_picsonym(fake_client)

        results = picsonym.rename_folder(tmp_path)

        assert [r.original.name for r in results] == ["IMG_1.png"]

    def test_recursive_descends_into_subfolders(
        self, fake_client: FakeOpenAIClient, tmp_path: Path
    ) -> None:
        (tmp_path / "IMG_1.png").write_bytes(_PNG_BYTES)
        subfolder = tmp_path / "sub"
        subfolder.mkdir()
        (subfolder / "IMG_2.png").write_bytes(_PNG_BYTES)
        picsonym = make_picsonym(fake_client)

        results = picsonym.rename_folder(tmp_path, recursive=True)

        assert {r.original.name for r in results} == {"IMG_1.png", "IMG_2.png"}

    def test_recursive_skips_dot_directories(
        self, fake_client: FakeOpenAIClient, tmp_path: Path
    ) -> None:
        (tmp_path / "IMG_1.png").write_bytes(_PNG_BYTES)
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "IMG_2.png").write_bytes(_PNG_BYTES)
        nested_hidden = tmp_path / "sub" / ".hidden"
        nested_hidden.mkdir(parents=True)
        (nested_hidden / "IMG_3.png").write_bytes(_PNG_BYTES)
        picsonym = make_picsonym(fake_client)

        results = picsonym.rename_folder(tmp_path, recursive=True)

        assert [r.original.name for r in results] == ["IMG_1.png"]

    def test_is_lazy_not_computed_all_at_once(
        self, fake_client: FakeOpenAIClient, tmp_path: Path
    ) -> None:
        """Consuming one result must not touch every file up front."""
        (tmp_path / "IMG_1.png").write_bytes(_PNG_BYTES)
        (tmp_path / "IMG_2.png").write_bytes(_PNG_BYTES)
        (tmp_path / "IMG_3.png").write_bytes(_PNG_BYTES)
        picsonym = make_picsonym(fake_client)

        results = picsonym.rename_folder(tmp_path)
        next(results)

        assert len(fake_client.completions.calls) == 1

    def test_per_file_failure_does_not_abort_the_batch(self, tmp_path: Path) -> None:
        (tmp_path / "IMG_1.png").write_bytes(_PNG_BYTES)
        (tmp_path / "IMG_2.png").write_bytes(_PNG_BYTES)

        # A client whose completions.create fails on the first call only,
        # so exactly one of the two files fails and the other still renames.
        fake_client = FakeOpenAIClient()
        original_create = fake_client.completions.create
        call_count = {"n": 0}

        def flaky_create(**kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return original_create(**kwargs)

        fake_client.completions.create = flaky_create  # type: ignore[method-assign]
        picsonym = Picsonym(client=as_client(fake_client), model="test-model")

        results = list(picsonym.rename_folder(tmp_path))

        errors = [r for r in results if r.error is not None]
        successes = [r for r in results if r.error is None]
        assert len(errors) == 1
        assert len(successes) == 1
        assert successes[0].renamed is not None and successes[0].renamed.exists()

    def test_raises_for_non_directory(
        self, fake_client: FakeOpenAIClient, tmp_path: Path
    ) -> None:
        not_a_dir = tmp_path / "missing"
        picsonym = make_picsonym(fake_client)
        with pytest.raises(NotADirectoryError):
            picsonym.rename_folder(not_a_dir)
