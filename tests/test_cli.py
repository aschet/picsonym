# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Tests for the command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from _pytest.capture import CaptureFixture

from picsonym.backends import DEFAULT_TEMPERATURE
from picsonym.cli import main
from picsonym.renaming import RenameResult


class FakePicsonym:
    """Stand-in for :class:`picsonym.renaming.Picsonym`, records its args."""

    last_init_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        FakePicsonym.last_init_kwargs = kwargs

    def title_from_prompt(self, prompt: str) -> str:
        return "Quiet Morning Light"

    def title_from_image(self, image: Any, **_: Any) -> str:
        return "Quiet Morning Light"

    def rename_file(self, path: Path) -> RenameResult:
        path = Path(path)
        renamed = path.with_name("Quiet Morning Light.png")
        return RenameResult(path, renamed, "Quiet Morning Light", None)

    def rename_folder(
        self, folder: Path, *, recursive: bool = False
    ) -> list[RenameResult]:
        return [
            RenameResult(
                Path(folder) / "a.png",
                Path(folder) / "Quiet Morning Light.png",
                "Quiet Morning Light",
                None,
            )
        ]


@pytest.fixture(autouse=True)
def patch_picsonym(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("picsonym.cli.Picsonym", FakePicsonym)


def test_title_from_prompt(capsys: CaptureFixture[str]) -> None:
    assert main(["title", "--model", "test-model", "--prompt", "a scene"]) == 0
    assert capsys.readouterr().out == "Quiet Morning Light\n"


def test_title_from_image(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"data")
    assert main(["title", "--model", "test-model", "--image", str(image)]) == 0
    assert capsys.readouterr().out == "Quiet Morning Light\n"


def test_title_requires_prompt_or_image() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["title", "--model", "test-model"])
    assert excinfo.value.code == 2


def test_model_is_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["title", "--prompt", "a scene"])
    assert excinfo.value.code == 2


def test_rename_single_file(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    image = tmp_path / "IMG_1.png"
    image.write_bytes(b"data")

    assert main(["rename", "--model", "test-model", str(image)]) == 0

    out = capsys.readouterr().out
    assert "->" in out
    assert "Renamed 1, failed 0 of 1." in out


def test_rename_folder(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["rename", "--model", "test-model", str(tmp_path)]) == 0
    assert "Renamed 1, failed 0 of 1." in capsys.readouterr().out


def test_rename_missing_path_is_reported_as_failure(
    capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.png"
    assert main(["rename", "--model", "test-model", str(missing)]) == 1
    out = capsys.readouterr().out
    assert "error" in out
    assert "Renamed 0, failed 1 of 1." in out


def test_rename_forwards_flags_to_picsonym_and_constructor(tmp_path: Path) -> None:
    image = tmp_path / "IMG_1.png"
    image.write_bytes(b"data")

    main(
        [
            "rename",
            "--model",
            "my-model",
            "--base-url",
            "http://example.invalid/v1",
            "--recursive",
            str(image),
        ]
    )

    assert FakePicsonym.last_init_kwargs == {
        "model": "my-model",
        "base_url": "http://example.invalid/v1",
        "temperature": DEFAULT_TEMPERATURE,
    }


def test_temperature_flag_is_forwarded(tmp_path: Path) -> None:
    image = tmp_path / "IMG_1.png"
    image.write_bytes(b"data")

    main(["rename", "--model", "my-model", "--temperature", "1.1", str(image)])

    assert FakePicsonym.last_init_kwargs["temperature"] == 1.1


def test_version(capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    from picsonym import __version__

    assert __version__ in capsys.readouterr().out


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
