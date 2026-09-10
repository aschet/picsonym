# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Renaming image files to their generated titles."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from .filenames import IMAGE_EXTENSIONS, resolve_collision, sanitize_filename
from .titling import _TitleGenerator

__all__ = ["Picsonym", "RenameResult"]


class RenameResult(NamedTuple):
    """The outcome of renaming (or attempting to rename) one image file."""

    #: The file's path before this operation.
    original: Path
    #: The file's new path; None if renaming failed.
    renamed: Path | None
    #: The generated title, if title generation ran and succeeded.
    title: str | None
    #: A human-readable failure reason, or None if there was no failure.
    error: str | None


class Picsonym(_TitleGenerator):
    """Generates artistic titles for images and renames files to them."""

    def rename_file(self, path: str | os.PathLike[str]) -> RenameResult:
        """Rename one image file to its generated title.

        :param path: Image file path.
        :returns: A :class:`RenameResult`. Never raises for expected
            failures (title-generation error, unsupported image format,
            filesystem error) — these are reported via `.error`, matching
            :meth:`rename_folder`'s contract so callers handle both
            uniformly.
        """
        path = Path(path)

        try:
            title = self.title_from_image(path)
        except Exception as exc:  # reported, not raised
            return RenameResult(path, None, None, str(exc))

        sanitized = sanitize_filename(title, extension=path.suffix)
        candidate = resolve_collision(path.with_name(sanitized))

        try:
            path.rename(candidate)
        except OSError as exc:
            return RenameResult(path, None, title, str(exc))

        return RenameResult(path, candidate, title, None)

    def rename_folder(
        self, folder: str | os.PathLike[str], *, recursive: bool = False
    ) -> Iterator[RenameResult]:
        """Rename every image file inside `folder`.

        Results are yielded lazily, one per file as it finishes, rather
        than computed all at once — a caller printing each result as it
        arrives gets live progress instead of silence until the whole
        folder is done. Wrap in `list(...)` for a materialized list.

        :param folder: Directory to scan.
        :param recursive: False (default) — only immediate children;
            True — descend into subfolders too, skipping any subfolder
            whose name starts with a dot (e.g. ``.git``, ``.venv``).
        :returns: One :class:`RenameResult` per file whose suffix
            (case-insensitive) is in :data:`picsonym.filenames.IMAGE_EXTENSIONS`,
            in sorted-path order. A per-file failure (including an
            unexpected exception) is captured in its result and does not
            abort the remaining files.
        :raises NotADirectoryError: `folder` is not an existing directory
            — raised immediately, not deferred to first iteration.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder!r} is not a directory")
        return self._rename_images_in(folder, recursive=recursive)

    def _rename_images_in(
        self, folder: Path, *, recursive: bool
    ) -> Iterator[RenameResult]:
        candidates = folder.rglob("*") if recursive else folder.iterdir()
        images = sorted(
            path
            for path in candidates
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(
                part.startswith(".") for part in path.relative_to(folder).parts[:-1]
            )
        )

        for path in images:
            try:
                yield self.rename_file(path)
            except Exception as exc:  # reported, not raised
                yield RenameResult(path, None, None, str(exc))
