# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Command line interface for :mod:`picsonym`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import __version__
from ._client import DEFAULT_BASE_URL
from .backends import DEFAULT_TEMPERATURE
from .renaming import Picsonym, RenameResult

__all__ = ["main"]


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model name to use (must be vision-capable for --image/rename)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "sampling temperature; omitted by default so the model uses its own default"
        ),
    )
    return parser


def _build_parser() -> argparse.ArgumentParser:
    common = _common_parser()

    parser = argparse.ArgumentParser(
        prog="picsonym",
        description="Generate evocative, artistic titles for images.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    title_parser = subparsers.add_parser(
        "title", parents=[common], help="generate a title"
    )
    source = title_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", metavar="TEXT", help="image-generation prompt")
    source.add_argument("--image", metavar="PATH", type=Path, help="image file")

    rename_parser = subparsers.add_parser(
        "rename", parents=[common], help="rename image files to generated titles"
    )
    rename_parser.add_argument(
        "paths", metavar="PATH", nargs="+", type=Path, help="image file or folder"
    )
    rename_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="also rename images in subfolders",
    )

    return parser


def _run_title(args: argparse.Namespace) -> int:
    picsonym = Picsonym(
        model=args.model, base_url=args.base_url, temperature=args.temperature
    )
    try:
        if args.prompt is not None:
            title = picsonym.title_from_prompt(args.prompt)
        else:
            title = picsonym.title_from_image(args.image)
    except Exception as exc:  # reported to the user, not a crash
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(title)
    return 0


def _describe(result: RenameResult) -> str:
    if result.error is not None:
        return f"{result.original} -> error: {result.error}"
    return f"{result.original} -> {result.renamed}"


def _run_rename(args: argparse.Namespace) -> int:
    picsonym = Picsonym(
        model=args.model, base_url=args.base_url, temperature=args.temperature
    )

    def targets() -> Iterator[RenameResult]:
        for path in args.paths:
            if path.is_dir():
                yield from picsonym.rename_folder(path, recursive=args.recursive)
            elif path.is_file():
                yield picsonym.rename_file(path)
            else:
                yield RenameResult(path, None, None, "no such file or directory")

    results: list[RenameResult] = []
    for result in targets():
        print(_describe(result), flush=True)
        results.append(result)

    renamed = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    print(f"Renamed {renamed}, failed {failed} of {len(results)}.")

    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    :param argv: Arguments to parse; defaults to :data:`sys.argv` when None.
    :returns: 0 on full success; 1 if a `rename` target failed or `title`
        raised; 2 on an argparse usage error.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "title":
        return _run_title(args)
    return _run_rename(args)
