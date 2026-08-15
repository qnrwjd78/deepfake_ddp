#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def check_png(path_string):
    try:
        with Image.open(path_string) as image:
            image.verify()
        with Image.open(path_string) as image:
            image.load()
        return None
    except Exception as exc:
        return path_string, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data_precrop")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--output", default="bad_png_files.txt")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"directory does not exist: {root}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = sorted(str(path) for path in root.rglob("*.png"))
    total = len(paths)
    print(f"Found {total:,} PNG files under {root}", flush=True)

    bad = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(check_png, paths, chunksize=64)
        for checked, result in enumerate(results, 1):
            if result is not None:
                bad.append(result)
                print(f"BAD  {result[0]}  ({result[1]})", flush=True)
            if checked % 10000 == 0 or checked == total:
                print(
                    f"Checked {checked:,}/{total:,} | bad={len(bad):,}",
                    flush=True,
                )

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as report:
        for path, error in bad:
            report.write(f"{path}\t{error}\n")

    print(f"Done: checked={total:,}, bad={len(bad):,}")
    print(f"Report: {output.resolve()}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
