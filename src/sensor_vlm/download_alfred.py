from __future__ import annotations

import argparse
from pathlib import Path

import py7zr
import requests
from tqdm import tqdm

from .paths import DATA_DIR, ensure_project_dirs


ALFRED_DOWNLOADS = {
    "json": "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z",
    "json_feat": "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_feat_2.1.0.7z",
    "full": "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z",
}


def download_file(url: str, output_path: str | Path, *, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with resume support."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing_bytes = output.stat().st_size if output.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    mode = "ab" if existing_bytes and response.status_code == 206 else "wb"
    if mode == "wb":
        existing_bytes = 0

    total = response.headers.get("content-length")
    total_bytes = int(total) + existing_bytes if total is not None else None
    with output.open(mode + "") as file_handle:
        with tqdm(
            total=total_bytes,
            initial=existing_bytes,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {output.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_handle.write(chunk)
                progress.update(len(chunk))
    return output


def extract_7z(archive_path: str | Path, output_dir: str | Path, *, remove_archive: bool = False) -> Path:
    archive = Path(archive_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as zip_file:
        zip_file.extractall(path=output)
    if remove_archive:
        archive.unlink()
    return output


def download_alfred(
    kind: str,
    *,
    output_dir: str | Path = DATA_DIR,
    extract: bool = True,
    remove_archive: bool = False,
) -> Path:
    if kind not in ALFRED_DOWNLOADS:
        raise ValueError(f"Unknown ALFRED kind '{kind}'. Choose one of {sorted(ALFRED_DOWNLOADS)}")

    ensure_project_dirs()
    output_root = Path(output_dir)
    archive_path = output_root / f"{kind}_2.1.0.7z"
    download_file(ALFRED_DOWNLOADS[kind], archive_path)
    if extract:
        extract_7z(archive_path, output_root, remove_archive=remove_archive)
        return output_root / f"{kind}_2.1.0"
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and extract ALFRED datasets.")
    parser.add_argument(
        "kind",
        choices=sorted(ALFRED_DOWNLOADS),
        help="json is metadata only, json_feat has ResNet features, full includes raw images.",
    )
    parser.add_argument("--output-dir", default=DATA_DIR)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--remove-archive", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = download_alfred(
        args.kind,
        output_dir=args.output_dir,
        extract=not args.no_extract,
        remove_archive=args.remove_archive,
    )
    print(f"ALFRED {args.kind} ready at: {path}")


if __name__ == "__main__":
    main()

