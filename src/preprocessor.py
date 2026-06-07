"""
Pre-processing: lossless FFmpeg concat merge of multi-part raw flight videos.

Merges consecutive segments from ``data/raw/{folder_name}/`` into a single
``data/merged/{folder_name}/{folder_name}_full.mp4`` using the concat demuxer
with stream copy (no re-encode).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MERGED_ROOT = PROJECT_ROOT / "data" / "merged"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV", ".AVI", ".MKV", ".M4V"}

logger = logging.getLogger(__name__)


class PreprocessorError(Exception):
    """Raised when folder validation or FFmpeg merge fails."""


def _natural_sort_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def resolve_raw_folder(folder_name: str) -> Path:
    """
    Resolve and validate a raw input folder under ``data/raw/``.

    Accepts a bare folder name (``flight_01``) or a path such as
    ``data/raw/flight_01``.
    """
    candidate = Path(folder_name)
    if candidate.is_dir():
        return candidate.resolve()

    direct = RAW_ROOT / folder_name
    if direct.is_dir():
        return direct.resolve()

    if candidate.name and (RAW_ROOT / candidate.name).is_dir():
        return (RAW_ROOT / candidate.name).resolve()

    raise PreprocessorError(
        f"Raw input folder not found: '{folder_name}'. "
        f"Expected a directory under {RAW_ROOT.resolve()}"
    )


def discover_video_parts(raw_folder: Path) -> List[Path]:
    """Return video files in ``raw_folder``, sorted in natural part order."""
    if not raw_folder.is_dir():
        raise PreprocessorError(f"Not a directory: {raw_folder}")

    parts = [
        p.resolve()
        for p in raw_folder.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS
    ]
    if not parts:
        raise PreprocessorError(
            f"No video files found in {raw_folder}. "
            f"Supported extensions: {', '.join(sorted({e.lower() for e in VIDEO_EXTENSIONS}))}"
        )
    return sorted(parts, key=_natural_sort_key)


def merged_output_path(folder_name: str) -> Path:
    """Target path for the concatenated full video."""
    return MERGED_ROOT / folder_name / f"{folder_name}_full.mp4"


def _escape_concat_path(path: Path) -> str:
    """Escape a path for FFmpeg concat demuxer ``file '...'`` lines."""
    return str(path.resolve()).replace("'", "'\\''")


def _write_concat_list(video_paths: Sequence[Path], list_path: Path) -> None:
    lines = [f"file '{_escape_concat_path(p)}'" for p in video_paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreprocessorError(
            "ffmpeg not found on PATH. Install ffmpeg to run the merge step."
        )
    return ffmpeg


def merge_videos_copy(
    video_paths: Sequence[Path],
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """
    Concatenate videos with FFmpeg concat demuxer and ``-c copy``.

    Args:
        video_paths: Ordered list of source files.
        output_path: Destination ``.mp4`` path.
        force: Re-run merge even if ``output_path`` already exists.

    Returns:
        Resolved path to the merged output file.
    """
    if not video_paths:
        raise PreprocessorError("merge_videos_copy requires at least one input file")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Merged video already exists, skipping FFmpeg: %s", output_path)
        return output_path

    ffmpeg = _ensure_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="pedestrian_concat_") as tmp_dir:
        list_file = Path(tmp_dir) / "inputs.txt"
        _write_concat_list(video_paths, list_file)

        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
        ]
        logger.info("Running FFmpeg concat (stream copy): %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as err:
            raise PreprocessorError(f"Failed to execute ffmpeg: {err}") from err

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise PreprocessorError(
                f"FFmpeg concat failed (exit {result.returncode}).\n{stderr}"
            )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise PreprocessorError(f"FFmpeg produced no output at {output_path}")

    logger.info(
        "Merge complete: %d parts -> %s (%.1f MB)",
        len(video_paths),
        output_path,
        output_path.stat().st_size / (1024 * 1024),
    )
    return output_path


def preprocess_raw_folder(
    folder_name: str,
    *,
    force_merge: bool = False,
) -> Path:
    """
    Validate ``data/raw/{folder_name}/``, merge parts, return merged video path.

    Workflow:
        1. Resolve and check raw folder.
        2. Create ``data/merged/{folder_name}/``.
        3. Concat parts to ``{folder_name}_full.mp4`` via FFmpeg copy merge.
    """
    raw_folder = resolve_raw_folder(folder_name)
    folder_key = raw_folder.name

    parts = discover_video_parts(raw_folder)
    logger.info(
        "Found %d video part(s) in %s: %s",
        len(parts),
        raw_folder,
        ", ".join(p.name for p in parts),
    )

    output_path = merged_output_path(folder_key)
    return merge_videos_copy(parts, output_path, force=force_merge)
