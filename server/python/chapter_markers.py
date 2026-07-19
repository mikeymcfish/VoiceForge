#!/usr/bin/env python3
"""Shared chapter-marker parsing and sample-accurate timing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


CHAPTER_MARKER_PATTERN = re.compile(r"\[chapter\]", flags=re.IGNORECASE)
CHAPTER_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ChapterSection:
    """One marker-delimited block before engine-specific text splitting."""

    text: str
    starts_chapter: bool = False
    chapter_title: Optional[str] = None


@dataclass(frozen=True)
class ChapterSegment:
    """One engine-sized spoken segment with optional chapter metadata."""

    text: str
    starts_chapter: bool = False
    chapter_title: Optional[str] = None


@dataclass(frozen=True)
class AudioMergeTiming:
    """Exact sample positions and preceding gaps for merged audio segments."""

    start_samples: tuple[int, ...]
    gap_samples_before: tuple[int, ...]
    total_samples: int


def sanitize_chapter_title(value: object) -> Optional[str]:
    """Normalize a metadata title without changing the spoken segment text."""

    if value is None:
        return None
    normalized = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value))
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    return normalized or None


def split_chapter_sections(raw_text: str) -> list[ChapterSection]:
    """Split text on case-insensitive ``[CHAPTER]`` markers.

    The marker is control syntax and is removed. Text after a marker remains in
    the chapter section, including an inline title, so that title is spoken.
    Only text on the same line as the marker is treated as chapter metadata.
    """

    normalized = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(CHAPTER_MARKER_PATTERN.finditer(normalized))
    if not matches:
        plain_text = normalized.strip()
        return [ChapterSection(plain_text)] if plain_text else []

    sections: list[ChapterSection] = []
    preface = normalized[: matches[0].start()].strip()
    if preface:
        sections.append(ChapterSection(preface))

    for index, match in enumerate(matches):
        section_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(normalized)
        )
        raw_section = normalized[match.end() : section_end]
        spoken_text = raw_section.strip()
        if not spoken_text:
            continue
        inline_title = raw_section.split("\n", 1)[0]
        sections.append(
            ChapterSection(
                text=spoken_text,
                starts_chapter=True,
                chapter_title=sanitize_chapter_title(inline_title),
            )
        )
    return sections


def split_marked_text(
    raw_text: str,
    split_section: Callable[[str], Sequence[str]],
) -> list[ChapterSegment]:
    """Apply an engine splitter independently to each marker-delimited block."""

    segments: list[ChapterSegment] = []
    for section in split_chapter_sections(raw_text):
        section_segments = [
            str(segment).strip()
            for segment in split_section(section.text)
            if str(segment).strip()
        ]
        for index, text in enumerate(section_segments):
            segments.append(
                ChapterSegment(
                    text=text,
                    starts_chapter=section.starts_chapter and index == 0,
                    chapter_title=(
                        section.chapter_title
                        if section.starts_chapter and index == 0
                        else None
                    ),
                )
            )
    return segments


def build_audio_merge_timing(
    segment_sample_counts: Sequence[int],
    sample_rate: int,
    gap_ms: int,
    chapter_segment_indices: Sequence[int] = (),
    chapter_pause_ms: int = 0,
) -> AudioMergeTiming:
    """Build exact segment starts for base gaps plus extra chapter gaps.

    Chapter indices are zero-based. No silence is inserted before the first
    segment, even when it begins a chapter.
    """

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(gap_ms, bool) or not isinstance(gap_ms, int) or gap_ms < 0:
        raise ValueError("gap_ms must be a non-negative integer")
    if (
        isinstance(chapter_pause_ms, bool)
        or not isinstance(chapter_pause_ms, int)
        or chapter_pause_ms < 0
    ):
        raise ValueError("chapter_pause_ms must be a non-negative integer")

    counts: list[int] = []
    for raw_count in segment_sample_counts:
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise ValueError("segment sample counts must be positive integers")
        counts.append(raw_count)
    if not counts:
        raise ValueError("at least one audio segment is required")

    chapter_indices: set[int] = set()
    for raw_index in chapter_segment_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("chapter segment indices must be integers")
        if raw_index < 0 or raw_index >= len(counts):
            raise ValueError("chapter segment index is outside the segment list")
        chapter_indices.add(raw_index)

    base_gap_samples = int(sample_rate * gap_ms / 1000)
    chapter_gap_samples = int(sample_rate * chapter_pause_ms / 1000)
    starts: list[int] = []
    gaps: list[int] = []
    cursor = 0
    for index, sample_count in enumerate(counts):
        gap = 0
        if index > 0:
            gap = base_gap_samples
            if index in chapter_indices:
                gap += chapter_gap_samples
        cursor += gap
        gaps.append(gap)
        starts.append(cursor)
        cursor += sample_count

    return AudioMergeTiming(
        start_samples=tuple(starts),
        gap_samples_before=tuple(gaps),
        total_samples=cursor,
    )


def build_chapter_manifest(
    segments: Sequence[ChapterSegment],
    start_samples: Sequence[int],
    sample_rate: int,
    total_samples: int,
) -> dict[str, Any]:
    """Create and validate the JSON contract consumed by audio packaging."""

    if len(segments) != len(start_samples):
        raise ValueError("segments and start_samples must have the same length")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if (
        isinstance(total_samples, bool)
        or not isinstance(total_samples, int)
        or total_samples <= 0
    ):
        raise ValueError("total_samples must be a positive integer")
    chapters: list[dict[str, Any]] = []
    for segment, raw_start in zip(segments, start_samples):
        if not segment.starts_chapter:
            continue
        if isinstance(raw_start, bool) or not isinstance(raw_start, int):
            raise ValueError("chapter start samples must be integers")
        chapters.append(
            {
                "title": sanitize_chapter_title(segment.chapter_title),
                "start_sample": raw_start,
                "start_ms": raw_start * 1000.0 / sample_rate,
                "start_seconds": raw_start / float(sample_rate),
            }
        )
    manifest = {
        "version": CHAPTER_MANIFEST_VERSION,
        "sample_rate": sample_rate,
        "total_samples": total_samples,
        "chapters": chapters,
    }
    validate_chapter_manifest(manifest)
    return manifest


def validate_chapter_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject malformed or internally inconsistent timing manifests."""

    if set(manifest) != {"version", "sample_rate", "total_samples", "chapters"}:
        raise ValueError("chapter manifest has unexpected top-level fields")
    if manifest.get("version") != CHAPTER_MANIFEST_VERSION:
        raise ValueError("unsupported chapter manifest version")

    sample_rate = manifest.get("sample_rate")
    total_samples = manifest.get("total_samples")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("chapter manifest sample_rate must be a positive integer")
    if (
        isinstance(total_samples, bool)
        or not isinstance(total_samples, int)
        or total_samples <= 0
    ):
        raise ValueError("chapter manifest total_samples must be a positive integer")

    chapters = manifest.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("chapter manifest chapters must be a list")
    prior_start = -1
    required_fields = {"title", "start_sample", "start_ms", "start_seconds"}
    for chapter in chapters:
        if not isinstance(chapter, Mapping) or set(chapter) != required_fields:
            raise ValueError("chapter manifest entry has unexpected fields")
        title = chapter.get("title")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError("chapter title must be null or a non-empty string")
        start_sample = chapter.get("start_sample")
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or start_sample < 0
            or start_sample >= total_samples
        ):
            raise ValueError("chapter start_sample is outside the audio")
        if start_sample <= prior_start:
            raise ValueError("chapter start samples must be strictly increasing")
        prior_start = start_sample

        start_ms = chapter.get("start_ms")
        start_seconds = chapter.get("start_seconds")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, (int, float))
            or not math.isfinite(float(start_ms))
            or isinstance(start_seconds, bool)
            or not isinstance(start_seconds, (int, float))
            or not math.isfinite(float(start_seconds))
        ):
            raise ValueError("chapter time values must be finite numbers")
        if not math.isclose(
            float(start_ms),
            start_sample * 1000.0 / sample_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("chapter start_ms does not match start_sample")
        if not math.isclose(
            float(start_seconds),
            start_sample / float(sample_rate),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("chapter start_seconds does not match start_sample")


def write_chapter_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Validate and atomically write a UTF-8 chapter timing manifest."""

    validate_chapter_manifest(manifest)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
        temporary_path.replace(destination)
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
