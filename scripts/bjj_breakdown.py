#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import statistics
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np

ALGO_VERSION = 3            # bumped 2->3 in M6: transcript reconstruction changed (see §7 amendments)
SCHEMA_VERSION = 1
STEPS_SCHEMA_VERSION = 2    # steps.json only (agent-written; v2 adds one_thing + explanation)
TRANSCRIPT_SEG_OVERLAP_FRAC = 0.33   # show segment under a step if >=33% of the segment overlaps it
TRANSCRIPT_STEP_OVERLAP_FRAC = 0.5   # ...or if the overlap covers >=50% of the step
ONE_THING_MAX_CHARS = 300
EXPLANATION_MAX_CHARS = 700
TECHNIQUE_TITLE_MAX_CHARS = 80     # techniques[].title (M8)
TECHNIQUE_SUMMARY_MAX_CHARS = 300  # techniques[].summary (M8)
DEMO_DRILL_FRAMES_WARN_BELOW = 2   # check warns when a demonstration/drill step has fewer picks (M9)
DEMO_DRILL_MEDIAN_TARGET = 3       # ...and when the median across demo/drill steps falls below this (M9)
CJK_TEXT_FRAC = 0.3        # tokenize grounding-lint text as char bigrams when >30% of its non-space chars are CJK
YOUTUBE_HOSTS = {"youtube.com", "youtu.be"}   # timestamped_url hosts, after stripping one leading "www." or "m." (M10)
BILIBILI_HOSTS = {"bilibili.com"}             # (M10)
DEFAULT_SCENE_THRESHOLD = 0.3
DEFAULT_MIN_SCENE_GAP = 10.0
UNIFORM_SAMPLES_PER_STEP = 8
MAX_CANDIDATES_PER_STEP = 12
MIN_CANDIDATES_PER_STEP = 4
NEAR_CUT_EXCLUSION = 1.0
BLUR_REL_THRESHOLD = 0.5
BLUR_ABS_FLOOR = 20.0
DHASH_HAMMING_DUP = 4
THUMB_W, THUMB_H = 960, 540
SHEET_CELL_W, SHEET_CELL_H = 480, 270
SHEET_COLS = 4
SHEET_LABEL_H = 36
OCR_CONF_MIN = 65.0

CATEGORY_STYLES = {
    "intro": {"icon": "🎬", "label": "Introduction", "accent": "#60a5fa"},
    "concept": {"icon": "💡", "label": "Core Concept", "accent": "#fbbf24"},
    "demonstration": {"icon": "🥋", "label": "Demonstration", "accent": "#34d399"},
    "drill": {"icon": "🔄", "label": "Drill", "accent": "#f472b6"},
    "recap": {"icon": "📝", "label": "Recap", "accent": "#a78bfa"},
}

VALID_CATEGORIES = set(CATEGORY_STYLES)


class ValidationError(Exception):
    def __init__(self, source_name: str, path: str, expected: str, actual: Any = None, message: Optional[str] = None):
        self.source_name = source_name
        self.path = path
        self.expected = expected
        self.actual = actual
        if message is not None:
            full_message = message
        elif actual is None:
            full_message = f"{source_name}: {path}: expected {expected}"
        else:
            full_message = f"{source_name}: {path}: expected {expected}, got {actual!r}"
        super().__init__(full_message)


def resolve_tool_path(env_var: str, tool_name: str) -> Optional[str]:
    override = os.environ.get(env_var)
    if override:
        return override
    return shutil.which(tool_name)


def ensure_required_tools() -> Tuple[str, str, Optional[str]]:
    ffmpeg = resolve_tool_path("BJJ_FFMPEG", "ffmpeg")
    ffprobe = resolve_tool_path("BJJ_FFPROBE", "ffprobe")
    tesseract = resolve_tool_path("BJJ_TESSERACT", "tesseract")
    missing = []
    if not ffmpeg:
        missing.append("ffmpeg")
    if not ffprobe:
        missing.append("ffprobe")
    if missing:
        print(
            "Error: required tool(s) not found: "
            + ", ".join(missing)
            + ". Set BJJ_FFMPEG / BJJ_FFPROBE or install them on PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not tesseract:
        print("Notice: tesseract not found; OCR title-card filtering will be skipped.", file=sys.stderr)
    return ffmpeg, ffprobe, tesseract


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_fraction(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        num_f = float(num)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return num_f / den_f
    return float(value)


def parse_time_str(t_str: str) -> float:
    parts = t_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(t_str)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def strip_video_suffix(video_base: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", video_base).strip()


def clean_vtt_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_word(word: str) -> str:
    return word.lower().translate(str.maketrans("", "", string.punctuation))


def find_overlap_words(words1: Sequence[str], words2: Sequence[str]) -> int:
    norm1 = [normalize_word(w) for w in words1]
    norm2 = [normalize_word(w) for w in words2]
    max_len = min(len(norm1), len(norm2))
    for size in range(max_len, 0, -1):
        if norm1[-size:] == norm2[:size]:
            return size
    return 0


STANDALONE_SINGLE_LETTER_WORDS = {"a", "i", "o"}


def repair_midword_splits(segments: Sequence[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """M6 amendment (§7): reconstruct_vtt can split a word across two output segments when
    overlap matching fails on a partial word - e.g. one segment ends "Okay, t" and the next
    begins "his shoulder" (the word was "this"). Repair pass: if segment N's final token is a
    SINGLE alphabetic letter with no trailing punctuation, that letter is not a standalone word
    (a, A, i, I, o, O), and segment N+1's text starts with a lowercase letter, move the letter
    off segment N and prepend it to segment N+1 joined without a space. Deliberately narrow:
    multi-letter fragments are left alone (rare, and a wider heuristic risks gluing real words
    together - e.g. a segment legitimately ending in "to")."""
    repaired = [list(seg) for seg in segments]
    for i in range(len(repaired) - 1):
        text = repaired[i][2]
        next_text = repaired[i + 1][2]
        if not text or not next_text:
            continue
        words = text.split(" ")
        last_word = words[-1] if words else ""
        if len(last_word) != 1 or not last_word.isalpha():
            continue
        if last_word.lower() in STANDALONE_SINGLE_LETTER_WORDS:
            continue
        if not next_text[0].islower():
            continue
        repaired[i][2] = " ".join(words[:-1]).rstrip()
        repaired[i + 1][2] = last_word + next_text
    return [(start, end, text) for start, end, text in repaired]


def reconstruct_vtt(vtt_path: Path) -> List[Tuple[float, float, str]]:
    with vtt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks: List[Tuple[str, str, str]] = []
    current_time: Optional[Tuple[str, str]] = None
    current_text: List[str] = []
    time_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )

    stripped_lines = [raw_line.strip() for raw_line in lines]
    for i, line in enumerate(stripped_lines):
        if not line:
            continue
        match = time_pattern.search(line)
        if match:
            if current_time and current_text:
                txt = clean_vtt_text(" ".join(current_text))
                if txt:
                    blocks.append((current_time[0], current_time[1], txt))
            current_time = (match.group(1), match.group(2))
            current_text = []
        elif (
            "-->" not in line
            and i + 1 < len(stripped_lines)
            and time_pattern.search(stripped_lines[i + 1])
            and (i == 0 or not stripped_lines[i - 1])
        ):
            # WebVTT cue identifier: its own line directly above a timestamp line, after
            # a blank line (or at file start) - e.g. the numeric ids SRT->VTT converters
            # keep. Not caption text; without this check it gets glued onto the previous
            # cue. The blank-line requirement means real caption text can never match.
            continue
        elif current_time:
            if line != "WEBVTT" and not line.startswith("Kind:") and not line.startswith("Language:"):
                current_text.append(line)

    if current_time and current_text:
        txt = clean_vtt_text(" ".join(current_text))
        if txt:
            blocks.append((current_time[0], current_time[1], txt))

    if not blocks:
        return []

    reconstructed: List[Tuple[float, float, str]] = []
    start_sec = parse_time_str(blocks[0][0])
    end_sec = parse_time_str(blocks[0][1])
    reconstructed.append((start_sec, end_sec, blocks[0][2]))

    for next_start_str, next_end_str, next_text in blocks[1:]:
        next_start = parse_time_str(next_start_str)
        next_end = parse_time_str(next_end_str)
        prev_text = reconstructed[-1][2]
        words_prev = prev_text.split()
        words_next = next_text.split()
        overlap_size = find_overlap_words(words_prev, words_next)
        if overlap_size > 0:
            if overlap_size == len(words_next):
                reconstructed[-1] = (reconstructed[-1][0], next_end, prev_text)
            else:
                new_words = words_next[overlap_size:]
                merged_text = prev_text + " " + " ".join(new_words)
                reconstructed[-1] = (reconstructed[-1][0], next_end, merged_text)
        else:
            reconstructed.append((next_start, next_end, next_text))

    return repair_midword_splits(reconstructed)


_TXT_LINE_RE = re.compile(r"\[([0-9:]+\.[0-9]+)\s*-->\s*([0-9:]+\.[0-9]+)\]\s*(.*)")


def parse_transcript(transcript_path: Path) -> List[Tuple[float, float, str]]:
    if not transcript_path.exists():
        raise FileNotFoundError(str(transcript_path))
    if transcript_path.suffix.lower() == ".vtt":
        return reconstruct_vtt(transcript_path)
    # Clean-transcript .txt format: one "[HH:MM:SS.mmm --> HH:MM:SS.mmm] text" block per
    # line, already de-duplicated - no scrolling-line merge needed. (v1 had this branch;
    # it was lost in the M1 copy, which fed .txt files to reconstruct_vtt. That parser
    # treats every line as a timestamp line and discards same-line text, yielding zero
    # segments silently.)
    segments: List[Tuple[float, float, str]] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = _TXT_LINE_RE.search(line)
            if match:
                start_str, end_str, text = match.groups()
                text = text.strip()
                if not text:
                    continue
                try:
                    start_sec, end_sec = parse_time_str(start_str), parse_time_str(end_str)
                except ValueError:
                    # The loose [0-9:]+ pattern can match a malformed stamp (e.g. too many
                    # colon groups); skip the line rather than crash on it.
                    continue
                segments.append((start_sec, end_sec, text))
    return segments


def clean_text(raw_text: str) -> str:
    text = raw_text
    text = text.replace("&gt;&gt;", ">>")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    text = text.replace("[&nbsp;__&nbsp;]", "[***]")
    text = text.replace("&nbsp;", " ")
    # M6: strip literal '>>' speaker-change markers YouTube auto-captions insert, e.g.
    # ">> See, he was styling to keep holding the trouser." -> "See, he was styling ..."
    text = text.replace(">>", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_ocr_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.replace("\x0c", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"[^A-Za-z0-9'& -]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" -_'\".,:;!?")
    return text.strip()


def is_meaningful_ocr_title(text: str) -> bool:
    cleaned = clean_ocr_text(text)
    if not cleaned:
        return False
    tokens = cleaned.split()
    if len(cleaned) < 4 or len(cleaned) > 80:
        return False
    letters = sum(ch.isalpha() for ch in cleaned)
    digits = sum(ch.isdigit() for ch in cleaned)
    if letters < 3:
        return False
    if digits > letters and len(cleaned.split()) <= 2:
        return False
    if len(tokens) > 12:
        return False
    if len(cleaned.replace(" ", "")) < 4:
        return False
    short_tokens = [tok for tok in tokens if len(re.sub(r"[^A-Za-z0-9]", "", tok)) < 3]
    if len(short_tokens) > 1:
        return False
    generic_brand_words = {"project", "channel", "academy", "seminar", "logo", "studio"}
    lower_words = {word.lower() for word in tokens}
    if lower_words & generic_brand_words and len(cleaned.split()) <= 4:
        return False
    return True


def preprocess_ocr_image(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    white_ratio = float((thresh == 255).mean())
    if white_ratio <= 0.5:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def run_tesseract_ocr(image_path: Path, tesseract_path: str) -> Tuple[str, float]:
    cmd = [
        tesseract_path,
        str(image_path),
        "stdout",
        "--psm",
        "6",
        "-l",
        "eng",
        "tsv",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return "", 0.0
    words: List[str] = []
    confidences: List[float] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
            conf = float(parts[10])
        except ValueError:
            continue
        if level != 5 or conf < 0:
            continue
        word = parts[11].strip()
        if not word:
            continue
        if conf >= 40:
            words.append(word)
            confidences.append(conf)
    if not words:
        return "", 0.0
    return " ".join(words), sum(confidences) / len(confidences)


def get_video_duration(video_path: Path, ffprobe_path: str) -> float:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_command(cmd)
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
    return 0.0


def probe_video_stream_info(video_path: Path, ffprobe_path: str) -> Tuple[float, int, int]:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe did not return a video stream")
    stream = streams[0]
    fps = 0.0
    try:
        fps = parse_fraction(str(stream.get("r_frame_rate", "0")))
    except Exception:
        fps = 0.0
    if fps <= 0:
        fps = 29.97
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    return fps, width, height


def escape_lavfi_path(path: Path) -> str:
    escaped = str(path)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace(",", "\\,")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("[", "\\[")
    escaped = escaped.replace("]", "\\]")
    return escaped


def run_scene_detection(video_path: Path, threshold: float, ffprobe_path: str) -> List[Tuple[float, float]]:
    escaped_path = escape_lavfi_path(video_path)
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_frames",
        "-of",
        "compact=p=0",
        "-f",
        "lavfi",
        f"movie={escaped_path},select=gt(scene\\,{threshold})",
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe scene detection failed for {video_path}")
    time_re = re.compile(r"best_effort_timestamp_time=([0-9\.]+)")
    score_re = re.compile(r"tag:lavfi\.scene_score=([0-9\.]+)")
    scenes = []
    for line in result.stdout.splitlines():
        time_match = time_re.search(line)
        score_match = score_re.search(line)
        if time_match and score_match:
            scenes.append((float(time_match.group(1)), float(score_match.group(1))))
    return sorted(scenes, key=lambda item: item[0])


def filter_scenes(raw_scenes: Sequence[Tuple[float, float]], min_duration: float = DEFAULT_MIN_SCENE_GAP) -> List[Tuple[float, float]]:
    if not raw_scenes:
        return []
    filtered: List[Tuple[float, float]] = []
    for time_value, score in raw_scenes:
        if not filtered or (time_value - filtered[-1][0] >= min_duration):
            filtered.append((time_value, score))
        else:
            if score > filtered[-1][1] * 1.5:
                filtered[-1] = (time_value, score)
    return filtered


def build_work_dir(video_path: Path) -> Path:
    return video_path.parent / f"{video_path.stem}_review"


def ensure_work_dir(video_path: Path) -> Path:
    work_dir = build_work_dir(video_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


SUBTITLE_EXTENSIONS = ("txt", "vtt")   # priority within a tie: txt beats vtt (legacy order)


def match_subtitle_filename(filename: str, video_base: str) -> Optional[Tuple[Optional[str], str]]:
    """Pure match: does `filename` look like `<video_base>[.<lang>].txt|.vtt`?

    Returns (lang_or_None, ext) on a match, None otherwise. Deliberately does NOT use
    Path.glob/fnmatch: `video_base` can contain `[` and `]`, which are glob metacharacters
    (see spec §15) - callers must iterate the directory and compare names directly, as
    this function does.
    """
    prefix = video_base + "."
    if not filename.startswith(prefix):
        return None
    rest = filename[len(prefix):]
    if rest in SUBTITLE_EXTENSIONS:
        return None, rest
    for ext in SUBTITLE_EXTENSIONS:
        suffix = "." + ext
        if rest.endswith(suffix):
            lang = rest[: -len(suffix)]
            if lang:
                return lang, ext
    return None


def subtitle_priority(lang: Optional[str]) -> int:
    """Priority order per spec §5: (1) exact `.en`, (2) any other `.en*` tag,
    (3) bare (no tag - Whisper CLI output), (4) any other language tag."""
    if lang is None:
        return 3
    if lang == "en":
        return 1
    if lang.startswith("en"):
        return 2
    return 4


def find_subtitle_candidates(video_dir: Path, video_base: str) -> List[Tuple[Path, Optional[str], str]]:
    """Iterate the directory (never glob - see match_subtitle_filename) and return every
    sibling file matching `<video_base>[.<lang>].txt|.vtt` as (path, lang_or_None, ext)."""
    candidates = []
    for entry in sorted(video_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        match = match_subtitle_filename(entry.name, video_base)
        if match is not None:
            lang, ext = match
            candidates.append((entry, lang, ext))
    return candidates


def rank_subtitle_candidates(
    candidates: Sequence[Tuple[Path, Optional[str], str]]
) -> List[Tuple[Path, Optional[str], str]]:
    """Sort by priority (§5), then .txt-before-.vtt within a priority, then filename
    (alphabetically first) for deterministic tie-break."""
    ext_rank = {ext: i for i, ext in enumerate(SUBTITLE_EXTENSIONS)}
    return sorted(candidates, key=lambda c: (subtitle_priority(c[1]), ext_rank[c[2]], c[0].name))


def choose_subtitle_file(video_dir: Path, video_base: str) -> Tuple[Optional[Path], List[Path]]:
    """Pure: returns (best_path_or_None, other_candidates_in_priority_order)."""
    candidates = find_subtitle_candidates(video_dir, video_base)
    if not candidates:
        return None, []
    ranked = rank_subtitle_candidates(candidates)
    return ranked[0][0], [c[0] for c in ranked[1:]]


def subtitle_path_from_video(video_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    video_dir = video_path.parent
    video_base = video_path.stem
    chosen, others = choose_subtitle_file(video_dir, video_base)
    if chosen is None:
        raise FileNotFoundError(
            f"No subtitle file found next to {video_path.name} in {video_dir}. "
            f"Looked for (in priority order): {video_base}.en.txt / {video_base}.en.vtt; "
            f"{video_base}.<lang>.txt / {video_base}.<lang>.vtt (any en* tag); "
            f"{video_base}.txt / {video_base}.vtt (bare); "
            f"{video_base}.<lang>.txt / {video_base}.<lang>.vtt (any other tag). "
            "See SKILL.md Stage 0 (yt-dlp downloads subtitles for YouTube/Bilibili URLs) "
            "and Stage 1.5 (Whisper generates a .vtt when no subtitles exist)."
        )
    if others:
        print(
            f"Multiple subtitle candidates found for {video_base}; using {chosen.name} "
            f"(also found: {', '.join(o.name for o in others)})"
        )
    return chosen.resolve()


def resolve_source_url(existing_url: Optional[str], url_arg: Optional[str]) -> Optional[str]:
    """Pure store/preserve/clear resolution for `prepare --url` (§5, M10). `url_arg` is the
    raw CLI value: None means the flag was omitted (preserve whatever meta.json already has,
    so a cache-hit re-run never silently drops a stored link), "" means an explicit clear,
    and anything else must start with http:// or https:// and replaces the stored value."""
    if url_arg is None:
        return existing_url
    if url_arg == "":
        return None
    if not (url_arg.startswith("http://") or url_arg.startswith("https://")):
        raise ValueError(
            f"--url must start with http:// or https:// (got {url_arg!r}). Pass the video's "
            'online source URL, or --url "" to remove a stored source_url.'
        )
    return url_arg


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(
            path.name,
            "file",
            f"{path} to exist. Run the pipeline stage that writes it first (see SKILL.md), "
            "or check that --video points at the right file",
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(path.name, f"json[{exc.lineno}:{exc.colno}]", "valid JSON", exc.msg) from exc


def load_meta(work_dir: Path) -> Dict[str, Any]:
    return load_json(work_dir / "meta.json")


def load_steps(work_dir: Path) -> Dict[str, Any]:
    return load_json(work_dir / "steps.json")


def load_candidates(work_dir: Path) -> Dict[str, Any]:
    return load_json(work_dir / "candidates.json")


def load_approved(work_dir: Path) -> Dict[str, Any]:
    return load_json(work_dir / "approved_frames.json")


def validate_schema_header(data: Dict[str, Any], source_name: str, require_algo: bool = False) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(source_name, "schema_version", f"{SCHEMA_VERSION}", data.get("schema_version"))
    if require_algo and data.get("algo_version") != ALGO_VERSION:
        raise ValidationError(source_name, "algo_version", f"{ALGO_VERSION}", data.get("algo_version"))


EXPLANATION_REQUIRED_CATEGORIES = {"concept", "demonstration", "drill"}


def validate_steps_data(data: Dict[str, Any], meta: Dict[str, Any], source_name: str = "steps.json") -> List[Dict[str, Any]]:
    schema_version = data.get("schema_version")
    if schema_version == 1:
        raise ValidationError(
            source_name,
            "schema_version",
            f"{STEPS_SCHEMA_VERSION}",
            schema_version,
            message=(
                f'{source_name} is schema_version 1; write schema_version {STEPS_SCHEMA_VERSION}, which adds a '
                'top-level "one_thing" and per-step "explanation" fields - see SKILL.md §Segmentation'
            ),
        )
    if schema_version != STEPS_SCHEMA_VERSION:
        raise ValidationError(source_name, "schema_version", f"{STEPS_SCHEMA_VERSION}", schema_version)
    if data.get("video_base") != meta.get("video_base"):
        raise ValidationError(source_name, "video_base", repr(meta.get("video_base")), data.get("video_base"))
    one_thing = data.get("one_thing")
    if not isinstance(one_thing, str) or not one_thing.strip():
        raise ValidationError(source_name, "one_thing", "a non-empty string", one_thing)
    if len(one_thing) > ONE_THING_MAX_CHARS:
        raise ValidationError(source_name, "one_thing", f"{ONE_THING_MAX_CHARS} characters or fewer", len(one_thing))
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValidationError(source_name, "steps", "a non-empty list")

    # M8 (§6.4): optional top-level `techniques` list, paired with a per-step `technique`
    # field. Both absent is the common (single-technique) case and must validate exactly
    # as before. When present: sequential ids, length-capped title/summary, and (checked
    # per-step below) every step tagged with an existing id, contiguous & in list order.
    techniques_raw = data.get("techniques")
    techniques_present = techniques_raw is not None
    technique_ids: List[str] = []
    technique_index_by_id: Dict[str, int] = {}
    technique_step_counts: Dict[str, int] = {}
    if techniques_present:
        if not isinstance(techniques_raw, list) or not techniques_raw:
            raise ValidationError(source_name, "techniques", "a non-empty list", techniques_raw)
        for j, tech in enumerate(techniques_raw):
            tech_path = f"techniques[{j}]"
            if not isinstance(tech, dict):
                raise ValidationError(source_name, tech_path, "an object", tech)
            expected_tech_id = f"tech_{j + 1:02d}"
            if tech.get("id") != expected_tech_id:
                raise ValidationError(source_name, f"{tech_path}.id", repr(expected_tech_id), tech.get("id"))
            tech_title = tech.get("title")
            if not isinstance(tech_title, str) or not tech_title.strip():
                raise ValidationError(source_name, f"{tech_path}.title", "a non-empty string", tech_title)
            if len(tech_title) > TECHNIQUE_TITLE_MAX_CHARS:
                raise ValidationError(
                    source_name, f"{tech_path}.title", f"{TECHNIQUE_TITLE_MAX_CHARS} characters or fewer", len(tech_title)
                )
            tech_summary = tech.get("summary")
            if tech_summary is not None:
                if not isinstance(tech_summary, str) or not tech_summary.strip():
                    raise ValidationError(source_name, f"{tech_path}.summary", "a non-empty string", tech_summary)
                if len(tech_summary) > TECHNIQUE_SUMMARY_MAX_CHARS:
                    raise ValidationError(
                        source_name,
                        f"{tech_path}.summary",
                        f"{TECHNIQUE_SUMMARY_MAX_CHARS} characters or fewer",
                        len(tech_summary),
                    )
            technique_ids.append(expected_tech_id)
            technique_index_by_id[expected_tech_id] = j
            technique_step_counts[expected_tech_id] = 0

    duration_limit = float(meta.get("duration", 0.0)) + 1.0
    prev_start = -math.inf
    prev_end = -math.inf
    max_seen_technique_index = -1
    for index, step in enumerate(steps):
        step_path = f"steps[{index}]"
        if not isinstance(step, dict):
            raise ValidationError(source_name, step_path, "an object", step)
        required_fields = ["id", "title", "start", "end", "category", "key_points", "visual_cues"]
        for field in required_fields:
            if field not in step:
                raise ValidationError(source_name, f"{step_path}.{field}", "present")
        expected_id = f"step_{index + 1:02d}"
        if step.get("id") != expected_id:
            raise ValidationError(source_name, f"{step_path}.id", repr(expected_id), step.get("id"))
        start = step.get("start")
        end = step.get("end")
        if not isinstance(start, (int, float)):
            raise ValidationError(source_name, f"{step_path}.start", "a number", start)
        if not isinstance(end, (int, float)):
            raise ValidationError(source_name, f"{step_path}.end", "a number", end)
        if not (0 <= float(start) < float(end) <= duration_limit):
            raise ValidationError(source_name, f"{step_path}.end", f"greater than start and <= {duration_limit}", end)
        if float(start) < prev_start:
            raise ValidationError(source_name, f"{step_path}.start", "steps sorted by start time", start)
        if index > 0 and float(start) < float(prev_end) - 0.5:
            raise ValidationError(source_name, f"{step_path}.start", "overlap with previous step <= 0.5s", start)
        category = step.get("category")
        if category not in VALID_CATEGORIES:
            raise ValidationError(source_name, f"{step_path}.category", f"one of {sorted(VALID_CATEGORIES)}", category)
        title = step.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValidationError(source_name, f"{step_path}.title", "a non-empty string", title)
        if len(title) > 80:
            raise ValidationError(source_name, f"{step_path}.title", "80 characters or fewer", title)
        key_points = step.get("key_points")
        if not isinstance(key_points, list) or not (1 <= len(key_points) <= 4):
            raise ValidationError(source_name, f"{step_path}.key_points", "a list of 1-4 strings", key_points)
        for kp_index, point in enumerate(key_points):
            if not isinstance(point, str) or not point.strip():
                raise ValidationError(source_name, f"{step_path}.key_points[{kp_index}]", "a non-empty string", point)
        visual_cues = step.get("visual_cues")
        if not isinstance(visual_cues, str) or not visual_cues.strip():
            raise ValidationError(source_name, f"{step_path}.visual_cues", "a non-empty string", visual_cues)
        explanation = step.get("explanation")
        if explanation is None:
            if category in EXPLANATION_REQUIRED_CATEGORIES:
                raise ValidationError(
                    source_name,
                    f"{step_path}.explanation",
                    "present (required for concept/demonstration/drill steps)",
                )
        else:
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValidationError(source_name, f"{step_path}.explanation", "a non-empty string", explanation)
            if len(explanation) > EXPLANATION_MAX_CHARS:
                raise ValidationError(
                    source_name,
                    f"{step_path}.explanation",
                    f"{EXPLANATION_MAX_CHARS} characters or fewer",
                    len(explanation),
                )
        # M8 (§6.4): techniques + per-step technique are optional and paired.
        if techniques_present:
            tech_field = step.get("technique")
            # isinstance first: a non-string value (e.g. a list) is unhashable and the
            # dict-membership test below would raise a raw TypeError.
            if not isinstance(tech_field, str) or tech_field not in technique_index_by_id:
                raise ValidationError(
                    source_name,
                    f"{step_path}.technique",
                    f"one of {technique_ids} (techniques is defined at the top level)",
                    tech_field,
                )
            current_technique_index = technique_index_by_id[tech_field]
            if current_technique_index < max_seen_technique_index:
                raise ValidationError(
                    source_name,
                    f"{step_path}.technique",
                    "contiguous membership matching the techniques list order (all of one technique's "
                    "steps must precede the next technique's steps)",
                    tech_field,
                )
            max_seen_technique_index = max(max_seen_technique_index, current_technique_index)
            technique_step_counts[tech_field] += 1
        elif "technique" in step:
            raise ValidationError(
                source_name,
                f"{step_path}.technique",
                "absent ('techniques' is not defined at the top level of steps.json)",
                step.get("technique"),
            )
        prev_start = float(start)
        prev_end = float(end)
    if techniques_present:
        for j, tech_id in enumerate(technique_ids):
            if technique_step_counts[tech_id] == 0:
                raise ValidationError(
                    source_name,
                    f"techniques[{j}]",
                    "at least one step assigned to it via steps[].technique",
                    0,
                )
    return steps


def validate_step_lookup(steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for step in steps:
        lookup[step["id"]] = step
    return lookup


def validate_candidates_data(
    data: Dict[str, Any],
    steps: List[Dict[str, Any]],
    source_name: str = "candidates.json",
) -> Dict[str, Dict[str, Any]]:
    validate_schema_header(data, source_name, require_algo=True)
    step_lookup = validate_step_lookup(steps)
    entries = data.get("steps")
    if not isinstance(entries, dict):
        raise ValidationError(source_name, "steps", "an object")
    for key in entries:
        if key not in step_lookup:
            raise ValidationError(source_name, f"steps[{key}]", "a step id present in steps.json", key)
    for step in steps:
        key = step["id"]
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise ValidationError(source_name, f"steps[{key}]", "an object", entry)
        for field in ["contact_sheet", "window", "params", "cells"]:
            if field not in entry:
                raise ValidationError(source_name, f"steps[{key}].{field}", "present")
    return entries


def validate_approved_data(
    data: Dict[str, Any],
    steps: List[Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    source_name: str = "approved_frames.json",
) -> Dict[str, Dict[str, Any]]:
    validate_schema_header(data, source_name)
    step_lookup = validate_step_lookup(steps)
    entries = data.get("steps")
    if not isinstance(entries, dict):
        raise ValidationError(source_name, "steps", "an object")
    for key in entries:
        if key not in step_lookup:
            raise ValidationError(source_name, f"steps[{key}]", "a step id present in steps.json", key)
    for step in steps:
        key = step["id"]
        if key not in entries:
            raise ValidationError(source_name, f"steps[{key}]", "present in approved_frames.json")
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise ValidationError(source_name, f"steps[{key}]", "an object", entry)
        has_frames = "frames" in entry
        has_resample = bool(entry.get("needs_resample"))
        if has_frames and has_resample:
            raise ValidationError(source_name, f"steps[{key}]", "either frames or needs_resample, not both")
        if not has_frames and not has_resample:
            raise ValidationError(source_name, f"steps[{key}]", "either frames or needs_resample")
        if has_frames:
            frames = entry.get("frames")
            if not isinstance(frames, list):
                raise ValidationError(source_name, f"steps[{key}].frames", "a list")
            if len(frames) > 4:
                raise ValidationError(source_name, f"steps[{key}].frames", "4 letters or fewer", frames)
            seen = set()
            for idx, letter in enumerate(frames):
                # Type check first: a non-string entry (e.g. a nested list) is unhashable,
                # and the `in seen` duplicate check would raise a raw TypeError.
                if not isinstance(letter, str) or not re.fullmatch(r"[A-Z]", letter):
                    raise ValidationError(source_name, f"steps[{key}].frames[{idx}]", "an uppercase letter", letter)
                if letter in seen:
                    raise ValidationError(source_name, f"steps[{key}].frames[{idx}]", "unique letters", letter)
                seen.add(letter)
                candidate_entry = candidates.get(key, {}).get("cells", {})
                if letter not in candidate_entry:
                    raise ValidationError(source_name, f"steps[{key}].frames[{idx}]", "a letter present in candidates.json", letter)
            # M9 (§6.6): picked frames require notes - an agent that cannot describe its
            # picks did not look at them. Empty frames (text-only step) stay exempt.
            if frames:
                notes = entry.get("notes")
                if not isinstance(notes, str) or not notes.strip():
                    raise ValidationError(
                        source_name,
                        f"steps[{key}].notes",
                        "a non-empty string describing each picked letter (required when frames are picked)",
                        notes,
                    )
        else:
            window = entry.get("window")
            if not isinstance(window, list) or len(window) != 2:
                raise ValidationError(source_name, f"steps[{key}].window", "a 2-item list")
            if not all(isinstance(v, (int, float)) for v in window):
                raise ValidationError(source_name, f"steps[{key}].window", "two numeric values", window)
            step_start = float(step["start"])
            step_end = float(step["end"])
            if not (step_start - 5.0 <= float(window[0]) < float(window[1]) <= step_end + 5.0):
                raise ValidationError(
                    source_name,
                    f"steps[{key}].window",
                    f"within {step_start - 5.0}..{step_end + 5.0}",
                    window,
                )
    return entries


def validate_steps_file(steps_path: Path, meta_path: Path) -> List[Dict[str, Any]]:
    meta = load_json(meta_path)
    steps = validate_steps_data(load_json(steps_path), meta, steps_path.name)
    return steps


def validate_candidates_file(candidates_path: Path, steps_path: Path, meta_path: Path) -> Dict[str, Dict[str, Any]]:
    meta = load_json(meta_path)
    steps = validate_steps_data(load_json(steps_path), meta, steps_path.name)
    candidates = validate_candidates_data(load_json(candidates_path), steps, candidates_path.name)
    return candidates


def validate_approved_file(approved_path: Path, steps_path: Path, candidates_path: Path, meta_path: Path) -> Dict[str, Dict[str, Any]]:
    meta = load_json(meta_path)
    steps = validate_steps_data(load_json(steps_path), meta, steps_path.name)
    candidates = validate_candidates_data(load_json(candidates_path), steps, candidates_path.name)
    approved = validate_approved_data(load_json(approved_path), steps, candidates, approved_path.name)
    return approved


GROUNDING_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at", "for", "with",
    "from", "into", "onto", "over", "under", "is", "are", "was", "be", "been", "it", "its", "he", "his",
    "him", "she", "her", "you", "your", "i", "my", "we", "they", "them", "this", "that", "these", "those",
    "not", "no", "do", "does", "did", "can", "cannot", "will", "would", "going", "go", "get", "one", "two",
    "more", "again", "now", "just", "like", "when", "while", "until", "keep", "let", "very", "way", "what",
}

GROUNDING_MIN_OVERLAP = 0.15
GROUNDING_MEDIAN_OVERLAP = 0.30


# CJK-aware tokenization (M7, §5). Whitespace tokenization is meaningless for Chinese
# (Bilibili/ASR videos), so a text whose non-space characters are mostly CJK gets tokenized
# as character bigrams instead. Ranges: CJK unified ideographs, CJK extension A, kana, hangul.
_CJK_RUN_RE = re.compile("[一-鿿㐀-䶿぀-ヿ가-힯]+|[A-Za-z']+")


def _is_cjk_char(ch: str) -> bool:
    return (
        "一" <= ch <= "鿿"
        or "㐀" <= ch <= "䶿"
        or "぀" <= ch <= "ヿ"
        or "가" <= ch <= "힯"
    )


def _cjk_text_fraction(text: str) -> float:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    return sum(1 for ch in non_space if _is_cjk_char(ch)) / len(non_space)


def _cjk_aware_tokens(text: str) -> set:
    """Tokenize a CJK-dominant text: CJK runs become overlapping character bigrams (a
    stranded single CJK char is its own token); embedded Latin runs become normalized
    (lowercased, stopword-filtered) words, same as the Latin-only tokenizer below."""
    tokens = set()
    for run in _CJK_RUN_RE.findall(text):
        if _is_cjk_char(run[0]):
            if len(run) == 1:
                tokens.add(run)
            else:
                for i in range(len(run) - 1):
                    tokens.add(run[i:i + 2])
        else:
            word = run.lower()
            if len(word) > 2 and word not in GROUNDING_STOPWORDS:
                tokens.add(word)
    return tokens


def _is_bigram_token(tok: str) -> bool:
    return any(_is_cjk_char(ch) for ch in tok)


def _grounding_tokens(text: str) -> set:
    if _cjk_text_fraction(text) > CJK_TEXT_FRAC:
        return _cjk_aware_tokens(text)
    return {tok for tok in re.findall(r"[a-z']+", text.lower()) if len(tok) > 2 and tok not in GROUNDING_STOPWORDS}


def _grounding_overlap_ratio(claim_tokens: set, reference_tokens: set) -> float:
    """Fraction of claim_tokens found in reference_tokens. Latin words use the existing
    prefix-5 fuzzy match (unchanged below the CJK threshold); CJK bigrams match exactly
    only - the prefix-5 rule is Latin-only (§5)."""
    latin_reference = [tok for tok in reference_tokens if not _is_bigram_token(tok)]
    grounded = 0
    for tok in claim_tokens:
        if _is_bigram_token(tok):
            if tok in reference_tokens:
                grounded += 1
        elif tok in reference_tokens or any(
            w.startswith(tok[:5]) or tok.startswith(w[:5]) for w in latin_reference
        ):
            grounded += 1
    return grounded / len(claim_tokens)


def _load_transcript_segments(transcript_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not transcript_path.exists():
        return None
    try:
        return json.loads(transcript_path.read_text(encoding="utf-8")).get("segments", [])
    except (json.JSONDecodeError, OSError):
        return None


def step_grounding_warnings(steps: List[Dict[str, Any]], transcript_path: Path) -> List[str]:
    """Warn-only lint: flags steps whose title/key_points/explanation share almost no vocabulary
    with the transcript inside their time window — the signature of invented (ungrounded) content.
    visual_cues is deliberately not checked; it describes the picture, not the speech."""
    segments = _load_transcript_segments(transcript_path)
    if segments is None:
        return []
    warnings = []
    ratios = []
    for index, step in enumerate(steps):
        start, end = float(step["start"]), float(step["end"])
        window_text = " ".join(
            seg.get("text", "")
            for seg in segments
            if float(seg.get("start", 0.0)) < end and float(seg.get("end", 0.0)) > start
        )
        window_tokens = _grounding_tokens(window_text)
        claim_text = (
            str(step.get("title", ""))
            + " " + " ".join(step.get("key_points", []))
            + " " + str(step.get("explanation") or "")
        )
        claim_tokens = _grounding_tokens(claim_text)
        if not claim_tokens or not window_tokens:
            continue
        ratio = _grounding_overlap_ratio(claim_tokens, window_tokens)
        ratios.append(ratio)
        if ratio < GROUNDING_MIN_OVERLAP:
            warnings.append(
                f"steps[{index}] ({step['id']}): title/key_points/explanation share almost no vocabulary with the "
                f"transcript between {start:.0f}s and {end:.0f}s ({ratio:.0%} overlap). Re-read transcript.json and "
                "describe what the instructor actually says in that window - do not invent content."
            )
    if ratios:
        median_ratio = statistics.median(ratios)
        if median_ratio < GROUNDING_MEDIAN_OVERLAP:
            warnings.append(
                f"overall: median vocabulary overlap between steps and their transcript windows is "
                f"{median_ratio:.0%} (expected at least {GROUNDING_MEDIAN_OVERLAP:.0%}). The segmentation as a whole "
                "does not look grounded in this video's transcript - re-read transcript.json and rewrite steps.json "
                "from what the instructor actually says."
            )
    return warnings


def one_thing_grounding_warning(one_thing: str, transcript_path: Path) -> Optional[str]:
    """Warn-only lint (§5): the top-level one_thing is linted against the WHOLE transcript
    (not a step window) - below 15% vocabulary overlap is a WARNING."""
    segments = _load_transcript_segments(transcript_path)
    if segments is None:
        return None
    whole_text = " ".join(seg.get("text", "") for seg in segments)
    transcript_tokens = _grounding_tokens(whole_text)
    claim_tokens = _grounding_tokens(one_thing or "")
    if not claim_tokens or not transcript_tokens:
        return None
    ratio = _grounding_overlap_ratio(claim_tokens, transcript_tokens)
    if ratio < GROUNDING_MIN_OVERLAP:
        return (
            f"one_thing: shares almost no vocabulary with the transcript ({ratio:.0%} overlap). Re-read "
            "transcript.json (check the intro and recap first) and ground it in what the instructor actually says."
        )
    return None


def technique_grounding_warnings(
    steps_doc: Dict[str, Any], steps: List[Dict[str, Any]], transcript_path: Path
) -> List[str]:
    """M8 technique lint (§5): when steps.json has a `techniques` list, lint each technique's
    title + summary (tokenized together, pooled) against the transcript inside that technique's
    window - the earliest `start` to the latest `end` of its member steps. Reuses the same
    tokenizer/overlap machinery as the per-step lint (CJK-aware for free). Warn-only, never
    blocks: one WARNING per technique below 15% overlap, naming the technique id."""
    techniques = steps_doc.get("techniques")
    if not isinstance(techniques, list) or not techniques:
        return []
    segments = _load_transcript_segments(transcript_path)
    if segments is None:
        return []
    warnings = []
    for index, tech in enumerate(techniques):
        tech_id = tech.get("id")
        member_steps = [step for step in steps if step.get("technique") == tech_id]
        if not member_steps:
            continue
        window_start = min(float(step["start"]) for step in member_steps)
        window_end = max(float(step["end"]) for step in member_steps)
        window_text = " ".join(
            seg.get("text", "")
            for seg in segments
            if float(seg.get("start", 0.0)) < window_end and float(seg.get("end", 0.0)) > window_start
        )
        window_tokens = _grounding_tokens(window_text)
        claim_text = str(tech.get("title", "")) + " " + str(tech.get("summary") or "")
        claim_tokens = _grounding_tokens(claim_text)
        if not claim_tokens or not window_tokens:
            continue
        ratio = _grounding_overlap_ratio(claim_tokens, window_tokens)
        if ratio < GROUNDING_MIN_OVERLAP:
            warnings.append(
                f"techniques[{index}] ({tech_id}): title/summary share almost no vocabulary with the transcript "
                f"between {window_start:.0f}s and {window_end:.0f}s ({ratio:.0%} overlap). Re-read transcript.json "
                "and describe what the instructor actually says in that window - do not invent content."
            )
    return warnings


def approved_frames_warnings(steps: List[Dict[str, Any]], approved_doc: Dict[str, Any]) -> List[str]:
    """M9 judging lint (§5, warn-only): a demonstration/drill step approved with fewer than
    DEMO_DRILL_FRAMES_WARN_BELOW frames is the shallow-judging signature of a weak model
    pass - the rubric targets 3-4 picks for those categories. needs_resample entries and
    the other categories are exempt."""
    warnings: List[str] = []
    entries = approved_doc.get("steps", {})
    if not isinstance(entries, dict):
        return warnings
    demo_drill_counts: List[int] = []
    for step in steps:
        entry = entries.get(step["id"])
        if not isinstance(entry, dict) or "frames" not in entry:
            continue
        if step.get("category") not in ("demonstration", "drill"):
            continue
        frames = entry.get("frames") or []
        demo_drill_counts.append(len(frames))
        if len(frames) < DEMO_DRILL_FRAMES_WARN_BELOW:
            warnings.append(
                f"steps[{step['id']}] ({step['category']}): only {len(frames)} approved frame(s) - "
                "the rubric targets 3-4 for demonstration/drill steps. Re-judge the contact sheet, "
                "or resample if the window genuinely lacks distinct moments."
            )
    # File-level shallow-pass signature: one 2-pick demo is legitimate, but a median of 2
    # across all demonstration/drill steps means the judging pass stopped at the floor
    # everywhere (the observed weak-model failure) rather than working each sheet.
    if len(demo_drill_counts) >= 2:
        ordered = sorted(demo_drill_counts)
        mid = len(ordered) // 2
        median = float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        if median < DEMO_DRILL_MEDIAN_TARGET:
            warnings.append(
                f"overall: median approved frames across {len(demo_drill_counts)} demonstration/drill "
                f"steps is {median:g} - the rubric targets 3-4 per step. This is the signature of a "
                "shallow judging pass; re-check each contact sheet for distinct setup/grip/movement/"
                "finish moments before shipping."
            )
    return warnings


def cache_entry_valid(entry: Dict[str, Any], window: Sequence[float], params: Dict[str, Any], algo_version: int) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("algo_version") == algo_version
        and list(entry.get("window", [])) == list(window)
        and dict(entry.get("params", {})) == dict(params)
    )


def frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash_image(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_value = 0
    for bit in diff.flatten():
        hash_value = (hash_value << 1) | int(bool(bit))
    return hash_value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_uniform_sample_times(start: float, end: float, per_step: int = UNIFORM_SAMPLES_PER_STEP) -> List[float]:
    duration = max(0.0, end - start)
    if duration <= 0:
        return []
    return [start + duration * (i + 0.5) / per_step for i in range(per_step)]


def step_midpoint(start: float, end: float) -> float:
    return (start + end) / 2.0


@dataclass
class Sample:
    step_id: str
    time: float
    source: str
    frame: Optional[np.ndarray] = None
    sharpness: float = 0.0
    hash_value: Optional[int] = None
    rejected_reason: Optional[str] = None
    rejected_category: Optional[str] = None
    ocr_text: str = ""
    ocr_conf: float = 0.0


def add_time_if_far(samples: List[Tuple[float, str]], new_time: float, source: str, min_gap: float = 0.75) -> None:
    for existing_time, _ in samples:
        if abs(existing_time - new_time) < min_gap:
            return
    samples.append((new_time, source))


def motion_informed_times_for_step(
    cap: cv2.VideoCapture,
    step: Dict[str, Any],
    fps: float,
) -> List[Tuple[float, str]]:
    start = float(step["start"])
    end = float(step["end"])
    times = np.arange(start, end + 0.0001, 1.0).tolist()
    if len(times) <= 1:
        return []
    scores: List[float] = []
    score_times: List[float] = []
    prev_gray = None
    for t in times:
        frame = read_frame_at_time(cap, fps, t, low_res=True)
        if frame is None:
            continue
        if prev_gray is not None:
            diff = cv2.absdiff(frame, prev_gray)
            scores.append(float(np.mean(diff)))
            score_times.append(t)
        prev_gray = frame
    if not scores:
        return []
    peak_idx = int(np.argmax(scores))
    peak_val = float(scores[peak_idx])
    peak_time = float(score_times[peak_idx])
    mean_val = float(np.mean(scores))
    samples: List[Tuple[float, str]] = [(peak_time, "peak")]
    threshold = max(mean_val, peak_val * 0.4)
    settle_time = None
    for idx in range(peak_idx, len(scores)):
        if scores[idx] < threshold:
            settle_time = float(score_times[idx])
            break
    if settle_time is not None and abs(settle_time - peak_time) >= 0.75:
        samples.append((settle_time, "settle"))
    return samples


def collect_sample_times_for_step(
    cap: cv2.VideoCapture,
    step: Dict[str, Any],
    fps: float,
    per_step: int,
    max_candidates: int = MAX_CANDIDATES_PER_STEP,
) -> List[Tuple[float, str]]:
    start = float(step["start"])
    end = float(step["end"])
    selected: List[Tuple[float, str]] = []
    for time_value in build_uniform_sample_times(start, end, per_step):
        add_time_if_far(selected, time_value, "uniform")
    for time_value, source in motion_informed_times_for_step(cap, step, fps):
        add_time_if_far(selected, time_value, source)
    if len(selected) > max_candidates:
        uniform_indices = [idx for idx, (_, source) in enumerate(selected) if source == "uniform"]
        while len(selected) > max_candidates and uniform_indices:
            remove_index = uniform_indices[0] if len(uniform_indices) % 2 else uniform_indices[-1]
            selected.pop(remove_index)
            uniform_indices = [idx for idx, (_, source) in enumerate(selected) if source == "uniform"]
    selected.sort(key=lambda item: item[0])
    return selected


def read_frame_at_time(cap: cv2.VideoCapture, fps: float, time_value: float, low_res: bool = False) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_value * fps))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if low_res:
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def nearest_scene_boundary(time_value: float, scenes: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not scenes:
        return None
    best = min(scenes, key=lambda item: abs(item[0] - time_value))
    return best


def run_title_ocr(frame: np.ndarray, tesseract_path: Optional[str]) -> Tuple[str, float]:
    if not tesseract_path:
        return "", 0.0
    height = frame.shape[0]
    top = int(height * 0.15)
    bottom = int(height * 0.85)
    cropped = frame[top:bottom, :]
    processed = preprocess_ocr_image(cropped)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cv2.imwrite(str(tmp_path), processed)
        return run_tesseract_ocr(tmp_path, tesseract_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def dedupe_samples(samples: List[Sample]) -> List[Sample]:
    kept: List[Sample] = []
    for sample in sorted(samples, key=lambda item: item.time):
        if sample.frame is None:
            continue
        sample.hash_value = dhash_image(sample.frame)
        matched_index = None
        matched_distance = None
        for idx, existing in enumerate(kept):
            if existing.frame is None or existing.hash_value is None:
                continue
            distance = hamming_distance(sample.hash_value, existing.hash_value)
            if distance <= DHASH_HAMMING_DUP:
                matched_index = idx
                matched_distance = distance
                break
        if matched_index is None:
            kept.append(sample)
            continue
        existing = kept[matched_index]
        if existing.frame is not None and sample.sharpness > existing.sharpness:
            kept[matched_index] = sample
        else:
            sample.rejected_reason = f"duplicate hash distance {matched_distance}"
            sample.rejected_category = "duplicate"
            print(f"{sample.step_id} {format_mmss(sample.time)} {sample.source} dropped duplicate: {sample.rejected_reason}")
    return kept


def save_frame_image(frame: np.ndarray, path: Path, width: int, height: int) -> None:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def build_contact_sheet(step_id: str, samples: List[Sample], output_path: Path) -> None:
    if not samples:
        canvas = np.zeros((SHEET_LABEL_H + SHEET_CELL_H, SHEET_CELL_W, 3), dtype=np.uint8)
        cv2.putText(canvas, f"{step_id}: NO CANDIDATES", (12, SHEET_CELL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return
    cell_w, cell_h = SHEET_CELL_W, SHEET_CELL_H
    label_h = SHEET_LABEL_H
    cols = SHEET_COLS
    rows = math.ceil(len(samples) / cols)
    spacing = 4
    sheet_w = cols * cell_w + (cols + 1) * spacing
    sheet_h = rows * (label_h + cell_h) + (rows + 1) * spacing
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    canvas[:] = (14, 14, 18)
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = spacing + col * (cell_w + spacing)
        y = spacing + row * (label_h + cell_h + spacing)
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + label_h + cell_h), (28, 28, 36), thickness=-1)
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + label_h), (0, 0, 0), thickness=-1)
        label = f"{chr(ord('A') + idx)}  {format_mmss(sample.time)}"
        cv2.putText(canvas, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if sample.frame is not None:
            thumb = cv2.resize(sample.frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            canvas[y + label_h : y + label_h + cell_h, x : x + cell_w] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def candidate_entry_for_step(
    step: Dict[str, Any],
    samples: List[Sample],
    work_dir: Path,
    ocr_filter_enabled: bool,
    per_step: int,
) -> Dict[str, Any]:
    step_id = step["id"]
    step_dir = work_dir / "candidates" / step_id
    sheet_path = work_dir / "contact_sheets" / f"contact_{step_id}.jpg"
    if step_dir.exists():
        shutil.rmtree(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    letters = {}
    ordered = sorted(samples, key=lambda item: item.time)
    for idx, sample in enumerate(ordered):
        letter = chr(ord("A") + idx)
        out_path = step_dir / f"{letter}.jpg"
        if sample.frame is not None:
            save_frame_image(sample.frame, out_path, THUMB_W, THUMB_H)
        letters[letter] = {
            "time": round(float(sample.time), 3),
            "sharpness": round(float(sample.sharpness), 3),
            "source": sample.source,
            "thumb": str(out_path.relative_to(work_dir)).replace("\\", "/"),
        }
    build_contact_sheet(step_id, ordered, sheet_path)
    return {
        "contact_sheet": str(sheet_path.relative_to(work_dir)).replace("\\", "/"),
        "window": [float(step["start"]), float(step["end"])],
        "params": {"per_step": per_step, "ocr_filter": ocr_filter_enabled},
        "cells": letters,
    }


def load_scenes_from_workdir(work_dir: Path) -> List[Tuple[float, float]]:
    scenes_data = load_json(work_dir / "scenes.json")
    validate_schema_header(scenes_data, "scenes.json", require_algo=True)
    boundaries = scenes_data.get("boundaries")
    if not isinstance(boundaries, list):
        raise ValidationError("scenes.json", "boundaries", "a list")
    result = []
    for idx, item in enumerate(boundaries):
        if not isinstance(item, dict):
            raise ValidationError("scenes.json", f"boundaries[{idx}]", "an object", item)
        if "time" not in item or "score" not in item:
            raise ValidationError("scenes.json", f"boundaries[{idx}]", "time and score present")
        result.append((float(item["time"]), float(item["score"])))
    return result


def prepare_command(video_path: Path, args: argparse.Namespace) -> int:
    ffmpeg_path, ffprobe_path, _ = ensure_required_tools()
    # Video first: a typo'd --video must say so, not fail subtitle discovery -
    # and it must not leave an empty <typo>_review/ directory behind.
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        return 1
    subtitles_path = subtitle_path_from_video(video_path, args.subtitles)
    if not subtitles_path.exists():
        print(f"Error: Subtitle file not found at {subtitles_path}", file=sys.stderr)
        return 1
    work_dir = ensure_work_dir(video_path)
    meta_path = work_dir / "meta.json"
    transcript_path = work_dir / "transcript.json"
    scenes_path = work_dir / "scenes.json"
    # M10 (§5): --url stores/preserves/clears meta.json's optional source_url. meta.json is
    # rewritten on every prepare run, so an omitted --url must carry the stored value forward
    # (never silently strip it) - read whatever is already on disk before overwriting.
    existing_source_url = None
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing_meta, dict):
                existing_source_url = existing_meta.get("source_url")
        except (OSError, json.JSONDecodeError):
            existing_source_url = None
    try:
        resolved_source_url = resolve_source_url(existing_source_url, args.url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    duration = get_video_duration(video_path, ffprobe_path)
    fps, width, height = probe_video_stream_info(video_path, ffprobe_path)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "algo_version": ALGO_VERSION,
        "video_path": str(video_path.resolve()),
        "video_base": video_path.stem,
        "subtitles_path": str(subtitles_path),
    }
    if resolved_source_url:
        meta["source_url"] = resolved_source_url
    meta.update({
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
    })
    write_json(meta_path, meta)
    transcript_segments = parse_transcript(subtitles_path)
    cleaned_segments = []
    for start, end, text in transcript_segments:
        cleaned = clean_text(text)
        if cleaned:
            cleaned_segments.append({"start": round(start, 3), "end": round(end, 3), "text": cleaned})
    if not cleaned_segments:
        # An empty transcript poisons every later stage (ungroundable steps, empty
        # per-step transcripts) - fail loudly now, per the error contract.
        print(
            f"Error: no transcript segments could be parsed from {subtitles_path}. "
            "The file may be empty or in an unsupported format. Expected WebVTT (.vtt) or "
            "'[HH:MM:SS.mmm --> HH:MM:SS.mmm] text' lines (.txt). Try --subtitles with a "
            "different sibling subtitle file, or generate one per SKILL.md Stage 1.5 (Whisper).",
            file=sys.stderr,
        )
        return 1
    write_json(
        transcript_path,
        {
            "schema_version": SCHEMA_VERSION,
            "algo_version": ALGO_VERSION,
            "segments": cleaned_segments,
        },
    )
    cached_scenes = None
    if not args.force_refresh and scenes_path.exists():
        try:
            existing = load_json(scenes_path)
            if (
                existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("algo_version") == ALGO_VERSION
                and existing.get("threshold") == args.threshold
                and existing.get("min_gap") == args.min_duration
            ):
                cached_scenes = existing
        except Exception:
            cached_scenes = None
    if cached_scenes is None:
        raw_scenes = run_scene_detection(video_path, args.threshold, ffprobe_path)
        filtered = filter_scenes(raw_scenes, args.min_duration)
        write_json(
            scenes_path,
            {
                "schema_version": SCHEMA_VERSION,
                "algo_version": ALGO_VERSION,
                "threshold": args.threshold,
                "min_gap": args.min_duration,
                "boundaries": [{"time": round(t, 3), "score": round(s, 3)} for t, s in filtered],
            },
        )
        print(f"Scene detection wrote {scenes_path}")
    else:
        print(f"Using cached scenes.json at {scenes_path}")
    print(f"Prepared {work_dir}")
    print("NEXT: write steps.json per SKILL.md §Segmentation, then run check --what steps")
    return 0


def resolve_step_requests(
    steps: List[Dict[str, Any]],
    candidates_data: Dict[str, Dict[str, Any]],
    candidates_algo_version: Optional[int],
    per_step: int,
    ocr_filter_enabled: bool,
    force_refresh: bool,
) -> List[Dict[str, Any]]:
    requests = []
    for step in steps:
        step_id = step["id"]
        entry = candidates_data.get(step_id)
        cache_entry = dict(entry) if entry else None
        if cache_entry is not None:
            cache_entry["algo_version"] = candidates_algo_version
        cache_valid = (
            cache_entry
            and cache_entry_valid(cache_entry, [step["start"], step["end"]], {"per_step": per_step, "ocr_filter": ocr_filter_enabled}, ALGO_VERSION)
            and not force_refresh
        )
        if cache_valid:
            continue
        requests.append(step)
    return requests


def parse_resample_args(
    resample: Optional[Sequence[str]], steps: List[Dict[str, Any]]
) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Validate `--resample STEP_ID START END` before any video work. Returns
    (step_id, (start, end)) or None when --resample was not passed. Raises ValueError
    with a printable message on a non-numeric or inverted window and on an unknown
    step id - argparse cannot check any of these itself, and an unguarded float()
    here used to surface as a raw traceback."""
    if not resample:
        return None
    step_id, start_s, end_s = resample
    try:
        window = (float(start_s), float(end_s))
    except ValueError:
        raise ValueError(
            f"--resample START and END must be numbers in seconds (got {start_s!r}, {end_s!r})"
        ) from None
    if not (0 <= window[0] < window[1]):
        raise ValueError(
            f"--resample window must satisfy 0 <= START < END (got {window[0]:g}, {window[1]:g})"
        )
    if not any(step["id"] == step_id for step in steps):
        raise ValueError(f"unknown step id {step_id}")
    return step_id, window


def candidates_command(video_path: Path, args: argparse.Namespace) -> int:
    ffmpeg_path, ffprobe_path, tesseract_path = ensure_required_tools()
    work_dir = build_work_dir(video_path)
    meta = load_json(work_dir / "meta.json")
    steps = validate_steps_data(load_json(work_dir / "steps.json"), meta, "steps.json")
    try:
        resample_request = parse_resample_args(args.resample, steps)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    scenes = load_scenes_from_workdir(work_dir)
    existing_candidates = {}
    existing_algo_version = None
    candidates_path = work_dir / "candidates.json"
    if candidates_path.exists():
        try:
            candidates_doc = load_json(candidates_path)
            existing_candidates = candidates_doc.get("steps", {})
            existing_algo_version = candidates_doc.get("algo_version")
        except Exception:
            existing_candidates = {}
    ocr_filter_enabled = bool(tesseract_path) and not args.no_ocr_filter
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(
            f"Error: could not open video {video_path}. "
            "OpenCV likely lacks a decoder for this codec (AV1 downloads from YouTube are a "
            "common cause - check with: ffprobe -select_streams v:0 -show_entries stream=codec_name). "
            "Fix: transcode a same-timeline H.264 proxy with "
            "'ffmpeg -i <video> -vf scale=1280:720 -c:v libx264 -preset ultrafast -an <proxy>.mp4' "
            "and run candidates against the proxy (timestamps carry over; render can still use the "
            "original), or re-download preferring H.264 per SKILL.md Stage 0.",
            file=sys.stderr,
        )
        return 1
    fps = float(meta.get("fps") or 0.0) or 29.97
    try:
        updated_steps = dict(existing_candidates)
        resample_step_id = None
        resample_window = None
        if resample_request:
            resample_step_id, resample_window = resample_request
            steps_to_compute = [step for step in steps if step["id"] == resample_step_id]
        else:
            steps_to_compute = resolve_step_requests(
                steps,
                existing_candidates,
                existing_algo_version,
                args.per_step,
                ocr_filter_enabled,
                args.force_refresh,
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sample_map: Dict[str, List[Tuple[float, str]]] = {}
        for step in steps_to_compute:
            if resample_window and step["id"] == resample_step_id:
                step = dict(step)
                step["start"], step["end"] = resample_window
            sample_map[step["id"]] = collect_sample_times_for_step(cap, step, fps, args.per_step)
        all_requests = []
        for step in steps_to_compute:
            if resample_window and step["id"] == resample_step_id:
                step = dict(step)
                step["start"], step["end"] = resample_window
            for time_value, source in sample_map.get(step["id"], []):
                all_requests.append((time_value, step["id"], source))
        all_requests.sort(key=lambda item: item[0])
        sample_frames: Dict[Tuple[str, float, str], np.ndarray] = {}
        for time_value, step_id, source in all_requests:
            frame = read_frame_at_time(cap, fps, time_value, low_res=False)
            if frame is not None:
                sample_frames[(step_id, time_value, source)] = frame
        for step in steps_to_compute:
            if resample_window and step["id"] == resample_step_id:
                step = dict(step)
                step["start"], step["end"] = resample_window
            sample_objs = []
            for time_value, source in sample_map.get(step["id"], []):
                frame = sample_frames.get((step["id"], time_value, source))
                if frame is None:
                    continue
                sample_objs.append(Sample(step_id=step["id"], time=time_value, source=source, frame=frame, sharpness=frame_sharpness(frame)))
            step_samples, _ = finalize_step_samples(step, sample_objs, scenes, tesseract_path, ocr_filter_enabled)
            if not step_samples:
                midpoint = step_midpoint(float(step["start"]), float(step["end"]))
                frame = read_frame_at_time(cap, fps, midpoint, low_res=False)
                if frame is not None:
                    step_samples = [
                        Sample(step_id=step["id"], time=midpoint, source="fallback", frame=frame, sharpness=frame_sharpness(frame))
                    ]
                else:
                    print(f"Warning: {step['id']}: no readable frames in its window; contact sheet will be empty", file=sys.stderr)
            entry = candidate_entry_for_step(step, step_samples, work_dir, ocr_filter_enabled, args.per_step)
            updated_steps[step["id"]] = entry
            print(f"Wrote candidates for {step['id']} -> {entry['contact_sheet']}")
        if resample_request:
            for step in steps:
                if step["id"] != resample_step_id and step["id"] in existing_candidates:
                    updated_steps[step["id"]] = existing_candidates[step["id"]]
            approved_path = work_dir / "approved_frames.json"
            if approved_path.exists():
                try:
                    approved_doc = json.loads(approved_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    approved_doc = {}
                if "frames" in (approved_doc.get("steps", {}).get(resample_step_id) or {}):
                    print(
                        f"Warning: approved_frames.json already has letters for {resample_step_id}, but they were "
                        "chosen from the OLD contact sheet. Judge the new sheet and update that entry before render.",
                        file=sys.stderr,
                    )
        ordered_updated_steps = {
            step["id"]: updated_steps[step["id"]]
            for step in steps
            if step["id"] in updated_steps
        }
        write_json(
            candidates_path,
            {
                "schema_version": SCHEMA_VERSION,
                "algo_version": ALGO_VERSION,
                "steps": ordered_updated_steps,
            },
        )
    finally:
        cap.release()
    print(f"NEXT: judge contact sheets per SKILL.md §Judging, write approved_frames.json, then run check --what approved")
    return 0


def finalize_step_samples(
    step: Dict[str, Any],
    sample_objs: List[Sample],
    scenes: Sequence[Tuple[float, float]],
    tesseract_path: Optional[str],
    ocr_filter_enabled: bool,
) -> Tuple[List[Sample], List[Sample]]:
    if not sample_objs:
        return [], []
    step_id = step["id"]
    remaining: List[Sample] = []
    for sample in sample_objs:
        boundary = nearest_scene_boundary(sample.time, scenes)
        if boundary is not None and abs(sample.time - boundary[0]) < NEAR_CUT_EXCLUSION:
            print(f"{step_id} {format_mmss(sample.time)} {sample.source} dropped near-cut: within {abs(sample.time - boundary[0]):.2f}s of boundary {boundary[0]:.2f}")
            continue
        if ocr_filter_enabled and tesseract_path:
            ocr_text, mean_conf = run_title_ocr(sample.frame, tesseract_path)
            sample.ocr_text = ocr_text
            sample.ocr_conf = mean_conf
            if mean_conf >= OCR_CONF_MIN and is_meaningful_ocr_title(ocr_text):
                print(f"{step_id} {format_mmss(sample.time)} {sample.source} dropped title-card: OCR '{clean_ocr_text(ocr_text)}' (conf {mean_conf:.1f})")
                continue
        remaining.append(sample)
    if not remaining:
        return [], []
    median_sharpness = statistics.median(item.sharpness for item in remaining)
    blur_threshold = max(BLUR_ABS_FLOOR, BLUR_REL_THRESHOLD * median_sharpness)
    blur_rejected: List[Sample] = []
    blur_kept: List[Sample] = []
    for sample in remaining:
        if sample.sharpness < blur_threshold:
            print(f"{step_id} {format_mmss(sample.time)} {sample.source} dropped blur: blur {sample.sharpness:.1f} < {blur_threshold:.1f}")
            blur_rejected.append(sample)
        else:
            blur_kept.append(sample)
    deduped = dedupe_samples(blur_kept)
    if len(deduped) < MIN_CANDIDATES_PER_STEP:
        for sample in sorted(blur_rejected, key=lambda item: item.sharpness, reverse=True):
            if len(deduped) >= MIN_CANDIDATES_PER_STEP:
                break
            deduped.append(sample)
        deduped = dedupe_samples(deduped)
    deduped.sort(key=lambda item: item.time)
    return deduped, blur_rejected


# Plain string (not an f-string) so CSS/JS braces need no escaping. Injected before </body>.
LIGHTBOX_SNIPPET = """
<style>
  .image-strip figure { cursor: zoom-in; }
  #lightbox {
    position: fixed; inset: 0; z-index: 1000;
    display: none; align-items: center; justify-content: center;
    flex-direction: column; gap: .6rem; padding: 1.5rem;
    background: rgba(2, 6, 12, .94); cursor: zoom-out;
  }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 96vw; max-height: 86vh; object-fit: contain; border-radius: 10px; }
  #lightbox .lb-caption { color: #cbd5e1; font-size: .95rem; text-align: center; }
  #lightbox .lb-hint { color: #64748b; font-size: .8rem; }
</style>
<div id="lightbox" role="dialog" aria-modal="true">
  <img src="" alt="">
  <div class="lb-caption"></div>
  <div class="lb-hint">Click anywhere or press Esc to close</div>
</div>
<script>
(function () {
  var lb = document.getElementById('lightbox');
  var lbImg = lb.querySelector('img');
  var lbCap = lb.querySelector('.lb-caption');
  document.querySelectorAll('.image-strip figure').forEach(function (fig) {
    fig.addEventListener('click', function () {
      var img = fig.querySelector('img');
      var cap = fig.querySelector('figcaption');
      lbImg.src = img.src;
      lbImg.alt = img.alt;
      lbCap.textContent = (img.alt ? img.alt + ' - ' : '') + (cap ? cap.textContent : '');
      lb.classList.add('open');
      document.body.style.overflow = 'hidden';
    });
  });
  function closeLightbox() {
    lb.classList.remove('open');
    lbImg.src = '';
    document.body.style.overflow = '';
  }
  lb.addEventListener('click', closeLightbox);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeLightbox(); });
})();
</script>
"""

# Plain string (not an f-string, same pattern as LIGHTBOX_SNIPPET) so CSS/JS braces need no
# escaping. Injected before </body> only when 2+ techniques are rendered (§9, M8). Progressive
# enhancement: JS adds a `js` class to <html>; CSS then hides `.js .tech-panel:not(.active)`, so
# with JS disabled - and in @media print - every panel stays visible, stacked, with its <h2> as
# the divider (§15 "Ungrouped output must not change" / no-JS fallback pitfall).
TABS_SNIPPET = """
<style>
  .tech-tabs { display: flex; flex-wrap: wrap; gap: .6rem; margin: 0 0 1.75rem; padding: 0; list-style: none; }
  .tech-tab {
    cursor: pointer; text-align: left; display: flex; flex-direction: column; gap: .2rem;
    padding: .65rem 1.05rem; border-radius: 14px; border: 1px solid rgba(255,255,255,.1);
    background: rgba(17,24,39,.6); color: inherit; font: inherit;
  }
  .tech-tab-index { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--primary); font-weight: 700; }
  .tech-tab-title { font-weight: 600; }
  .tech-tab-time { font-size: .78rem; color: var(--muted); }
  .js .tech-tab.active { border-color: var(--primary); background: rgba(251,191,36,.14); }
  .js .tech-panel:not(.active) { display: none; }
  .tech-panel h2 { font-family: var(--font-display); margin: 0 0 .9rem; }
  .tech-summary {
    background: linear-gradient(135deg, rgba(251,191,36,.1), rgba(251,191,36,.02));
    border: 1px solid rgba(251,191,36,.25); border-left: 3px solid var(--primary);
    border-radius: 14px; padding: 1rem 1.3rem; margin: 0 0 1.5rem;
  }
  .tech-summary-label {
    display: block; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--primary); font-weight: 700; margin-bottom: .35rem;
  }
  .tech-summary-text { margin: 0; color: var(--text); }
  .tech-panel-toc h3 { margin: 0 0 .75rem; font-size: 1rem; color: var(--muted); font-family: var(--font-display); }
  @media print {
    .tech-tabs { display: none; }
    .tech-panel { display: block !important; }
  }
</style>
<script>
(function () {
  document.documentElement.classList.add('js');
  var panels = Array.prototype.slice.call(document.querySelectorAll('.tech-panel'));
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tech-tabs .tech-tab'));
  if (!panels.length) { return; }
  function activatePanel(panelId) {
    panels.forEach(function (panel) {
      panel.classList.toggle('active', panel.id === panelId);
    });
    tabs.forEach(function (tab) {
      tab.classList.toggle('active', tab.getAttribute('data-target') === panelId);
    });
  }
  function panelIdForSection(sectionId) {
    var el = document.getElementById(sectionId);
    var panel = el ? el.closest('.tech-panel') : null;
    return panel ? panel.id : null;
  }
  function applyHash(scrollToTarget) {
    var hash = window.location.hash.slice(1);
    if (!hash) { return; }
    if (hash.indexOf('tech_') === 0) {
      activatePanel(hash);
    } else if (hash.indexOf('section-') === 0) {
      var panelId = panelIdForSection(hash);
      if (panelId) { activatePanel(panelId); }
      if (scrollToTarget) {
        var target = document.getElementById(hash);
        if (target) { target.scrollIntoView(); }
      }
    }
  }
  function ensureActivePanel() {
    // Unrecognized or stale hashes (e.g. #tech_99 from an old link) must never
    // leave every panel hidden - fall back to the first panel.
    var anyActive = panels.some(function (panel) { return panel.classList.contains('active'); });
    if (!anyActive) { activatePanel(panels[0].id); }
  }
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      var targetId = tab.getAttribute('data-target');
      activatePanel(targetId);
      history.replaceState(null, '', '#' + targetId);
    });
  });
  window.addEventListener('hashchange', function () { applyHash(true); ensureActivePanel(); });
  applyHash(true);
  ensureActivePanel();
})();
</script>
"""

# M10 (§9): injected into the page <style> block only when meta.json carries a source_url —
# a no-URL render must not contain the string "video-link" anywhere (acceptance regression
# check), so this cannot live in the static stylesheet. Plain string, not an f-string:
# the page template doubles its braces and CSS braces inside it are a bug factory (§15).
SOURCE_LINK_CSS = """
    .video-link { margin: .9rem auto 0; }
    .video-link a {
      color: var(--primary);
      font-weight: 600;
      text-decoration: none;
      border-bottom: 1px solid rgba(251,191,36,.45);
    }
    .video-link a:hover { border-bottom-color: var(--primary); }
    .watch-step {
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      padding: .25rem .7rem;
      border: 1px solid rgba(251,191,36,.45);
      border-radius: 999px;
      color: var(--primary);
      font-size: .78rem;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }
    .watch-step:hover { background: rgba(251,191,36,.14); border-color: var(--primary); }
    figcaption a { color: inherit; text-decoration: none; }
    figcaption a .play-glyph { color: var(--primary); margin-right: .35rem; font-size: .7rem; }
    figcaption a .watch-hint { color: var(--primary); opacity: .85; }
    figcaption a:hover { color: var(--text); }
    figcaption a:hover .watch-hint { text-decoration: underline; }"""


def render_image_source(source_path: Path, display_path: Path, embed_images: bool) -> str:
    if not embed_images:
        return html_escape(str(display_path).replace("\\", "/"), quote=True)
    data = base64.b64encode(source_path.read_bytes()).decode("ascii")
    return html_escape(f"data:image/jpeg;base64,{data}", quote=True)


def timestamped_url(source_url: Optional[str], seconds: float) -> Optional[str]:
    """Pure helper (§9, M10): source_url with its query `t` parameter set to a click-to-watch
    deep link at `seconds`, replacing any existing `t` and preserving every other query
    parameter (and the fragment). Returns None when `source_url` is falsy, unparseable, or its
    host (after lowercasing and stripping one leading "www." or "m.") is not in YOUTUBE_HOSTS
    or BILIBILI_HOSTS. Never string-appends "&t=" - the stored URL may already carry `?t=`,
    other params, or a fragment (§15)."""
    if not source_url:
        return None
    try:
        parts = urlsplit(source_url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[len("www."):]
    elif host.startswith("m."):
        host = host[len("m."):]
    if host not in YOUTUBE_HOSTS and host not in BILIBILI_HOSTS:
        return None
    t_value = str(max(0, int(seconds)))
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    replaced = False
    new_pairs = []
    for key, value in pairs:
        if key == "t":
            new_pairs.append((key, t_value))
            replaced = True
        else:
            new_pairs.append((key, value))
    if not replaced:
        new_pairs.append(("t", t_value))
    new_query = urlencode(new_pairs)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def transcript_segments_for_step(
    segments: Sequence[Dict[str, Any]], step_start: float, step_end: float
) -> List[Dict[str, Any]]:
    """Overlap-based transcript-to-step assignment (§9), replacing the v1 midpoint rule which
    dropped all text for some steps and misfiled it for others. A segment appears under a step
    when overlap_seconds >= TRANSCRIPT_SEG_OVERLAP_FRAC * segment_duration OR
    overlap_seconds >= TRANSCRIPT_STEP_OVERLAP_FRAC * step_duration. Safety net: if no segment
    qualifies but at least one segment overlaps the step at all, include the single segment with
    the largest overlap - a step that has speech must never render an empty transcript. Pure
    function: takes segments + a step window, has no I/O."""
    step_duration = max(0.0, step_end - step_start)
    qualifying: List[Dict[str, Any]] = []
    best_overlap = 0.0
    best_segment: Optional[Dict[str, Any]] = None
    for segment in segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))
        overlap = min(seg_end, step_end) - max(seg_start, step_start)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment
        seg_duration = max(0.0, seg_end - seg_start)
        qualifies = (seg_duration > 0 and overlap >= TRANSCRIPT_SEG_OVERLAP_FRAC * seg_duration) or (
            step_duration > 0 and overlap >= TRANSCRIPT_STEP_OVERLAP_FRAC * step_duration
        )
        if qualifies:
            qualifying.append(segment)
    if not qualifying and best_segment is not None:
        qualifying = [best_segment]
    qualifying.sort(key=lambda seg: float(seg.get("start", 0.0)))
    return qualifying


def render_transcript(transcript_segments: List[Dict[str, Any]], start: float, end: float) -> str:
    assigned = transcript_segments_for_step(transcript_segments, start, end)
    return " ".join(segment.get("text", "") for segment in assigned)


def render_command(video_path: Path, args: argparse.Namespace) -> int:
    _, _, _ = ensure_required_tools()
    work_dir = build_work_dir(video_path)
    meta = load_json(work_dir / "meta.json")
    steps_doc = load_json(work_dir / "steps.json")
    steps = validate_steps_data(steps_doc, meta, "steps.json")
    one_thing = steps_doc.get("one_thing", "")
    candidates = validate_candidates_data(load_json(work_dir / "candidates.json"), steps, "candidates.json")
    approved = load_json(work_dir / "approved_frames.json")
    approved_entries = validate_approved_data(approved, steps, candidates, "approved_frames.json")
    unresolved = [step_id for step_id, entry in approved_entries.items() if entry.get("needs_resample")]
    if unresolved and not args.allow_incomplete:
        for step_id in unresolved:
            window = approved_entries[step_id]["window"]
            print(
                f"Run: python3 \"{Path(__file__).resolve()}\" candidates --video \"{video_path}\" --resample {step_id} {window[0]} {window[1]}",
                file=sys.stderr,
            )
        return 2
    if unresolved and args.allow_incomplete:
        for step_id in unresolved:
            print(f"Warning: skipping image strip for {step_id} because it still needs resample", file=sys.stderr)
    transcript_segments = load_json(work_dir / "transcript.json").get("segments", [])
    images_dir = work_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = resolve_tool_path("BJJ_FFMPEG", "ffmpeg")
    if not ffmpeg_path:
        print("Error: required tool ffmpeg not found", file=sys.stderr)
        return 1
    html_parts = build_render_html(
        video_path,
        meta,
        steps,
        one_thing,
        candidates,
        approved_entries,
        transcript_segments,
        images_dir,
        args.embed_images,
        ffmpeg_path,
        techniques=steps_doc.get("techniques"),
    )
    html_path = video_path.parent / f"{video_path.stem}_review.html"
    md_path = video_path.parent / f"{video_path.stem}_review.md"
    write_text(html_path, html_parts["html"])
    write_text(md_path, html_parts["md"])
    print(f"Wrote HTML: {html_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"NEXT: open {html_path}")
    return 0


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_render_html(
    video_path: Path,
    meta: Dict[str, Any],
    steps: List[Dict[str, Any]],
    one_thing: str,
    candidates: Dict[str, Dict[str, Any]],
    approved_entries: Dict[str, Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    images_dir: Path,
    embed_images: bool,
    ffmpeg_path: str,
    techniques: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    video_title = strip_video_suffix(video_path.stem)
    badge = "BJJ Technique Review"
    subtitle = "A staged review sheet built from transcript segmentation and human frame judgment."
    # M10 (§9): optional source_url -> header watch-link + per-step/frame timestamp anchors.
    # Absent source_url must leave the page byte-for-byte identical to pre-M10 output.
    source_url = meta.get("source_url")
    video_link_html = ""
    video_link_md_line = None
    source_link_css = ""
    if source_url:
        escaped_source_url = html_escape(source_url, quote=True)
        video_link_html = (
            f'<p class="video-link"><a href="{escaped_source_url}" target="_blank" '
            'rel="noopener">▶ Watch the source video</a></p>'
        )
        video_link_md_line = f"[▶ Watch the source video]({source_url})"
        source_link_css = SOURCE_LINK_CSS
    toc_items = []
    html_cards = []
    md_sections = []
    work_dir_name = images_dir.parent.name
    # M8 (§9): tabs render only with 2+ techniques - zero or one technique must produce the
    # exact pre-M8 structure (no tech-tabs / tech-panel markup anywhere in HTML or MD).
    has_tabs = bool(techniques) and len(techniques) >= 2
    step_heading_prefix = "###" if has_tabs else "##"
    steps_by_technique: Dict[str, List[int]] = {}
    for idx, step in enumerate(steps, start=1):
        step_id = step["id"]
        category = step["category"]
        cat_style = CATEGORY_STYLES[category]
        toc_items.append(
            f'<li><a href="#section-{idx:02d}"><span class="toc-icon">{cat_style["icon"]}</span> <span class="toc-label">{cat_style["label"]}</span> {html_escape(step["title"])}</a></li>'
        )
        approved_entry = approved_entries[step_id]
        selected_letters = approved_entry.get("frames", []) if "frames" in approved_entry else []
        image_figures = []
        md_images = []
        if selected_letters:
            cells = candidates[step_id]["cells"]
            for frame_index, letter in enumerate(selected_letters, start=1):
                cell = cells[letter]
                source_thumb = images_dir / f"{step_id}_{frame_index}.jpg"
                result = run_command(
                    [
                        ffmpeg_path,
                        "-y",
                        "-ss",
                        str(cell["time"]),
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(source_thumb),
                    ]
                )
                if result.returncode != 0 or not source_thumb.exists():
                    stderr_tail = (result.stderr or "").strip().splitlines()[-1:]
                    raise RuntimeError(
                        f"ffmpeg failed to extract {step_id} frame {letter} at {cell['time']}s: "
                        + " ".join(stderr_tail)
                    )
                relative_thumb = Path(images_dir.parent.name) / "images" / source_thumb.name
                img_src = render_image_source(source_thumb, relative_thumb, embed_images)
                caption = format_mmss(cell["time"])
                cell_ts_url = timestamped_url(source_url, cell["time"])
                if cell_ts_url:
                    escaped_cell_ts_url = html_escape(cell_ts_url, quote=True)
                    figcaption_html = (
                        f'<a href="{escaped_cell_ts_url}" target="_blank" rel="noopener" '
                        f'title="Watch this moment in the video">'
                        f'<span class="play-glyph">▶</span>{caption}<span class="watch-hint"> · watch</span></a>'
                    )
                    caption_md = f"[{caption}]({cell_ts_url})"
                else:
                    figcaption_html = caption
                    caption_md = caption
                image_figures.append(
                    f'<figure><img src="{img_src}" alt="{html_escape(step["title"])} {letter}" loading="lazy"><figcaption>{figcaption_html}</figcaption></figure>'
                )
                md_images.append(f"![{step['title']} {letter}]({str(relative_thumb).replace(os.sep, '/')})\n\n*{caption_md}*")
        transcript_text = render_transcript(transcript_segments, float(step["start"]), float(step["end"]))
        key_points_html = "<ul class=\"key-points\">" + "".join(f"<li>{html_escape(point)}</li>" for point in step["key_points"]) + "</ul>"
        transcript_html = (
            "<details class=\"transcript-toggle\"><summary>Verbatim transcript (may overlap adjacent steps)</summary>"
            "<div class=\"transcript-content\">"
            + html_escape(transcript_text)
            + "</div></details>"
        )
        image_strip_html = ""
        if image_figures:
            image_strip_html = '<div class="image-strip">' + "".join(image_figures) + "</div>"
        explanation_text = step.get("explanation")
        explanation_html = f'<p class="explanation">{html_escape(explanation_text)}</p>' if explanation_text else ""
        step_time_text = f"{format_time(float(step['start']))} → {format_time(float(step['end']))}"
        step_ts_url = timestamped_url(source_url, float(step["start"]))
        # A link hidden inside the muted timestamp text is not discoverable — students
        # need an explicit affordance. The time range stays plain text; the link is a
        # visible "▶ Watch this step" pill at the end of the header row.
        watch_step_html = ""
        if step_ts_url:
            escaped_step_ts_url = html_escape(step_ts_url, quote=True)
            watch_step_html = (
                f'<a class="watch-step" href="{escaped_step_ts_url}" target="_blank" rel="noopener" '
                f'title="Watch this step in the video">▶ Watch this step</a>'
            )
        html_cards.append(
            f"""
            <section class="step-card" id="section-{idx:02d}" style="--card-accent: {cat_style['accent']}">
              <div class="step-header-row">
                <span class="step-badge">{cat_style['icon']} {cat_style['label']}</span>
                <span class="step-number">Step {idx:02d}</span>
                <span class="step-time">{step_time_text}</span>{watch_step_html}
              </div>
              <h3>{html_escape(step['title'])}</h3>
              {key_points_html}
              {image_strip_html}
              {explanation_html}
              {transcript_html}
            </section>
            """
        )
        md_meta_line = f"_{cat_style['icon']} {cat_style['label']}_ | `{format_time(float(step['start']))}` → `{format_time(float(step['end']))}`"
        if step_ts_url:
            md_meta_line += f" · [watch]({step_ts_url})"
        md_block = [f"{step_heading_prefix} {idx:02d}. {step['title']}", md_meta_line, ""]
        md_block.extend(f"- {point}" for point in step["key_points"])
        md_block.append("")
        md_block.extend(md_images)
        if explanation_text:
            md_block.append("")
            md_block.append(explanation_text)
        md_block.append("")
        if transcript_text:
            md_block.extend(
                ["<details><summary>Verbatim transcript (may overlap adjacent steps)</summary>", "", transcript_text, "", "</details>", ""]
            )
        md_sections.append("\n".join(md_block))
        if has_tabs:
            steps_by_technique.setdefault(step.get("technique"), []).append(idx - 1)

    nav_and_steps_html = f"""<nav class="toc">
    <h2>Table of Contents</h2>
    <ol>
      {''.join(toc_items)}
    </ol>
  </nav>
  <div class="steps">
    {''.join(html_cards)}
  </div>"""
    tabs_snippet_block = ""
    toc_md_lines: List[str] = []
    part_md_sections: List[str] = []
    if has_tabs:
        tab_buttons = []
        panels_html = []
        for t_index, tech in enumerate(techniques, start=1):
            tech_id = tech["id"]
            member_positions = steps_by_technique.get(tech_id, [])
            member_steps = [steps[p] for p in member_positions]
            if member_steps:
                window_start = min(float(s["start"]) for s in member_steps)
                window_end = max(float(s["end"]) for s in member_steps)
            else:
                window_start = window_end = 0.0
            tab_buttons.append(
                f'<button type="button" class="tech-tab" data-target="{tech_id}">'
                f'<span class="tech-tab-index">Part {t_index}</span>'
                f'<span class="tech-tab-title">{html_escape(tech["title"])}</span>'
                f'<span class="tech-tab-time">{format_time(window_start)} → {format_time(window_end)}</span>'
                f'</button>'
            )
            tech_summary = tech.get("summary")
            summary_html = ""
            if tech_summary:
                summary_html = (
                    '<div class="tech-summary">'
                    f'<span class="tech-summary-label">Part {t_index} summary</span>'
                    f'<p class="tech-summary-text">{html_escape(tech_summary)}</p>'
                    '</div>'
                )
            panel_toc = "".join(toc_items[p] for p in member_positions)
            panel_cards = "".join(html_cards[p] for p in member_positions)
            panels_html.append(
                f'<section class="tech-panel" id="{tech_id}">'
                f'<h2>{html_escape(tech["title"])}</h2>'
                f'{summary_html}'
                f'<nav class="toc tech-panel-toc"><h3>Steps in this part</h3><ol>{panel_toc}</ol></nav>'
                f'<div class="steps">{panel_cards}</div>'
                '</section>'
            )
            toc_md_lines.append(f"- **Part {t_index}: {tech['title']}**")
            for p in member_positions:
                member_step = steps[p]
                icon = CATEGORY_STYLES[member_step["category"]]["icon"]
                toc_md_lines.append(f"  - {icon} [{member_step['title']}](#section-{p + 1:02d})")
            part_md = [f"## Part {t_index}: {tech['title']}"]
            if tech_summary:
                part_md.append("")
                part_md.append(f"_{tech_summary}_")
            part_md.append("")
            part_md.extend(md_sections[p] for p in member_positions)
            part_md_sections.append("\n".join(part_md))
        nav_and_steps_html = f'<nav class="tech-tabs">{"".join(tab_buttons)}</nav>{"".join(panels_html)}'
        tabs_snippet_block = TABS_SNIPPET
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_escape(video_title)} - Review</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #081019;
      --card-bg: rgba(17, 24, 39, 0.82);
      --card-border: rgba(255,255,255,0.08);
      --text: #f3f4f6;
      --muted: #9ca3af;
      --primary: #fbbf24;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 2rem 1rem 4rem;
      color: var(--text);
      font-family: var(--font-body);
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.16), transparent 35%),
        radial-gradient(circle at bottom right, rgba(251,191,36,0.10), transparent 40%),
        var(--bg);
    }}
    .container {{ max-width: 1180px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 2.5rem; }}
    .badge {{
      display: inline-block;
      padding: .4rem 1rem;
      border-radius: 999px;
      border: 1px solid rgba(251,191,36,.35);
      background: linear-gradient(135deg, rgba(251,191,36,.18), rgba(59,130,246,.12));
      color: var(--primary);
      letter-spacing: .08em;
      font-size: .8rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 1rem 0 .75rem;
      font-family: var(--font-display);
      font-size: clamp(2.2rem, 4vw, 3.8rem);
      font-weight: 800;
      background: linear-gradient(135deg, #fff 20%, var(--primary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{ color: var(--muted); max-width: 760px; margin: 0 auto; }}
    .one-thing {{
      background: linear-gradient(135deg, rgba(251,191,36,.14), rgba(251,191,36,.03));
      border: 1px solid rgba(251,191,36,.3);
      border-left: 4px solid var(--primary);
      border-radius: 16px;
      box-shadow: 0 0 40px rgba(251,191,36,.08);
      padding: 1.4rem 1.75rem;
      margin: 0 0 2rem;
    }}
    .one-thing-label {{
      display: block;
      font-size: .75rem;
      letter-spacing: .14em;
      text-transform: uppercase;
      font-variant: small-caps;
      color: var(--primary);
      font-weight: 700;
      margin-bottom: .5rem;
    }}
    .one-thing-text {{
      margin: 0;
      font-family: var(--font-display);
      font-size: clamp(1.15rem, 2.2vw, 1.6rem);
      font-weight: 600;
      line-height: 1.4;
    }}
    .toc, .step-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      backdrop-filter: blur(14px);
      box-shadow: 0 20px 60px rgba(0,0,0,.25);
    }}
    .toc {{ padding: 1.5rem 1.75rem; margin: 2rem 0; }}
    .toc h2 {{ margin: 0 0 1rem; font-family: var(--font-display); }}
    .toc ol {{ margin: 0; padding: 0; list-style: none; display: grid; gap: .4rem; }}
    .toc a {{
      display: flex;
      gap: .65rem;
      align-items: center;
      color: var(--muted);
      text-decoration: none;
      padding: .45rem .65rem;
      border-radius: 10px;
    }}
    .toc a:hover {{ background: rgba(255,255,255,.05); color: var(--text); }}
    .toc-icon {{ width: 1.4rem; text-align: center; }}
    .toc-label {{ font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; min-width: 94px; opacity: .78; }}
    .steps {{ display: grid; gap: 1.4rem; }}
    .step-card {{ padding: 1.3rem 1.4rem 1.5rem; border-left: 4px solid var(--card-accent, var(--primary)); }}
    .step-header-row {{
      display: flex;
      gap: .75rem;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: .7rem;
    }}
    .step-badge {{
      padding: .25rem .7rem;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 999px;
      color: var(--primary);
      font-size: .78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .step-number, .step-time {{ color: var(--muted); font-size: .85rem; }}
    h3 {{
      margin: .1rem 0 .75rem;
      font-family: var(--font-display);
      font-size: 1.5rem;
      line-height: 1.2;
    }}
    .key-points {{ margin: 0 0 1rem 0; padding-left: 1.1rem; }}
    .key-points li {{ margin-bottom: .35rem; }}
    .image-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: .75rem;
      margin: 1rem 0;
    }}
    figure {{
      margin: 0;
      overflow: hidden;
      border-radius: 14px;
      background: rgba(0,0,0,.35);
      border: 1px solid rgba(255,255,255,.06);
    }}
    figure img {{ width: 100%; display: block; object-fit: cover; }}
    figcaption {{
      padding: .45rem .65rem .55rem;
      font-size: .83rem;
      color: var(--muted);
      border-top: 1px solid rgba(255,255,255,.06);
    }}
    .explanation {{
      margin: .9rem 0 0;
      color: #e5e7eb;
      font-size: 1.02rem;
      line-height: 1.7;
    }}
    details {{ margin-top: .75rem; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    .transcript-content {{
      margin-top: .65rem;
      padding: .9rem 1rem;
      border-radius: 12px;
      background: rgba(0,0,0,.28);
      color: #cbd5e1;
      line-height: 1.7;
      white-space: pre-wrap;
    }}
    .footer {{ color: var(--muted); text-align: center; margin-top: 2rem; }}{source_link_css}
  </style>
</head>
<body>
<div class="container">
  <header>
    <span class="badge">{badge}</span>
    <h1>{html_escape(video_title)}</h1>
    <p class="subtitle">{html_escape(subtitle)}</p>{video_link_html}
  </header>
  <section class="one-thing">
    <span class="one-thing-label">If you remember one thing</span>
    <p class="one-thing-text">{html_escape(one_thing)}</p>
  </section>
  {nav_and_steps_html}
  <p class="footer">{html_escape(video_title)} review sheet &middot; click any picture to enlarge</p>
</div>
{LIGHTBOX_SNIPPET}
{tabs_snippet_block}
</body>
</html>
"""
    md = [
        f"# {badge}: {video_title}",
        "",
        subtitle,
        "",
    ]
    if video_link_md_line:
        md.extend([video_link_md_line, ""])
    md.extend([
        f"> **If you remember one thing:** {one_thing}",
        "",
        "## Table of Contents",
        "",
    ])
    if has_tabs:
        md.extend(toc_md_lines)
        md.append("")
        md.extend(part_md_sections)
    else:
        for idx, step in enumerate(steps, start=1):
            md.append(f"- {CATEGORY_STYLES[step['category']]['icon']} [{step['title']}](#section-{idx:02d})")
        md.append("")
        md.extend(md_sections)
    return {"html": html, "md": "\n".join(md)}


def check_command(video_path: Path, args: argparse.Namespace) -> int:
    work_dir = build_work_dir(video_path)
    meta = load_json(work_dir / "meta.json")
    steps_doc = load_json(work_dir / "steps.json")
    steps = validate_steps_data(steps_doc, meta, "steps.json")
    for warning in step_grounding_warnings(steps, work_dir / "transcript.json"):
        print(f"WARNING: {warning}", file=sys.stderr)
    one_thing_warning = one_thing_grounding_warning(steps_doc.get("one_thing", ""), work_dir / "transcript.json")
    if one_thing_warning:
        print(f"WARNING: {one_thing_warning}", file=sys.stderr)
    for warning in technique_grounding_warnings(steps_doc, steps, work_dir / "transcript.json"):
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.what == "steps":
        print("PASS")
        print("NEXT: run candidates")
        return 0
    candidates = validate_candidates_data(load_json(work_dir / "candidates.json"), steps, "candidates.json")
    approved_doc = load_json(work_dir / "approved_frames.json")
    validate_approved_data(approved_doc, steps, candidates, "approved_frames.json")
    for warning in approved_frames_warnings(steps, approved_doc):
        print(f"WARNING: {warning}", file=sys.stderr)
    print("PASS")
    print("NEXT: run render")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage BJJ review sheet generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--subtitles")
    prepare.add_argument("--url", default=None)
    prepare.add_argument("--threshold", type=float, default=DEFAULT_SCENE_THRESHOLD)
    prepare.add_argument("--min-duration", type=float, default=DEFAULT_MIN_SCENE_GAP)
    prepare.add_argument("--force-refresh", action="store_true")

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--video", required=True)
    candidates.add_argument("--per-step", type=int, default=UNIFORM_SAMPLES_PER_STEP)
    candidates.add_argument("--no-ocr-filter", action="store_true")
    candidates.add_argument("--resample", nargs=3, metavar=("STEP_ID", "START", "END"))
    candidates.add_argument("--force-refresh", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--video", required=True)
    render.add_argument("--embed-images", action="store_true")
    render.add_argument("--allow-incomplete", action="store_true")

    check = subparsers.add_parser("check")
    check.add_argument("--video", required=True)
    check.add_argument("--what", choices=["steps", "approved", "all"], default="all")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    video_path = Path(args.video).expanduser().resolve()
    try:
        if args.command == "prepare":
            return prepare_command(video_path, args)
        if args.command == "candidates":
            return candidates_command(video_path, args)
        if args.command == "render":
            return render_command(video_path, args)
        if args.command == "check":
            return check_command(video_path, args)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: file not found: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.error("unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
