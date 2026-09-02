from __future__ import annotations

import argparse
import csv
import contextlib
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


LEXICAL_RE = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
TIMESTAMP_UNIT_RE = re.compile(r"[A-Za-z]+|[0-9]+|[\u3400-\u9fff]")
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_CHANNELS = 1


def _process_exists(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def single_run_lock(output_root: Path) -> Iterable[None]:
    """Prevent two workflow runs from competing for the same GPU/output root."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".voice_qc.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing_pid = None
        try:
            match = re.search(r"pid=(\d+)", lock_path.read_text(encoding="ascii"))
            existing_pid = int(match.group(1)) if match else None
        except (OSError, ValueError):
            pass
        if existing_pid is not None and _process_exists(existing_pid):
            raise RuntimeError(f"流程已在运行：{lock_path}（pid={existing_pid}）") from exc
        if existing_pid is not None:
            lock_path.unlink(missing_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"流程已在运行或存在无法确认的锁文件：{lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class Segment:
    segment_id: str
    text: str
    paragraph_after: bool = False


@dataclass
class Candidate:
    segment_id: str
    text: str
    round_index: int
    audio_path: str
    seed: int = 0
    recognized_text: str = ""
    timestamps_ms: list[list[int]] = field(default_factory=list)
    acoustic: dict[str, Any] = field(default_factory=dict)
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def score(self) -> tuple[float, float, float, float, float]:
        return (
            float(len(self.hard_failures)),
            float(len(self.warnings)),
            float(self.metrics.get("cer", math.inf)),
            float(self.metrics.get("maximum_speed_ratio", math.inf)),
            float(self.metrics.get("maximum_internal_gap_ms", math.inf)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoxCPM2 automatic QC and regeneration flow")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument("--check-only", action="store_true", help="Validate configuration without generating audio")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for section in ("paths", "generation", "quality"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"配置缺少对象：{section}")
    config_dir = path.resolve().parent
    for key, value in list(payload["paths"].items()):
        if not isinstance(value, str):
            continue
        expanded = os.path.expandvars(value)
        path_value = Path(expanded)
        if not path_value.is_absolute():
            path_value = config_dir / path_value
        payload["paths"][key] = str(path_value.resolve())
    return payload


def lexical_units(text: str) -> list[tuple[str, int]]:
    return [(match.group(0).casefold(), match.start()) for match in LEXICAL_RE.finditer(text)]


def timestamp_units(text: str) -> list[tuple[str, int]]:
    return [(match.group(0).casefold(), match.start()) for match in TIMESTAMP_UNIT_RE.finditer(text)]


def normalized_text(text: str) -> str:
    return "".join(unit for unit, _ in lexical_units(text))


def load_reviewed_segments(text: str) -> list[Segment]:
    """Load generation segments whose boundaries were reviewed before the run.

    This function deliberately does not infer, split, merge, or shorten semantic
    units. Each non-empty input line is only the serialized form of one segment
    that has already been selected by a human or language model after reading the
    complete source text. Blank lines preserve paragraph boundaries for reports.
    """
    source_lines = text.splitlines()
    nonempty_indices = [index for index, line in enumerate(source_lines) if line.strip()]
    if not nonempty_indices:
        raise ValueError("input.txt没有可生成文本")
    reviewed_segments: list[tuple[str, bool]] = []
    for position, source_index in enumerate(nonempty_indices):
        line = source_lines[source_index].strip()
        next_index = nonempty_indices[position + 1] if position + 1 < len(nonempty_indices) else None
        paragraph_after = next_index is None or next_index > source_index + 1
        reviewed_segments.append((line, paragraph_after))
    return [
        Segment(f"S{index:03d}", segment_text, paragraph_after)
        for index, (segment_text, paragraph_after) in enumerate(reviewed_segments, start=1)
    ]


def _segment_text_without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _validate_segment_payload(payload: Any, source_text: str) -> list[Segment]:
    raw_segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("语义分段器必须返回非空segments数组")
    segments: list[Segment] = []
    for index, item in enumerate(raw_segments, start=1):
        if isinstance(item, str):
            text = item.strip()
            paragraph_after = False
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            paragraph_after = bool(item.get("paragraph_after", False))
        else:
            raise ValueError(f"语义分段器返回了无效片段：第{index}项")
        if not text:
            raise ValueError(f"语义分段器返回了空片段：第{index}项")
        segments.append(Segment(f"S{index:03d}", text, paragraph_after))
    source_normalized = _segment_text_without_whitespace(source_text)
    segmented_normalized = _segment_text_without_whitespace("".join(item.text for item in segments))
    if segmented_normalized != source_normalized:
        raise ValueError("语义分段器修改了原文；只允许改变边界，不允许增删或改写文字")
    if not segments[-1].paragraph_after:
        segments[-1] = Segment(segments[-1].segment_id, segments[-1].text, True)
    return segments


def load_semantic_segments(source_text: str, segmentation: dict[str, Any], run_dir: Path) -> list[Segment]:
    mode = str(segmentation.get("mode", "reviewed")).strip().lower()
    if mode in ("reviewed", "manual"):
        return load_reviewed_segments(source_text)
    if mode != "ai_command":
        raise ValueError("segmentation.mode必须是reviewed或ai_command")
    command_template = segmentation.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError("ai_command模式必须配置非空的segmentation.command数组")
    segmentation_dir = run_dir / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=False)
    input_path = segmentation_dir / "source.txt"
    output_path = segmentation_dir / "segments.json"
    input_path.write_text(source_text, encoding="utf-8")
    replacements = {
        "{input}": str(input_path),
        "{output}": str(output_path),
        "{project_root}": str(PROJECT_ROOT),
    }
    command = [str(part).format(**replacements) for part in command_template]
    _run_command(command, segmentation_dir / "segmenter.log", cwd=PROJECT_ROOT)
    if not output_path.is_file():
        raise FileNotFoundError(f"语义分段器没有生成结果：{output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    segments = _validate_segment_payload(payload, source_text)
    (segmentation_dir / "validated_segments.json").write_text(
        json.dumps([asdict(item) for item in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return segments


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _alignment(
    target: Sequence[str], recognized: Sequence[str]
) -> tuple[dict[int, int], list[tuple[str, int, int, int, int]]]:
    matcher = difflib.SequenceMatcher(a=target, b=recognized, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping, matcher.get_opcodes()


def _apply_known_asr_substitutions(
    target: str,
    recognized: str,
    substitutions: dict[str, str],
) -> tuple[str, list[str]]:
    matcher = difflib.SequenceMatcher(a=target, b=recognized, autojunk=False)
    corrected: list[str] = []
    applied: list[str] = []
    for tag, target_start, target_end, recognized_start, recognized_end in matcher.get_opcodes():
        target_span = target[target_start:target_end]
        recognized_span = recognized[recognized_start:recognized_end]
        if tag == "replace" and substitutions.get(target_span) == recognized_span:
            corrected.append(target_span)
            applied.append(f"{target_span}→{recognized_span}")
        else:
            corrected.append(recognized_span)
    return "".join(corrected), applied


def _punctuation_kind(text: str, left_position: int, right_position: int) -> str | None:
    between = text[left_position + 1 : right_position]
    if re.search(r"[。！？!?；;\n]", between):
        return "sentence"
    if re.search(r"[，,、：:]", between):
        return "comma"
    return None


def _speech_rate_from_timestamps(timestamps: list[list[int]]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration_seconds = (timestamps[-1][1] - timestamps[0][0]) / 1000.0
    return len(timestamps) / duration_seconds if duration_seconds > 0 else 0.0


def evaluate_candidate(
    raw: dict[str, Any],
    round_index: int,
    config: dict[str, Any],
    reference_speech_rate: float = 0.0,
    check_internal_prosody: bool = True,
) -> Candidate:
    quality = config["quality"]
    target_text = str(raw.get("target_text") or "")
    recognized_text = str(raw.get("recognized_text") or "")
    candidate = Candidate(
        segment_id=str(raw.get("segment_id") or ""),
        text=target_text,
        round_index=round_index,
        audio_path=str(raw.get("audio_path") or ""),
        seed=int(raw.get("seed") or 0),
        recognized_text=recognized_text,
        timestamps_ms=[[int(pair[0]), int(pair[1])] for pair in raw.get("timestamps_ms") or [] if len(pair) >= 2],
        acoustic=dict(raw.get("acoustic") or {}),
    )
    if raw.get("error"):
        candidate.hard_failures.append("音频分析失败")
        candidate.metrics["worker_error"] = raw.get("error")
        return candidate

    target_units = lexical_units(target_text)
    target = "".join(unit for unit, _ in target_units)
    recognized_raw = normalized_text(recognized_text)
    recognized, known_substitutions = _apply_known_asr_substitutions(
        target,
        recognized_raw,
        {
            str(target_span): str(recognized_span)
            for target_span, recognized_span in dict(quality.get("known_asr_substitutions") or {}).items()
        },
    )
    candidate.warnings.extend(f"ASR已知近音替换：{item}" for item in known_substitutions)
    if not target:
        candidate.hard_failures.append("目标文本为空")
        return candidate

    distance = levenshtein_distance(target, recognized)
    cer = distance / max(len(target), 1)
    mapping, opcodes = _alignment(target, recognized)
    missing_or_extra_span = 0
    for tag, target_start, target_end, recognized_start, recognized_end in opcodes:
        if tag != "equal":
            missing_or_extra_span = max(
                missing_or_extra_span,
                target_end - target_start,
                recognized_end - recognized_start,
            )

    timestamps = candidate.timestamps_ms
    timing_target_units = timestamp_units(target_text)
    timing_recognized_units = timestamp_units(recognized_text)
    timing_target = [unit for unit, _ in timing_target_units]
    timing_recognized = [unit for unit, _ in timing_recognized_units]
    timing_mapping, _ = _alignment(timing_target, timing_recognized)
    timing_mapping = {
        target_index: recognized_index
        for target_index, recognized_index in timing_mapping.items()
        if recognized_index < len(timestamps)
    }
    timestamp_coverage = len(timing_mapping) / max(len(timing_target), 1)
    internal_gaps: list[float] = []
    gap_details: list[dict[str, Any]] = []
    punctuation_gaps: list[dict[str, Any]] = []
    for target_index in range(len(timing_target_units) - 1):
        recognized_left = timing_mapping.get(target_index)
        recognized_right = timing_mapping.get(target_index + 1)
        if recognized_left is None or recognized_right is None or recognized_right != recognized_left + 1:
            continue
        if recognized_right >= len(timestamps):
            continue
        gap_ms = max(0, timestamps[recognized_right][0] - timestamps[recognized_left][1])
        punctuation_kind = _punctuation_kind(
            target_text,
            timing_target_units[target_index][1],
            timing_target_units[target_index + 1][1],
        )
        if punctuation_kind:
            left_duration_ms = max(0, timestamps[recognized_left][1] - timestamps[recognized_left][0])
            punctuation_gaps.append(
                {
                    "kind": punctuation_kind,
                    "left": timing_target_units[target_index][0],
                    "right": timing_target_units[target_index + 1][0],
                    "gap_ms": gap_ms,
                    "left_duration_ms": left_duration_ms,
                    "boundary_duration_ms": left_duration_ms + gap_ms,
                }
            )
            continue
        internal_gaps.append(float(gap_ms))
        gap_details.append(
            {
                "left": timing_target_units[target_index][0],
                "right": timing_target_units[target_index + 1][0],
                "gap_ms": gap_ms,
            }
        )

    gap_median = statistics.median(internal_gaps) if internal_gaps else 0.0
    gap_mad = statistics.median([abs(value - gap_median) for value in internal_gaps]) if internal_gaps else 0.0
    gap_threshold = max(
        float(quality["internal_gap_floor_ms"]),
        gap_median + float(quality["internal_gap_mad_multiplier"]) * gap_mad,
    )
    abnormal_gaps = [item for item in gap_details if item["gap_ms"] > gap_threshold]
    short_punctuation_gaps: list[dict[str, Any]] = []
    long_punctuation_gaps: list[dict[str, Any]] = []
    for item in punctuation_gaps:
        prefix = "sentence" if item["kind"] == "sentence" else "comma"
        minimum = float(quality[f"{prefix}_pause_min_ms"])
        maximum = float(quality[f"{prefix}_pause_max_ms"])
        boundary_minimum = float(quality["punctuation_boundary_min_ms"])
        if item["gap_ms"] < minimum and item["boundary_duration_ms"] < boundary_minimum:
            short_punctuation_gaps.append(item)
        elif item["gap_ms"] > maximum:
            long_punctuation_gaps.append(item)

    window_size = int(quality["speed_window_chars"])
    window_rates: list[float] = []
    speed_window_texts: list[str] = []
    for start in range(0, max(0, len(timing_target) - window_size + 1)):
        speed_window_texts.append("".join(timing_target[start : start + window_size]))
        mapped = [timing_mapping.get(index) for index in range(start, start + window_size)]
        if any(index is None for index in mapped):
            window_rates.append(0.0)
            continue
        recognized_indices = [int(index) for index in mapped if index is not None]
        if recognized_indices != list(range(recognized_indices[0], recognized_indices[0] + window_size)):
            window_rates.append(0.0)
            continue
        if recognized_indices[-1] >= len(timestamps):
            window_rates.append(0.0)
            continue
        duration_seconds = (timestamps[recognized_indices[-1]][1] - timestamps[recognized_indices[0]][0]) / 1000.0
        window_rates.append(window_size / duration_seconds if duration_seconds > 0 else 0.0)
    valid_rates = [rate for rate in window_rates if rate > 0]
    median_rate = statistics.median(valid_rates) if valid_rates else 0.0
    speed_ratios = [rate / median_rate if rate > 0 and median_rate > 0 else 0.0 for rate in window_rates]
    limit = float(quality["speed_ratio_limit"])
    required_consecutive = int(quality["speed_consecutive_windows"])
    consecutive = 0
    speed_failed = False
    for ratio in speed_ratios:
        consecutive = consecutive + 1 if ratio > limit else 0
        if consecutive >= required_consecutive:
            speed_failed = True
            break

    acoustic = candidate.acoustic
    duration_seconds = float(acoustic.get("duration_seconds") or 0.0)
    sample_rate = int(acoustic.get("sample_rate") or 0)
    channels = int(acoustic.get("channels") or 0)
    clipping_fraction = float(acoustic.get("clipping_fraction") or 0.0)
    overall_speech_rate = _speech_rate_from_timestamps(timestamps)
    speed_ratio_to_reference = (
        overall_speech_rate / reference_speech_rate
        if overall_speech_rate > 0 and reference_speech_rate > 0
        else 0.0
    )
    first_half = dict(acoustic.get("first_half") or {})
    second_half = dict(acoustic.get("second_half") or {})
    first_centroid = float(first_half.get("spectral_centroid_hz") or 0.0)
    second_centroid = float(second_half.get("spectral_centroid_hz") or 0.0)
    first_high = float(first_half.get("high_frequency_ratio_4khz") or 0.0)
    second_high = float(second_half.get("high_frequency_ratio_4khz") or 0.0)
    first_rms = float(first_half.get("rms_db") or -120.0)
    second_rms = float(second_half.get("rms_db") or -120.0)
    tail_centroid_ratio = second_centroid / first_centroid if first_centroid > 0 else 1.0
    tail_high_ratio = second_high / first_high if first_high > 0 else 1.0
    tail_rms_drop_db = first_rms - second_rms

    if timestamp_coverage < 0.70:
        candidate.hard_failures.append("逐字时间戳覆盖不足")
    if check_internal_prosody and abnormal_gaps:
        candidate.hard_failures.append("句内异常停顿")
    if check_internal_prosody and short_punctuation_gaps:
        candidate.hard_failures.append("标点停顿过短")
    if check_internal_prosody and long_punctuation_gaps:
        candidate.hard_failures.append("标点停顿过长")
    if check_internal_prosody and speed_failed:
        candidate.hard_failures.append("局部语速异常")
    if cer > float(quality["cer_limit"]):
        candidate.hard_failures.append("转写差异过大")
    if check_internal_prosody and missing_or_extra_span >= int(quality["missing_or_extra_span_limit"]):
        candidate.hard_failures.append("存在连续漏字、错字或重复")
    if duration_seconds < float(quality["minimum_audio_seconds"]):
        candidate.hard_failures.append("音频为空或过短")
    if sample_rate != 48000 or channels != 1:
        candidate.hard_failures.append("音频格式异常")
    if clipping_fraction > float(quality["clipping_fraction_limit"]):
        candidate.hard_failures.append("音频削波")
    if overall_speech_rate > 0 and (
        overall_speech_rate < float(quality["overall_speed_min_chars_per_second"])
        or overall_speech_rate > float(quality["overall_speed_max_chars_per_second"])
    ):
        candidate.hard_failures.append("整体语速异常")
    if reference_speech_rate > 0 and speed_ratio_to_reference > float(quality["overall_speed_ratio_max"]):
        candidate.hard_failures.append("整体语速明显快于参考")
    if duration_seconds >= float(quality["tail_analysis_min_seconds"]):
        if (
            tail_centroid_ratio < float(quality["tail_centroid_ratio_limit"])
            and tail_high_ratio < float(quality["tail_high_frequency_ratio_limit"])
        ):
            candidate.hard_failures.append("后半段频谱明显变暗")
        if tail_rms_drop_db > float(quality["tail_rms_drop_db_limit"]):
            candidate.hard_failures.append("后半段响度异常下降")

    candidate.metrics.update(
        {
            "target_normalized": target,
            "recognized_normalized_raw": recognized_raw,
            "recognized_normalized": recognized,
            "known_asr_substitutions": known_substitutions,
            "cer": cer,
            "edit_distance": distance,
            "maximum_changed_span": missing_or_extra_span,
            "timestamp_coverage": timestamp_coverage,
            "internal_gap_median_ms": gap_median,
            "internal_gap_mad_ms": gap_mad,
            "internal_gap_threshold_ms": gap_threshold,
            "maximum_internal_gap_ms": max(internal_gaps, default=0.0),
            "abnormal_gaps": abnormal_gaps,
            "punctuation_gaps": punctuation_gaps,
            "short_punctuation_gaps": short_punctuation_gaps,
            "long_punctuation_gaps": long_punctuation_gaps,
            "window_rates": window_rates,
            "speed_window_texts": speed_window_texts,
            "median_window_rate": median_rate,
            "maximum_speed_ratio": max(speed_ratios, default=0.0),
            "internal_prosody_enforced": check_internal_prosody,
            "overall_speech_rate": overall_speech_rate,
            "reference_speech_rate": reference_speech_rate,
            "overall_speed_ratio_to_reference": speed_ratio_to_reference,
            "tail_centroid_ratio": tail_centroid_ratio,
            "tail_high_frequency_ratio": tail_high_ratio,
            "tail_rms_drop_db": tail_rms_drop_db,
        }
    )
    return candidate


def add_muffle_warning(candidate: Candidate, reference_acoustic: dict[str, Any], batch_acoustics: list[dict[str, Any]]) -> None:
    if not candidate.acoustic or not reference_acoustic:
        return
    batch_centroids = [float(item.get("spectral_centroid_hz") or 0.0) for item in batch_acoustics]
    batch_centroids = [value for value in batch_centroids if value > 0]
    reference_centroid = float(reference_acoustic.get("spectral_centroid_hz") or 0.0)
    reference_high = float(reference_acoustic.get("high_frequency_ratio_4khz") or 0.0)
    centroid = float(candidate.acoustic.get("spectral_centroid_hz") or 0.0)
    high_ratio = float(candidate.acoustic.get("high_frequency_ratio_4khz") or 0.0)
    batch_median = statistics.median(batch_centroids) if batch_centroids else reference_centroid
    baseline_centroid = statistics.median([value for value in (reference_centroid, batch_median) if value > 0])
    if baseline_centroid > 0 and reference_high > 0 and centroid < baseline_centroid * 0.75 and high_ratio < reference_high * 0.65:
        candidate.warnings.append("频谱偏暗（仅提示）")
    candidate.metrics["muffle"] = {
        "reference_centroid_hz": reference_centroid,
        "batch_centroid_median_hz": batch_median,
        "centroid_ratio": centroid / baseline_centroid if baseline_centroid > 0 else None,
        "high_frequency_ratio_vs_reference": high_ratio / reference_high if reference_high > 0 else None,
    }


def preflight(config: dict[str, Any]) -> None:
    paths = config["paths"]
    required_files = (
        "voxcpm_python",
        "voxcpm_batch_script",
        "asr_python",
        "reference_audio",
        "reference_text_file",
        "input_text",
        "ffmpeg",
        "ffprobe",
    )
    required_directories = ("voxcpm_model", "asr_model", "vad_model")
    missing = [key for key in required_files if not Path(paths[key]).is_file()]
    missing.extend(key for key in required_directories if not Path(paths[key]).is_dir())
    if missing:
        raise FileNotFoundError("环境预检失败：" + ", ".join(missing))
    cfg = float(config["generation"]["cfg"])
    steps = int(config["generation"]["steps"])
    if not 1.0 <= cfg <= 3.0:
        raise ValueError("CFG必须在1.0到3.0之间")
    if not 4 <= steps <= 30:
        raise ValueError("steps必须在4到30之间")
    if config["generation"].get("reference_mode") not in ("combined", "reference"):
        raise ValueError("reference_mode必须是combined或reference")
    seed = int(config["generation"]["base_seed"])
    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("base_seed必须在0到2^63-1之间")
    if int(config["generation"]["retry_seed_stride"]) <= 0:
        raise ValueError("retry_seed_stride必须大于0")
    if int(config["generation"]["max_quality_retries"]) < 0:
        raise ValueError("max_quality_retries不能小于0")
    if int(config["generation"].get("max_parallel", 1)) != 1:
        raise ValueError("max_parallel必须固定为1；流程禁止并行GPU生成")
    if config["generation"]["reference_mode"] == "combined":
        reference_text = Path(paths["reference_text_file"]).read_text(encoding="utf-8").strip()
        if not reference_text:
            raise ValueError("combined参考模式的参考文字不能为空")
    segmentation = config.get("segmentation") or {"mode": "reviewed"}
    mode = str(segmentation.get("mode", "reviewed")).strip().lower()
    if mode not in ("reviewed", "manual", "ai_command"):
        raise ValueError("segmentation.mode必须是reviewed或ai_command")
    if mode == "ai_command" and (not isinstance(segmentation.get("command"), list) or not segmentation["command"]):
        raise ValueError("ai_command模式必须配置segmentation.command数组")
    preferred_min = int(segmentation.get("preferred_min_chars", 120))
    preferred_max = int(segmentation.get("preferred_max_chars", 180))
    hard_max = int(segmentation.get("hard_max_chars", config["generation"].get("max_segment_chars", 240)))
    if preferred_min < 1 or preferred_max < preferred_min:
        raise ValueError("segmentation.preferred_min_chars和preferred_max_chars范围无效")
    if hard_max < preferred_max:
        raise ValueError("segmentation.hard_max_chars不能小于preferred_max_chars")


def runtime_resource_preflight(segments: Sequence[Segment], config: dict[str, Any]) -> None:
    generation = config["generation"]
    segmentation = config.get("segmentation") or {}
    max_segment_chars = int(
        segmentation.get("hard_max_chars", generation.get("max_segment_chars", 240))
    )
    preferred_min = int(segmentation.get("preferred_min_chars", 120))
    preferred_max = int(segmentation.get("preferred_max_chars", 180))
    oversized = [segment for segment in segments if len(segment.text) > max_segment_chars]
    if oversized:
        details = "、".join(f"{segment.segment_id}={len(segment.text)}字" for segment in oversized[:8])
        raise ValueError(
            f"安全检查拒绝生成：存在超过{max_segment_chars}字的单段（{details}）。"
            "请先让AI按语义分段后再运行。"
        )
    if str(generation.get("device", "cuda")).startswith("cuda"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                encoding="ascii",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            raise RuntimeError("安全检查失败：找不到nvidia-smi，拒绝启动CUDA生成") from exc
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"安全检查失败：无法读取GPU显存：{result.stderr.strip()[-500:]}")
        free_mb = int(float(result.stdout.strip().splitlines()[0].strip()))
        required_mb = int(generation.get("min_free_vram_mb", 4096))
        if free_mb < required_mb:
            raise RuntimeError(f"安全检查拒绝生成：当前可用显存{free_mb}MB，低于安全下限{required_mb}MB")
        print(
            f"资源检查通过：可用显存{free_mb}MB；"
            f"语义分段参考{preferred_min}-{preferred_max}字；"
            f"硬上限{max_segment_chars}字；"
            f"本次最大单段{max((len(item.text) for item in segments), default=0)}字"
        )


def _run_command(
    command: list[str],
    log_path: Path,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    process_env = os.environ.copy()
    if env_overrides:
        process_env.update(env_overrides)
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=process_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + process.stdout + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    if process.stdout:
        print(process.stdout, end="")
    if process.returncode != 0:
        raise RuntimeError(f"命令失败，exit={process.returncode}，日志={log_path}\n{process.stderr[-2000:]}")


def detect_silence_intervals_ffmpeg(
    audio_path: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    """Run the fixed silencedetect command and return intervals in sample frames."""
    import soundfile as sf

    ffmpeg_path = Path(config["paths"]["ffmpeg"])
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio_path),
        "-af",
        "silencedetect=noise=-50dB:d=0.05",
        "-f",
        "null",
        "NUL",
    ]
    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path = run_dir / "silence_detection.log"
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n"
        + process.stdout + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg静音检测失败，exit={process.returncode}，日志={log_path}")
    info = sf.info(audio_path)
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", process.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", process.stderr)]
    intervals: list[tuple[int, int]] = []
    for index, start_seconds in enumerate(starts):
        end_seconds = ends[index] if index < len(ends) else float(info.duration)
        start_frame = max(0, int(round(start_seconds * info.samplerate)))
        end_frame = min(int(info.frames), int(round(end_seconds * info.samplerate)))
        if end_frame > start_frame:
            intervals.append((start_frame, end_frame))
    return intervals


def compress_silence_samples(
    samples: Any,
    intervals: Sequence[tuple[int, int]],
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    threshold_ms: float = 200.0,
    target_ms: float = 200.0,
    silence_level: float = 10.0 ** (-50.0 / 20.0),
) -> tuple[Any, list[dict[str, Any]]]:
    """Remove only the excess samples inside confirmed long silent intervals."""
    import numpy as np

    source = np.ascontiguousarray(samples, dtype=np.float32)
    threshold_frames = int(round(threshold_ms * sample_rate / 1000.0))
    target_frames = int(round(target_ms * sample_rate / 1000.0))
    if target_frames <= 0 or target_frames > threshold_frames:
        raise ValueError("静音目标长度必须大于0且不超过200毫秒")
    deletions: list[tuple[int, int]] = []
    edits: list[dict[str, Any]] = []
    for raw_start, raw_end in sorted(intervals):
        start = max(0, min(len(source), int(raw_start)))
        end = max(start, min(len(source), int(raw_end)))
        duration = end - start
        if duration <= threshold_frames:
            continue
        delete_start = start + target_frames
        delete_end = end
        deleted = source[delete_start:delete_end]
        if deleted.size and float(np.max(np.abs(deleted))) >= silence_level:
            raise RuntimeError(
                f"FFmpeg静音区间包含有声采样，拒绝剪辑：{start / sample_rate:.3f}-"
                f"{end / sample_rate:.3f}秒"
            )
        deletions.append((delete_start, delete_end))
        edits.append(
            {
                "start_frame": start,
                "end_frame": end,
                "duration_ms_before": duration * 1000.0 / sample_rate,
                "target_duration_ms": target_frames * 1000.0 / sample_rate,
                "delete_start_frame": delete_start,
                "delete_end_frame": delete_end,
                "deleted_frames": delete_end - delete_start,
            }
        )
    if not deletions:
        return source, edits
    keep_ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in deletions:
        if start > cursor:
            keep_ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(source):
        keep_ranges.append((cursor, len(source)))
    edited = np.concatenate([source[start:end] for start, end in keep_ranges]) if keep_ranges else np.array([], dtype=np.float32)
    for item in edits:
        item["start_seconds"] = float(item["start_frame"]) / sample_rate
        item["end_seconds"] = float(item["end_frame"]) / sample_rate
    return np.ascontiguousarray(edited, dtype=np.float32), edits


def _remove_ranges(samples: Any, deletions: Sequence[tuple[int, int]]) -> Any:
    import numpy as np

    source = np.ascontiguousarray(samples, dtype=np.float32)
    pieces: list[Any] = []
    cursor = 0
    for start, end in deletions:
        if start > cursor:
            pieces.append(source[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(source):
        pieces.append(source[cursor:])
    return np.ascontiguousarray(np.concatenate(pieces) if pieces else np.array([], dtype=np.float32))


def apply_silence_edits_task(
    assembly: dict[str, Any],
    segments: list[dict[str, Any]],
    run_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create the post-processed derivative while preserving the original assembly."""
    import numpy as np
    import soundfile as sf

    run_path = Path(run_dir)
    original_path = Path(assembly["final_candidate_path"])
    original, sample_rate = sf.read(original_path, dtype="float32", always_2d=False)
    if sample_rate != OUTPUT_SAMPLE_RATE or original.ndim != 1:
        raise RuntimeError("原始拼接文件格式异常，不能执行静音后期处理")
    intervals = detect_silence_intervals_ffmpeg(original_path, run_path, config)
    edited, edits = compress_silence_samples(original, intervals)
    final_path = run_path / "final_candidate.wav"
    sf.write(final_path, edited, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
    written, written_rate = sf.read(final_path, dtype="float32", always_2d=False)
    if written_rate != OUTPUT_SAMPLE_RATE or not np.array_equal(written, edited):
        raise RuntimeError("静音后期输出未能逐采样保留")
    remaining_intervals = detect_silence_intervals_ffmpeg(final_path, run_path, config)

    deletion_ranges = [
        (int(item["delete_start_frame"]), int(item["delete_end_frame"])) for item in edits
    ]
    post_dir = run_path / "postprocessed_segments"
    post_dir.mkdir(parents=False, exist_ok=False)
    timeline: list[dict[str, Any]] = []
    frame_cursor = 0
    for item in segments:
        segment_id = str(item["segment_id"])
        source_start = int(next(row["start_frame"] for row in assembly["timeline"] if row["segment_id"] == segment_id))
        source_end = int(next(row["end_frame"] for row in assembly["timeline"] if row["segment_id"] == segment_id))
        segment_samples = original[source_start:source_end]
        local_deletions = [
            (max(0, start - source_start), min(source_end, end) - source_start)
            for start, end in deletion_ranges
            if start < source_end and end > source_start
        ]
        post_samples = _remove_ranges(segment_samples, local_deletions)
        segment_path = post_dir / f"{segment_id}.wav"
        sf.write(segment_path, post_samples, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
        reread, reread_rate = sf.read(segment_path, dtype="float32", always_2d=False)
        if reread_rate != OUTPUT_SAMPLE_RATE or not np.array_equal(reread, post_samples):
            raise RuntimeError(f"后期片段未能逐采样保留：{segment_id}")
        start_frame = frame_cursor
        end_frame = start_frame + len(post_samples)
        timeline.append(
            {
                "segment_id": segment_id,
                "text": str(item["text"]),
                "seed": int(next(row["seed"] for row in assembly["timeline"] if row["segment_id"] == segment_id)),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start_frame / OUTPUT_SAMPLE_RATE,
                "end_seconds": end_frame / OUTPUT_SAMPLE_RATE,
                "duration_seconds": len(post_samples) / OUTPUT_SAMPLE_RATE,
                "sample_sha256": hashlib.sha256(post_samples.tobytes()).hexdigest(),
                "audio_path": str(segment_path),
                "original_start_frame": source_start,
                "original_end_frame": source_end,
            }
        )
        frame_cursor = end_frame

    edits_path = run_path / "silence_edits.csv"
    fields = [
        "start_frame", "end_frame", "start_seconds", "end_seconds", "duration_ms_before",
        "target_duration_ms", "delete_start_frame", "delete_end_frame", "deleted_frames",
    ]
    with edits_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edits)
    return {
        "final_candidate_path": str(final_path),
        "timeline": timeline,
        "silence_intervals": intervals,
        "remaining_silence_intervals": remaining_intervals,
        "silence_edits": edits,
        "review_audio_paths": {item["segment_id"]: item["audio_path"] for item in timeline},
        "silence_edits_path": str(edits_path),
    }


def generate_batch_task(
    round_index: int,
    segments: list[dict[str, Any]],
    run_dir: str,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    paths = config["paths"]
    generation = config["generation"]
    base_dir = Path(run_dir) / "candidates" / f"round_{round_index}"
    attempt_dir = base_dir / f"tech_{uuid.uuid4().hex[:8]}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    input_path = attempt_dir / "input.txt"
    input_path.write_text("\n".join(str(item["text"]).replace("\n", " ") for item in segments) + "\n", encoding="utf-8")
    seed = int(generation["base_seed"]) + round_index * int(generation["retry_seed_stride"])
    seeds = [seed for _ in segments]
    seeds_path = attempt_dir / "seeds.txt"
    seeds_path.write_text("\n".join(str(value) for value in seeds) + "\n", encoding="ascii")
    output_dir = attempt_dir / "generated"
    command = [
        paths["voxcpm_python"],
        paths["voxcpm_batch_script"],
        "--input",
        str(input_path),
        "--seeds-file",
        str(seeds_path),
        "--output-dir",
        str(output_dir),
        "--model-path",
        paths["voxcpm_model"],
        "--reference-audio",
        paths["reference_audio"],
        "--reference-text-file",
        paths["reference_text_file"],
        "--cfg-value",
        str(generation["cfg"]),
        "--steps",
        str(generation["steps"]),
        "--device",
        str(generation["device"]),
        "--no-normalize" if not generation.get("normalize", False) else "--normalize",
        "--reference-mode",
        str(generation["reference_mode"]),
    ]
    _run_command(
        command,
        attempt_dir / "generation.log",
        cwd=Path(paths["voxcpm_batch_script"]).parent,
        env_overrides={"PYTHONHASHSEED": str(seed)},
    )
    mapping: dict[str, dict[str, Any]] = {}
    for index, (item, item_seed) in enumerate(zip(segments, seeds, strict=True), start=1):
        output = output_dir / f"{index:02d}.wav"
        if not output.is_file():
            raise FileNotFoundError(f"生成结果缺失：{output}")
        mapping[str(item["segment_id"])] = {"audio_path": str(output), "seed": item_seed}
    return mapping


def analyze_batch_task(
    round_index: int,
    segments: list[dict[str, Any]],
    generated: dict[str, dict[str, Any]],
    run_dir: str,
    config: dict[str, Any],
    include_reference: bool,
) -> dict[str, Any]:
    paths = config["paths"]
    attempt_id = uuid.uuid4().hex[:8]
    analysis_dir = Path(run_dir) / "candidates" / f"round_{round_index}" / f"analysis_{attempt_id}"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    items = [
        {
            "segment_id": str(item["segment_id"]),
            "text": str(item["text"]),
            "audio_path": generated[str(item["segment_id"])]["audio_path"],
            "seed": generated[str(item["segment_id"])]["seed"],
        }
        for item in segments
    ]
    if include_reference:
        items.append(
            {
                "segment_id": "__REFERENCE__",
                "text": Path(paths["reference_text_file"]).read_text(encoding="utf-8").strip(),
                "audio_path": paths["reference_audio"],
            }
        )
    manifest_path = analysis_dir / "manifest.json"
    result_path = analysis_dir / "result.json"
    manifest_path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        paths["asr_python"],
        str(PROJECT_ROOT / "asr_worker.py"),
        "--manifest",
        str(manifest_path),
        "--output",
        str(result_path),
        "--asr-model",
        paths["asr_model"],
        "--vad-model",
        paths["vad_model"],
        "--model-cache",
        str(Path(paths["asr_model"]).parents[3]),
        "--device",
        str(config["generation"].get("asr_device", "cuda:0")),
    ]
    _run_command(command, analysis_dir / "analysis.log", cwd=PROJECT_ROOT)
    return json.loads(result_path.read_text(encoding="utf-8"))


def format_audio_time(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000.0)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def candidate_failure_details(candidate: Candidate) -> list[str]:
    details: list[str] = []
    metrics = candidate.metrics
    for issue in candidate.hard_failures:
        if issue == "逐字时间戳覆盖不足":
            coverage = float(metrics.get("timestamp_coverage") or 0.0) * 100.0
            details.append(f"逐字时间戳覆盖不足（{coverage:.1f}%）")
        elif issue == "句内异常停顿":
            gaps = list(metrics.get("abnormal_gaps") or [])
            if gaps:
                gap_text = "、".join(
                    f"“{gap.get('left', '')}”到“{gap.get('right', '')}”{float(gap.get('gap_ms') or 0):.0f}毫秒"
                    for gap in gaps
                )
                details.append(f"句内异常停顿（{gap_text}）")
            else:
                details.append(issue)
        elif issue == "局部语速异常":
            rates = list(metrics.get("window_rates") or [])
            window_texts = list(metrics.get("speed_window_texts") or [])
            ratio = float(metrics.get("maximum_speed_ratio") or 0.0)
            window_text = ""
            if rates and window_texts:
                fastest_index = max(range(len(rates)), key=lambda index: float(rates[index]))
                if fastest_index < len(window_texts):
                    window_text = str(window_texts[fastest_index])
            location = f"“{window_text}”，" if window_text else ""
            details.append(f"局部语速异常（{location}最高比值{ratio:.3f}）")
        elif issue in ("标点停顿过短", "标点停顿过长"):
            key = "short_punctuation_gaps" if issue.endswith("过短") else "long_punctuation_gaps"
            gaps = list(metrics.get(key) or [])
            gap_text = "、".join(
                f"“{gap.get('left', '')}”到“{gap.get('right', '')}”"
                f"静音{float(gap.get('gap_ms') or 0):.0f}毫秒，"
                f"含前字拉长共{float(gap.get('boundary_duration_ms') or 0):.0f}毫秒"
                for gap in gaps
            )
            details.append(f"{issue}（{gap_text}）" if gap_text else issue)
        elif issue.startswith("整体语速"):
            rate = float(metrics.get("overall_speech_rate") or 0.0)
            ratio = float(metrics.get("overall_speed_ratio_to_reference") or 0.0)
            details.append(f"{issue}（{rate:.2f}字/秒，参考比值{ratio:.3f}）")
        elif issue == "后半段频谱明显变暗":
            details.append(
                f"{issue}（频谱质心比{float(metrics.get('tail_centroid_ratio') or 0.0):.3f}，"
                f"高频占比比{float(metrics.get('tail_high_frequency_ratio') or 0.0):.3f}）"
            )
        elif issue == "后半段响度异常下降":
            details.append(f"{issue}（下降{float(metrics.get('tail_rms_drop_db') or 0.0):.2f}dB）")
        elif issue == "跨段停顿异常":
            boundary = dict(metrics.get("boundary_issue") or {})
            details.append(
                f"{issue}（{boundary.get('left_segment_id', '')}到{boundary.get('right_segment_id', '')}，"
                f"{float(boundary.get('gap_ms') or 0):.0f}毫秒）"
            )
        elif issue == "转写差异过大":
            details.append(f"转写差异过大（CER {float(metrics.get('cer') or 0.0):.3f}）")
        else:
            details.append(issue)
    return details


def assemble_task(
    segments: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    run_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    run_path = Path(run_dir)
    arrays: list[Any] = []
    timeline: list[dict[str, Any]] = []
    boundary_gaps: list[dict[str, Any]] = []
    frame_cursor = 0

    for index, item in enumerate(segments):
        segment = Segment(**item)
        candidate = selected[segment.segment_id]
        source = Path(candidate["audio_path"])
        info = sf.info(source)
        if info.samplerate != OUTPUT_SAMPLE_RATE or info.channels != OUTPUT_CHANNELS or info.subtype != "FLOAT":
            raise RuntimeError(f"候选不是原生float32格式：{source}，{info}")
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        if sample_rate != OUTPUT_SAMPLE_RATE or audio.shape[1] != OUTPUT_CHANNELS:
            raise RuntimeError(f"候选读取格式异常：{source}")
        samples = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
        arrays.append(samples)
        start_frame = frame_cursor
        end_frame = start_frame + len(samples)
        timeline.append(
            {
                "segment_id": segment.segment_id,
                "text": segment.text,
                "seed": int(candidate.get("seed") or 0),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start_frame / OUTPUT_SAMPLE_RATE,
                "end_seconds": end_frame / OUTPUT_SAMPLE_RATE,
                "duration_seconds": len(samples) / OUTPUT_SAMPLE_RATE,
                "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            }
        )
        frame_cursor = end_frame

        if index > 0:
            previous = segments[index - 1]
            previous_candidate = selected[str(previous["segment_id"])]
            gap_ms = float(previous_candidate.get("acoustic", {}).get("trailing_silence_ms") or 0.0)
            gap_ms += float(candidate.get("acoustic", {}).get("leading_silence_ms") or 0.0)
            boundary_gaps.append(
                {
                    "left_segment_id": str(previous["segment_id"]),
                    "right_segment_id": segment.segment_id,
                    "gap_ms": gap_ms,
                }
            )

    if not arrays:
        raise RuntimeError("没有可拼接的合格候选")
    final_samples = np.concatenate(arrays)
    final_path = run_path / "final_candidate.wav"
    sf.write(final_path, final_samples, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
    original_path = run_path / "original_assembly.wav"
    sf.write(original_path, final_samples, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
    written, written_rate = sf.read(final_path, dtype="float32", always_2d=False)
    if written_rate != OUTPUT_SAMPLE_RATE or written.ndim != 1 or not np.array_equal(written, final_samples):
        raise RuntimeError("最终文件未能逐采样保留合格候选")
    for array, item in zip(arrays, timeline, strict=True):
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if not np.array_equal(written[start:end], array):
            raise RuntimeError(f"最终文件中的{item['segment_id']}采样发生变化")
    return {
        "final_candidate_path": str(final_path),
        "original_assembly_path": str(original_path),
        "timeline": timeline,
        "boundary_gaps": boundary_gaps,
        "sample_sha256": hashlib.sha256(final_samples.tobytes()).hexdigest(),
    }


def export_review_segments(
    segments: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    timeline: list[dict[str, Any]],
    run_dir: str,
    review_audio_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    run_path = Path(run_dir)
    review_dir = run_path / "review_segments"
    review_dir.mkdir(parents=False, exist_ok=False)
    manifest_path = run_path / "review_segments.csv"
    timeline_by_id = {str(item["segment_id"]): item for item in timeline}
    rows: list[dict[str, Any]] = []

    for item in segments:
        segment = Segment(**item)
        candidate = selected[segment.segment_id]
        source = Path((review_audio_paths or {}).get(segment.segment_id, candidate["audio_path"]))
        destination = review_dir / f"{segment.segment_id}.wav"
        shutil.copyfile(source, destination)
        source_file_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        copied_file_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        if copied_file_sha256 != source_file_sha256:
            raise RuntimeError(f"审听片段复制后文件发生变化：{segment.segment_id}")
        timeline_item = timeline_by_id[segment.segment_id]
        timeline_item["review_audio_path"] = str(destination)
        rows.append(
            {
                "segment_id": segment.segment_id,
                "audio_file": str(destination),
                "text": segment.text,
                "seed": int(candidate.get("seed") or 0),
                "round_index": int(candidate.get("round_index") or 0),
                "duration_seconds": float(timeline_item["duration_seconds"]),
                "final_start_seconds": float(timeline_item["start_seconds"]),
                "final_end_seconds": float(timeline_item["end_seconds"]),
                "sample_sha256": str(timeline_item["sample_sha256"]),
                "file_sha256": copied_file_sha256,
                "source_file_sha256": source_file_sha256,
                "postprocessed": bool(review_audio_paths),
            }
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"review_dir": str(review_dir), "manifest_path": str(manifest_path)}


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["passed"] = candidate.passed
    payload["score"] = list(candidate.score())
    return payload


def write_metrics(
    path: Path,
    candidates: Iterable[Candidate],
    selected: dict[str, Candidate],
    timeline_by_id: dict[str, dict[str, Any]] | None = None,
    final_qc: Candidate | None = None,
) -> None:
    timeline_by_id = timeline_by_id or {}
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            record = candidate_to_dict(candidate)
            record["record_type"] = "candidate"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for segment_id, candidate in selected.items():
            timeline = timeline_by_id.get(segment_id, {})
            handle.write(
                json.dumps(
                    {
                        "record_type": "selection",
                        "segment_id": segment_id,
                        "round_index": candidate.round_index,
                        "audio_path": candidate.audio_path,
                        "seed": candidate.seed,
                        "passed": candidate.passed,
                        "start_seconds": timeline.get("start_seconds"),
                        "end_seconds": timeline.get("end_seconds"),
                        "duration_seconds": timeline.get("duration_seconds"),
                        "review_audio_path": timeline.get("review_audio_path"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if final_qc is not None:
            record = candidate_to_dict(final_qc)
            record["record_type"] = "final_qc"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    segments: list[Segment],
    history: dict[str, list[Candidate]],
    selected: dict[str, Candidate],
    final_path: str | None,
    timeline_by_id: dict[str, dict[str, Any]] | None = None,
    final_qc: Candidate | None = None,
    review_export: dict[str, str] | None = None,
) -> None:
    timeline_by_id = timeline_by_id or {}
    lines = ["音频质量报告", "", f"本次生成块数：{len(segments)}段", ""]
    total_regenerations = 0
    initially_failed = 0
    final_failed = 0
    for segment in segments:
        attempts = history[segment.segment_id]
        regenerations = max(0, len(attempts) - 1)
        total_regenerations += regenerations
        chosen = selected.get(segment.segment_id)
        if not attempts[0].passed:
            initially_failed += 1
        if chosen is None or not chosen.passed:
            final_failed += 1
        timeline = timeline_by_id.get(segment.segment_id)
        lines.append(f"对应文字：{segment.text}")
        if timeline:
            start_seconds = float(timeline["start_seconds"])
            end_seconds = float(timeline["end_seconds"])
            lines.append(
                f"成品位置：{format_audio_time(start_seconds)} - {format_audio_time(end_seconds)}"
                f"（{start_seconds:.3f}秒 - {end_seconds:.3f}秒）"
            )
        lines.append(f"重新生成：{regenerations}次")
        for attempt_index, candidate in enumerate(attempts):
            attempt_name = "首次生成" if attempt_index == 0 else f"第{attempt_index}次重新生成"
            issue_count = len(candidate.hard_failures)
            result = (
                "通过（检测问题0项）"
                if candidate.passed
                else f"未通过（检测问题{issue_count}项）：" + "、".join(candidate_failure_details(candidate))
            )
            lines.append(f"{attempt_name}（seed={candidate.seed}）：{result}")
        lines.extend(["最终状态：" + ("通过" if chosen and chosen.passed else "仍未通过"), ""])
    if final_qc is not None:
        final_result = (
            "通过"
            if final_qc.passed
            else "未通过：" + "、".join(candidate_failure_details(final_qc))
        )
        lines.extend([f"最终整条质检：{final_result}", ""])
    lines.extend(
        [
            f"首次检测未达标：{initially_failed}段",
            f"未达标音频：{final_failed}段",
            f"总重新生成次数：{total_regenerations}次",
            f"最终音频：{final_path or '未交付'}",
            f"逐段审听目录：{review_export['review_dir'] if review_export else '未交付'}",
            f"逐段审听清单：{review_export['manifest_path'] if review_export else '未交付'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_segments_csv(path: Path, segments: list[Segment]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment_id", "text", "paragraph_after"])
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))


def technical_error_summary(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if "os error 1455" in message or "页面文件太小" in message:
        return "Windows提交内存不足（错误1455：页面文件太小），VoxCPM2模型未能载入"
    if "CUDA" in message and ("out of memory" in message.lower() or "内存" in message):
        return "GPU显存不足，模型进程未能完成"
    return message.splitlines()[0][:1000]


def select_passed_candidates(
    segments: list[Segment],
    history: dict[str, list[Candidate]],
) -> dict[str, Candidate]:
    selected: dict[str, Candidate] = {}
    for segment in segments:
        passed = [candidate for candidate in history[segment.segment_id] if candidate.passed]
        if passed:
            selected[segment.segment_id] = min(passed, key=lambda candidate: candidate.score())
    return selected


def apply_boundary_failures(
    segments: list[Segment],
    history: dict[str, list[Candidate]],
    quality: dict[str, Any],
) -> list[str]:
    failed_ids: list[str] = []
    selected = select_passed_candidates(segments, history)
    if len(selected) != len(segments):
        return failed_ids
    for left_segment, right_segment in zip(segments, segments[1:]):
        left = selected[left_segment.segment_id]
        right = selected[right_segment.segment_id]
        trailing = float(left.acoustic.get("trailing_silence_ms") or 0.0)
        leading = float(right.acoustic.get("leading_silence_ms") or 0.0)
        gap_ms = trailing + leading
        minimum = float(quality["cross_segment_gap_min_ms"])
        maximum = float(quality["cross_segment_gap_max_ms"])
        if minimum <= gap_ms <= maximum:
            continue
        victim = right if gap_ms < minimum or leading >= trailing else left
        issue = {
            "left_segment_id": left_segment.segment_id,
            "right_segment_id": right_segment.segment_id,
            "gap_ms": gap_ms,
            "minimum_ms": minimum,
            "maximum_ms": maximum,
        }
        if "跨段停顿异常" not in victim.hard_failures:
            victim.hard_failures.append("跨段停顿异常")
            victim.metrics["boundary_issue"] = issue
            failed_ids.append(victim.segment_id)
    return failed_ids


def run_pipeline(config_path: str) -> str:
    config = load_config(Path(config_path))
    output_root = Path(config["paths"]["output_root"])
    with single_run_lock(output_root):
        return _run_pipeline(config_path)


def _run_pipeline(config_path: str) -> str:
    config = load_config(Path(config_path))
    preflight(config)
    source_text = Path(config["paths"]["input_text"]).read_text(encoding="utf-8")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config["paths"]["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prompt_path = PROJECT_ROOT / "SEGMENTATION_PROMPT.md"
    if prompt_path.is_file():
        shutil.copyfile(prompt_path, run_dir / prompt_path.name)
    segments = load_semantic_segments(source_text, config.get("segmentation") or {}, run_dir)
    runtime_resource_preflight(segments, config)
    (run_dir / "reviewed_input.txt").write_text("\n".join(segment.text for segment in segments) + "\n", encoding="utf-8")
    write_segments_csv(run_dir / "segments.csv", segments)
    (run_dir / "effective_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    history: dict[str, list[Candidate]] = {segment.segment_id: [] for segment in segments}
    all_candidates: list[Candidate] = []
    reference_acoustic: dict[str, Any] = {}
    reference_speech_rate = 0.0
    pending = list(segments)
    max_round = int(config["generation"]["max_quality_retries"])

    for round_index in range(max_round + 1):
        if not pending:
            break
        segment_payload = [asdict(segment) for segment in pending]
        print(f"第{round_index + 1}轮：生成{len(pending)}段")
        generated = generate_batch_task(round_index, segment_payload, str(run_dir), config)
        analysis = analyze_batch_task(round_index, segment_payload, generated, str(run_dir), config, round_index == 0)
        raw_results = list(analysis.get("results") or [])
        if round_index == 0:
            reference_result = next((item for item in raw_results if item.get("segment_id") == "__REFERENCE__"), None)
            if reference_result and not reference_result.get("error"):
                reference_acoustic = dict(reference_result.get("acoustic") or {})
                reference_speech_rate = _speech_rate_from_timestamps(
                    [[int(pair[0]), int(pair[1])] for pair in reference_result.get("timestamps_ms") or []]
                )
        raw_results = [item for item in raw_results if item.get("segment_id") != "__REFERENCE__"]
        result_by_id = {str(item.get("segment_id")): item for item in raw_results}
        evaluated: list[Candidate] = []
        for segment in pending:
            raw = result_by_id.get(segment.segment_id)
            if raw is None:
                raw = {
                    "segment_id": segment.segment_id,
                    "target_text": segment.text,
                    "audio_path": generated[segment.segment_id]["audio_path"],
                    "seed": generated[segment.segment_id]["seed"],
                    "error": "ASR worker did not return this segment",
                }
            candidate = evaluate_candidate(raw, round_index, config, reference_speech_rate)
            evaluated.append(candidate)
        batch_acoustics = [candidate.acoustic for candidate in evaluated if candidate.acoustic]
        for candidate in evaluated:
            add_muffle_warning(candidate, reference_acoustic, batch_acoustics)
            history[candidate.segment_id].append(candidate)
            all_candidates.append(candidate)
            print(
                f"{candidate.segment_id}："
                + ("通过" if candidate.passed else "未通过：" + "、".join(candidate.hard_failures))
            )
        boundary_failed = apply_boundary_failures(segments, history, config["quality"])
        for segment_id in boundary_failed:
            print(f"{segment_id}：未通过：跨段停顿异常")
        selected = select_passed_candidates(segments, history)
        pending = [
            segment
            for segment in segments
            if segment.segment_id not in selected and round_index < max_round
        ]

    selected = select_passed_candidates(segments, history)
    unresolved = [segment for segment in segments if segment.segment_id not in selected]
    if unresolved:
        write_metrics(run_dir / "metrics.jsonl", all_candidates, selected)
        write_report(run_dir / "report.txt", segments, history, selected, None)
        names = "、".join(segment.segment_id for segment in unresolved)
        raise RuntimeError(f"质检未通过且已达到最大重做次数：{names}；未生成最终音频")

    selected_payload = {segment_id: candidate_to_dict(candidate) for segment_id, candidate in selected.items()}
    assembly = assemble_task([asdict(segment) for segment in segments], selected_payload, str(run_dir), config)
    postprocess = apply_silence_edits_task(
        assembly,
        [asdict(segment) for segment in segments],
        str(run_dir),
        config,
    )
    final_candidate_path = str(postprocess["final_candidate_path"])
    timeline_by_id = {str(item["segment_id"]): item for item in postprocess["timeline"]}

    final_segment = Segment("__FINAL__", "\n".join(segment.text for segment in segments), True)
    final_generated = {
        final_segment.segment_id: {
            "audio_path": final_candidate_path,
            "seed": 0,
        }
    }
    final_analysis = analyze_batch_task(
        "final",
        [asdict(final_segment)],
        final_generated,
        str(run_dir),
        config,
        False,
    )
    final_raw = next(
        (item for item in final_analysis.get("results") or [] if item.get("segment_id") == "__FINAL__"),
        {
            "segment_id": "__FINAL__",
            "target_text": final_segment.text,
            "audio_path": final_candidate_path,
            "error": "ASR worker did not return final audio",
        },
    )
    final_qc = evaluate_candidate(
        final_raw,
        max_round + 1,
        config,
        reference_speech_rate,
        check_internal_prosody=False,
    )
    add_muffle_warning(final_qc, reference_acoustic, [final_qc.acoustic] if final_qc.acoustic else [])
    long_silences = [
        item for item in postprocess["remaining_silence_intervals"]
        if (item[1] - item[0]) * 1000.0 / OUTPUT_SAMPLE_RATE > 200.0
    ]
    if long_silences:
        final_qc.hard_failures.append("后期仍存在超过200毫秒的连续无声区")
        final_qc.metrics["remaining_long_silences"] = long_silences

    if not final_qc.passed:
        write_metrics(run_dir / "metrics.jsonl", all_candidates, selected, timeline_by_id, final_qc)
        write_report(run_dir / "report.txt", segments, history, selected, None, timeline_by_id, final_qc)
        raise RuntimeError("最终整条音频质检未通过；保留final_candidate.wav用于诊断，不交付final.wav")

    review_export = export_review_segments(
        [asdict(segment) for segment in segments],
        selected_payload,
        postprocess["timeline"],
        str(run_dir),
        postprocess["review_audio_paths"],
    )
    final_path_obj = run_dir / "final.wav"
    Path(final_candidate_path).replace(final_path_obj)
    final_path = str(final_path_obj)
    final_qc.audio_path = final_path
    write_metrics(run_dir / "metrics.jsonl", all_candidates, selected, timeline_by_id, final_qc)
    write_report(
        run_dir / "report.txt",
        segments,
        history,
        selected,
        final_path,
        timeline_by_id,
        final_qc,
        review_export,
    )
    print(f"完成：{final_path}")
    print(f"逐段审听：{review_export['review_dir']}")
    print(f"逐段清单：{review_export['manifest_path']}")
    print(f"报告：{run_dir / 'report.txt'}")
    return final_path


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output_root: Path | None = None
    existing_runs: set[Path] = set()
    try:
        config = load_config(config_path)
        if args.check_only:
            preflight(config)
            source_text = Path(config["paths"]["input_text"]).read_text(encoding="utf-8")
            mode = str((config.get("segmentation") or {}).get("mode", "reviewed")).lower()
            if mode == "ai_command":
                print("检查通过：配置、程序、模型、参考音频和输入文件均可用；AI语义分段器已配置但未在check-only中调用；未生成音频。")
            else:
                segments = load_reviewed_segments(source_text)
                runtime_resource_preflight(segments, config)
                lengths = [len(segment.text) for segment in segments]
                print(
                    "检查通过：配置、程序、模型、参考音频和输入文件均可用；"
                    f"将按已审定的输入边界生成{len(segments)}段，"
                    f"长度{min(lengths)}-{max(lengths)}字；未生成音频。"
                )
            return 0
        output_root = Path(config["paths"]["output_root"])
        existing_runs = set(output_root.iterdir()) if output_root.is_dir() else set()
        final_path = run_pipeline(str(config_path))
        print(final_path)
        return 0
    except Exception as exc:
        if output_root is not None and output_root.is_dir():
            new_runs = [path for path in output_root.iterdir() if path.is_dir() and path not in existing_runs]
            if new_runs:
                run_dir = max(new_runs, key=lambda path: path.stat().st_mtime)
                report_path = run_dir / "report.txt"
                if not report_path.exists():
                    report_path.write_text(
                        "音频质量报告\n\n"
                        f"流程状态：技术失败\n"
                        f"错误：{technical_error_summary(exc)}\n"
                        "最终音频：未生成\n",
                        encoding="utf-8",
                    )
                    (run_dir / "technical_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print(f"技术失败报告：{report_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
