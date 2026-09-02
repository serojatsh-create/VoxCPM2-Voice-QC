from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ASR and acoustic analysis worker for VoxCPM2 QC")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asr-model", required=True, type=Path)
    parser.add_argument("--vad-model", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _extract_item(raw: Any) -> dict[str, Any]:
    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        raise RuntimeError("FunASR returned a non-object result")
    return item


def _normalize_timestamps(raw: Any) -> list[list[int]]:
    result: list[list[int]] = []
    if not isinstance(raw, (list, tuple)):
        return result
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        start = int(round(float(pair[0])))
        end = int(round(float(pair[1])))
        if start >= 0 and end >= start:
            result.append([start, end])
    return result


def acoustic_metrics(audio_path: Path) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    if audio.size == 0 or sample_rate <= 0:
        raise RuntimeError("Audio is empty or has an invalid sample rate")
    channels = int(audio.shape[1])
    mono = np.mean(audio, axis=1, dtype=np.float32)
    duration_seconds = float(len(mono) / sample_rate)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    clipping_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if len(mono) else 0.0
    rms_db = float(20.0 * math.log10(max(rms, 1e-12)))

    active_threshold = 10.0 ** (-50.0 / 20.0)
    active_indices = np.flatnonzero(np.abs(mono) >= active_threshold)
    if active_indices.size:
        leading_silence_ms = float(active_indices[0] * 1000.0 / sample_rate)
        trailing_silence_ms = float((len(mono) - 1 - active_indices[-1]) * 1000.0 / sample_rate)
    else:
        leading_silence_ms = duration_seconds * 1000.0
        trailing_silence_ms = duration_seconds * 1000.0

    def spectral_summary(values: Any) -> dict[str, float]:
        analysis_audio = np.asarray(values, dtype=np.float32)
        if len(analysis_audio) == 0:
            return {
                "rms_db": -240.0,
                "clipping_fraction": 0.0,
                "spectral_centroid_hz": 0.0,
                "spectral_rolloff_85_hz": 0.0,
                "high_frequency_ratio_4khz": 0.0,
            }
        analysis_rate = int(sample_rate)
        if analysis_rate > 24000:
            analysis_audio = librosa.resample(analysis_audio, orig_sr=analysis_rate, target_sr=24000)
            analysis_rate = 24000
        n_fft = 2048 if len(analysis_audio) >= 2048 else 256
        hop_length = max(64, n_fft // 4)
        magnitude = np.abs(librosa.stft(analysis_audio, n_fft=n_fft, hop_length=hop_length))
        power = np.square(magnitude, dtype=np.float64)
        frequencies = librosa.fft_frequencies(sr=analysis_rate, n_fft=n_fft)
        total_power = float(np.sum(power))
        high_mask = frequencies >= 4000.0
        high_frequency_ratio = float(np.sum(power[high_mask, :]) / total_power) if total_power > 0 else 0.0
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=analysis_rate)
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=analysis_rate, roll_percent=0.85)
        segment_rms = float(np.sqrt(np.mean(np.square(analysis_audio), dtype=np.float64))) if len(analysis_audio) else 0.0
        return {
            "rms_db": float(20.0 * math.log10(max(segment_rms, 1e-12))),
            "clipping_fraction": float(np.mean(np.abs(analysis_audio) >= 0.999)) if len(analysis_audio) else 0.0,
            "spectral_centroid_hz": float(np.nanmedian(centroid)) if centroid.size else 0.0,
            "spectral_rolloff_85_hz": float(np.nanmedian(rolloff)) if rolloff.size else 0.0,
            "high_frequency_ratio_4khz": high_frequency_ratio,
        }

    midpoint = max(1, len(mono) // 2)
    overall_spectrum = spectral_summary(mono)
    first_half = spectral_summary(mono[:midpoint])
    second_half = spectral_summary(mono[midpoint:])

    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak": peak,
        "clipping_fraction": clipping_fraction,
        "rms_db": rms_db,
        "leading_silence_ms": leading_silence_ms,
        "trailing_silence_ms": trailing_silence_ms,
        "spectral_centroid_hz": overall_spectrum["spectral_centroid_hz"],
        "spectral_rolloff_85_hz": overall_spectrum["spectral_rolloff_85_hz"],
        "high_frequency_ratio_4khz": overall_spectrum["high_frequency_ratio_4khz"],
        "first_half": first_half,
        "second_half": second_half,
    }


def main() -> int:
    args = parse_args()
    for required in (args.manifest, args.asr_model, args.vad_model, args.model_cache):
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ["MODELSCOPE_CACHE"] = str(args.model_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from funasr import AutoModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the ASR environment")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list")

    model = AutoModel(
        model=str(args.asr_model),
        vad_model=str(args.vad_model),
        device=args.device,
        disable_update=True,
    )

    results: list[dict[str, Any]] = []
    for entry in items:
        segment_id = str(entry.get("segment_id") or "")
        audio_path = Path(str(entry.get("audio_path") or ""))
        result: dict[str, Any] = {
            "segment_id": segment_id,
            "target_text": str(entry.get("text") or ""),
            "audio_path": str(audio_path),
            "seed": int(entry.get("seed") or 0),
        }
        try:
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            raw = model.generate(
                input=str(audio_path),
                cache={},
                language="zh",
                use_itn=False,
                batch_size_s=300,
                merge_vad=True,
                merge_length_s=15,
            )
            item = _extract_item(raw)
            timestamps = _normalize_timestamps(item.get("timestamp") or item.get("timestamps") or [])
            result.update(
                {
                    "recognized_text": str(item.get("text") or "").strip(),
                    "timestamps_ms": timestamps,
                    "acoustic": acoustic_metrics(audio_path),
                    "raw": _jsonable(raw),
                    "error": None,
                }
            )
        except Exception as exc:
            result.update(
                {
                    "recognized_text": "",
                    "timestamps_ms": [],
                    "acoustic": {},
                    "raw": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        results.append(result)

    payload = {
        "worker": {
            "python": sys.executable,
            "torch": torch.__version__,
            "device": args.device,
            "asr_model": str(args.asr_model),
            "vad_model": str(args.vad_model),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
