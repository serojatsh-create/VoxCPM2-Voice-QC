from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from voice_qc_flow import (
    Candidate,
    Segment,
    apply_boundary_failures,
    compress_silence_samples,
    assemble_task,
    candidate_to_dict,
    candidate_failure_details,
    evaluate_candidate,
    export_review_segments,
    format_audio_time,
    levenshtein_distance,
    load_reviewed_segments,
    runtime_resource_preflight,
    single_run_lock,
    _validate_segment_payload,
    technical_error_summary,
    write_report,
)


def config() -> dict:
    return {
        "quality": {
            "internal_gap_floor_ms": 280,
            "internal_gap_mad_multiplier": 4.0,
            "comma_pause_min_ms": 40,
            "comma_pause_max_ms": 900,
            "sentence_pause_min_ms": 100,
            "sentence_pause_max_ms": 1600,
            "punctuation_boundary_min_ms": 240,
            "known_asr_substitutions": {"情分": "勤奋"},
            "speed_window_chars": 6,
            "speed_ratio_limit": 1.45,
            "speed_consecutive_windows": 2,
            "cer_limit": 0.12,
            "missing_or_extra_span_limit": 2,
            "clipping_fraction_limit": 0.001,
            "minimum_audio_seconds": 0.35,
            "overall_speed_min_chars_per_second": 2.0,
            "overall_speed_max_chars_per_second": 7.0,
            "overall_speed_ratio_max": 1.5,
            "tail_analysis_min_seconds": 4.0,
            "tail_centroid_ratio_limit": 0.65,
            "tail_high_frequency_ratio_limit": 0.5,
            "tail_rms_drop_db_limit": 10.0,
            "cross_segment_gap_min_ms": 60,
            "cross_segment_gap_max_ms": 1200,
        }
    }


def raw_candidate(text: str, timestamps: list[list[int]]) -> dict:
    return {
        "segment_id": "S001",
        "target_text": text,
        "audio_path": "S001.wav",
        "recognized_text": text,
        "timestamps_ms": timestamps,
        "acoustic": {
            "duration_seconds": max(pair[1] for pair in timestamps) / 1000.0,
            "sample_rate": 48000,
            "channels": 1,
            "clipping_fraction": 0.0,
            "rms_db": -22.0,
        },
        "error": None,
    }


class SegmentationTests(unittest.TestCase):
    def test_reviewed_segment_boundaries_are_preserved(self) -> None:
        first_line = "第一行包含句号。也包含问号？仍然必须作为一次完整生成。"
        second_line = "第二行是另一次生成。"
        segments = load_reviewed_segments(first_line + "\n\n" + second_line)
        self.assertEqual(
            segments,
            [
                Segment("S001", first_line, True),
                Segment("S002", second_line, True),
            ],
        )

    def test_long_reviewed_segment_is_never_automatically_split(self) -> None:
        long_line = "这是一整段由人工或者大模型提前切好的内容，" * 8 + "必须保持为一次生成。"
        segments = load_reviewed_segments(long_line)
        self.assertGreater(len(long_line), 90)
        self.assertEqual(segments, [Segment("S001", long_line, True)])

    def test_ai_segment_payload_preserves_text_and_assigns_ids(self) -> None:
        source = "第一句完整表达。第二句继续说明。"
        segments = _validate_segment_payload(
            {"segments": [{"text": "第一句完整表达。"}, {"text": "第二句继续说明。"}]},
            source,
        )
        self.assertEqual([item.segment_id for item in segments], ["S001", "S002"])
        self.assertEqual("".join(item.text for item in segments), source)
        self.assertTrue(segments[-1].paragraph_after)

    def test_ai_segment_payload_rejects_rewritten_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "修改了原文"):
            _validate_segment_payload({"segments": ["被改写的文本。"]}, "原始文本。")


class ResourceGuardTests(unittest.TestCase):
    def test_segment_over_240_characters_is_rejected(self) -> None:
        config = {
            "generation": {"device": "cpu"},
            "segmentation": {"hard_max_chars": 240},
        }
        runtime_resource_preflight([Segment("S001", "字" * 240)], config)
        with self.assertRaisesRegex(ValueError, "超过240字"):
            runtime_resource_preflight([Segment("S001", "字" * 241)], config)

    def test_cuda_run_is_rejected_below_4096_mb_free_vram(self) -> None:
        config = {
            "generation": {"device": "cuda", "min_free_vram_mb": 4096},
            "segmentation": {"hard_max_chars": 240},
        }
        result = subprocess.CompletedProcess([], 0, stdout="4000\n", stderr="")
        with patch("voice_qc_flow.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "低于安全下限4096MB"):
                runtime_resource_preflight([Segment("S001", "安全片段。")], config)

    def test_single_run_lock_rejects_a_live_owner(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with single_run_lock(root):
                with self.assertRaisesRegex(RuntimeError, "流程已在运行"):
                    with single_run_lock(root):
                        pass


class OutputFormatTests(unittest.TestCase):
    def test_production_source_contains_no_audio_polish_filters(self) -> None:
        source = (PROJECT_ROOT / "voice_qc_flow.py").read_text(encoding="utf-8")
        for forbidden in ("silenceremove", "afade=", "volume=", "pcm_s16le", "@flow", "@task"):
            self.assertNotIn(forbidden, source)

    def test_exact_float32_concatenation_preserves_every_sample(self) -> None:
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left_path = root / "left.wav"
            right_path = root / "right.wav"
            left = np.array([0.0, 0.125, -0.25, 0.5], dtype=np.float32)
            right = np.array([-0.75, 0.25, 0.0], dtype=np.float32)
            sf.write(left_path, left, 48000, subtype="FLOAT")
            sf.write(right_path, right, 48000, subtype="FLOAT")
            segments = [
                {"segment_id": "S001", "text": "第一段。", "paragraph_after": False},
                {"segment_id": "S002", "text": "第二段。", "paragraph_after": True},
            ]
            selected = {
                "S001": candidate_to_dict(
                    Candidate("S001", "第一段。", 0, str(left_path), seed=11, acoustic={"trailing_silence_ms": 80})
                ),
                "S002": candidate_to_dict(
                    Candidate("S002", "第二段。", 0, str(right_path), seed=22, acoustic={"leading_silence_ms": 70})
                ),
            }
            result = assemble_task(segments, selected, str(root), {})
            combined, rate = sf.read(result["final_candidate_path"], dtype="float32")
            self.assertEqual(rate, 48000)
            self.assertTrue(np.array_equal(combined, np.concatenate([left, right])))

    def test_review_segments_are_byte_identical_and_manifested(self) -> None:
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "candidate.wav"
            samples = np.array([0.0, 0.25, -0.5, 0.75], dtype=np.float32)
            sf.write(source, samples, 48000, subtype="FLOAT")
            segments = [{"segment_id": "S001", "text": "完整的一句话。", "paragraph_after": True}]
            selected = {
                "S001": candidate_to_dict(Candidate("S001", "完整的一句话。", 1, str(source), seed=22))
            }
            timeline = [
                {
                    "segment_id": "S001",
                    "start_seconds": 0.0,
                    "end_seconds": len(samples) / 48000,
                    "duration_seconds": len(samples) / 48000,
                    "sample_sha256": "sample-hash",
                }
            ]
            exported = export_review_segments(segments, selected, timeline, str(root))
            copied = Path(exported["review_dir"]) / "S001.wav"
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            manifest = Path(exported["manifest_path"]).read_text(encoding="utf-8-sig")
            self.assertIn("S001", manifest)
            self.assertIn("完整的一句话。", manifest)
            self.assertEqual(timeline[0]["review_audio_path"], str(copied))

    def test_long_silence_is_reduced_to_200ms_and_voice_is_preserved(self) -> None:
        import numpy as np

        rate = 48000
        voice_left = np.array([0.25, -0.25], dtype=np.float32)
        silence = np.zeros(round(rate * 0.5), dtype=np.float32)
        voice_right = np.array([0.5, -0.5], dtype=np.float32)
        edited, edits = compress_silence_samples(
            np.concatenate([voice_left, silence, voice_right]),
            [(len(voice_left), len(voice_left) + len(silence))],
            sample_rate=rate,
        )
        expected_silence = round(rate * 0.2)
        self.assertEqual(len(edits), 1)
        self.assertEqual(len(edited), len(voice_left) + expected_silence + len(voice_right))
        self.assertTrue(np.array_equal(edited[: len(voice_left)], voice_left))
        self.assertTrue(np.array_equal(edited[-len(voice_right) :], voice_right))

    def test_short_silence_is_left_unchanged(self) -> None:
        import numpy as np

        samples = np.zeros(48000 // 5, dtype=np.float32)
        edited, edits = compress_silence_samples(samples, [(0, len(samples))])
        self.assertEqual(edits, [])
        self.assertTrue(np.array_equal(edited, samples))

    def test_bad_cross_segment_gap_rejects_the_right_candidate(self) -> None:
        segments = [Segment("S001", "第一段。"), Segment("S002", "第二段。", True)]
        left = Candidate("S001", "第一段。", 0, "left.wav", acoustic={"trailing_silence_ms": 10})
        right = Candidate("S002", "第二段。", 0, "right.wav", acoustic={"leading_silence_ms": 10})
        history = {"S001": [left], "S002": [right]}
        failed = apply_boundary_failures(segments, history, config()["quality"])
        self.assertEqual(failed, ["S002"])
        self.assertIn("跨段停顿异常", right.hard_failures)

    def test_all_bad_boundaries_are_rejected_in_the_same_round(self) -> None:
        segments = [Segment("S001", "一。"), Segment("S002", "二。"), Segment("S003", "三。", True)]
        history = {
            segment.segment_id: [
                Candidate(
                    segment.segment_id,
                    segment.text,
                    0,
                    f"{segment.segment_id}.wav",
                    acoustic={"leading_silence_ms": 5, "trailing_silence_ms": 5},
                )
            ]
            for segment in segments
        }
        failed = apply_boundary_failures(segments, history, config()["quality"])
        self.assertEqual(failed, ["S002", "S003"])


class QualityRuleTests(unittest.TestCase):
    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("语速不会突然加快", "语速不会加快"), 2)

    def test_internal_350ms_gap_is_rejected(self) -> None:
        text = "语速不会突然加快"
        timestamps: list[list[int]] = []
        cursor = 0
        for index, _ in enumerate(text):
            if index == 4:
                cursor += 350
            timestamps.append([cursor, cursor + 120])
            cursor += 130
        candidate = evaluate_candidate(raw_candidate(text, timestamps), 0, config())
        self.assertIn("句内异常停顿", candidate.hard_failures)

    def test_gap_after_comma_is_not_internal_failure(self) -> None:
        text = "语速不会，突然加快"
        lexical = [character for character in text if character != "，"]
        timestamps: list[list[int]] = []
        cursor = 0
        for index, _ in enumerate(lexical):
            if index == 4:
                cursor += 350
            timestamps.append([cursor, cursor + 120])
            cursor += 130
        candidate = evaluate_candidate(raw_candidate(text, timestamps), 0, config())
        self.assertNotIn("句内异常停顿", candidate.hard_failures)

    def test_comma_pause_that_is_too_short_is_rejected(self) -> None:
        text = "前面，后面"
        timestamps = [[index * 120, index * 120 + 120] for index in range(4)]
        candidate = evaluate_candidate(raw_candidate(text, timestamps), 0, config())
        self.assertIn("标点停顿过短", candidate.hard_failures)

    def test_lengthened_syllable_can_express_punctuation_boundary(self) -> None:
        text = "前面，后面"
        timestamps = [[0, 120], [120, 360], [360, 480], [480, 600]]
        candidate = evaluate_candidate(raw_candidate(text, timestamps), 0, config())
        self.assertNotIn("标点停顿过短", candidate.hard_failures)

    def test_final_mode_does_not_rejudge_frozen_internal_prosody(self) -> None:
        text = "前面，后面"
        timestamps = [[index * 120, index * 120 + 120] for index in range(4)]
        candidate = evaluate_candidate(
            raw_candidate(text, timestamps),
            0,
            config(),
            check_internal_prosody=False,
        )
        self.assertNotIn("标点停顿过短", candidate.hard_failures)
        self.assertFalse(candidate.metrics["internal_prosody_enforced"])

    def test_final_mode_does_not_rejudge_frozen_local_text_spans(self) -> None:
        target = "这是一段已经逐段核对通过的完整文字内容用于最终检查"
        recognized = target.replace("完整", "天气")
        timestamps = [[index * 130, index * 130 + 120] for index in range(len(recognized))]
        raw = raw_candidate(target, timestamps)
        raw["recognized_text"] = recognized
        candidate = evaluate_candidate(raw, 0, config(), check_internal_prosody=False)
        self.assertNotIn("存在连续漏字、错字或重复", candidate.hard_failures)

    def test_english_words_use_one_timestamp_unit(self) -> None:
        text = "好好gap或者AI健身"
        timestamps = [[index * 130, index * 130 + 120] for index in range(8)]
        candidate = evaluate_candidate(raw_candidate(text, timestamps), 0, config())
        self.assertEqual(candidate.metrics["timestamp_coverage"], 1.0)
        self.assertNotIn("逐字时间戳覆盖不足", candidate.hard_failures)

    def test_two_character_omission_is_rejected(self) -> None:
        target = "语速不会突然加快"
        recognized = "语速不会加快"
        timestamps = [[index * 130, index * 130 + 120] for index in range(len(recognized))]
        raw = raw_candidate(target, timestamps)
        raw["recognized_text"] = recognized
        candidate = evaluate_candidate(raw, 0, config())
        self.assertIn("存在连续漏字、错字或重复", candidate.hard_failures)

    def test_verified_asr_substitution_is_warning_without_relaxing_other_errors(self) -> None:
        target = "这个过程叫情分清零"
        timestamps = [[index * 130, index * 130 + 120] for index in range(len(target))]
        verified_raw = raw_candidate(target, timestamps)
        verified_raw["recognized_text"] = "这个过程叫勤奋清零"
        verified = evaluate_candidate(verified_raw, 0, config())
        self.assertNotIn("存在连续漏字、错字或重复", verified.hard_failures)
        self.assertIn("ASR已知近音替换：情分→勤奋", verified.warnings)

        other_raw = raw_candidate(target, timestamps)
        other_raw["recognized_text"] = "这个过程叫天气清零"
        other = evaluate_candidate(other_raw, 0, config())
        self.assertIn("存在连续漏字、错字或重复", other.hard_failures)

    def test_windows_pagefile_error_is_summarized(self) -> None:
        summary = technical_error_summary(RuntimeError("页面文件太小，无法完成操作。 (os error 1455)"))
        self.assertIn("错误1455", summary)


class ReportTests(unittest.TestCase):
    def test_audio_time_format(self) -> None:
        self.assertEqual(format_audio_time(65.432), "01:05.432")

    def test_failure_details_include_location(self) -> None:
        candidate = Candidate(
            segment_id="S001",
            text="前后",
            round_index=0,
            audio_path="S001.wav",
            hard_failures=["句内异常停顿"],
            metrics={"abnormal_gaps": [{"left": "前", "right": "后", "gap_ms": 360}]},
        )
        self.assertEqual(candidate_failure_details(candidate), ["句内异常停顿（“前”到“后”360毫秒）"])

    def test_report_uses_final_timeline_and_text_instead_of_internal_id(self) -> None:
        import tempfile

        segment = Segment("S001", "用于试听定位的文字", True)
        failed = Candidate(
            segment_id="S001",
            text=segment.text,
            round_index=0,
            audio_path="first.wav",
            hard_failures=["句内异常停顿"],
            metrics={"abnormal_gaps": [{"left": "试", "right": "听", "gap_ms": 360}]},
        )
        passed = Candidate(
            segment_id="S001",
            text=segment.text,
            round_index=1,
            audio_path="retry.wav",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.txt"
            write_report(
                report_path,
                [segment],
                {"S001": [failed, passed]},
                {"S001": passed},
                "final.wav",
                {
                    "S001": {
                        "start_seconds": 12.345,
                        "end_seconds": 17.89,
                        "duration_seconds": 5.545,
                    }
                },
            )
            report = report_path.read_text(encoding="utf-8")
        self.assertIn("成品位置：00:12.345 - 00:17.890（12.345秒 - 17.890秒）", report)
        self.assertIn("对应文字：用于试听定位的文字", report)
        self.assertIn("重新生成：1次", report)
        self.assertIn("第1次重新生成（seed=0）：通过", report)
        self.assertNotIn("S001", report)

    def test_report_includes_blocks_that_pass_on_first_generation(self) -> None:
        import tempfile

        segment = Segment("S001", "一次通过也必须出现在报告里。", True)
        passed = Candidate(
            segment_id="S001",
            text=segment.text,
            round_index=0,
            audio_path="first.wav",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.txt"
            write_report(
                report_path,
                [segment],
                {"S001": [passed]},
                {"S001": passed},
                "final.wav",
                {
                    "S001": {
                        "start_seconds": 0.0,
                        "end_seconds": 3.5,
                        "duration_seconds": 3.5,
                    }
                },
            )
            report = report_path.read_text(encoding="utf-8")
        self.assertIn("本次生成块数：1段", report)
        self.assertIn("对应文字：一次通过也必须出现在报告里。", report)
        self.assertIn("首次生成（seed=0）：通过（检测问题0项）", report)
        self.assertIn("重新生成：0次", report)


if __name__ == "__main__":
    unittest.main()
