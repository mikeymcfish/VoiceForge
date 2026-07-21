#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chapter_markers as chapters
import moss_tts_worker as moss
import qwen_tts_worker as qwen


class _FakeProcessor:
    def __init__(self) -> None:
        self.kwargs = None
        self.user_kwargs = []
        self.assistant_kwargs = []

    def build_user_message(self, **kwargs):
        self.kwargs = kwargs
        self.user_kwargs.append(kwargs)
        return kwargs

    def build_assistant_message(self, **kwargs):
        self.assistant_kwargs.append(kwargs)
        return {"assistant": kwargs}


class WorkerHelpersTest(unittest.TestCase):
    @staticmethod
    def _create_fake_snapshot(snapshot: Path) -> None:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"fake-weights")

    def test_qwen_pins_are_full_commits(self) -> None:
        self.assertEqual(qwen.DEFAULT_MODEL_ID, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.assertTrue(qwen.PINNED_MODEL_REVISIONS)
        for revision in qwen.PINNED_MODEL_REVISIONS.values():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_all_moss_model_and_codec_pins_are_full_commits(self) -> None:
        self.assertEqual(moss.DEFAULT_MODEL_ID, moss.MODEL_REPO_ID)
        self.assertIn(moss.LOCAL_MODEL_REPO_ID, moss.PINNED_MODEL_SPECS)
        for model_id, spec in moss.PINNED_MODEL_SPECS.items():
            with self.subTest(model_id=model_id):
                self.assertIn("manifest_name", spec)
                self.assertTrue(spec["artifacts"])
                for artifact in spec["artifacts"].values():
                    self.assertRegex(artifact["revision"], r"^[0-9a-f]{40}$")

    def test_qwen_split_handles_cjk_without_spaces(self) -> None:
        text = (
            "这是第一句，它有足够多的文字来帮助测试中文断句行为。"
            "这是第二句，它同样没有在句号后面添加任何空格。"
            "这是第三句，用于确保输出会被分成多个安全长度的片段。"
        )
        chunks = qwen.split_text(text, 50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk and len(chunk) <= 50 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_long_unbroken_token_is_hard_split(self) -> None:
        chunks = moss.split_text("x" * 135, 50)
        self.assertEqual([len(chunk) for chunk in chunks], [50, 50, 35])

    def test_chapter_markers_are_case_insensitive_control_syntax(self) -> None:
        source = (
            "Opening text.\n\n"
            "[ChApTeR] Part One\n"
            "Body one.\n"
            "[chapter]\n"
            "Body two."
        )
        sections = chapters.split_chapter_sections(source)
        self.assertEqual(
            [section.text for section in sections],
            ["Opening text.", "Part One\nBody one.", "Body two."],
        )
        self.assertEqual(
            [section.starts_chapter for section in sections],
            [False, True, True],
        )
        self.assertEqual(
            [section.chapter_title for section in sections],
            [None, "Part One", None],
        )
        self.assertNotIn("[chapter]", " ".join(section.text for section in sections).lower())

    def test_worker_splitters_force_chapter_boundaries_and_speak_inline_titles(
        self,
    ) -> None:
        source = "Before marker. [CHAPTER] Chapter One\nThe chapter body."
        for splitter in (
            qwen.split_text_with_chapters,
            moss.split_text_with_chapters,
        ):
            with self.subTest(splitter=splitter.__module__):
                segments = splitter(source, 500)
                self.assertEqual(len(segments), 2)
                self.assertEqual(segments[0].text, "Before marker.")
                self.assertFalse(segments[0].starts_chapter)
                self.assertEqual(
                    segments[1].text,
                    "Chapter One The chapter body.",
                )
                self.assertTrue(segments[1].starts_chapter)
                self.assertEqual(segments[1].chapter_title, "Chapter One")
                self.assertNotIn("[CHAPTER]", segments[1].text.upper())

    def test_chapter_control_syntax_is_opt_in_for_synthesis(self) -> None:
        source = "Literal [CHAPTER] text stays spoken unless chapters are enabled."
        for worker in (qwen, moss):
            with self.subTest(worker=worker.__name__):
                plain = worker.split_synthesis_text(
                    source,
                    500,
                    use_chapter_markers=False,
                )
                self.assertIn("[CHAPTER]", " ".join(segment.text for segment in plain))
                self.assertFalse(any(segment.starts_chapter for segment in plain))

                chaptered = worker.split_synthesis_text(
                    source,
                    500,
                    use_chapter_markers=True,
                )
                self.assertNotIn(
                    "[CHAPTER]",
                    " ".join(segment.text for segment in chaptered),
                )
                self.assertTrue(any(segment.starts_chapter for segment in chaptered))

    def test_audio_merge_timing_adds_only_the_extra_chapter_pause(self) -> None:
        timing = chapters.build_audio_merge_timing(
            [100, 200, 300],
            sample_rate=1_000,
            gap_ms=100,
            chapter_segment_indices=[1],
            chapter_pause_ms=500,
        )
        self.assertEqual(timing.gap_samples_before, (0, 600, 100))
        self.assertEqual(timing.start_samples, (0, 700, 1_000))
        self.assertEqual(timing.total_samples, 1_300)

    def test_chapter_manifest_is_validated_and_written_atomically(self) -> None:
        segments = [
            chapters.ChapterSegment("Preface"),
            chapters.ChapterSegment("Part One", True, "Part One"),
            chapters.ChapterSegment("Body", True, None),
        ]
        manifest = chapters.build_chapter_manifest(
            segments,
            start_samples=[0, 800, 1_800],
            sample_rate=1_000,
            total_samples=2_500,
        )
        self.assertEqual(
            set(manifest),
            {"version", "sample_rate", "total_samples", "chapters"},
        )
        self.assertEqual(
            manifest["chapters"],
            [
                {
                    "title": "Part One",
                    "start_sample": 800,
                    "start_ms": 800.0,
                    "start_seconds": 0.8,
                },
                {
                    "title": None,
                    "start_sample": 1_800,
                    "start_ms": 1_800.0,
                    "start_seconds": 1.8,
                },
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "chapters.json"
            written = chapters.write_chapter_manifest(destination, manifest)
            self.assertEqual(written, destination.resolve())
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                manifest,
            )
            self.assertEqual(
                [path.name for path in destination.parent.iterdir()],
                ["chapters.json"],
            )

        invalid = json.loads(json.dumps(manifest))
        invalid["chapters"][1].update(
            {
                "start_sample": 800,
                "start_ms": 800.0,
                "start_seconds": 0.8,
            }
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            chapters.validate_chapter_manifest(invalid)
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            chapters.build_chapter_manifest(
                segments,
                start_samples=[0, 800, 1_800],
                sample_rate=0,
                total_samples=2_500,
            )

    def test_worker_parsers_accept_chapter_output_arguments(self) -> None:
        qwen_args = qwen.build_parser().parse_args(
            [
                "synthesize",
                "--voice",
                "voice.wav",
                "--text",
                "input.txt",
                "--output",
                "output.wav",
                "--chapter-pause-ms",
                "750",
                "--chapter-manifest",
                "chapters.json",
            ]
        )
        moss_args = moss.build_parser().parse_args(
            [
                "synthesize",
                "--text",
                "input.txt",
                "--output",
                "output.wav",
                "--chapter-pause-ms",
                "750",
                "--chapter-manifest",
                "chapters.json",
            ]
        )
        for parsed in (qwen_args, moss_args):
            self.assertEqual(parsed.chapter_pause_ms, 750)
            self.assertEqual(parsed.chapter_manifest, "chapters.json")

    def test_language_normalization(self) -> None:
        self.assertIsNone(moss.normalize_language("Auto"))
        self.assertIsNone(moss.normalize_language("Auto (omit)"))
        self.assertEqual(moss.normalize_language(" French "), "French")

    def test_shared_ffmpeg_validation_rejects_static_and_ffmpeg_8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ffmpeg.exe").write_bytes(b"exe")
            self.assertFalse(moss._is_compatible_shared_ffmpeg_bin(root))
            for name in (
                "avcodec-62.dll",
                "avformat-62.dll",
                "avutil-60.dll",
                "avfilter-11.dll",
                "swscale-9.dll",
                "swresample-6.dll",
            ):
                (root / name).write_bytes(b"dll")
            self.assertFalse(moss._is_compatible_shared_ffmpeg_bin(root))

    def test_shared_ffmpeg_validation_accepts_ffmpeg_7(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "ffmpeg.exe",
                "avcodec-61.dll",
                "avformat-61.dll",
                "avutil-59.dll",
                "avfilter-10.dll",
                "swscale-8.dll",
                "swresample-5.dll",
            ):
                (root / name).write_bytes(b"fixture")
            self.assertTrue(moss._is_compatible_shared_ffmpeg_bin(root))

    def test_optional_reference_and_language_message(self) -> None:
        processor = _FakeProcessor()
        message = moss.build_user_message(
            processor,
            "Bonjour",
            "reference.wav",
            "French",
        )
        self.assertEqual(
            message,
            {
                "text": "Bonjour",
                "reference": ["reference.wav"],
                "language": "French",
            },
        )

        direct_message = moss.build_user_message(processor, "Hello", None, "Auto")
        self.assertEqual(direct_message, {"text": "Hello"})

    def test_first_moss_segment_preserves_direct_and_clone_generation(self) -> None:
        direct_processor = _FakeProcessor()
        direct_conversation, direct_mode = moss.build_segment_conversation(
            direct_processor,
            "Hello",
            None,
            "Auto",
        )
        self.assertEqual(direct_mode, "generation")
        self.assertEqual(direct_conversation, [[{"text": "Hello"}]])
        self.assertEqual(direct_processor.assistant_kwargs, [])

        clone_processor = _FakeProcessor()
        clone_conversation, clone_mode = moss.build_segment_conversation(
            clone_processor,
            "Bonjour",
            "clean-reference.wav",
            "French",
        )
        self.assertEqual(clone_mode, "generation")
        self.assertEqual(
            clone_conversation,
            [[{
                "text": "Bonjour",
                "reference": ["clean-reference.wav"],
                "language": "French",
            }]],
        )
        self.assertEqual(clone_processor.assistant_kwargs, [])

    def test_rolling_moss_continuation_pairs_exact_text_and_audio(self) -> None:
        processor = _FakeProcessor()
        prefix = moss.RollingPrefix(
            "The exact transcript of segment one.",
            "rolling-prefix.wav",
        )
        conversation, mode = moss.build_segment_conversation(
            processor,
            "Segment two starts here.",
            "clean-reference.wav",
            "English",
            prefix,
        )
        self.assertEqual(mode, "continuation")
        self.assertEqual(
            conversation,
            [[
                {
                    "text": (
                        "The exact transcript of segment one. "
                        "Segment two starts here."
                    ),
                    "reference": ["clean-reference.wav"],
                    "language": "English",
                },
                {
                    "assistant": {
                        "audio_codes_list": ["rolling-prefix.wav"],
                    },
                },
            ]],
        )

    def test_rolling_moss_continuation_works_without_clone_anchor(self) -> None:
        processor = _FakeProcessor()
        conversation, mode = moss.build_segment_conversation(
            processor,
            "The new segment.",
            None,
            "Auto",
            moss.RollingPrefix("Prior segment?", "prior.wav"),
        )
        self.assertEqual(mode, "continuation")
        self.assertEqual(
            conversation[0][0],
            {"text": "Prior segment? The new segment."},
        )
        self.assertEqual(
            conversation[0][1],
            {"assistant": {"audio_codes_list": ["prior.wav"]}},
        )

    def test_rolling_moss_prefix_must_be_complete(self) -> None:
        processor = _FakeProcessor()
        for prefix in (
            moss.RollingPrefix("", "prior.wav"),
            moss.RollingPrefix("Prior segment.", ""),
        ):
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    moss.build_segment_conversation(
                        processor,
                        "The new segment.",
                        "clean-reference.wav",
                        "English",
                        prefix,
                    )

    def test_sampling_validation(self) -> None:
        self.assertEqual(moss.DEFAULT_TEMPERATURE, 1.3)
        self.assertEqual(moss.DEFAULT_TOP_P, 0.75)
        moss.validate_sampling(1.3, 0.75, 25, 1.0, 4096, 800, 120)
        with self.assertRaises(moss.WorkerError):
            moss.validate_sampling(1.3, 1.1, 25, 1.0, 4096, 800, 120)
        with self.assertRaises(moss.WorkerError):
            moss.validate_sampling(1.3, 0.75, 0, 1.0, 4096, 800, 120)

    def test_moss_duration_outlier_detection_is_conservative(self) -> None:
        text = "a" * 100
        normal_audio = moss.normalize_audio_channels([0.0] * 1_000)
        slow_audio = moss.normalize_audio_channels([0.0] * 2_000)
        fast_audio = moss.normalize_audio_channels([0.0] * 500)

        normal_rate = moss.segment_speaking_rate(text, normal_audio, 100)
        slow_rate = moss.segment_speaking_rate(text, slow_audio, 100)
        fast_rate = moss.segment_speaking_rate(text, fast_audio, 100)
        self.assertEqual(normal_rate, 10.0)
        self.assertEqual(slow_rate, 5.0)
        self.assertEqual(fast_rate, 20.0)

        history = [9.5, 10.0, 10.5]
        self.assertIsNone(
            moss.duration_outlier_analysis(normal_rate, history, 1.35)
        )
        slow_outlier = moss.duration_outlier_analysis(slow_rate, history, 1.35)
        fast_outlier = moss.duration_outlier_analysis(fast_rate, history, 1.35)
        self.assertIsNotNone(slow_outlier)
        self.assertIsNotNone(fast_outlier)
        self.assertAlmostEqual(slow_outlier.rate_ratio, 0.5)
        self.assertAlmostEqual(fast_outlier.rate_ratio, 2.0)

        self.assertIsNone(
            moss.segment_speaking_rate("a" * 79, normal_audio, 100)
        )
        self.assertIsNone(
            moss.segment_speaking_rate(
                ("a" * 100) + " [pause 2s]",
                normal_audio,
                100,
            )
        )
        self.assertIsNone(
            moss.duration_outlier_analysis(fast_rate, [10.0], 1.35)
        )

    def test_moss_duration_retry_keeps_the_closest_pace(self) -> None:
        self.assertTrue(
            moss.duration_candidate_is_closer(11.0, 20.0, 10.0)
        )
        self.assertFalse(
            moss.duration_candidate_is_closer(25.0, 20.0, 10.0)
        )
        self.assertFalse(
            moss.duration_candidate_is_closer(None, 20.0, 10.0)
        )
        moss.validate_duration_outlier_controls(1, 1.35)
        with self.assertRaises(moss.WorkerError):
            moss.validate_duration_outlier_controls(4, 1.35)
        with self.assertRaises(moss.WorkerError):
            moss.validate_duration_outlier_controls(1, 1.0)

    def test_moss_parser_uses_consistency_defaults(self) -> None:
        parsed = moss.build_parser().parse_args(
            ["synthesize", "--text", "input.txt", "--output", "output.wav"]
        )
        self.assertEqual(parsed.temperature, 1.3)
        self.assertEqual(parsed.top_p, 0.75)
        self.assertEqual(parsed.duration_outlier_retries, 0)
        self.assertEqual(parsed.duration_outlier_ratio, 1.35)

    def test_moss_review_parser_commands(self) -> None:
        rerun = moss.build_parser().parse_args([
            "rerun-segment",
            "--review-manifest",
            "review.json",
            "--segment-index",
            "2",
        ])
        self.assertEqual(rerun.segment_index, 2)
        self.assertEqual(rerun.temperature, 1.3)
        compile_args = moss.build_parser().parse_args([
            "compile-review",
            "--review-manifest",
            "review.json",
            "--output",
            "output.wav",
        ])
        self.assertEqual(compile_args.output, "output.wav")

    def test_snapshot_paths_include_pinned_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            qwen_path = qwen._expected_snapshot_path(root, qwen.DEFAULT_MODEL_ID)
            self.assertEqual(
                qwen_path.name,
                qwen.PINNED_MODEL_REVISIONS[qwen.DEFAULT_MODEL_ID],
            )
            moss_model_path = moss._expected_snapshot_path(root, "model")
            moss_codec_path = moss._expected_snapshot_path(root, "codec")
            self.assertEqual(moss_model_path.name, moss.MODEL_REVISION)
            self.assertEqual(moss_codec_path.name, moss.CODEC_REVISION)
            local_model_path = moss._expected_snapshot_path(
                root,
                "model",
                moss.LOCAL_MODEL_REPO_ID,
            )
            local_codec_path = moss._expected_snapshot_path(
                root,
                "codec",
                moss.LOCAL_MODEL_REPO_ID,
            )
            self.assertEqual(local_model_path.name, moss.LOCAL_MODEL_REVISION)
            self.assertEqual(local_codec_path.name, moss.LOCAL_CODEC_REVISION)
            self.assertNotEqual(
                moss._manifest_path(root),
                moss._manifest_path(root, moss.LOCAL_MODEL_REPO_ID),
            )

    def test_moss_parser_selects_each_pinned_checkpoint(self) -> None:
        for command in ("check", "setup", "download", "synthesize"):
            for model_id in moss.PINNED_MODEL_SPECS:
                argv = [command, "--model-id", model_id]
                if command == "synthesize":
                    argv.extend(["--text", "input.txt", "--output", "output.wav"])
                with self.subTest(command=command, model_id=model_id):
                    parsed = moss.build_parser().parse_args(argv)
                    self.assertEqual(parsed.model_id, model_id)

    def test_moss_audio_normalization_preserves_mono_and_stereo(self) -> None:
        mono = moss.normalize_audio_channels([0.0, 0.25, -0.25])
        self.assertEqual(mono.shape, (1, 3))

        stereo = moss.normalize_audio_channels(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]
        )
        self.assertEqual(stereo.shape, (2, 3))

        sample_first = moss.normalize_audio_channels(
            [[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]]
        )
        self.assertEqual(sample_first.shape, (2, 3))
        self.assertEqual(moss.soundfile_waveform(stereo).shape, (3, 2))
        self.assertEqual(moss.soundfile_waveform(mono).shape, (3,))

    def test_moss_audio_concatenation_keeps_channels_and_gap(self) -> None:
        first = moss.normalize_audio_channels([[0.1, 0.2], [0.3, 0.4]])
        second = moss.normalize_audio_channels([[0.5], [0.6]])
        combined = moss.combine_audio_segments([first, second], 1_000, 10)
        self.assertEqual(combined.shape, (2, 13))
        self.assertTrue((combined[:, 2:12] == 0).all())
        with self.assertRaises(moss.WorkerError):
            moss.combine_audio_segments(
                [first, moss.normalize_audio_channels([0.5])],
                1_000,
                0,
            )

    def test_moss_audio_concatenation_reports_exact_chapter_start(self) -> None:
        first = moss.normalize_audio_channels([[0.1, 0.2], [0.3, 0.4]])
        second = moss.normalize_audio_channels([[0.5], [0.6]])
        combined, timing = moss.combine_audio_segments_with_timing(
            [first, second],
            sample_rate=1_000,
            gap_ms=10,
            chapter_segment_indices=[1],
            chapter_pause_ms=20,
        )
        self.assertEqual(combined.shape, (2, 33))
        self.assertEqual(timing.gap_samples_before, (0, 30))
        self.assertEqual(timing.start_samples, (0, 32))
        self.assertTrue((combined[:, 2:32] == 0).all())

    def test_moss_streaming_writer_preserves_samples_and_gaps(self) -> None:
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "streamed.wav"
            writer = moss.StreamingWaveWriter(output, 1_000)
            writer.write(moss.normalize_audio_channels([0.1, 0.2]))
            writer.write(moss.normalize_audio_channels([0.5]), gap_samples=10)
            self.assertEqual(writer.total_samples, 13)
            writer.finalize()

            samples, sample_rate = sf.read(output, dtype="float32")
            self.assertEqual(sample_rate, 1_000)
            self.assertEqual(samples.shape, (13,))
            self.assertTrue(np.allclose(samples[:2], [0.1, 0.2], atol=1e-4))
            self.assertTrue(np.allclose(samples[2:12], 0.0, atol=1e-4))
            self.assertTrue(np.allclose(samples[12:], [0.5], atol=1e-4))

    def test_moss_streaming_writer_discards_incompatible_partial_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "streamed.wav"
            writer = moss.StreamingWaveWriter(output, 1_000)
            writer.write(moss.normalize_audio_channels([0.1, 0.2]))
            with self.assertRaises(moss.WorkerError):
                writer.write(moss.normalize_audio_channels([[0.1], [0.2]]))
            writer.discard()
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).glob("*.wav")), [])

    def test_moss_review_manifest_compiles_current_segment_takes(self) -> None:
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            segment_dir = root / "segments"
            segment_dir.mkdir()
            sf.write(
                segment_dir / "segment-0001.wav",
                np.full(1_000, 0.1, dtype=np.float32),
                1_000,
                subtype="FLOAT",
            )
            sf.write(
                segment_dir / "segment-0002.wav",
                np.full(500, 0.2, dtype=np.float32),
                1_000,
                subtype="FLOAT",
            )
            review_path = root / "review.json"
            manifest = {
                "version": moss.REVIEW_MANIFEST_VERSION,
                "sample_rate": 1_000,
                "gap_ms": 100,
                "chapter_pause_ms": 200,
                "segments": [
                    {
                        "index": 0,
                        "text": "First segment.",
                        "audio_file": "segments/segment-0001.wav",
                        "sample_count": 1_000,
                        "duration_seconds": 1.0,
                        "attempt": 1,
                        "starts_chapter": False,
                        "chapter_title": None,
                        "speaking_rate": None,
                        "pace_ratio": None,
                        "pace_status": "not-compared",
                        "updated_at": 1,
                    },
                    {
                        "index": 1,
                        "text": "Chapter Two.",
                        "audio_file": "segments/segment-0002.wav",
                        "sample_count": 500,
                        "duration_seconds": 0.5,
                        "attempt": 2,
                        "starts_chapter": True,
                        "chapter_title": "Chapter Two",
                        "speaking_rate": None,
                        "pace_ratio": None,
                        "pace_status": "not-compared",
                        "updated_at": 2,
                    },
                ],
            }
            moss.write_review_manifest(review_path, manifest)
            self.assertEqual(
                moss.read_review_manifest(review_path)["segments"][1]["attempt"],
                2,
            )

            output_path = root / "compiled.wav"
            chapter_path = root / "chapters.json"
            moss.compile_review(
                str(review_path),
                str(output_path),
                str(chapter_path),
            )
            output_info = sf.info(output_path)
            self.assertEqual(output_info.samplerate, 1_000)
            self.assertEqual(output_info.frames, 1_800)
            chapter_manifest = json.loads(chapter_path.read_text(encoding="utf-8"))
            self.assertEqual(chapter_manifest["chapters"][0]["start_sample"], 1_300)

            unsafe = json.loads(json.dumps(manifest))
            unsafe["segments"][0]["audio_file"] = "../outside.wav"
            with self.assertRaises(moss.WorkerError):
                moss.validate_review_manifest(
                    unsafe,
                    review_path,
                    require_audio=False,
                )

    def test_qwen_manifest_detects_snapshot_tampering(self) -> None:
        original_minimum = qwen.MINIMUM_MODEL_BYTES
        qwen.MINIMUM_MODEL_BYTES = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                snapshot = qwen._expected_snapshot_path(root, qwen.DEFAULT_MODEL_ID)
                self._create_fake_snapshot(snapshot)
                qwen._write_manifest_entry(root, qwen.DEFAULT_MODEL_ID, snapshot)
                self.assertEqual(
                    qwen.load_pinned_snapshot(root, qwen.DEFAULT_MODEL_ID),
                    snapshot,
                )
                (snapshot / "model.safetensors").write_bytes(b"tampered")
                with self.assertRaises(qwen.WorkerError):
                    qwen.load_pinned_snapshot(root, qwen.DEFAULT_MODEL_ID)
        finally:
            qwen.MINIMUM_MODEL_BYTES = original_minimum

    def test_moss_manifest_verifies_model_and_codec(self) -> None:
        original_minimums = {
            name: artifact["minimum_bytes"]
            for name, artifact in moss.PINNED_ARTIFACTS.items()
        }
        for artifact in moss.PINNED_ARTIFACTS.values():
            artifact["minimum_bytes"] = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                snapshots = {
                    name: moss._expected_snapshot_path(root, name)
                    for name in moss.PINNED_ARTIFACTS
                }
                for snapshot in snapshots.values():
                    self._create_fake_snapshot(snapshot)
                moss._write_manifest(root, snapshots)
                self.assertEqual(
                    moss.load_pinned_snapshots(root),
                    (snapshots["model"], snapshots["codec"]),
                )
                (snapshots["codec"] / "model.safetensors").write_bytes(b"tampered")
                with self.assertRaises(moss.WorkerError):
                    moss.load_pinned_snapshots(root)
        finally:
            for name, minimum in original_minimums.items():
                moss.PINNED_ARTIFACTS[name]["minimum_bytes"] = minimum

    def test_moss_checkpoint_manifests_coexist_and_validate_independently(self) -> None:
        original_minimums = {
            (model_id, name): artifact["minimum_bytes"]
            for model_id, spec in moss.PINNED_MODEL_SPECS.items()
            for name, artifact in spec["artifacts"].items()
        }
        for spec in moss.PINNED_MODEL_SPECS.values():
            for artifact in spec["artifacts"].values():
                artifact["minimum_bytes"] = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                installed = {}
                for model_id, spec in moss.PINNED_MODEL_SPECS.items():
                    snapshots = {
                        name: moss._expected_snapshot_path(root, name, model_id)
                        for name in spec["artifacts"]
                    }
                    for snapshot in snapshots.values():
                        self._create_fake_snapshot(snapshot)
                    moss._write_manifest(root, snapshots, model_id)
                    installed[model_id] = snapshots

                self.assertTrue(moss._manifest_path(root).is_file())
                self.assertTrue(
                    moss._manifest_path(root, moss.LOCAL_MODEL_REPO_ID).is_file()
                )
                for model_id, snapshots in installed.items():
                    self.assertEqual(
                        moss.load_pinned_snapshots(root, model_id),
                        (snapshots["model"], snapshots["codec"]),
                    )

                local_codec = installed[moss.LOCAL_MODEL_REPO_ID]["codec"]
                (local_codec / "model.safetensors").write_bytes(b"tampered")
                with self.assertRaises(moss.WorkerError):
                    moss.load_pinned_snapshots(root, moss.LOCAL_MODEL_REPO_ID)
                default_snapshots = installed[moss.DEFAULT_MODEL_ID]
                self.assertEqual(
                    moss.load_pinned_snapshots(root),
                    (default_snapshots["model"], default_snapshots["codec"]),
                )
        finally:
            for model_id, spec in moss.PINNED_MODEL_SPECS.items():
                for name, artifact in spec["artifacts"].items():
                    artifact["minimum_bytes"] = original_minimums[(model_id, name)]


if __name__ == "__main__":
    unittest.main(verbosity=2)
