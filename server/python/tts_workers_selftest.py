#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import moss_tts_worker as moss
import qwen_tts_worker as qwen


class _FakeProcessor:
    def __init__(self) -> None:
        self.kwargs = None

    def build_user_message(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


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

    def test_sampling_validation(self) -> None:
        moss.validate_sampling(1.7, 0.8, 25, 1.0, 4096, 800, 120)
        with self.assertRaises(moss.WorkerError):
            moss.validate_sampling(1.7, 1.1, 25, 1.0, 4096, 800, 120)
        with self.assertRaises(moss.WorkerError):
            moss.validate_sampling(1.7, 0.8, 0, 1.0, 4096, 800, 120)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
