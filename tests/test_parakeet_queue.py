"""Tests for reusable input/output queue semantics."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tools.run_parakeet_queue import build_command, build_parser, pending_media


def parse_cli(monkeypatch, *arguments):
    # The production parser imports the optional ASR stack at module load.
    # Keep the queue helper tests runnable in a minimal Windows checkout while
    # exercising parser integration whenever the normal CLI dependencies exist.
    pytest.importorskip("stable_whisper")
    from whisperjav.main import parse_arguments

    monkeypatch.setattr(sys, "argv", ["whisperjav", *arguments])
    return parse_arguments()


def test_queue_defaults_to_repo_local_input_and_output(monkeypatch):
    args = parse_cli(monkeypatch, "--queue")

    assert args.input == ["input"]
    assert args.queue_input_dir == "input"
    assert args.output_dir == "output"
    assert args.skip_existing is True


def test_queue_accepts_explicit_input_and_output_directories(monkeypatch, tmp_path):
    input_dir = tmp_path / "incoming"
    output_dir = tmp_path / "subtitles"
    args = parse_cli(
        monkeypatch,
        "--queue",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    )

    assert args.input == [str(input_dir)]
    assert args.queue_input_dir == str(input_dir)
    assert args.output_dir == str(output_dir)
    assert args.skip_existing is True


def test_input_dir_cannot_be_combined_with_positional_input(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        parse_cli(monkeypatch, "clip.wav", "--input-dir", str(tmp_path / "incoming"))


def test_non_queue_defaults_remain_backward_compatible(monkeypatch):
    args = parse_cli(monkeypatch, "clip.wav")

    assert args.input == ["clip.wav"]
    assert args.output_dir == "source"
    assert args.skip_existing is False


def test_pending_media_uses_srt_or_vtt_as_completion_marker(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "done.wav").touch()
    (input_dir / "pending.flac").touch()
    (input_dir / "notes.txt").touch()
    (output_dir / "done.ja.whisperjav.srt").touch()

    assert pending_media(input_dir, output_dir) == [input_dir / "pending.flac"]


def test_queue_defaults_to_ten_segmenter():
    assert build_parser().parse_args([]).qwen_segmenter == "ten"


def test_queue_defaults_to_full_scene_framer():
    assert build_parser().parse_args([]).qwen_framer == "full-scene"


def test_queue_accepts_explicit_vad_grouped_framer():
    args = build_parser().parse_args(["--qwen-framer", "vad-grouped"])

    assert args.qwen_framer == "vad-grouped"


def test_queue_command_uses_validated_baseline_parakeet_settings(tmp_path):
    args = Namespace(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
        keep_temp=False,
        debug=False,
    )

    command = build_command(args)

    assert "--mode" in command and command[command.index("--mode") + 1] == "qwen"
    assert "--qwen-generator" in command and command[command.index("--qwen-generator") + 1] == "parakeet"
    assert "--qwen-segmenter" in command and command[command.index("--qwen-segmenter") + 1] == "ten"
    assert "--qwen-framer" in command and command[command.index("--qwen-framer") + 1] == "full-scene"
    assert "--qwen-regroup" in command and command[command.index("--qwen-regroup") + 1] == "off"
    assert "--parakeet-regroup" in command
    assert "--qwen-dtype" in command and command[command.index("--qwen-dtype") + 1] == "float16"
    assert "--pass1-qwen-params" not in command


def test_queue_command_preserves_explicit_vad_grouped_ab_path(tmp_path):
    args = Namespace(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
        qwen_framer="vad-grouped",
        keep_temp=False,
        debug=False,
    )

    command = build_command(args)

    assert command[command.index("--qwen-framer") + 1] == "vad-grouped"
