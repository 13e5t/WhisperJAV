#!/usr/bin/env python3
"""Run the baseline Japanese Parakeet pipeline against a reusable file queue.

The queue is a directory, not a second transcription architecture:
WhisperJAV receives the directory in one invocation and uses the final SRT/VTT
as the completion marker. The existing Parakeet generator lifecycle remains
unchanged and loads/unloads its GPU model per input file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


MODEL_ID = "grider-transwithai/parakeet-ctc-1.1b-ja"
MEDIA_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma",
    ".m4a", ".m4b", ".opus",
}


def repo_root() -> Path:
    """Return the repository root when this helper is run from the checkout."""
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run the reusable Parakeet input queue using the baseline settings."
    )
    parser.add_argument("--input-dir", type=Path, default=root / "input")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--temp-dir", type=Path, default=root / "temp" / "parakeet_queue")
    parser.add_argument(
        "--qwen-segmenter",
        choices=["none", "silero", "silero-v4.0", "silero-v3.1", "silero-v6.2",
                 "nemo", "nemo-lite", "whisper-vad", "ten", "whisperseg"],
        default="ten",
        help="Speech segmentation backend for VAD-based chunking (default: ten).",
    )
    parser.add_argument("--watch", action="store_true",
                        help="Keep polling for newly added files after each batch.")
    parser.add_argument("--poll-seconds", type=float, default=15.0,
                        help="Polling interval for --watch (default: 15).")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep per-file diagnostics under --temp-dir.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable WhisperJAV debug logging.")
    return parser


def pending_media(input_dir: Path, output_dir: Path) -> List[Path]:
    """Return media files without a completed SRT/VTT in deterministic order."""
    if not input_dir.exists():
        return []

    pending: List[Path] = []
    for path in sorted(input_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        stem = path.stem
        if has_completed_output(path, output_dir):
            continue
        pending.append(path)
    return pending


def has_completed_output(media_path: Path, output_dir: Path) -> bool:
    """Return whether the durable subtitle marker exists for one media file."""
    stem = media_path.stem
    return (
        (output_dir / f"{stem}.ja.whisperjav.srt").exists()
        or (output_dir / f"{stem}.ja.whisperjav.vtt").exists()
    )


def _configure_cuda_runtime(environment: dict[str, str]) -> None:
    """Expose PyTorch's bundled CUDA libraries to ONNX Runtime on Linux.

    The RunPod PyTorch environment keeps cuDNN/cuBLAS under the venv's
    ``nvidia`` site packages rather than on the system linker path.  WhisperSeg
    uses ONNX Runtime in a child process, so pass those library directories to
    it without changing the host or the user's shell permanently.
    """
    if os.name == "nt":
        return

    try:
        import site

        site_packages = [Path(path) for path in site.getsitepackages()]
    except (ImportError, AttributeError):
        return

    library_dirs = []
    for component in ("cudnn", "cublas", "cuda_runtime", "cufft"):
        for site_package in site_packages:
            library_dir = site_package / "nvidia" / component / "lib"
            if library_dir.is_dir():
                library_dirs.append(str(library_dir))
                break

    if not library_dirs:
        return

    existing = environment.get("LD_LIBRARY_PATH", "")
    existing_dirs = [path for path in existing.split(os.pathsep) if path]
    merged = library_dirs + [path for path in existing_dirs if path not in library_dirs]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(merged)


def build_command(args: argparse.Namespace) -> List[str]:
    """Build the exact baseline Parakeet command for one queue pass."""
    command = [
        sys.executable,
        "-m",
        "whisperjav.main",
        "--queue",
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--mode",
        "qwen",
        "--qwen-generator",
        "parakeet",
        "--qwen-segmenter",
        getattr(args, "qwen_segmenter", "ten"),
        "--qwen-framer",
        "vad-grouped",
        "--qwen-model-id",
        MODEL_ID,
        "--device",
        "cuda",
        "--qwen-device",
        "cuda",
        "--qwen-dtype",
        "float16",
        "--qwen-timestamps",
        "word",
        "--qwen-regroup",
        "off",
        "--parakeet-regroup",
        "--temp-dir",
        str(args.temp_dir),
    ]
    if args.keep_temp:
        command.append("--keep-temp")
    if args.debug:
        command.append("--debug")
    return command


def run_once(args: argparse.Namespace) -> int:
    """Process the current queue snapshot, returning the CLI exit code."""
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    pending = pending_media(args.input_dir, args.output_dir)
    if not pending:
        print(f"[parakeet-queue] no pending media in {args.input_dir}")
        return 0

    print(f"[parakeet-queue] processing {len(pending)} file(s)")
    environment = os.environ.copy()
    root = str(repo_root())
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        root if not existing_pythonpath else root + os.pathsep + existing_pythonpath
    )
    # Keep model downloads on the persistent RunPod volume. This also makes a
    # fresh shell after Pod start safe when HF_HOME was not exported again.
    environment.setdefault(
        "HF_HOME",
        str(repo_root().parent / ".cache" / "huggingface"),
    )
    _configure_cuda_runtime(environment)
    return_code = subprocess.run(
        build_command(args),
        cwd=repo_root(),
        env=environment,
        check=False,
    ).returncode
    if return_code == 0:
        incomplete = [path for path in pending if not has_completed_output(path, args.output_dir)]
        if incomplete:
            print(
                "[parakeet-queue] command finished but these files have no output: "
                + ", ".join(path.name for path in incomplete),
                file=sys.stderr,
            )
            return 1
    return return_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero")

    if not args.watch:
        return run_once(args)

    print(f"[parakeet-queue] watching {args.input_dir} (Ctrl-C to stop)")
    try:
        while True:
            return_code = run_once(args)
            if return_code != 0:
                print(f"[parakeet-queue] batch exited with code {return_code}", file=sys.stderr)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[parakeet-queue] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
