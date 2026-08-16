"""Focused tests for ensemble backend-specific speech-segmenter defaults."""

from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("librosa")

from whisperjav.ensemble import pass_worker


class CapturePipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _build_qwen_capture(monkeypatch, tmp_path, backend, speech_segmenter=None):
    resolved_backends = []

    def fake_resolve(segmenter_backend, sensitivity, user_overrides=None):
        resolved_backends.append((segmenter_backend, sensitivity))
        return {}

    # _build_pipeline uses the dedicated QwenPipeline symbol after resolving
    # the backend-specific parameter bundle.
    monkeypatch.setattr(pass_worker, "QwenPipeline", CapturePipeline)
    monkeypatch.setattr(pass_worker, "resolve_qwen_sensitivity", fake_resolve)

    pass_config = {
        "pipeline": "qwen",
        "qwen_params": {"generator_backend": backend},
    }
    if speech_segmenter is not None:
        pass_config["speech_segmenter"] = speech_segmenter

    pipeline = pass_worker._build_pipeline(
        pass_config=pass_config,
        pass_number=1,
        output_dir=str(tmp_path / "out"),
        keep_temp_files=False,
        subs_language="native",
        extra_kwargs={},
        pass_temp_dir=Path(tmp_path / "tmp"),
    )
    return pipeline.kwargs, resolved_backends


def test_parakeet_defaults_to_ten_before_sensitivity_resolution(monkeypatch, tmp_path):
    kwargs, resolved = _build_qwen_capture(monkeypatch, tmp_path, "parakeet")

    assert kwargs["speech_segmenter"] == "ten"
    assert resolved == [("ten", "balanced")]


@pytest.mark.parametrize("speech_segmenter", ["whisperseg", "silero-v6.2"])
def test_explicit_parakeet_segmenter_overrides_ten(
    monkeypatch, tmp_path, speech_segmenter
):
    kwargs, resolved = _build_qwen_capture(
        monkeypatch,
        tmp_path,
        "parakeet",
        speech_segmenter=speech_segmenter,
    )

    assert kwargs["speech_segmenter"] == speech_segmenter
    assert resolved == [(speech_segmenter, "balanced")]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("qwen3", "whisperseg"),
        ("anime-whisper", "whisperseg"),
        ("cohere", "whisperseg"),
    ],
)
def test_other_generator_segmenter_defaults_are_unchanged(
    monkeypatch, tmp_path, backend, expected
):
    kwargs, resolved = _build_qwen_capture(monkeypatch, tmp_path, backend)

    assert kwargs["speech_segmenter"] == expected
    assert resolved == [(expected, "balanced")]
