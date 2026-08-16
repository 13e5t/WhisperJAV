"""Mocked tests for the optional Parakeet CTC Japanese backend.

These tests deliberately install a tiny fake ``nemo.collections.asr`` module;
they never download checkpoints or require a GPU.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


class FakeASRModel:
    """Small stand-in for ``nemo.collections.asr.models.ASRModel``."""

    from_pretrained = MagicMock()
    restore_from = MagicMock()


class FakeNeMoModel:
    def __init__(self, outputs=None, window_stride=0.01, reject_timestamps=False):
        self.outputs = outputs or []
        self.reject_timestamps = reject_timestamps
        self.transcribe_calls = []
        self.to_calls = []
        self.eval_called = False
        self.cfg = SimpleNamespace(
            preprocessor=SimpleNamespace(window_stride=window_stride),
        )

    def to(self, value):
        self.to_calls.append(value)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def transcribe(self, **kwargs):
        self.transcribe_calls.append(kwargs)
        if self.reject_timestamps and "timestamps" in kwargs:
            raise TypeError("timestamps is not supported by this checkpoint")
        return self.outputs


class LegacySignatureNeMoModel(FakeNeMoModel):
    """NeMo 2.0-style transcribe signature without ``**kwargs``."""

    def transcribe(self, audio, batch_size=1, return_hypotheses=False):
        self.transcribe_calls.append(
            {
                "audio": audio,
                "batch_size": batch_size,
                "return_hypotheses": return_hypotheses,
            }
        )
        return self.outputs


class PublicStrategyNeMoModel(FakeNeMoModel):
    """NeMo model exposing the public decoding-strategy API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg.decoding = {
            "preserve_alignments": False,
            "compute_timestamps": False,
            "ctc_timestamp_type": "char",
        }
        self.change_decoding_strategy_calls = []

    def change_decoding_strategy(self, decoding_cfg):
        self.change_decoding_strategy_calls.append(decoding_cfg)


class RaisingPublicStrategyNeMoModel(PublicStrategyNeMoModel):
    """Public API failure should fall back to the legacy decoder flags."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoding = SimpleNamespace(cfg={}, compute_timestamps=False)

    def change_decoding_strategy(self, decoding_cfg):
        self.change_decoding_strategy_calls.append(decoding_cfg)
        raise RuntimeError("decoder config is incompatible with this NeMo runtime")


def _install_fake_nemo(monkeypatch, model):
    """Install enough package structure for ``import nemo.collections.asr``."""
    nemo = types.ModuleType("nemo")
    nemo.__path__ = []
    collections = types.ModuleType("nemo.collections")
    collections.__path__ = []
    asr = types.ModuleType("nemo.collections.asr")
    asr.models = SimpleNamespace(ASRModel=FakeASRModel)
    nemo.collections = collections
    collections.asr = asr

    FakeASRModel.from_pretrained.reset_mock()
    FakeASRModel.restore_from.reset_mock()
    FakeASRModel.from_pretrained.return_value = model
    FakeASRModel.restore_from.return_value = model

    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline test")),
    )
    return FakeASRModel


def _force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


class TestParakeetRegistration:
    def test_import_is_lazy_with_respect_to_nemo(self):
        module_name = "whisperjav.modules.subtitle_pipeline.generators.parakeet"
        sys.modules.pop(module_name, None)
        real_import = builtins.__import__

        def reject_nemo(name, *args, **kwargs):
            if name == "nemo" or name.startswith("nemo."):
                raise AssertionError("NeMo was imported while importing the adapter")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=reject_nemo):
            module = importlib.import_module(module_name)
            generator = module.ParakeetTextGenerator()

        assert generator.is_loaded is False
        assert "import nemo.collections.asr" in inspect_source(module)

    def test_factory_registration_and_protocol(self):
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )
        from whisperjav.modules.subtitle_pipeline.protocols import TextGenerator

        assert "parakeet" in TextGeneratorFactory.available()
        generator = TextGeneratorFactory.create("parakeet")
        assert isinstance(generator, TextGenerator)
        assert generator._config["model_id"] == (
            "grider-transwithai/parakeet-ctc-1.1b-ja"
        )

    def test_configuration_validation(self):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        with pytest.raises(ValueError, match="timestamp_level"):
            ParakeetTextGenerator(timestamp_level="phoneme")
        with pytest.raises(ValueError, match="batch_size"):
            ParakeetTextGenerator(batch_size=0)

    def test_qwen_cli_exposes_parakeet_backend(self):
        source = Path("whisperjav/main.py").read_text(encoding="utf-8")
        assert 'choices=["qwen3", "anime-whisper", "parakeet"]' in source
        assert "parakeet uses native NeMo CTC timestamps" in source


def inspect_source(module):
    """Return source without importing any optional runtime."""
    import inspect

    return inspect.getsource(module)


class TestParakeetInference:
    def test_load_generate_batch_and_native_char_timestamps(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[
                SimpleNamespace(
                    text="こんにちは",
                    timestamp={
                        "char": [
                            {"char": "こ", "start": 0.0, "end": 0.20},
                            {"char": "ん", "start": 0.20, "end": 0.40},
                        ],
                    },
                ),
            ],
        )
        asr_model = _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        audio = tmp_path / "frame.wav"

        generator = ParakeetTextGenerator(batch_size=4)
        generator.load()
        results = generator.generate_batch([audio], language="ja")

        assert generator.is_loaded is True
        assert asr_model.from_pretrained.call_args.kwargs["model_name"] == (
            "grider-transwithai/parakeet-ctc-1.1b-ja"
        )
        assert asr_model.from_pretrained.call_args.kwargs["map_location"] == "cpu"
        assert model.eval_called is True
        assert model.transcribe_calls[0]["audio"] == [str(audio)]
        assert model.transcribe_calls[0]["batch_size"] == 4
        assert model.transcribe_calls[0]["return_hypotheses"] is True
        assert model.transcribe_calls[0]["timestamps"] is True
        assert results[0].text == "こんにちは"
        assert [(w.word, w.start, w.end) for w in results[0].words] == [
            ("こ", 0.0, 0.20),
            ("ん", 0.20, 0.40),
        ]
        assert results[0].metadata["timestamp_status"] == "native"
        assert results[0].metadata["timing_source"] == "native_ctc"

    def test_public_decoding_strategy_api_configures_native_timestamps(
        self, monkeypatch
    ):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = PublicStrategyNeMoModel()
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)

        generator = ParakeetTextGenerator()
        generator.load()

        assert len(model.change_decoding_strategy_calls) == 1
        configured = model.change_decoding_strategy_calls[0]
        assert configured["preserve_alignments"] is True
        assert configured["compute_timestamps"] is True
        assert configured["ctc_timestamp_type"] == "all"

    def test_public_decoding_strategy_failure_uses_legacy_setup(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = RaisingPublicStrategyNeMoModel()
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)

        generator = ParakeetTextGenerator()
        generator.load()

        assert len(model.change_decoding_strategy_calls) == 1
        assert model.decoding.compute_timestamps is True
        assert model.decoding.cfg["ctc_timestamp_type"] == "all"

    def test_generate_is_a_single_item_batch_wrapper(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[SimpleNamespace(text="声", timestamp={"word": [
                {"word": "声", "start": 0.1, "end": 0.3},
            ]})],
        )
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator(timestamp_level="word")
        generator.load()

        result = generator.generate(tmp_path / "one.wav")

        assert result.text == "声"
        assert result.words[0].word == "声"
        assert len(model.transcribe_calls) == 1

    def test_local_nemo_checkpoint_uses_restore_from(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel()
        asr_model = _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        checkpoint = tmp_path / "parakeet-ja.nemo"

        generator = ParakeetTextGenerator(model_id=str(checkpoint))
        generator.load()

        asr_model.restore_from.assert_called_once_with(
            restore_path=str(checkpoint),
            map_location="cpu",
        )
        asr_model.from_pretrained.assert_not_called()

    def test_model_card_repo_restores_published_nemo_archive(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel()
        asr_model = _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        checkpoint = tmp_path / "parakeet-ja.nemo"
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **kwargs: str(checkpoint),
        )

        generator = ParakeetTextGenerator()
        generator.load()

        asr_model.restore_from.assert_called_once_with(
            restore_path=str(checkpoint),
            map_location="cpu",
        )
        asr_model.from_pretrained.assert_not_called()

    def test_offset_timestamps_are_converted_using_model_stride(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[SimpleNamespace(text="音", timestamp={
                "char": [
                    {"char": "音", "start_offset": 10, "end_offset": 20},
                    {"char": "壊", "start_offset": "bad", "end_offset": 2},
                ],
            })],
            window_stride=0.01,
        )
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        result = generator.generate_batch([Path("offset.wav")])[0]

        assert len(result.words) == 1
        assert result.words[0].start == pytest.approx(0.8)
        assert result.words[0].end == pytest.approx(1.6)

    def test_missing_and_malformed_timestamp_provenance_is_explicit(
        self, monkeypatch, tmp_path
    ):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[
                SimpleNamespace(text="音声", timestamp=None),
                {"text": "壊れ", "timestamp": {"char": [
                    {"char": "壊", "start": 1.0, "end": 0.5},
                ]}},
            ],
        )
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        results = generator.generate_batch([
            tmp_path / "no-timestamp.wav",
            tmp_path / "malformed.wav",
        ])

        assert [result.text for result in results] == ["音声", "壊れ"]
        assert all(result.words == [] for result in results)
        assert results[0].metadata["timestamp_status"] == "unavailable"
        assert results[0].metadata["timing_source"] == "frame_fallback_required"
        assert results[1].metadata["timestamp_status"] == "malformed"
        assert results[1].metadata["timing_source"] == "native_ctc_malformed"
        assert results[1].metadata["native_timestamp_invalid_count"] == 1

    def test_partial_malformed_timestamps_are_reported(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[SimpleNamespace(text="混在", timestamp={"char": [
                {"char": "混", "start": 0.0, "end": 0.2},
                {"char": "在", "start": 0.4, "end": 0.1},
            ]})],
        )
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        result = generator.generate_batch([tmp_path / "partial.wav"])[0]

        assert len(result.words) == 1
        assert result.metadata["timestamp_status"] == "malformed"
        assert result.metadata["timing_source"] == "native_ctc_malformed"
        assert result.metadata["native_timestamp_invalid_count"] == 1

    def test_finer_malformed_level_falls_back_to_clean_lower_level(
        self, monkeypatch, tmp_path
    ):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(
            outputs=[SimpleNamespace(text="下位", timestamp={
                "char": [{"char": "下", "start": 0.2, "end": 0.1}],
                "word": [{"word": "下位", "start": 0.0, "end": 0.4}],
            })],
        )
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        result = generator.generate_batch([tmp_path / "lower-level.wav"])[0]

        assert [(word.word, word.start, word.end) for word in result.words] == [
            ("下位", 0.0, 0.4),
        ]
        assert result.metadata["timestamp_level"] == "word"
        assert result.metadata["native_timestamp_invalid_count"] == 0
        assert result.metadata["timing_source"] == "native_ctc"

    def test_timestamp_none_skips_timestamp_request(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(outputs=["テスト"])
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator(timestamp_level="none")
        generator.load()

        result = generator.generate_batch([tmp_path / "plain.wav"])[0]

        assert result.text == "テスト"
        assert result.words == []
        assert result.metadata["timestamp_status"] == "disabled"
        assert result.metadata["timing_source"] == "disabled"
        assert "timestamps" not in model.transcribe_calls[0]
        assert "return_hypotheses" not in model.transcribe_calls[0]

    def test_unsupported_native_timestamps_retry_text_only(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel(outputs=["テキスト"], reject_timestamps=True)
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        result = generator.generate_batch([tmp_path / "legacy.wav"])[0]

        assert result.text == "テキスト"
        assert result.words == []
        assert result.metadata["timestamp_status"] == "unavailable"
        assert result.metadata["timing_source"] == "frame_fallback_required"
        assert result.metadata["timestamp_request_fallback"] is True
        assert len(model.transcribe_calls) == 2
        assert "timestamps" not in model.transcribe_calls[1]

    def test_legacy_nemo_decoder_timestamps_are_used(self, monkeypatch, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = LegacySignatureNeMoModel(
            outputs=[
                SimpleNamespace(
                    text="声",
                    timestep={
                        "char": [
                            {"char": "声", "start_offset": 10, "end_offset": 20},
                        ],
                    },
                ),
            ],
            window_stride=0.01,
        )
        model.decoding = SimpleNamespace(cfg={}, compute_timestamps=False)
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        result = generator.generate(tmp_path / "legacy-native.wav")

        assert result.text == "声"
        assert [(word.word, word.start, word.end) for word in result.words] == [
            ("声", 0.8, 1.6),
        ]
        assert result.metadata["timestamp_status"] == "native"
        assert result.metadata["timing_source"] == "native_ctc"
        assert result.metadata["timestamp_request_fallback"] is True
        assert model.decoding.compute_timestamps is True
        assert model.decoding.cfg["ctc_timestamp_type"] == "all"
        assert len(model.transcribe_calls) == 1
        assert model.transcribe_calls[0]["return_hypotheses"] is True
        assert "timestamps" not in model.transcribe_calls[0]


class TestParakeetLifecycleAndPipelineBridge:
    def test_device_and_dtype_fallbacks(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        assert generator._detect_device("auto") == "cpu"
        assert generator._detect_device("cuda") == "cpu"
        assert generator._detect_dtype("cpu", "float16") is torch.float32
        assert generator._detect_dtype("cpu", "bfloat16") is torch.float32

    def test_unload_and_cleanup_release_model(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.parakeet import (
            ParakeetTextGenerator,
        )

        model = FakeNeMoModel()
        _install_fake_nemo(monkeypatch, model)
        _force_cpu(monkeypatch)
        generator = ParakeetTextGenerator()
        generator.load()

        with patch("whisperjav.utils.gpu_utils.safe_cuda_cleanup") as cleanup:
            generator.unload()
            cleanup.assert_called_once()
        assert generator.is_loaded is False
        assert generator._model is None
        generator.cleanup()  # idempotent

    def test_native_words_reach_existing_frame_fallback_path(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import (
            DecoupledSubtitlePipeline,
        )
        from whisperjav.modules.subtitle_pipeline.types import TemporalFrame

        frames = [[TemporalFrame(0.0, 1.0), TemporalFrame(2.0, 4.0)]]
        texts = [["こ", "フォールバック"]]
        native = [[[
            {"word": "こ", "start": 0.1, "end": 0.2, "source": "native"},
        ], None]]

        alignments = DecoupledSubtitlePipeline._native_or_frame_fallback(
            frames, texts, native,
        )

        assert alignments[0][0][0]["start"] == pytest.approx(0.1)
        assert alignments[0][0][0]["source"] == "native"
        assert alignments[0][1][0]["start"] == pytest.approx(0.0)
        assert alignments[0][1][0]["end"] == pytest.approx(2.0)
        assert alignments[0][1][0]["source"] == "frame_fallback"

    def test_orchestrator_rejects_generator_marked_malformed_native_timing(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import (
            DecoupledSubtitlePipeline,
        )

        result = SimpleNamespace(
            words=[SimpleNamespace(word="有効", start=0.1, end=0.2)],
            metadata={"native_timestamp_invalid_count": 1},
        )

        assert DecoupledSubtitlePipeline._extract_native_words(result) is None

    def test_orchestrator_preserves_native_ctc_provenance(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import (
            DecoupledSubtitlePipeline,
        )

        result = SimpleNamespace(
            words=[SimpleNamespace(word="有効", start=0.1, end=0.2)],
            metadata={"timing_source": "native_ctc"},
        )

        words = DecoupledSubtitlePipeline._extract_native_words(result)

        assert words[0]["source"] == "native_ctc"

    def test_text_only_generators_keep_existing_branch_b_path(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import (
            DecoupledSubtitlePipeline,
        )
        from whisperjav.modules.subtitle_pipeline.types import TemporalFrame

        assert DecoupledSubtitlePipeline._native_or_frame_fallback(
            [[TemporalFrame(0.0, 1.0)]],
            [["テキストのみ"]],
            [[None]],
        ) is None

    def test_qwen_pipeline_uses_parakeet_without_forced_aligner(self, tmp_path):
        try:
            from whisperjav.pipelines.qwen_pipeline import QwenPipeline
        except ModuleNotFoundError as exc:
            pytest.skip(f"Qwen pipeline dependency is not installed: {exc.name}")

        pipeline = QwenPipeline(
            output_dir=str(tmp_path / "out"),
            temp_dir=str(tmp_path / "tmp"),
            generator_backend="parakeet",
            qwen_framer="full-scene",
            scene_detector="none",
        )

        generator = pipeline._subtitle_pipeline.generator
        assert generator.__class__.__name__ == "ParakeetTextGenerator"
        assert generator._config["model_id"] == (
            "grider-transwithai/parakeet-ctc-1.1b-ja"
        )
        assert pipeline._subtitle_pipeline.aligner is None
        assert pipeline.assembly_cleaner_enabled is False
        assert pipeline.stepdown_enabled is False

    @pytest.mark.parametrize("backend", ["qwen3", "anime-whisper"])
    def test_native_regroup_request_is_ignored_by_other_backends(self, backend, tmp_path):
        try:
            from whisperjav.pipelines.qwen_pipeline import QwenPipeline
        except ModuleNotFoundError as exc:
            pytest.skip(f"Qwen pipeline dependency is not installed: {exc.name}")

        # Bypass component construction; this test only exercises the backend
        # scoping performed by QwenPipeline.__init__.
        with patch.object(QwenPipeline, "_build_subtitle_pipeline", return_value=None):
            pipeline = QwenPipeline(
                output_dir=str(tmp_path / f"{backend}-out"),
                temp_dir=str(tmp_path / f"{backend}-tmp"),
                generator_backend=backend,
                parakeet_regroup=True,
                scene_detector="none",
            )

        assert pipeline.parakeet_regroup_enabled is False
        assert pipeline.parakeet_regroup_config is None

    def test_pass_worker_contains_parakeet_defaults(self):
        source = Path("whisperjav/ensemble/pass_worker.py").read_text(encoding="utf-8")
        assert 'elif _gen_backend == "parakeet":' in source
        assert '"grider-transwithai/parakeet-ctc-1.1b-ja"' in source
        assert 'qwen_pipeline_params["regroup_mode"] = "standard"' in source
