"""Parakeet CTC TextGenerator adapter.

NeMo is deliberately imported only from :meth:`load`.  Installing or
importing WhisperJAV therefore does not require the optional NeMo runtime.

The NeMo ASR API loads Hugging Face checkpoints with
``ASRModel.from_pretrained(model_name=...)`` and local ``.nemo`` files with
``ASRModel.restore_from(restore_path=...)``.  Newer releases accept
``transcribe(..., return_hypotheses=True, timestamps=True)``; the pinned
Japanese checkpoint runtime enables the same native CTC timing through
``model.decoding.compute_timestamps`` and returns it under ``Hypothesis.timestep``.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult, WordTimestamp
from whisperjav.utils.logger import logger


class ParakeetTextGenerator:
    """TextGenerator backed by NVIDIA NeMo Parakeet ASR models."""

    DEFAULT_MODEL_ID = "grider-transwithai/parakeet-ctc-1.1b-ja"
    _TIMESTAMP_LEVELS = {"auto", "char", "token", "word", "segment", "none"}

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        timestamp_level: str = "char",
    ):
        """Store configuration for deferred NeMo model construction.

        Args:
            model_id: Hugging Face model ID or local ``.nemo`` checkpoint.
            device: ``auto``, ``cuda``, ``cuda:N``, ``mps``, or ``cpu``.
            dtype: ``auto``, ``float16``, ``bfloat16``, or ``float32``.
            batch_size: NeMo transcription batch size.
            timestamp_level: Preferred native level. ``char``/``token``/``auto``
                keep the finest available level; ``none`` disables timestamp
                requests and intentionally uses the pipeline fallback.
        """
        if timestamp_level not in self._TIMESTAMP_LEVELS:
            raise ValueError(
                f"Unknown Parakeet timestamp_level '{timestamp_level}'. "
                f"Expected one of {sorted(self._TIMESTAMP_LEVELS)}."
            )
        if batch_size < 1:
            raise ValueError("Parakeet batch_size must be at least 1")

        self._config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "timestamp_level": timestamp_level,
        }
        self._model = None
        self._device: str | None = None
        self._dtype = None
        self._loaded = False
        self._legacy_transcribe_api = False

    @property
    def is_loaded(self) -> bool:
        """Whether the NeMo model is currently loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Device and dtype resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device(device: str) -> str:
        """Resolve a requested device, falling back safely when unavailable."""
        import torch

        requested = (device or "auto").lower()
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if requested == "auto":
            if cuda_available:
                return "cuda:0"
            if mps_available:
                return "mps"
            return "cpu"

        if requested == "cuda":
            if cuda_available:
                return "cuda:0"
            logger.warning("Parakeet requested CUDA but CUDA is unavailable; falling back to CPU")
            return "cpu"

        if requested.startswith("cuda") and not cuda_available:
            logger.warning(
                "Parakeet requested %s but CUDA is unavailable; falling back to CPU",
                device,
            )
            return "cpu"

        if requested == "mps" and not mps_available:
            logger.warning("Parakeet requested MPS but MPS is unavailable; falling back to CPU")
            return "cpu"

        return device

    @staticmethod
    def _detect_dtype(device: str, dtype: str):
        """Resolve a safe inference dtype for the resolved device."""
        import torch

        requested = (dtype or "auto").lower()
        if requested not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                f"Unknown Parakeet dtype '{dtype}'. Expected auto, float16, bfloat16, or float32."
            )

        is_cuda = str(device).startswith("cuda")
        if not is_cuda:
            if requested in {"float16", "bfloat16"}:
                logger.warning(
                    "Parakeet %s is not supported reliably on %s; using float32",
                    requested,
                    device,
                )
            return torch.float32

        if requested == "auto":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if bf16_supported else torch.float16

        if requested == "bfloat16":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if not bf16_supported:
                logger.warning("Parakeet bfloat16 is unavailable on %s; using float16", device)
                return torch.float16

        return getattr(torch, requested)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _is_local_checkpoint(model_id: str) -> bool:
        """Return whether ``model_id`` denotes a local path/checkpoint."""
        path = Path(model_id).expanduser()
        return path.suffix.lower() == ".nemo" or path.exists()

    @classmethod
    def _load_hf_nemo_checkpoint(cls, nemo_asr, model_id: str, device: str):
        """Load the Japanese model-card checkpoint from the Hugging Face repo."""
        if model_id != cls.DEFAULT_MODEL_ID:
            return None

        try:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id=model_id,
                filename="parakeet-ja.nemo",
            )
        except Exception as exc:
            # Keep custom NeMo repositories on the regular from_pretrained path.
            # The Japanese model card publishes a .nemo archive rather than the
            # model_config.yaml files expected by NeMo's repo loader.
            logger.debug(
                "[ParakeetTextGenerator] Hugging Face .nemo checkpoint lookup "
                "unavailable for %s: %s",
                model_id,
                exc,
            )
            return None

        logger.info(
            "[ParakeetTextGenerator] Restoring Hugging Face checkpoint: %s",
            checkpoint,
        )
        return nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoint),
            map_location=device,
        )

    def load(self) -> None:
        """Load the selected Parakeet checkpoint using the current NeMo API."""
        if self._loaded:
            logger.debug("[ParakeetTextGenerator] Already loaded")
            return

        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError as exc:
            raise ImportError(
                "Parakeet support requires the optional NeMo ASR dependency. "
                "Install it with `pip install 'whisperjav[parakeet]'`."
            ) from exc

        cfg = self._config
        device = self._detect_device(cfg["device"])
        dtype = self._detect_dtype(device, cfg["dtype"])
        model_id = str(cfg["model_id"])

        logger.info(
            "[ParakeetTextGenerator] Loading model=%s device=%s dtype=%s",
            model_id,
            device,
            dtype,
        )

        try:
            if self._is_local_checkpoint(model_id):
                self._model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=str(Path(model_id).expanduser()),
                    map_location=device,
                )
            else:
                self._model = self._load_hf_nemo_checkpoint(
                    nemo_asr,
                    model_id,
                    device,
                )
                if self._model is None:
                    self._model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=model_id,
                        map_location=device,
                    )

            self._model.to(device)
            if dtype != torch.float32:
                self._model.to(dtype)
            if hasattr(self._model, "eval"):
                self._model.eval()

            # NeMo 2.0.0rc0 (the runtime pinned for the Japanese checkpoint)
            # configures CTC timestamps on the decoder rather than accepting a
            # `timestamps=True` transcribe keyword.  Newer NeMo versions keep
            # accepting the keyword, so this is deliberately best-effort.
            self._configure_native_timestamps()
            self._legacy_transcribe_api = self._uses_legacy_transcribe_api()
            if self._legacy_transcribe_api:
                logger.debug("[ParakeetTextGenerator] Using legacy NeMo hypothesis API")

            self._device = device
            self._dtype = dtype
            self._loaded = True
        except Exception:
            self._model = None
            self._device = None
            self._dtype = None
            self._loaded = False
            raise

        logger.info("[ParakeetTextGenerator] Model loaded")

    def _uses_legacy_transcribe_api(self) -> bool:
        """Detect NeMo releases whose transcribe method lacks timestamp kwargs."""
        if self._model is None:
            return False
        try:
            parameters = tuple(inspect.signature(self._model.transcribe).parameters.values())
        except (TypeError, ValueError):
            # Unknown/custom callables retain the modern attempt-and-fallback path.
            return False

        parameter_names = {parameter.name for parameter in parameters}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        return "timestamps" not in parameter_names and not accepts_kwargs

    def _configure_native_timestamps(self) -> bool:
        """Enable legacy NeMo CTC timestamp generation when available."""
        if self._model is None or self._config["timestamp_level"] == "none":
            return False

        decoding = getattr(self._model, "decoding", None)
        if decoding is None:
            return False

        try:
            decoding_cfg = getattr(decoding, "cfg", None)
            if decoding_cfg is not None:
                # `all` exposes both char and word levels; extraction still
                # follows the caller's preferred level and falls back safely.
                decoding_cfg["ctc_timestamp_type"] = "all"
            if not hasattr(decoding, "compute_timestamps"):
                return False
            decoding.compute_timestamps = True
            logger.debug("[ParakeetTextGenerator] Enabled legacy NeMo CTC timestamps")
            return True
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug(
                "[ParakeetTextGenerator] Legacy NeMo timestamp setup unavailable: %s",
                exc,
            )
            return False

    def unload(self) -> None:
        """Release the NeMo model and clear CUDA allocations."""
        if self._model is None and not self._loaded:
            return

        model = self._model
        self._model = None
        if model is not None:
            del model

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup

        safe_cuda_cleanup()
        self._device = None
        self._dtype = None
        self._loaded = False
        logger.info("[ParakeetTextGenerator] Model unloaded")

    # ------------------------------------------------------------------
    # Inference and timestamp conversion
    # ------------------------------------------------------------------

    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe one audio file and preserve native CTC timing when present."""
        results = self.generate_batch(
            [audio_path],
            language=language,
            contexts=[context] if context else None,
            **kwargs,
        )
        if results:
            return results[0]
        return TranscriptionResult(
            text="",
            language=language or "ja",
            metadata={
                "generator": "parakeet",
                "audio_path": str(audio_path),
                "timestamp_status": "unavailable",
            },
        )

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """Transcribe a batch through NeMo's native batched ``transcribe`` API."""
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "ParakeetTextGenerator.generate_batch() called before load(). Call load() first."
            )

        if contexts and any(contexts):
            logger.debug("[ParakeetTextGenerator] context is ignored; CTC decoding has no prompt input")

        paths = list(audio_paths)
        if not paths:
            return []

        timestamp_level = self._config["timestamp_level"]
        timestamp_enabled = timestamp_level != "none"
        transcribe_kwargs: dict[str, Any] = {
            "audio": [str(path) for path in paths],
            "batch_size": self._config["batch_size"],
        }
        if timestamp_enabled:
            # NeMo documents that timestamps require Hypothesis objects.
            transcribe_kwargs["return_hypotheses"] = True
            if not self._legacy_transcribe_api:
                transcribe_kwargs["timestamps"] = True

        timestamp_request_fallback = self._legacy_transcribe_api
        try:
            raw_outputs = self._model.transcribe(**transcribe_kwargs)
        except (NotImplementedError, TypeError) as exc:
            if not timestamp_enabled:
                raise
            # NeMo 2.0.0rc0 rejects the newer `timestamps=True` keyword but
            # returns native timing when the decoder was configured above.
            logger.warning(
                "[ParakeetTextGenerator] Timestamp keyword unavailable (%s); "
                "retrying NeMo hypothesis transcription",
                exc,
            )
            timestamp_request_fallback = True
            self._legacy_transcribe_api = True
            self._configure_native_timestamps()
            try:
                raw_outputs = self._model.transcribe(
                    audio=[str(path) for path in paths],
                    batch_size=self._config["batch_size"],
                    return_hypotheses=True,
                )
            except (NotImplementedError, TypeError) as legacy_exc:
                # Some older/custom checkpoints reject hypotheses entirely.
                # Preserve the text path and let the orchestrator apply its
                # explicit frame-boundary timing fallback.
                logger.warning(
                    "[ParakeetTextGenerator] Native hypotheses unavailable (%s); "
                    "retrying text-only transcription",
                    legacy_exc,
                )
                raw_outputs = self._model.transcribe(
                    audio=[str(path) for path in paths],
                    batch_size=self._config["batch_size"],
                )
        outputs = self._normalise_outputs(raw_outputs)
        results: list[TranscriptionResult] = []

        for index, path in enumerate(paths):
            output = outputs[index] if index < len(outputs) else None
            text = self._extract_text(output)
            words, resolved_level, invalid_timestamp_count = self._extract_timestamps(output)
            if not timestamp_enabled:
                timestamp_status = "disabled"
            elif words and invalid_timestamp_count == 0:
                timestamp_status = "native"
            elif words:
                timestamp_status = "malformed"
            else:
                timestamp_status = "unavailable"

            results.append(
                TranscriptionResult(
                    text=text,
                    language=language or "ja",
                    words=words,
                    metadata={
                        "generator": "parakeet",
                        "model_id": self._config["model_id"],
                        "audio_path": str(path),
                        "timestamp_status": timestamp_status,
                        "timestamp_level": resolved_level,
                        "native_timestamp_count": len(words),
                        "native_timestamp_invalid_count": invalid_timestamp_count,
                        "timestamp_request_fallback": timestamp_request_fallback,
                    },
                )
            )

        return results

    @staticmethod
    def _normalise_outputs(raw_outputs: Any) -> list[Any]:
        """Normalize NeMo's list-like output without assuming Hypothesis types."""
        if raw_outputs is None:
            return []
        if isinstance(raw_outputs, (str, bytes, dict)):
            return [raw_outputs]
        try:
            return list(raw_outputs)
        except TypeError:
            return [raw_outputs]

    @staticmethod
    def _extract_text(output: Any) -> str:
        """Extract text from a NeMo Hypothesis, dict, string, or malformed result."""
        if output is None:
            return ""
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            value = output.get("text", output.get("pred_text", output.get("transcript", "")))
        else:
            value = getattr(output, "text", None)
            if value is None:
                value = getattr(output, "pred_text", "")
        return str(value).strip() if value is not None else ""

    def _extract_timestamps(
        self,
        output: Any,
    ) -> tuple[list[WordTimestamp], str | None, int]:
        """Convert NeMo timestamp dictionaries to shared ``WordTimestamp`` values."""
        if output is None or self._config["timestamp_level"] == "none":
            return [], None, 0

        if isinstance(output, dict):
            timestamp_data = output.get(
                "timestamp",
                output.get("timestep", output.get("timestamps")),
            )
        else:
            timestamp_data = getattr(output, "timestamp", None)
            if timestamp_data is None:
                timestamp_data = getattr(output, "timestep", None)
            if timestamp_data is None:
                timestamp_data = getattr(output, "timestamps", None)

        # Do not use truthiness here: NeMo may expose a tensor-like timestamp
        # container whose boolean value is ambiguous.
        if timestamp_data is None:
            return [], None, 0

        preferred = self._preferred_timestamp_levels()
        if isinstance(timestamp_data, dict):
            candidates = [(level, timestamp_data.get(level)) for level in preferred]
        else:
            candidates = [(preferred[0], timestamp_data)]

        invalid_timestamp_count = 0
        partial_words: list[WordTimestamp] = []
        partial_level: str | None = None
        for level, items in candidates:
            if items is None:
                continue
            words, invalid_count = self._parse_timestamp_items(items, level)
            invalid_timestamp_count += invalid_count
            if words and invalid_count == 0:
                # A fully valid lower-priority level is preferable to mixing
                # malformed records from a finer level with usable timing.
                return words, level, 0
            if words and not partial_words:
                partial_words = words
                partial_level = level

        if partial_words:
            # Preserve valid records for generator-level observability, but
            # mark the result malformed so the orchestrator uses the complete
            # parent-frame fallback instead of mixing partial native timing.
            return partial_words, partial_level, invalid_timestamp_count
        return [], None, invalid_timestamp_count

    def _preferred_timestamp_levels(self) -> list[str]:
        level = self._config["timestamp_level"]
        if level in {"auto", "char"}:
            return ["char", "token", "word", "segment"]
        if level == "token":
            return ["token", "char", "word", "segment"]
        if level == "word":
            return ["word", "token", "char", "segment"]
        return ["segment"]

    def _parse_timestamp_items(
        self,
        items: Any,
        level: str,
    ) -> tuple[list[WordTimestamp], int]:
        """Parse timestamp records, rejecting malformed records explicitly."""
        if isinstance(items, dict):
            items = [items]
        try:
            iterable = list(items)
        except TypeError:
            return [], 1

        words: list[WordTimestamp] = []
        invalid_count = 0
        for item in iterable:
            if isinstance(item, dict):
                value = item.get(
                    "char",
                    item.get("word", item.get("token", item.get("text", item.get("segment")))),
                )
                start = item.get("start")
                end = item.get("end")
                if start is None or end is None:
                    start = item.get("start_offset")
                    end = item.get("end_offset")
                    if start is not None and end is not None:
                        stride = self._timestamp_stride_seconds()
                        if stride is None:
                            invalid_count += 1
                            continue
                        start = self._as_float(start)
                        end = self._as_float(end)
                        start = start * stride if start is not None else None
                        end = end * stride if end is not None else None
            else:
                value = getattr(item, "char", None)
                if value is None:
                    value = getattr(item, "word", None)
                if value is None:
                    value = getattr(item, "text", None)
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)

            start_value = self._as_float(start)
            end_value = self._as_float(end)
            if value is None or start_value is None or end_value is None:
                invalid_count += 1
                continue
            if not math.isfinite(start_value) or not math.isfinite(end_value):
                invalid_count += 1
                continue
            if start_value < 0.0 or end_value <= start_value:
                invalid_count += 1
                continue

            token = str(value).strip()
            if token.startswith("▁"):
                token = token[1:]
            if token.startswith("##"):
                token = token[2:]
            if not token:
                invalid_count += 1
                continue

            words.append(WordTimestamp(word=token, start=start_value, end=end_value))

        return words, invalid_count

    def _timestamp_stride_seconds(self) -> float | None:
        """Resolve NeMo's encoder-step stride for offset-only timestamps."""
        try:
            window_stride = self._model.cfg.preprocessor.window_stride
            stride = 8.0 * float(window_stride)
            return stride if stride > 0 else None
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """Convert scalar tensors/numbers to float, rejecting non-scalars."""
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
