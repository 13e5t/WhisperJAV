"""
Decoupled Subtitle Pipeline orchestrator.

Composes TemporalFramer, TextGenerator, TextCleaner, and TextAligner
protocol implementations into a working pipeline.  Handles the 9-step
processing flow defined in ADR-006 Section 9.2:

    1. Temporal framing (per scene)
    2. Audio slicing (per frame → temp WAV)
    3. Text generation (batch, with VRAM lifecycle)
    4. Text cleaning (batch, lightweight)
    5-7. Alignment (batch, with VRAM lifecycle)
    8. Word merging (frame-relative → scene-relative)
    9. Sentinel + Reconstruction + Hardening (per scene)

VRAM swap pattern:
    generator.load() → generate → generator.unload()
    → safe_cuda_cleanup()
    → aligner.load() → align → aligner.unload()
    → safe_cuda_cleanup()
"""

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.hardening import harden_scene_result
from whisperjav.modules.subtitle_pipeline.protocols import (
    TemporalFramer,
    TextAligner,
    TextCleaner,
    TextGenerator,
)
from whisperjav.modules.subtitle_pipeline.reconstruction import (
    REGROUP_VAD_ONLY,
    reconstruct_frame_native,
    reconstruct_from_words,
    resolve_regroup,
    split_frame_to_words,
)
from whisperjav.modules.subtitle_pipeline.regroupers.japanese import (
    JapaneseNativeRegrouper,
)
from whisperjav.modules.subtitle_pipeline.types import (
    HardeningConfig,
    RegroupMode,
    SceneDiagnostics,
    StepDownConfig,
    TemporalFrame,
    TimestampMode,
)
from whisperjav.utils.logger import logger

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


class DecoupledSubtitlePipeline:
    """
    Model-agnostic subtitle generation pipeline.

    Composes protocol-based components into a working pipeline with
    explicit VRAM lifecycle management, alignment sentinel integration,
    and per-scene diagnostics.
    """

    def __init__(
        self,
        framer: TemporalFramer,
        generator: TextGenerator,
        cleaner: TextCleaner,
        aligner: Optional[TextAligner],
        hardening_config: HardeningConfig,
        artifacts_dir: Optional[Path] = None,
        language: str = "ja",
        context: str = "",
        stepdown_config: Optional[StepDownConfig] = None,
        native_regrouper: Optional[JapaneseNativeRegrouper] = None,
    ):
        """
        Initialize the pipeline with protocol components.

        Args:
            framer: Produces temporal frames from scene audio.
            generator: Produces text from audio.
            cleaner: Cleans raw transcription text.
            aligner: Aligns text to audio for word-level timestamps.
                     None for aligner-free workflows.
            hardening_config: Timestamp resolution and boundary config.
            artifacts_dir: Directory for debug artifacts (None = no artifacts).
            language: Language code for generation and alignment.
            context: User-provided context for ASR (cast names, terminology).
            stepdown_config: Optional step-down retry config. When enabled
                and alignment collapses on a scene, the orchestrator re-frames
                with tighter grouping and retries generation + alignment.
            native_regrouper: Optional Parakeet-only native timestamp
                regrouper. Other generators leave this as ``None``.
        """
        self.framer = framer
        self.generator = generator
        self.cleaner = cleaner
        self.aligner = aligner
        self.hardening_config = hardening_config
        self.artifacts_dir = artifacts_dir
        self.language = language
        self.context = context
        self.stepdown_config = stepdown_config
        self.native_regrouper = native_regrouper

        # Sentinel stats accumulated across all scenes
        self.sentinel_stats: dict[str, Any] = {
            "total_scenes": 0,
            "collapsed_scenes": 0,
            "recovered_scenes": 0,
            "recovery_strategies": {"vad_guided": 0, "proportional": 0},
        }

        # Temp files for cleanup
        self._temp_files: list[Path] = []

    def process_scenes(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Process all scenes through the decoupled pipeline.

        When step-down retry is enabled, collapsed scenes are automatically
        retried with tighter temporal framing (Pass 2).

        Args:
            scene_audio_paths: Paths to per-scene audio files (WAV, 16kHz).
                Used for ASR generation and alignment.
            scene_durations: Duration of each scene in seconds.
            scene_speech_regions: Optional per-scene VAD speech regions
                (from Phase 4 or VadGroupedFramer metadata).
            vad_audio_paths: Optional per-scene audio files for VAD/framing
                only (dual-track ``--enhance-for-vad`` mode).  When provided,
                the framer uses these (enhanced) files for temporal framing
                while ASR slicing reads from *scene_audio_paths* (original).
                When None (default), framing and slicing use the same files.

        Returns:
            List of (WhisperResult_or_None, diagnostics_dict) per scene.
        """
        n_scenes = len(scene_audio_paths)
        if n_scenes != len(scene_durations):
            raise ValueError(f"scene_audio_paths ({n_scenes}) and scene_durations ({len(scene_durations)}) must match")
        if vad_audio_paths is not None and len(vad_audio_paths) != n_scenes:
            raise ValueError(f"vad_audio_paths ({len(vad_audio_paths)}) and scene_audio_paths ({n_scenes}) must match")

        logger.info(
            "[DecoupledPipeline] Processing %d scenes (aligner=%s, step-down=%s%s)",
            n_scenes,
            "yes" if self.aligner else "none",
            "enabled" if (self.stepdown_config and self.stepdown_config.enabled) else "disabled",
            ", dual-track=yes" if vad_audio_paths else "",
        )

        # --- Pass 1: Normal processing ---
        results = self._run_pass(
            scene_audio_paths, scene_durations, scene_speech_regions,
            vad_audio_paths=vad_audio_paths,
        )

        # --- Identify collapsed scenes ---
        collapsed_indices = [
            i for i, (_result, diag) in enumerate(results)
            if self._is_collapsed(diag)
        ]

        if not collapsed_indices:
            return results

        # --- Step-down decision ---
        can_reframe = hasattr(self.framer, "reframe")
        stepdown_enabled = (
            self.stepdown_config is not None
            and self.stepdown_config.enabled
            and can_reframe
        )

        if stepdown_enabled:
            logger.info(
                "[DecoupledPipeline] Step-down: %d/%d scenes collapsed, "
                "retrying with tighter framing (%.1fs max group)",
                len(collapsed_indices), n_scenes,
                self.stepdown_config.fallback_max_group_s,
            )
            retry_results = self._run_stepdown_pass(
                collapsed_indices, scene_audio_paths,
                scene_durations, scene_speech_regions,
                vad_audio_paths=vad_audio_paths,
            )
            # Replace Pass 1 results for retried scenes with Pass 2 results
            for idx, retry_result in zip(collapsed_indices, retry_results):
                _pass1_diag = results[idx][1]
                _pass2_result, _pass2_diag = retry_result
                # Annotate step-down outcome
                improved = not self._is_collapsed(_pass2_diag)
                _pass2_diag["stepdown"] = {
                    "attempted": True,
                    "enabled": True,
                    "improved": improved,
                    "pass1_sentinel": _pass1_diag.get("sentinel_status", "N/A"),
                    "pass2_sentinel": _pass2_diag.get("sentinel_status", "N/A"),
                    "fallback_max_group_s": self.stepdown_config.fallback_max_group_s,
                }
                if improved:
                    results[idx] = retry_result
                    logger.info(
                        "[DecoupledPipeline] Step-down: Scene %d improved (was COLLAPSED → %s)",
                        idx, _pass2_diag.get("sentinel_status", "?"),
                    )
                else:
                    # Pass 2 also collapsed — keep Pass 2 result anyway
                    # (proportional recovery was already applied in _step9)
                    results[idx] = retry_result
                    logger.warning(
                        "[DecoupledPipeline] Step-down: Scene %d still collapsed after retry",
                        idx,
                    )
        else:
            # Step-down disabled or not available
            if self.stepdown_config and not self.stepdown_config.enabled:
                reason = "disabled by user configuration"
            elif not can_reframe:
                reason = "framer does not support reframing"
            else:
                reason = "not configured"
            logger.warning(
                "[DecoupledPipeline] %d/%d scenes collapsed but step-down retry is %s. "
                "Proportional recovery applied.",
                len(collapsed_indices), n_scenes, reason,
            )
            # Annotate diagnostics for collapsed scenes
            for idx in collapsed_indices:
                results[idx][1]["stepdown"] = {
                    "attempted": False,
                    "enabled": False if (self.stepdown_config and not self.stepdown_config.enabled) else None,
                    "improved": False,
                }

        return results

    # -----------------------------------------------------------------------
    # Pass execution (shared by Pass 1 and step-down Pass 2)
    # -----------------------------------------------------------------------

    def _run_pass(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
        framer_override_max_group: Optional[float] = None,
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Execute a single pass of the pipeline (framing → generation → alignment → hardening).

        Args:
            scene_audio_paths: Per-scene audio file paths (used for ASR).
            scene_durations: Per-scene durations.
            scene_speech_regions: Optional per-scene VAD speech regions.
            framer_override_max_group: When set, calls framer.reframe()
                with this max group duration instead of framer.frame().
            vad_audio_paths: Optional per-scene audio for framing/VAD
                (dual-track mode).  When None, framing uses scene_audio_paths.
        """
        try:
            scene_frames, frame_audio_paths, frame_speech_regions = self._step1_frame_and_slice(
                scene_audio_paths, scene_durations,
                framer_override_max_group=framer_override_max_group,
                vad_audio_paths=vad_audio_paths,
            )
            scene_texts, scene_native_words = self._step2_4_generate_and_clean(
                scene_frames, frame_audio_paths, scene_durations,
            )
            scene_alignments = self._step5_7_align(
                scene_frames,
                frame_audio_paths,
                scene_texts,
                scene_durations,
                native_frame_words=scene_native_words,
            )
            results = self._step9_reconstruct_and_harden(
                scene_frames, scene_texts, scene_alignments,
                scene_audio_paths, scene_durations,
                frame_speech_regions, scene_speech_regions,
            )
        finally:
            self._cleanup_temp_files()
        return results

    def _run_stepdown_pass(
        self,
        collapsed_indices: list[int],
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Re-process collapsed scenes with tighter framing (step-down retry)."""
        retry_audio_paths = [scene_audio_paths[i] for i in collapsed_indices]
        retry_durations = [scene_durations[i] for i in collapsed_indices]
        retry_speech_regions = (
            [scene_speech_regions[i] for i in collapsed_indices]
            if scene_speech_regions else None
        )
        retry_vad_paths = (
            [vad_audio_paths[i] for i in collapsed_indices]
            if vad_audio_paths else None
        )
        return self._run_pass(
            retry_audio_paths, retry_durations, retry_speech_regions,
            framer_override_max_group=self.stepdown_config.fallback_max_group_s,
            vad_audio_paths=retry_vad_paths,
        )

    @staticmethod
    def _is_collapsed(diag: dict[str, Any]) -> bool:
        """Check if a scene's diagnostics indicate alignment collapse."""
        return diag.get("sentinel_status") == "COLLAPSED"

    # -----------------------------------------------------------------------
    # Step 1: Temporal framing + audio slicing
    # -----------------------------------------------------------------------

    def _step1_frame_and_slice(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        framer_override_max_group: Optional[float] = None,
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> tuple[
        list[list[TemporalFrame]],
        list[list[Path]],
        list[Optional[list[list[tuple[float, float]]]]],
    ]:
        """
        Frame each scene and slice audio per frame into temp WAV files.

        Args:
            framer_override_max_group: When set and the framer supports
                ``reframe()``, uses tighter grouping (step-down retry).
            vad_audio_paths: Optional per-scene audio for framing/VAD
                (dual-track ``--enhance-for-vad`` mode).  When provided,
                the framer runs on these (enhanced) files while audio
                slicing reads from *scene_audio_paths* (original quality).

        Returns:
            scene_frames: Per-scene list of TemporalFrame objects.
            frame_audio_paths: Per-scene, per-frame temp WAV paths.
            frame_speech_regions: Per-scene speech regions from framer metadata.
        """
        n_scenes = len(scene_audio_paths)
        dual_track = vad_audio_paths is not None
        logger.info(
            "[DecoupledPipeline] Step 1: Framing %d scenes%s%s", n_scenes,
            f" (reframe override={framer_override_max_group}s)" if framer_override_max_group else "",
            " (dual-track: enhanced→framer, original→ASR)" if dual_track else "",
        )

        scene_frames: list[list[TemporalFrame]] = []
        frame_audio_paths: list[list[Path]] = []
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]] = []

        for scene_idx, audio_path in enumerate(scene_audio_paths):
            # Dual-track: framer runs on enhanced audio, slicing on original
            framing_path = vad_audio_paths[scene_idx] if dual_track else audio_path
            framing_audio, sr = self._load_audio(framing_path)

            # Run framer (or reframe for step-down)
            if framer_override_max_group is not None and hasattr(self.framer, "reframe"):
                framing_result = self.framer.reframe(
                    framing_audio, sr, max_group_duration_s=framer_override_max_group,
                )
            else:
                framing_result = self.framer.frame(framing_audio, sr)
            frames = framing_result.frames
            scene_frames.append(frames)

            # Extract speech regions from framer metadata (VadGroupedFramer provides these)
            regions = framing_result.metadata.get("speech_regions")
            frame_speech_regions.append(regions)

            # Load ASR audio (only needed if dual-track and we have frames to slice)
            if dual_track and not (len(frames) == 1 and frames[0].start == 0.0):
                asr_audio, sr = self._load_audio(audio_path)
            else:
                asr_audio = framing_audio  # same source, no extra load

            # Slice audio per frame → temp WAV (always from ASR audio)
            frame_paths = []
            for frame_idx, frame in enumerate(frames):
                if len(frames) == 1 and frame.start == 0.0:
                    # Full-scene frame — use original ASR audio path (no slicing needed)
                    frame_paths.append(audio_path)
                else:
                    # Slice and write temp WAV from ASR audio
                    start_sample = int(frame.start * sr)
                    end_sample = int(frame.end * sr)
                    frame_audio = asr_audio[start_sample:end_sample]
                    temp_path = self._write_temp_wav(frame_audio, sr, scene_idx, frame_idx)
                    frame_paths.append(temp_path)

            frame_audio_paths.append(frame_paths)

            logger.debug(
                "[DecoupledPipeline] Scene %d: %d frames (%.1fs)",
                scene_idx,
                len(frames),
                scene_durations[scene_idx],
            )

        total_frames = sum(len(f) for f in scene_frames)
        logger.info(
            "[DecoupledPipeline] Step 1: Complete — %d scenes, %d total frames",
            n_scenes, total_frames,
        )

        return scene_frames, frame_audio_paths, frame_speech_regions

    # -----------------------------------------------------------------------
    # Steps 2-4: Text generation + cleaning
    # -----------------------------------------------------------------------

    def _step2_4_generate_and_clean(
        self,
        scene_frames: list[list[TemporalFrame]],
        frame_audio_paths: list[list[Path]],
        scene_durations: list[float],
    ) -> tuple[list[list[str]], list[list[Optional[list[dict[str, Any]]]]]]:
        """
        Generate text for each frame, then clean.

        VRAM lifecycle: generator.load() → generate all → generator.unload()

        Returns:
            A tuple of per-scene, per-frame cleaned text strings and optional
            native timestamp word dictionaries.  Native timestamps are kept
            in frame-relative coordinates and are offset in Step 9 just like
            TextAligner output.  ``None`` means the generator did not return
            usable native timestamps for that frame.
        """
        import time as _time

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Steps 2-4: Generating + cleaning text for %d scenes",
            n_scenes,
        )
        step24_start = _time.monotonic()

        # Collect frames that need generation (no pre-existing text)
        needs_generation = False
        for frames in scene_frames:
            for frame in frames:
                if frame.text is None:
                    needs_generation = True
                    break
            if needs_generation:
                break

        # Phase 1: Generation
        scene_raw_texts: list[list[str]] = []
        scene_native_words: list[list[Optional[list[dict[str, Any]]]]] = []

        if needs_generation:
            self.generator.load()

        try:
            for scene_idx in range(n_scenes):
                frames = scene_frames[scene_idx]
                audio_paths = frame_audio_paths[scene_idx]
                raw_texts = []
                raw_native_words: list[Optional[list[dict[str, Any]]]] = []

                logger.info(
                    "[DecoupledPipeline] Generating scene %d/%d (%.1fs audio)...",
                    scene_idx + 1, n_scenes, scene_durations[scene_idx],
                )

                # Separate framer-provided and needs-generation frames
                gen_indices = []
                gen_audio_paths = []
                for frame_idx, frame in enumerate(frames):
                    if frame.text is not None:
                        raw_texts.append(frame.text)
                        raw_native_words.append(None)
                    else:
                        raw_texts.append(None)  # placeholder
                        raw_native_words.append(None)
                        gen_indices.append(frame_idx)
                        gen_audio_paths.append(audio_paths[frame_idx])

                # Batch generate for frames without text
                if gen_indices:
                    try:
                        gen_contexts = [self.context] * len(gen_audio_paths) if self.context else None
                        gen_results = self.generator.generate_batch(
                            audio_paths=gen_audio_paths,
                            language=self.language,
                            contexts=gen_contexts,
                            audio_durations=[frames[i].duration for i in gen_indices],
                        )
                        for i, gen_idx in enumerate(gen_indices):
                            result = gen_results[i] if i < len(gen_results) else None
                            raw_texts[gen_idx] = getattr(result, "text", "") if result is not None else ""
                            raw_native_words[gen_idx] = self._extract_native_words(result)
                    except Exception:
                        # Batch failed — fall back to per-frame
                        logger.warning(
                            "[DecoupledPipeline] Batch generation failed for scene %d, falling back to per-frame",
                            scene_idx,
                            exc_info=True,
                        )
                        for i, gen_idx in enumerate(gen_indices):
                            try:
                                result = self.generator.generate(
                                    audio_path=gen_audio_paths[i],
                                    language=self.language,
                                    context=self.context if self.context else None,
                                )
                                raw_texts[gen_idx] = getattr(result, "text", "")
                                raw_native_words[gen_idx] = self._extract_native_words(result)
                            except Exception:
                                logger.error(
                                    "[DecoupledPipeline] Generation failed for scene %d frame %d",
                                    scene_idx,
                                    gen_idx,
                                    exc_info=True,
                                )
                                raw_texts[gen_idx] = ""
                                raw_native_words[gen_idx] = None

                # Replace any remaining None with empty string
                raw_texts = [t if t is not None else "" for t in raw_texts]
                scene_raw_texts.append(raw_texts)
                scene_native_words.append(raw_native_words)

                # Per-scene generation result
                scene_chars = sum(len(t) for t in raw_texts)
                if scene_chars == 0:
                    logger.info(
                        "[DecoupledPipeline]   Scene %d/%d: empty (no text generated)",
                        scene_idx + 1, n_scenes,
                    )
                else:
                    logger.debug(
                        "[DecoupledPipeline]   Scene %d/%d: %d chars",
                        scene_idx + 1, n_scenes, scene_chars,
                    )

                # Save raw text artifacts
                if self.artifacts_dir:
                    self._save_artifact(scene_idx, "raw", "\n---\n".join(raw_texts))

        finally:
            if needs_generation:
                self.generator.unload()
                self._safe_cuda_cleanup()

        # Phase 2: Cleaning
        logger.info("[DecoupledPipeline] Cleaning %d scenes", n_scenes)
        scene_texts: list[list[str]] = []
        for scene_idx, raw_texts in enumerate(scene_raw_texts):
            clean_texts = self.cleaner.clean_batch(raw_texts)
            scene_texts.append(clean_texts)

            # Save clean text artifacts
            if self.artifacts_dir:
                self._save_artifact(scene_idx, "clean", "\n---\n".join(clean_texts))

        # Step summary
        total_raw_chars = sum(len(t) for texts in scene_raw_texts for t in texts)
        total_clean_chars = sum(len(t) for texts in scene_texts for t in texts)
        n_empty = sum(1 for texts in scene_texts if all(not t.strip() for t in texts))
        elapsed = _time.monotonic() - step24_start
        logger.info(
            "[DecoupledPipeline] Steps 2-4: Complete — %d scenes, %d chars (%d removed by cleaning), %d empty (%.1fs)",
            n_scenes, total_clean_chars, total_raw_chars - total_clean_chars, n_empty, elapsed,
        )

        return scene_texts, scene_native_words

    @staticmethod
    def _extract_native_words(result: Any) -> Optional[list[dict[str, Any]]]:
        """Normalize optional generator-native timing to the shared word shape.

        TextGenerator implementations return ``WordTimestamp`` instances via
        ``TranscriptionResult.words``.  The orchestrator converts them to the
        same small dictionaries used by TextAligner output so the existing
        reconstruction and regrouping code remains the single downstream
        path.  Invalid records are dropped; an all-invalid result becomes
        ``None`` and receives the documented frame-boundary fallback.
        """
        if result is None:
            return None

        metadata = getattr(result, "metadata", {}) or {}
        invalid_count = metadata.get("native_timestamp_invalid_count", 0)
        try:
            invalid_count = int(invalid_count)
        except (TypeError, ValueError):
            invalid_count = 0
        if invalid_count > 0:
            logger.warning(
                "[DecoupledPipeline] Native timestamp result contains %d malformed "
                "record(s); using the parent frame fallback",
                invalid_count,
            )
            return None

        native_words = getattr(result, "words", None)
        if native_words is None:
            native_words = metadata.get("native_timestamps")
        if not native_words:
            return None

        normalized: list[dict[str, Any]] = []
        for native in native_words:
            if isinstance(native, dict):
                token = native.get("word", native.get("text", native.get("char", "")))
                start = native.get("start")
                end = native.get("end")
            else:
                token = getattr(native, "word", getattr(native, "text", ""))
                start = getattr(native, "start", None)
                end = getattr(native, "end", None)

            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue

            token = str(token).strip() if token is not None else ""
            if (
                not token
                or not math.isfinite(start_value)
                or not math.isfinite(end_value)
                or start_value < 0.0
                or end_value <= start_value
            ):
                continue

            normalized.append({
                "word": token,
                "start": start_value,
                "end": end_value,
                "source": "native",
            })

        return normalized or None

    # -----------------------------------------------------------------------
    # Steps 5-7: Alignment
    # -----------------------------------------------------------------------

    def _step5_7_align(
        self,
        scene_frames: list[list[TemporalFrame]],
        frame_audio_paths: list[list[Path]],
        scene_texts: list[list[str]],
        scene_durations: list[float],
        native_frame_words: Optional[list[list[Optional[list[dict[str, Any]]]]]] = None,
    ) -> Optional[list[list[list[dict[str, Any]]]]]:
        """
        Align text to audio for word-level timestamps.

        VRAM lifecycle: aligner.load() → align all → aligner.unload()

        Returns:
            None if no aligner, otherwise:
            scene_alignments[scene_idx][frame_idx] = list of word dicts
            Each word dict: {'word': str, 'start': float, 'end': float}
        """
        if self.aligner is None:
            return self._native_or_frame_fallback(
                scene_frames,
                scene_texts,
                native_frame_words,
            )

        import time as _time

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Steps 5-7: Aligning %d scenes", n_scenes,
        )
        step57_start = _time.monotonic()
        scene_alignments: list[list[list[dict[str, Any]]]] = []

        self.aligner.load()
        try:
            for scene_idx in range(n_scenes):
                frames = scene_frames[scene_idx]
                audio_paths = frame_audio_paths[scene_idx]
                texts = scene_texts[scene_idx]

                # Separate frames that need alignment from empty ones
                batch_indices: list[int] = []
                batch_audio_paths: list[Path] = []
                batch_texts: list[str] = []
                batch_durations: list[float] = []

                for frame_idx, (frame, audio_path, text) in enumerate(zip(frames, audio_paths, texts)):
                    if text.strip():
                        batch_indices.append(frame_idx)
                        batch_audio_paths.append(audio_path)
                        batch_texts.append(text)
                        batch_durations.append(frame.duration)

                scene_chars = sum(len(t) for t in texts if t.strip())
                logger.info(
                    "[DecoupledPipeline] Aligning scene %d/%d (%.1fs audio, %d chars)...",
                    scene_idx + 1, n_scenes, scene_durations[scene_idx], scene_chars,
                )

                # Initialize all frames as empty
                frame_alignments: list[list[dict[str, Any]]] = [[] for _ in frames]

                if batch_indices:
                    try:
                        # Batch align all non-empty frames for this scene
                        batch_results = self.aligner.align_batch(
                            audio_paths=batch_audio_paths,
                            texts=batch_texts,
                            language=self.language,
                            audio_durations=batch_durations,
                        )
                        # Unpack results back to per-frame positions
                        for i, frame_idx in enumerate(batch_indices):
                            word_dicts = [
                                {
                                    "word": w.word,
                                    "start": w.start,
                                    "end": w.end,
                                }
                                for w in batch_results[i].words
                            ]
                            frame_alignments[frame_idx] = word_dicts
                    except Exception:
                        # Batch failed — fall back to per-frame alignment
                        logger.warning(
                            "[DecoupledPipeline] Batch alignment failed for scene %d, falling back to per-frame",
                            scene_idx,
                            exc_info=True,
                        )
                        for i, frame_idx in enumerate(batch_indices):
                            try:
                                align_result = self.aligner.align(
                                    audio_path=batch_audio_paths[i],
                                    text=batch_texts[i],
                                    language=self.language,
                                    audio_durations=[batch_durations[i]],
                                )
                                word_dicts = [
                                    {
                                        "word": w.word,
                                        "start": w.start,
                                        "end": w.end,
                                    }
                                    for w in align_result.words
                                ]
                                frame_alignments[frame_idx] = word_dicts
                            except Exception:
                                logger.error(
                                    "[DecoupledPipeline] Alignment failed for scene %d frame %d",
                                    scene_idx,
                                    frame_idx,
                                    exc_info=True,
                                )

                scene_alignments.append(frame_alignments)

                scene_words = sum(len(fa) for fa in frame_alignments)
                logger.debug(
                    "[DecoupledPipeline]   Scene %d/%d: %d words aligned",
                    scene_idx + 1, n_scenes, scene_words,
                )

                # Save alignment artifacts
                if self.artifacts_dir:
                    self._save_artifact(
                        scene_idx,
                        "aligned",
                        json.dumps(frame_alignments, ensure_ascii=False, indent=2),
                        ext=".json",
                    )

        finally:
            self.aligner.unload()
            self._safe_cuda_cleanup()

        total_words = sum(
            len(w) for fa_list in scene_alignments for w in fa_list
        )
        elapsed = _time.monotonic() - step57_start
        logger.info(
            "[DecoupledPipeline] Steps 5-7: Complete — %d scenes aligned, %d total words (%.1fs)",
            n_scenes, total_words, elapsed,
        )

        return scene_alignments

    @staticmethod
    def _native_or_frame_fallback(
        scene_frames: list[list[TemporalFrame]],
        scene_texts: list[list[str]],
        native_frame_words: Optional[list[list[Optional[list[dict[str, Any]]]]]],
    ) -> Optional[list[list[list[dict[str, Any]]]]]:
        """Use native generator timing, with a per-frame fallback when absent.

        A ``None`` return preserves the existing aligner-free Branch B for
        text-only generators.  Once any native timing is present, the method
        returns a complete per-frame alignment list: native entries are kept,
        while timestamp-less frames use their temporal frame bounds only.
        This avoids silently replacing valid native timing with scene-wide VAD
        distribution and allows the existing Japanese regrouping stage to
        split one speech region into multiple subtitle cues.
        """
        if not native_frame_words:
            return None

        has_native = any(
            frame_words
            for scene_words in native_frame_words
            for frame_words in scene_words
        )
        if not has_native:
            return None

        scene_alignments: list[list[list[dict[str, Any]]]] = []
        fallback_frames = 0
        for scene_idx, frames in enumerate(scene_frames):
            scene_words = (
                native_frame_words[scene_idx]
                if scene_idx < len(native_frame_words)
                else []
            )
            texts = scene_texts[scene_idx] if scene_idx < len(scene_texts) else []
            frame_alignments: list[list[dict[str, Any]]] = []

            for frame_idx, frame in enumerate(frames):
                native = scene_words[frame_idx] if frame_idx < len(scene_words) else None
                if native:
                    frame_alignments.append(native)
                    continue

                text = texts[frame_idx] if frame_idx < len(texts) else ""
                if not text.strip():
                    frame_alignments.append([])
                    continue

                # Native timestamps are relative to the sliced frame audio.
                # Generate the fallback in that same coordinate system; the
                # normal frame→scene offset is applied later.
                fallback_frames += 1
                fallback_words = split_frame_to_words(text, 0.0, frame.duration)
                for word in fallback_words:
                    word["source"] = "frame_fallback"
                frame_alignments.append(fallback_words)

            scene_alignments.append(frame_alignments)

        if fallback_frames:
            logger.warning(
                "[DecoupledPipeline] %d frame(s) had no valid native timestamps; "
                "using frame-boundary fallback for those frames",
                fallback_frames,
            )
        return scene_alignments

    # -----------------------------------------------------------------------
    # Step 9: Reconstruction + Sentinel + Hardening
    # -----------------------------------------------------------------------

    def _step9_reconstruct_and_harden(
        self,
        scene_frames: list[list[TemporalFrame]],
        scene_texts: list[list[str]],
        scene_alignments: Optional[list[list[list[dict[str, Any]]]]],
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]],
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Per-scene: merge words → sentinel → reconstruct → harden.

        Returns list of (WhisperResult_or_None, diagnostics) per scene.
        """
        from whisperjav.modules.alignment_sentinel import (
            assess_alignment_quality,
            redistribute_collapsed_words,
        )

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Step 9: Reconstructing %d scenes", n_scenes,
        )

        results: list[tuple[Any, dict[str, Any]]] = []
        total_segments = 0
        total_collapses = 0

        for scene_idx in range(n_scenes):
            frames = scene_frames[scene_idx]
            texts = scene_texts[scene_idx]
            duration = scene_durations[scene_idx]
            audio_path = scene_audio_paths[scene_idx]

            self.sentinel_stats["total_scenes"] += 1

            try:
                word_count = 0
                assessment = None
                recovery_info = None
                native_regroup_diagnostics = None

                if scene_alignments is not None:
                    # Branch A: Aligned workflow — merge frame-relative → scene-relative
                    # Resolve regroup mode for Branch A
                    regroup_a = resolve_regroup(self.hardening_config.regroup_mode, is_branch_b=False)

                    frame_word_groups, frame_parent_regions = self._group_frame_words_with_bounds(
                        frames,
                        scene_alignments[scene_idx],
                    )
                    native_flags = [
                        any(word.get("source") == "native" for word in group)
                        for group in frame_word_groups
                    ]

                    if self.native_regrouper is not None and any(native_flags):
                        # Regroup before stable-ts reconstruction. This keeps
                        # native Parakeet boundaries from being merged again
                        # by REGROUP_JAV's broader gap heuristic.
                        cue_groups, native_regroup_diagnostics = self.native_regrouper.regroup_scene(
                            frame_word_groups,
                            frame_parent_regions,
                            native_flags=native_flags,
                        )
                        word_count = sum(len(group) for group in cue_groups)
                        flat_words = [word for group in cue_groups for word in group]
                        assessment = assess_alignment_quality(flat_words, duration)
                        sentinel_status = assessment["status"]
                        result = reconstruct_frame_native(cue_groups, audio_path)
                    elif regroup_a is False:
                        # Frame-native path: one segment per frame, skip sentinel recovery.
                        # User explicitly set regroup_mode=OFF — they want raw frame output.
                        word_count = sum(len(g) for g in frame_word_groups)

                        # Sentinel assessment for diagnostics only (no recovery)
                        flat_words = [w for g in frame_word_groups for w in g]
                        assessment = assess_alignment_quality(flat_words, duration)
                        sentinel_status = assessment["status"]
                        if sentinel_status == "COLLAPSED":
                            self.sentinel_stats["collapsed_scenes"] += 1
                            total_collapses += 1

                        result = reconstruct_frame_native(frame_word_groups, audio_path)
                    else:
                        # Standard merge-and-regroup path
                        all_words = self._merge_frame_words(frames, scene_alignments[scene_idx])
                        word_count = len(all_words)

                        # Sentinel assessment (always run for diagnostics visibility)
                        assessment = assess_alignment_quality(all_words, duration)
                        sentinel_status = assessment["status"]

                        # G3 fix: aligner_only mode wants raw aligner output —
                        # assess for diagnostics but skip recovery even if collapsed.
                        skip_recovery = (
                            self.hardening_config.timestamp_mode == TimestampMode.ALIGNER_ONLY
                        )

                        if sentinel_status == "COLLAPSED" and not skip_recovery:
                            # Standard recovery path (aligner_interpolation, aligner_vad_fallback)
                            self.sentinel_stats["collapsed_scenes"] += 1
                            total_collapses += 1
                            logger.warning(
                                "[SENTINEL] Scene %d/%d: Alignment Collapse — "
                                "coverage=%.1f%%, CPS=%.1f, %d chars in %.3fs span",
                                scene_idx + 1, n_scenes,
                                assessment["coverage_ratio"] * 100,
                                assessment["aggregate_cps"],
                                assessment["char_count"],
                                assessment["word_span_sec"],
                            )

                            # Get speech regions for recovery
                            regions = self._get_speech_regions(scene_idx, frame_speech_regions, scene_speech_regions)

                            corrected_words = redistribute_collapsed_words(all_words, duration, regions)
                            self.sentinel_stats["recovered_scenes"] += 1
                            strategy = "vad_guided" if regions else "proportional"
                            self.sentinel_stats["recovery_strategies"][strategy] += 1
                            recovery_info = {
                                "strategy": strategy,
                                "words_redistributed": len(corrected_words),
                            }

                            # Reconstruct with suppress_silence=False (H3 fix)
                            result = reconstruct_from_words(
                                corrected_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )

                        elif sentinel_status == "COLLAPSED" and skip_recovery:
                            # aligner_only: log collapse but keep raw aligner timestamps
                            self.sentinel_stats["collapsed_scenes"] += 1
                            total_collapses += 1
                            logger.info(
                                "[SENTINEL] Scene %d/%d: COLLAPSED but aligner_only mode "
                                "— keeping raw aligner timestamps (no recovery)",
                                scene_idx + 1, n_scenes,
                            )
                            result = reconstruct_from_words(
                                all_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )
                        else:
                            # OK path — ForcedAligner timestamps are already accurate.
                            # Don't let stable-ts's crude loudness quantizer shrink them.
                            result = reconstruct_from_words(
                                all_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )

                else:
                    # Branch B: Aligner-free
                    # Resolve regroup mode for Branch B (aligner-free)
                    regroup = resolve_regroup(self.hardening_config.regroup_mode, is_branch_b=True)

                    if regroup is False:
                        # Frame-native: one segment per frame, each frame = one word entry.
                        # User explicitly set regroup_mode=OFF — they want raw frame output.
                        frame_word_groups = []
                        for frame, text in zip(frames, texts):
                            if text.strip():
                                frame_word_groups.append([{
                                    "word": text.strip(),
                                    "start": frame.start,
                                    "end": frame.end,
                                }])
                        word_count = len(frame_word_groups)
                        result = reconstruct_frame_native(frame_word_groups, audio_path)
                        sentinel_status = "N/A"
                    else:
                        # Standard split-and-regroup path
                        words = []
                        for frame, text in zip(frames, texts):
                            if text.strip():
                                words.extend(
                                    split_frame_to_words(text, frame.start, frame.end)
                                )
                        word_count = len(words)
                        # G1 complement: VAD_ONLY preserves exact frame boundaries —
                        # suppress_silence=False prevents stable-ts silence detection
                        # from shifting VAD group timing.
                        suppress = (
                            self.hardening_config.timestamp_mode != TimestampMode.VAD_ONLY
                        )
                        try:
                            result = reconstruct_from_words(
                                words, audio_path, suppress_silence=suppress, regroup=regroup,
                            )
                        except Exception as regroup_err:
                            logger.warning(
                                "[DecoupledPipeline] Scene %d: regrouping failed (%s), "
                                "retrying without regrouping",
                                scene_idx, regroup_err,
                            )
                            result = reconstruct_from_words(
                                words, audio_path, suppress_silence=suppress, regroup=False,
                            )
                        sentinel_status = "N/A"

                # Hardening (shared by all paths)
                # Resolve per-scene speech regions for VAD_ONLY mode
                per_scene_regions = self._get_speech_regions(
                    scene_idx, frame_speech_regions, scene_speech_regions,
                )
                config = HardeningConfig(
                    timestamp_mode=self.hardening_config.timestamp_mode,
                    scene_duration_sec=duration,
                    speech_regions=per_scene_regions,
                )
                hardening_diag = harden_scene_result(result, config)

                segment_count = len(result.segments) if result and result.segments else 0
                total_segments += segment_count

                # Per-scene progress
                logger.info(
                    "[DecoupledPipeline] Scene %d/%d: %d words → %d segments (sentinel: %s)",
                    scene_idx + 1, n_scenes, word_count, segment_count, sentinel_status,
                )

                # Canonical diagnostics (SceneDiagnostics v2.0.0)
                aligner_native_count = max(
                    0,
                    segment_count - hardening_diag.interpolated_count - hardening_diag.fallback_count,
                )

                # VAD regions for this scene (if available)
                vad_regions = None
                if per_scene_regions:
                    vad_regions = [
                        {"start": round(s, 3), "end": round(e, 3)}
                        for s, e in per_scene_regions
                    ]

                scene_diag = SceneDiagnostics(
                    schema_version="2.0.0",
                    scene_index=scene_idx,
                    scene_duration_sec=duration,
                    framer_backend=frames[0].source if frames else "",
                    frame_count=len(frames),
                    word_count=word_count,
                    segment_count=segment_count,
                    sentinel_status=sentinel_status,
                    sentinel_triggers=assessment.get("triggers", []) if assessment else [],
                    sentinel_recovery=recovery_info,
                    timing_aligner_native=aligner_native_count,
                    timing_interpolated=hardening_diag.interpolated_count,
                    timing_vad_fallback=hardening_diag.fallback_count,
                    timing_total_segments=segment_count,
                    hardening_clamped=hardening_diag.clamped_count,
                    hardening_sorted=hardening_diag.sorted,
                    vad_regions=vad_regions,
                    native_regroup=native_regroup_diagnostics,
                )
                diagnostics = asdict(scene_diag)

                # Save diagnostics artifact
                if self.artifacts_dir:
                    self._save_artifact(
                        scene_idx,
                        "diag",
                        json.dumps(diagnostics, ensure_ascii=False, indent=2),
                        ext=".json",
                    )

                results.append((result, diagnostics))

            except Exception as e:
                logger.error(
                    "[DecoupledPipeline] Scene %d failed: %s",
                    scene_idx,
                    e,
                    exc_info=True,
                )
                error_diag = SceneDiagnostics(
                    scene_index=scene_idx,
                    scene_duration_sec=scene_durations[scene_idx] if scene_idx < len(scene_durations) else 0.0,
                    error=str(e),
                )
                results.append((None, asdict(error_diag)))

        logger.info(
            "[DecoupledPipeline] Step 9: Complete — %d scenes, %d total segments, %d collapses",
            n_scenes, total_segments, total_collapses,
        )

        return results

    # -----------------------------------------------------------------------
    # Word merging: frame-relative → scene-relative
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge_frame_words(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Merge per-frame word lists into a single scene-relative word list.

        Each frame's words are in frame-relative coordinates.  We offset
        them by frame.start to convert to scene-relative coordinates.
        """
        all_words: list[dict[str, Any]] = []

        for frame, word_list in zip(frames, frame_word_lists):
            offset = frame.start
            for w in word_list:
                all_words.append(
                    {
                        "word": w["word"],
                        "start": w["start"] + offset,
                        "end": w["end"] + offset,
                        **({"source": w["source"]} if "source" in w else {}),
                    }
                )

        return all_words

    @staticmethod
    def _group_frame_words(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        """
        Group per-frame word lists into scene-relative word groups (one per frame).

        Like _merge_frame_words but keeps frame grouping instead of flattening.
        Each frame's words are offset by frame.start to convert from frame-relative
        to scene-relative coordinates.

        Returns:
            List of per-frame word groups (empty frames omitted).
        """
        groups, _ = DecoupledSubtitlePipeline._group_frame_words_with_bounds(
            frames,
            frame_word_lists,
        )
        return groups

    @staticmethod
    def _group_frame_words_with_bounds(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
    ) -> tuple[list[list[dict[str, Any]]], list[tuple[float, float]]]:
        """Return scene-relative frame groups and their parent bounds."""
        groups: list[list[dict[str, Any]]] = []
        parent_regions: list[tuple[float, float]] = []
        for frame, word_list in zip(frames, frame_word_lists):
            offset = frame.start
            frame_words = [
                {
                    "word": w["word"],
                    "start": w["start"] + offset,
                    "end": w["end"] + offset,
                    **({"source": w["source"]} if "source" in w else {}),
                }
                for w in word_list
            ]
            if frame_words:
                groups.append(frame_words)
                parent_regions.append((frame.start, frame.end))
        return groups, parent_regions

    # -----------------------------------------------------------------------
    # Speech regions resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_speech_regions(
        scene_idx: int,
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]],
    ) -> Optional[list[tuple[float, float]]]:
        """
        Get speech regions for a scene, preferring explicit over framer-derived.

        Priority:
            1. scene_speech_regions (passed by caller, e.g., from Phase 4 VAD)
            2. frame_speech_regions (from VadGroupedFramer metadata)
        """
        # Priority 1: Explicit scene-level regions
        if scene_speech_regions and scene_idx < len(scene_speech_regions):
            regions = scene_speech_regions[scene_idx]
            if regions:
                return regions

        # Priority 2: Framer-derived regions (flatten per-frame regions)
        if scene_idx < len(frame_speech_regions):
            per_frame_regions = frame_speech_regions[scene_idx]
            if per_frame_regions:
                flat: list[tuple[float, float]] = []
                for frame_regions in per_frame_regions:
                    flat.extend(frame_regions)
                return flat if flat else None

        return None

    # -----------------------------------------------------------------------
    # Audio I/O helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, int]:
        """Load audio from file as numpy array."""
        if sf is None:
            raise ImportError("soundfile is required for audio loading")
        audio, sr = sf.read(str(path), dtype="float32")
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    def _write_temp_wav(
        self,
        audio: np.ndarray,
        sample_rate: int,
        scene_idx: int,
        frame_idx: int,
    ) -> Path:
        """Write audio slice to a temporary WAV file."""
        if sf is None:
            raise ImportError("soundfile is required for audio writing")

        temp_dir = self.artifacts_dir or Path(tempfile.gettempdir())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"dsp_s{scene_idx:03d}_f{frame_idx:03d}.wav"
        sf.write(str(temp_path), audio, sample_rate)
        self._temp_files.append(temp_path)
        return temp_path

    def _cleanup_temp_files(self) -> None:
        """Delete temporary audio files."""
        cleaned = 0
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
                    cleaned += 1
            except OSError:
                pass
        if cleaned > 0:
            logger.debug("[DecoupledPipeline] Cleaned %d temp files", cleaned)
        self._temp_files.clear()

    # -----------------------------------------------------------------------
    # Artifact saving
    # -----------------------------------------------------------------------

    def _save_artifact(
        self,
        scene_idx: int,
        name: str,
        content: str,
        ext: str = ".txt",
    ) -> None:
        """Save a debug artifact to artifacts_dir."""
        if not self.artifacts_dir:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"scene{scene_idx:03d}_{name}{ext}"
        path.write_text(content, encoding="utf-8")

    # -----------------------------------------------------------------------
    # CUDA cleanup
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_cuda_cleanup() -> None:
        """Centralized CUDA cache cleanup."""
        from whisperjav.utils.gpu_utils import safe_cuda_cleanup

        safe_cuda_cleanup()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all component resources."""
        self.framer.cleanup()
        self.generator.cleanup()
        if self.aligner:
            self.aligner.cleanup()
        self._cleanup_temp_files()
