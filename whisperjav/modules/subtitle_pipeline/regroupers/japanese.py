"""Deterministic Japanese cue regrouping for native Parakeet timestamps.

This module deliberately operates on the decoupled pipeline's small word
dictionaries instead of stable-ts objects.  It turns one temporal speech frame
into one or more cue-sized word groups while preserving the native unit timing.
The caller can then reconstruct each group as an individual subtitle cue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from whisperjav.utils.logger import logger

WordDict = dict[str, Any]
_NATIVE_TIMING_SOURCES = frozenset({"native", "native_ctc"})


@dataclass(frozen=True)
class JapaneseRegroupConfig:
    """Configurable limits for native Japanese cue regrouping."""

    gap_split_ms: float = 400.0
    max_duration_s: float = 6.0
    max_chars: int = 30
    min_duration_s: float = 0.5
    timestamp_tolerance_s: float = 0.05

    def __post_init__(self) -> None:
        numeric_values = (
            self.gap_split_ms,
            self.max_duration_s,
            self.min_duration_s,
            self.timestamp_tolerance_s,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("regroup timing limits must be finite")
        if self.gap_split_ms < 0:
            raise ValueError("gap_split_ms must be non-negative")
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        if self.max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if self.min_duration_s < 0:
            raise ValueError("min_duration_s must be non-negative")
        if self.timestamp_tolerance_s < 0:
            raise ValueError("timestamp_tolerance_s must be non-negative")

    @property
    def gap_split_s(self) -> float:
        """Gap threshold in seconds used by the regrouping algorithm."""
        return self.gap_split_ms / 1000.0


@dataclass
class JapaneseRegroupDiagnostics:
    """Low-volume counters describing native regrouping decisions."""

    speech_regions_input: int = 0
    native_timestamp_regions: int = 0
    fallback_regions: int = 0
    malformed_timestamp_regions: int = 0
    clamped_timestamps: int = 0
    reordered_timestamp_regions: int = 0
    cues_before_regroup: int = 0
    cues_after_regroup: int = 0
    gap_splits: int = 0
    punctuation_splits: int = 0
    duration_splits: int = 0
    length_splits: int = 0
    repaired_micro_fragments: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable diagnostics dictionary."""
        return asdict(self)


@dataclass
class _Cue:
    """Internal cue representation retaining the reason for its boundary."""

    units: list[WordDict]
    boundary_before: str | None = None

    @property
    def start(self) -> float:
        return float(self.units[0]["start"])

    @property
    def end(self) -> float:
        return float(self.units[-1]["end"])

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text(self) -> str:
        return _join_units(self.units)


class JapaneseNativeRegrouper:
    """Regroup Parakeet-native units into readable Japanese subtitle cues.

    ``regroup_scene`` accepts one list per temporal frame.  Each frame is a
    parent speech region; native frames are split using their timing, while
    timestamp-less/fallback frames are kept intact so the Phase 1 fallback
    behavior remains available.
    """

    _SENTENCE_ENDINGS = frozenset("。！？!?")
    _CLOSING_MARKS = frozenset("」』】）》)]}\"'")

    def __init__(self, config: JapaneseRegroupConfig | None = None):
        self.config = config or JapaneseRegroupConfig()

    def regroup_scene(
        self,
        frame_word_groups: list[list[WordDict]],
        parent_regions: list[tuple[float, float]],
        native_flags: list[bool] | None = None,
    ) -> tuple[list[list[WordDict]], dict[str, int]]:
        """Regroup frame word-groups and return cue groups plus diagnostics.

        Args:
            frame_word_groups: Scene-relative words grouped by temporal frame.
            parent_regions: Scene-relative ``(start, end)`` bounds per frame.
            native_flags: Whether each frame has reliable native timing.  If
                omitted, ``source`` values ``native`` and ``native_ctc`` are
                used for detection.

        Returns:
            ``(cue_groups, diagnostics)``.  Each inner cue group contains the
            original native units, with only timestamp clamping/order repair
            applied.  Fallback groups retain their original Phase 1 words.
        """
        diagnostics = JapaneseRegroupDiagnostics()
        cue_groups: list[list[WordDict]] = []

        for index, words in enumerate(frame_word_groups):
            if not words:
                continue

            diagnostics.speech_regions_input += 1
            diagnostics.cues_before_regroup += 1
            parent_start, parent_end = self._parent_bounds(
                words,
                parent_regions[index] if index < len(parent_regions) else None,
            )
            is_native = (
                native_flags[index]
                if native_flags is not None and index < len(native_flags)
                else any(word.get("source") in _NATIVE_TIMING_SOURCES for word in words)
            )

            if not is_native:
                diagnostics.fallback_regions += 1
                fallback_group = self._fallback_group(words, parent_start, parent_end, diagnostics)
                if fallback_group:
                    cue_groups.append(fallback_group)
                continue

            normalized, malformed = self._normalize_native_units(
                words,
                parent_start,
                parent_end,
                diagnostics,
            )
            if malformed or not normalized:
                diagnostics.fallback_regions += 1
                if malformed:
                    diagnostics.malformed_timestamp_regions += 1
                fallback_group = self._fallback_group(
                    words,
                    parent_start,
                    parent_end,
                    diagnostics,
                    force_parent=malformed,
                )
                if fallback_group:
                    cue_groups.append(fallback_group)
                continue

            diagnostics.native_timestamp_regions += 1
            cues = self._split_native_units(normalized, diagnostics)
            cues = self._repair_micro_fragments(cues, diagnostics)
            cue_groups.extend(cue.units for cue in cues if cue.units)

        diagnostics.cues_after_regroup = len(cue_groups)
        return cue_groups, diagnostics.as_dict()

    def _split_native_units(
        self,
        units: list[WordDict],
        diagnostics: JapaneseRegroupDiagnostics,
    ) -> list[_Cue]:
        """Greedily split native units at strong deterministic boundaries."""
        cues: list[_Cue] = []
        current: list[WordDict] = []
        current_boundary_before: str | None = None

        for unit in units:
            if not current:
                current = [unit]
                continue

            reason = self._split_reason(current, unit)
            if reason is not None:
                cues.append(_Cue(current, boundary_before=current_boundary_before))
                current = [unit]
                current_boundary_before = reason
                setattr(diagnostics, f"{reason}_splits", getattr(diagnostics, f"{reason}_splits") + 1)
            else:
                current.append(unit)

        if current:
            cues.append(_Cue(current, boundary_before=current_boundary_before))
        return cues

    def _split_reason(self, current: list[WordDict], next_unit: WordDict) -> str | None:
        """Return the strongest split reason before ``next_unit``."""
        previous = current[-1]
        gap = max(0.0, float(next_unit["start"]) - float(previous["end"]))
        if gap >= self.config.gap_split_s:
            return "gap"
        if self._ends_sentence(_join_units(current)):
            return "punctuation"

        candidate = current + [next_unit]
        if self._cue_duration(candidate) > self.config.max_duration_s:
            return "duration"
        if len(_join_units(candidate)) > self.config.max_chars:
            return "length"
        return None

    def _repair_micro_fragments(
        self,
        cues: list[_Cue],
        diagnostics: JapaneseRegroupDiagnostics,
    ) -> list[_Cue]:
        """Conservatively merge tiny fragments without crossing real gaps."""
        if len(cues) < 2 or self.config.min_duration_s <= 0:
            return cues

        repaired = list(cues)
        index = 0
        while index < len(repaired):
            cue = repaired[index]
            if cue.duration >= self.config.min_duration_s:
                index += 1
                continue

            # A real silence or sentence ending is an intentional boundary;
            # do not merge short Japanese utterances such as ``うん`` or ``だめ``.
            if cue.boundary_before in {"gap", "punctuation"}:
                index += 1
                continue

            candidates: list[tuple[float, int, _Cue]] = []
            if index > 0:
                previous = repaired[index - 1]
                if self._can_merge(previous, cue, allow_small_overflow=True):
                    candidates.append((self._gap(previous, cue), index - 1, previous))
            if index + 1 < len(repaired):
                following = repaired[index + 1]
                if (
                    following.boundary_before not in {"gap", "punctuation"}
                    and self._can_merge(cue, following, allow_small_overflow=True)
                ):
                    candidates.append((self._gap(cue, following), index, following))

            if not candidates:
                index += 1
                continue

            _, merge_index, _ = min(candidates, key=lambda item: item[0])
            if merge_index == index - 1:
                previous = repaired[merge_index]
                repaired[merge_index] = _Cue(
                    previous.units + cue.units,
                    boundary_before=previous.boundary_before,
                )
                del repaired[index]
                diagnostics.repaired_micro_fragments += 1
                index = max(0, merge_index)
            else:
                following = repaired[index + 1]
                repaired[index] = _Cue(
                    cue.units + following.units,
                    boundary_before=cue.boundary_before,
                )
                del repaired[index + 1]
                diagnostics.repaired_micro_fragments += 1

        return repaired

    def _can_merge(
        self,
        left: _Cue,
        right: _Cue,
        allow_small_overflow: bool = False,
    ) -> bool:
        if self._gap(left, right) >= self.config.gap_split_s:
            return False
        if self._ends_sentence(left.text):
            return False

        duration_ok = self._cue_duration(left.units + right.units) <= self.config.max_duration_s
        chars = len(left.text + right.text)
        chars_ok = chars <= self.config.max_chars
        if allow_small_overflow and (len(left.text) <= 1 or len(right.text) <= 1):
            chars_ok = chars <= self.config.max_chars + 1
        return duration_ok and chars_ok

    @staticmethod
    def _gap(left: _Cue, right: _Cue) -> float:
        return max(0.0, right.start - left.end)

    @staticmethod
    def _cue_duration(units: list[WordDict]) -> float:
        return max(0.0, float(units[-1]["end"]) - float(units[0]["start"]))

    def _normalize_native_units(
        self,
        words: list[WordDict],
        parent_start: float,
        parent_end: float,
        diagnostics: JapaneseRegroupDiagnostics,
    ) -> tuple[list[WordDict], bool]:
        """Validate native timestamps, clamp bounds, and repair ordering."""
        normalized: list[tuple[int, WordDict]] = []
        malformed = False

        for index, word in enumerate(words):
            token = word.get("word", word.get("text", word.get("char", "")))
            try:
                start = float(word.get("start"))
                end = float(word.get("end"))
            except (TypeError, ValueError):
                malformed = True
                continue

            if (
                not token
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or end < start
            ):
                malformed = True
                continue

            tolerance = self.config.timestamp_tolerance_s
            if (
                start < parent_start - tolerance
                or end > parent_end + tolerance
                or end < parent_start - tolerance
                or start > parent_end + tolerance
            ):
                # A small amount of model rounding is safe to clamp. A larger
                # excursion means the native units do not belong to this
                # speech frame, so preserve the text with a parent-region
                # fallback instead of manufacturing misleading cue timing.
                malformed = True
                continue

            clamped_start = min(parent_end, max(parent_start, start))
            clamped_end = min(parent_end, max(parent_start, end))
            if clamped_start != start or clamped_end != end:
                diagnostics.clamped_timestamps += 1
            if clamped_end <= clamped_start:
                malformed = True
                continue

            normalized.append(
                (
                    index,
                    {
                        **word,
                        "word": str(token),
                        "start": clamped_start,
                        "end": clamped_end,
                        "source": word.get("source", "native"),
                    },
                )
            )

        if malformed:
            return [], True

        if any(
            current[1]["start"] < previous[1]["start"]
            or current[1]["end"] < previous[1]["end"]
            for previous, current in zip(normalized, normalized[1:])
        ):
            normalized.sort(key=lambda item: (item[1]["start"], item[1]["end"], item[0]))
            diagnostics.reordered_timestamp_regions += 1

        return [word for _, word in normalized], False

    def _fallback_group(
        self,
        words: list[WordDict],
        parent_start: float,
        parent_end: float,
        diagnostics: JapaneseRegroupDiagnostics,
        force_parent: bool = False,
    ) -> list[WordDict]:
        """Keep Phase 1 words, clamping them or preserving the full text."""
        if force_parent:
            text = _join_units(words)
            if text and parent_end > parent_start:
                return [{
                    "word": text,
                    "start": parent_start,
                    "end": parent_end,
                    "source": "region_fallback",
                }]

        fallback: list[WordDict] = []
        for word in words:
            token = word.get("word", word.get("text", word.get("char", "")))
            if not token:
                continue
            try:
                start = float(word.get("start", parent_start))
                end = float(word.get("end", parent_end))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                continue
            clamped_start = min(parent_end, max(parent_start, start))
            clamped_end = min(parent_end, max(parent_start, end))
            if clamped_end <= clamped_start:
                continue
            fallback.append({
                **word,
                "word": str(token),
                "start": clamped_start,
                "end": clamped_end,
                "source": word.get("source", "region_fallback"),
            })

        if fallback:
            return fallback

        text = _join_units(words)
        if not text or parent_end <= parent_start:
            return []
        logger.debug(
            "[ParakeetRegrouper] Native timing unavailable; using parent region %.3f-%.3f",
            parent_start,
            parent_end,
        )
        return [{
            "word": text,
            "start": parent_start,
            "end": parent_end,
            "source": "region_fallback",
        }]

    @staticmethod
    def _parent_bounds(
        words: list[WordDict],
        bounds: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if bounds is None:
            starts = [float(word.get("start", 0.0)) for word in words]
            ends = [float(word.get("end", 0.0)) for word in words]
            start = min(starts, default=0.0)
            end = max(ends, default=start)
        else:
            start, end = bounds
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            return 0.0, max(0.0, end if math.isfinite(end) else 0.0)
        return min(start, end), max(start, end)

    @classmethod
    def _ends_sentence(cls, text: str) -> bool:
        text = text.rstrip()
        while text and text[-1] in cls._CLOSING_MARKS:
            text = text[:-1].rstrip()
        return bool(text) and text[-1] in cls._SENTENCE_ENDINGS


def _join_units(units: list[WordDict]) -> str:
    """Join unit text without inventing spaces in Japanese."""
    return "".join(
        str(unit.get("word", unit.get("text", unit.get("char", ""))))
        for unit in units
    ).strip()
