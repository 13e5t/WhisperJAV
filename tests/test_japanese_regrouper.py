from __future__ import annotations

import math

import pytest

from whisperjav.modules.subtitle_pipeline.regroupers import (
    JapaneseNativeRegrouper,
    JapaneseRegroupConfig,
)


def _units(
    text: str,
    start: float,
    end: float,
    *,
    source: str = "native",
) -> list[dict[str, object]]:
    characters = list(text)
    step = (end - start) / max(1, len(characters))
    return [
        {
            "word": character,
            "start": start + index * step,
            "end": start + (index + 1) * step,
            "source": source,
        }
        for index, character in enumerate(characters)
    ]


def _texts(groups: list[list[dict[str, object]]]) -> list[str]:
    return ["".join(str(word["word"]) for word in group) for group in groups]


def test_one_speech_region_becomes_three_native_japanese_cues() -> None:
    words = (
        _units("気持ちいい", 0.0, 1.4)
        + _units("もっとして", 1.9, 3.0)
        + _units("うん", 3.7, 4.2)
    )

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(0.0, 5.0)], [True]
    )

    assert _texts(groups) == ["気持ちいい", "もっとして", "うん"]
    assert diagnostics["gap_splits"] == 2
    assert diagnostics["cues_before_regroup"] == 1
    assert diagnostics["cues_after_regroup"] == 3


def test_punctuation_splits_without_a_large_gap() -> None:
    words = _units("これは。", 0.0, 1.0) + _units("次です！", 1.05, 2.0)

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(0.0, 2.0)], [True]
    )

    assert _texts(groups) == ["これは。", "次です！"]
    assert diagnostics["punctuation_splits"] == 1


def test_duration_limit_splits_long_native_region() -> None:
    config = JapaneseRegroupConfig(max_duration_s=1.0, min_duration_s=0.0)
    words = _units("あいうえ", 0.0, 4.0)

    groups, diagnostics = JapaneseNativeRegrouper(config).regroup_scene(
        [words], [(0.0, 4.0)], [True]
    )

    assert len(groups) == 4
    assert diagnostics["duration_splits"] == 3


def test_character_limit_splits_long_native_region() -> None:
    config = JapaneseRegroupConfig(max_chars=3, min_duration_s=0.0)
    words = _units("あいうえお", 0.0, 1.0)

    groups, diagnostics = JapaneseNativeRegrouper(config).regroup_scene(
        [words], [(0.0, 1.0)], [True]
    )

    assert _texts(groups) == ["あいう", "えお"]
    assert diagnostics["length_splits"] == 1


def test_short_utterances_across_gaps_are_not_merged() -> None:
    words = (
        _units("うん", 0.0, 0.2)
        + _units("だめ", 0.7, 0.9)
        + _units("もっと", 1.4, 1.7)
    )

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(0.0, 2.0)], [True]
    )

    assert _texts(groups) == ["うん", "だめ", "もっと"]
    assert diagnostics["repaired_micro_fragments"] == 0


def test_micro_fragment_repairs_when_no_real_boundary_exists() -> None:
    config = JapaneseRegroupConfig(max_chars=2, min_duration_s=0.5)
    words = _units("あいう", 0.0, 0.2)

    groups, diagnostics = JapaneseNativeRegrouper(config).regroup_scene(
        [words], [(0.0, 0.2)], [True]
    )

    assert _texts(groups) == ["あいう"]
    assert diagnostics["repaired_micro_fragments"] == 1


def test_malformed_native_timestamps_fall_back_to_parent_region() -> None:
    words = [
        {"word": "これは", "start": 0.0, "end": 0.4, "source": "native"},
        {"word": "壊れた", "start": math.nan, "end": 0.8, "source": "native"},
    ]

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(2.0, 4.0)], [True]
    )

    assert _texts(groups) == ["これは壊れた"]
    assert groups[0][0]["start"] == 2.0
    assert groups[0][0]["end"] == 4.0
    assert diagnostics["malformed_timestamp_regions"] == 1
    assert diagnostics["fallback_regions"] == 1


def test_missing_native_timestamps_keep_fallback_behavior() -> None:
    words = [{"word": "フレーム字幕", "source": "frame_fallback"}]

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(1.0, 2.0)], [False]
    )

    assert _texts(groups) == ["フレーム字幕"]
    assert groups[0][0]["start"] == 1.0
    assert groups[0][0]["end"] == 2.0
    assert diagnostics["fallback_regions"] == 1


def test_out_of_parent_timestamps_are_clamped() -> None:
    config = JapaneseRegroupConfig(gap_split_ms=2000.0, min_duration_s=0.0)
    words = [
        {"word": "前", "start": 0.97, "end": 1.5, "source": "native"},
        {"word": "後", "start": 2.5, "end": 3.03, "source": "native"},
    ]

    groups, diagnostics = JapaneseNativeRegrouper(config).regroup_scene(
        [words], [(1.0, 3.0)], [True]
    )

    assert groups[0][0]["start"] == 1.0
    assert groups[0][-1]["end"] == 3.0
    assert diagnostics["clamped_timestamps"] == 2


def test_timestamp_far_outside_parent_falls_back() -> None:
    words = [
        {"word": "範囲外", "start": 0.0, "end": 0.5, "source": "native"},
    ]

    groups, diagnostics = JapaneseNativeRegrouper().regroup_scene(
        [words], [(2.0, 4.0)], [True]
    )

    assert _texts(groups) == ["範囲外"]
    assert groups[0][0]["start"] == 2.0
    assert groups[0][0]["end"] == 4.0
    assert diagnostics["malformed_timestamp_regions"] == 1


def test_out_of_order_native_units_are_sorted_deterministically() -> None:
    config = JapaneseRegroupConfig(gap_split_ms=2000.0, min_duration_s=0.0)
    words = [
        {"word": "後", "start": 1.0, "end": 1.2, "source": "native"},
        {"word": "前", "start": 0.0, "end": 0.2, "source": "native"},
    ]

    groups, diagnostics = JapaneseNativeRegrouper(config).regroup_scene(
        [words], [(0.0, 2.0)], [True]
    )

    assert _texts(groups) == ["前後"]
    assert diagnostics["reordered_timestamp_regions"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("gap_split_ms", -1.0),
        ("max_duration_s", 0.0),
        ("max_chars", 0),
        ("min_duration_s", -1.0),
        ("timestamp_tolerance_s", -1.0),
        ("max_duration_s", math.inf),
    ],
)
def test_regroup_config_rejects_invalid_limits(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        JapaneseRegroupConfig(**{field: value})
