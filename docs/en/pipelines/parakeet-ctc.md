# Parakeet CTC Japanese Pipeline

WhisperJAV can use the Japanese Parakeet CTC model as the Qwen-pipeline
generator backend. It keeps NeMo optional and consumes native character,
word, or segment timing when the selected checkpoint exposes it. The Qwen
ForcedAligner is not loaded for this backend.

## Installation

For the current Japanese checkpoint, use the model-card-era NeMo runtime in
a fresh Python 3.10 environment. Install the CUDA PyTorch wheels first, then
install the project extra:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 build-essential git

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install "nemo-toolkit==2.0.0rc0" "nemo-toolkit[asr]==2.0.0rc0"
python -m pip install -e ".[cli,parakeet]"
```

The repository's `[parakeet]` extra is pinned to the same NeMo version, so
`python -m pip install -e ".[cli,parakeet]"` is reproducible for this
checkpoint. Do not use `uv sync --locked` for this bring-your-own
PyTorch/CUDA stack.

The default checkpoint is
`grider-transwithai/parakeet-ctc-1.1b-ja`. A Hugging Face model ID or a local
`.nemo` checkpoint can be supplied with `--qwen-model-id`.

## CLI

### Baseline frame-boundary output

```bash
whisperjav input.mp4 --mode qwen --qwen-generator parakeet \
  --qwen-regroup off
```

The backend requests native timestamps at the finest available level. If a
checkpoint returns no usable timestamps, the pipeline uses its existing
frame-boundary fallback for that frame and continues without a second ASR or
ForcedAligner pass. `--qwen-timestamps none` disables timestamp requests.

### Native Japanese regrouping

Phase 2 native regrouping is opt-in and only activates for the Parakeet
backend:

```bash
whisperjav input.mp4 --mode qwen --qwen-generator parakeet \
  --qwen-regroup off --parakeet-regroup
```

It groups Parakeet's character/token/word timing into readable Japanese cues
using these defaults:

| Option | Default | Behavior |
| --- | ---: | --- |
| `--parakeet-gap-split-ms` | `400` | Split at a silence of at least this length. |
| `--parakeet-max-cue-duration` | `6.0` | Split cues longer than this many seconds. |
| `--parakeet-max-cue-chars` | `30` | Split cues longer than this many Japanese characters. |
| `--parakeet-min-cue-duration` | `0.5` | Repair tiny fragments when no real boundary would be crossed. |

The regrouping stage prefers valid native timestamps, clamps units to their
parent speech frame, and falls back to the Phase 1 frame output for missing or
malformed regions. It does not change Whisper, Anime-Whisper, or Qwen3
behavior, and it does not run the Qwen ForcedAligner.

The same settings can be supplied to the Python pipeline through
`qwen_params`:

```python
{
    "generator_backend": "parakeet",
    "parakeet_regroup": True,
    "parakeet_gap_split_ms": 400.0,
    "parakeet_max_cue_duration": 6.0,
    "parakeet_max_cue_chars": 30,
    "parakeet_min_cue_duration": 0.5,
}
```

Each scene records compact regrouping counters in its diagnostics, including
native/fallback regions, clamped or reordered timestamps, split reasons, and
micro-fragment repairs.

To compare an existing SRT against a native-regrouped SRT:

```bash
python scripts/compare_srt_metrics.py current.srt regrouped.srt
```

## Hardware notes

CUDA is selected automatically when available. CPU and unavailable-CUDA
requests fall back to `float32`; CUDA uses `bfloat16` when supported and
otherwise `float16`. Model downloads and end-to-end GPU quality validation
remain environment-dependent.
