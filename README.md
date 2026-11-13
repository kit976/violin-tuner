# 🎻 Violin Tuner

A reliable, microphone-agnostic violin tuner built with Python and [uv](https://docs.astral.sh/uv/).

Uses the **YIN pitch detection algorithm** — far more accurate than simple autocorrelation across all four violin strings (G3, D4, A4, E5).

## Quick Start

### Run with uv & Python
```bash
cd violin-tuner
uv run python main.py
```

## Features

- **YIN pitch detection** — handles all 4 strings reliably (G3 through E5)
- **Works with any microphone** — laptop built-in, USB condenser, clip-on contact mics
- **Bandpass filter** — isolates the violin range (150–2000 Hz), cuts room noise
- **Ring buffer accumulation** — large analysis window (4096 samples) for accurate low-frequency detection
- **Smooth analog meter** — color-coded zones (green/orange/red)
- **Violin string indicators** — highlights which open string you're tuning
- **Adjustable A4 reference** — 400–480 Hz range for different orchestras
- **Device selection** — pick any connected microphone, with live refresh

## Why YIN Instead of Autocorrelation?

The original tuner used FFT-based autocorrelation, which has known issues:

| Problem | Autocorrelation | YIN |
|---|---|---|
| E5 string (659 Hz) | Often locks onto harmonics (octave errors) | CMNDF suppresses harmonic peaks |
| G3 string (196 Hz) | Needs many cycles → large buffer | Works with 3–4 cycles |
| Noisy mic input | Raw peaks confused by noise | Threshold-based first-dip selection |
| Sub-sample accuracy | Basic peak picking | Parabolic interpolation |

## Dependencies

Managed automatically by `uv`:
- `numpy` — array operations
- `scipy` — Butterworth bandpass filter
- `sounddevice` — cross-platform audio input (no pyaudio C dependency)
- `tkinter` — GUI (included with Python)

## Settings

Click **⚙ Settings** to change the A4 reference pitch (default 440 Hz). Standard alternatives:
- 442 Hz — common in European orchestras
- 415 Hz — Baroque pitch
