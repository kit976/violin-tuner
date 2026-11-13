"""Smoke-test: verify YIN detects all 4 violin open strings from synthetic sine waves."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from main import YINDetector, SAMPLE_RATE, ANALYSIS_SIZE

def make_sine(freq: float, duration_s: float = 0.1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a clean sine wave at the given frequency."""
    t = np.arange(int(sr * duration_s)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)

def make_violin_like(freq: float, duration_s: float = 0.1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a violin-like waveform with harmonics."""
    t = np.arange(int(sr * duration_s)) / sr
    # Violin has strong odd harmonics + some even
    signal = (
        1.0 * np.sin(2 * np.pi * freq * t)       # fundamental
        + 0.7 * np.sin(2 * np.pi * 2 * freq * t)  # 2nd harmonic
        + 0.5 * np.sin(2 * np.pi * 3 * freq * t)  # 3rd harmonic
        + 0.3 * np.sin(2 * np.pi * 4 * freq * t)  # 4th harmonic
        + 0.15 * np.sin(2 * np.pi * 5 * freq * t)  # 5th harmonic
    )
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5
    return signal

def test_string(name: str, expected_freq: float, signal_func, tolerance_cents: float = 15.0):
    detector = YINDetector()
    audio = signal_func(expected_freq)
    detected = detector.detect(audio)

    if detected is None:
        print(f"  FAIL  {name}: expected ~{expected_freq:.1f} Hz, got None")
        return False

    # Calculate error in cents
    cents_error = 1200 * np.log2(detected / expected_freq)
    ok = abs(cents_error) < tolerance_cents
    status = "  OK  " if ok else "  FAIL"
    print(f"{status}  {name}: expected {expected_freq:.1f} Hz, got {detected:.1f} Hz (error: {cents_error:+.1f} cents)")
    return ok

def main():
    strings = [
        ("G3", 196.0),
        ("D4", 293.66),
        ("A4", 440.0),
        ("E5", 659.26),
    ]

    print("=" * 60)
    print("YIN Pitch Detection Test — Synthetic Signals")
    print("=" * 60)

    all_pass = True

    print("\n--- Pure Sine Waves ---")
    for name, freq in strings:
        if not test_string(name, freq, make_sine):
            all_pass = False

    print("\n--- Violin-like (with harmonics) ---")
    for name, freq in strings:
        if not test_string(name, freq, make_violin_like):
            all_pass = False

    print("\n--- Noisy Violin-like (SNR ~20dB) ---")
    for name, freq in strings:
        def noisy_signal(f):
            s = make_violin_like(f)
            noise = np.random.randn(len(s)) * 0.02
            return s + noise
        if not test_string(name, freq, noisy_signal, tolerance_cents=25.0):
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
