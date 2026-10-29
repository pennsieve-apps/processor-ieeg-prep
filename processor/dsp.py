"""
Signal processing: zero-phase IIR filters (notch at the line frequency and its harmonics,
optional high-pass and low-pass) and the per-channel quality metrics used to find bad
channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

NOTCH_Q = 30.0          # notch quality factor: -3 dB width = f0 / Q (2 Hz at 60 Hz)
HIGHPASS_ORDER = 4      # Butterworth order; applied forwards and backwards (zero phase, order 8 effective)
LOWPASS_ORDER = 4


@dataclass
class FilterPlan:
    sfreq: float
    line_freq: float
    notch: list[float]
    highpass: float
    lowpass: float

    def describe(self) -> dict:
        return {
            "notch_hz": self.notch, "notch": f"IIR notch (Q={NOTCH_Q:g}), zero-phase" if self.notch else "none",
            "highpass_hz": self.highpass or None,
            "highpass": f"Butterworth order {HIGHPASS_ORDER}, zero-phase (sosfiltfilt)" if self.highpass else "none",
            "lowpass_hz": self.lowpass or None,
            "lowpass": f"Butterworth order {LOWPASS_ORDER}, zero-phase (sosfiltfilt)" if self.lowpass else "none",
        }


def plan_filters(sfreq: float, line_freq: float, highpass: float, lowpass: float) -> FilterPlan:
    nyq = sfreq / 2.0
    notch = []
    if line_freq > 0:
        k = 1
        while k * line_freq < nyq - 1.0:   # harmonics strictly below Nyquist (with 1 Hz room)
            notch.append(float(k * line_freq))
            k += 1
    hp = highpass if 0 < highpass < nyq else 0.0
    lp = lowpass if 0 < lowpass < nyq * 0.98 else 0.0
    return FilterPlan(sfreq, line_freq, notch, hp, lp)


class Filter:
    """Applies a FilterPlan to one channel at a time (float64 in, float64 out)."""

    def __init__(self, plan: FilterPlan):
        self.plan = plan
        self.notches = [signal.iirnotch(f, NOTCH_Q, fs=plan.sfreq) for f in plan.notch]
        self.hp = (signal.butter(HIGHPASS_ORDER, plan.highpass, btype="highpass", fs=plan.sfreq, output="sos")
                   if plan.highpass else None)
        self.lp = (signal.butter(LOWPASS_ORDER, plan.lowpass, btype="lowpass", fs=plan.sfreq, output="sos")
                   if plan.lowpass else None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        if len(y) < 16:
            return y
        y = y - np.median(y)
        for b, a in self.notches:
            y = signal.filtfilt(b, a, y)
        if self.hp is not None:
            # odd-extension padding of ~3 time constants keeps the edge transient of a 0.5 Hz
            # high-pass short (the default pad is a few samples)
            pad = int(min(len(y) - 1, 3 * self.plan.sfreq / self.plan.highpass))
            y = signal.sosfiltfilt(self.hp, y, padlen=pad)
        if self.lp is not None:
            y = signal.sosfiltfilt(self.lp, y)
        return y


@dataclass
class Metrics:
    sd_raw: float               # µV, of the detrended unfiltered signal
    zero_diff_fraction: float   # fraction of consecutive samples that are identical
    line_length: float          # median |dx| per sample, µV, after the QC filter (1 Hz high-pass + notch)
    amplitude: float            # robust standard deviation (1.4826 × MAD), µV, after the QC filter
    line_noise_ratio: float     # power within ±1 Hz of the line frequency and harmonics / power 1 Hz – min(Nyquist, 250 Hz)


def qc_filter(sfreq: float, line_freq: float) -> Filter:
    return Filter(plan_filters(sfreq, line_freq, 1.0 if sfreq > 20 else 0.0, 0.0))


def metrics(x: np.ndarray, sfreq: float, line_freq: float, qcf: Filter) -> Metrics:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return Metrics(0.0, 1.0, 0.0, 0.0, 0.0)
    xd = signal.detrend(x, type="linear")
    sd_raw = float(np.std(xd))
    zdf = float(np.mean(np.diff(x) == 0))
    y = qcf(x)
    # robust versions: sparse transients (interictal spikes, stimulation pulses) occupy far fewer than
    # half of the samples and so do not move a median; persistent noise or a loose contact does
    ll = float(np.median(np.abs(np.diff(y))))
    amp = float(1.4826 * np.median(np.abs(y - np.median(y))))
    lnr = 0.0
    if line_freq > 0 and sd_raw > 0:
        nper = int(min(n, max(256, 2 ** int(np.ceil(np.log2(sfreq * 2))))))  # ~0.5 Hz resolution
        f, p = signal.welch(xd, fs=sfreq, nperseg=nper)
        top = min(sfreq / 2.0 - 1.0, 250.0)
        band = (f >= 1.0) & (f <= top)
        total = float(np.sum(p[band]))
        line = np.zeros_like(f, dtype=bool)
        k = 1
        while k * line_freq <= top:
            line |= np.abs(f - k * line_freq) <= 1.0
            k += 1
        lnr = float(np.sum(p[band & line]) / total) if total > 0 else 0.0
    return Metrics(sd_raw, zdf, ll, amp, lnr)


# Smallest spread used for the robust z of log10 line length / amplitude: 0.05 decades. When the
# channels are nearly identical (MAD → 0), a channel needs at least 10^(BAD_Z × 0.05) times the median
# (1.8× at BAD_Z = 5) to count as an outlier, instead of being flagged for a few percent.
MIN_LOG_SPREAD = 0.05


def robust_z(values: np.ndarray, min_spread: float = 0.0) -> np.ndarray:
    """(x - median) / max(1.4826 · MAD, min_spread); zeros when both are 0. NaN in, NaN out."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    med = np.median(v[ok])
    mad = max(1.4826 * np.median(np.abs(v[ok] - med)), min_spread)
    if mad <= 0:
        return np.where(ok, 0.0, np.nan)
    return (v - med) / mad
