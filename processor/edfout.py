"""
EDF+ output with a physical range per channel (edfio), and the read-back check with MNE.

Each channel's physical range is symmetric, ±L with L = 1.1 × robust max |x| (at least 1 µV),
where robust max = min(max |x|, 3 × the 99.99th percentile of |x|): a recording without
extreme outliers is stored without any clipping, while a single saturated artefact far
above the rest of the channel cannot coarsen its quantisation step. Digital range ±32767, so
0 µV is stored exactly and the step is 2L / 65534.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from edfio import Edf, EdfAnnotation, EdfSignal, Patient, Recording

DIGITAL = 32767
ROBUST_PCT = 99.99
ROBUST_FACTOR = 3.0
MARGIN = 1.1
MIN_RANGE_UV = 1.0


def _edf_number(value: float, round_func) -> float:
    """The value as EDF stores it in 8 characters (edfio's rounding)."""
    if float(value).is_integer():
        return float(value)
    s = str(float(value))
    ipl = s.find(".")
    if ipl >= 8:
        return float(round_func(value))
    f = 10 ** (8 - 1 - ipl)
    return round_func(value * f) / f


def symmetric_limit(lim: float) -> float:
    """
    The smallest L ≥ lim that edfio stores unchanged as both physical minimum (-L, rounded down to
    8 characters) and maximum (L, rounded up), so the range is exactly symmetric and 0 µV is digital 0.
    """
    from edfio.edf_signal import _round_float_to_8_characters as r8
    cand = -_edf_number(-lim, math.floor)
    for _ in range(50):
        lo, hi = r8(-cand, math.floor), r8(cand, math.ceil)
        if lo == -cand and hi == cand:
            return cand
        s = str(cand)
        decimals = len(s) - s.find(".") - 1 if "." in s else 0
        cand = round(cand + 10 ** -decimals, decimals) if decimals else cand + 1
    return float(math.ceil(lim))


@dataclass
class Range:
    limit: float        # L, µV, as stored in the header
    step: float         # quantisation step, µV
    clipped: int        # samples clipped to ±L
    max_abs: float      # max |x| before clipping


def channel_range(x: np.ndarray) -> Range:
    ax = np.abs(x)
    mx = float(ax.max()) if len(ax) else 0.0
    if not np.isfinite(mx):
        mx = float(np.nanmax(np.where(np.isfinite(ax), ax, 0.0)))
    p = float(np.percentile(ax, ROBUST_PCT)) if len(ax) else 0.0
    robust = min(mx, ROBUST_FACTOR * p) if p > 0 else mx
    lim = max(MARGIN * robust, MIN_RANGE_UV)
    lim = symmetric_limit(lim)
    clipped = int(np.count_nonzero(ax > lim))
    return Range(limit=lim, step=2 * lim / (2 * DIGITAL), clipped=clipped, max_abs=mx)


def make_signal(x: np.ndarray, sfreq: float, label: str, rng: Range, prefiltering: str,
                transducer: str = "") -> EdfSignal:
    y = np.clip(np.nan_to_num(x, nan=0.0), -rng.limit, rng.limit)
    return EdfSignal(y, sfreq, label=label[:16], transducer_type=transducer[:80], physical_dimension="uV",
                     physical_range=(-rng.limit, rng.limit), digital_range=(-DIGITAL, DIGITAL),
                     prefiltering=prefiltering[:80])


def record_duration(sfreq: float) -> tuple[float, int]:
    """(data record duration in s, samples per record) such that sfreq × duration is a whole number."""
    for d in (1.0, 0.5, 0.25, 0.2, 0.1, 2.0, 4.0, 5.0, 10.0):
        spr = sfreq * d
        if abs(spr - round(spr)) < 1e-9 and round(spr) >= 1:
            return d, int(round(spr))
    return 1.0, int(round(sfreq))


def write_edf(path: Path, signals: list[EdfSignal], annotations: list[tuple[float, float | None, str]],
              startdate: dt.date | None, starttime: dt.time | None, record_s: float) -> None:
    ann = [EdfAnnotation(max(0.0, on), du if (du is not None and du > 0) else None, text[:60])
           for on, du, text in annotations]
    edf = Edf(signals, patient=Patient(), recording=Recording(startdate=startdate),
              starttime=starttime or dt.time(0, 0, 0), data_record_duration=record_s, annotations=ann)
    path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(path)


def verify(path: Path, checks: list[tuple[int, int, np.ndarray]], names: list[str],
           steps: list[float]) -> dict:
    """
    Read the EDF back with MNE and compare windows of it with the values that were written.
    `checks`: (channel index, start sample, expected µV after clipping). Returns the largest
    error in µV and as a fraction of that channel's quantisation step (≤ 0.5 when correct).
    """
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    if raw.ch_names[: len(names)] != [n[:16] for n in names]:
        raise ValueError("channel names read back differ from those written")
    worst_uv, worst_frac, worst_ch = 0.0, 0.0, ""
    by_window: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for ch, start, exp in checks:
        by_window.setdefault((start, start + len(exp)), []).append((ch, exp))
    for (start, stop), items in by_window.items():
        picks = [c for c, _ in items]
        got = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
        for (ch, exp), g in zip(items, got):
            err = float(np.max(np.abs(g - exp))) if len(exp) else 0.0
            frac = err / steps[ch] if steps[ch] > 0 else 0.0
            if frac > worst_frac:
                worst_uv, worst_frac, worst_ch = err, frac, names[ch]
    raw.close()
    return {"max_abs_error_uv": worst_uv, "max_error_in_steps": worst_frac, "worst_channel": worst_ch,
            "windows_checked": len(by_window), "channel_windows_checked": len(checks)}
