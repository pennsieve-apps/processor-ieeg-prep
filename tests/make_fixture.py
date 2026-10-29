"""
A small synthetic input that exercises the contract: a BIDS-iEEG dataset with sidecars and
labels, a plain (non-BIDS) EDF with exporter-style channel names, and a labels.csv.

  <out>/bids/dataset_description.json
  <out>/bids/sub-P01/ses-implant1/ieeg/sub-P01_ses-implant1_task-ictal01_run-01_{ieeg.edf,ieeg.json,channels.tsv,events.tsv}
  <out>/bids/sub-P01/ses-implant1/ieeg/sub-P01_ses-implant1_space-MNI152NLin2009aSym_{electrodes.tsv,coordsystem.json}
  <out>/plain/patient02.edf
  <out>/labels.csv

Signals (µV, 1000 Hz, 60 s): background noise per contact plus a common reference component;
spikes on LA02/LA03; a seizure from 40 s on LA01-LA04 (fast activity growing into 6 Hz
spike-waves); LB05 dominated by 60 Hz; LB07 very noisy; LB08 flat; LB03 marked bad in
channels.tsv; an EKG, a DC channel and scalp C3/C4.

Usage: python -m tests.make_fixture data/fixture
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from edfio import Edf, EdfSignal
from scipy import signal

SFREQ = 1000.0
DURATION = 60.0
ONSET = 40.0
CONTACTS = [f"LA{i:02d}" for i in range(1, 9)] + [f"LB{i:02d}" for i in range(1, 9)]


def background(rng: np.random.Generator, n: int, amp: float = 30.0) -> np.ndarray:
    """1/f-like noise: white noise through a first-order low-pass, scaled to `amp` µV sd."""
    w = rng.standard_normal(n)
    b, a = signal.butter(1, 8.0, fs=SFREQ)
    x = signal.lfilter(b, a, w) + 0.05 * w
    return amp * x / np.std(x)


def make_signals(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(SFREQ * DURATION)
    t = np.arange(n) / SFREQ
    common = background(rng, n, 15.0)
    out = {}
    for name in CONTACTS:
        out[name] = background(rng, n) + common
    # spikes on LA02 and LA03 every ~2 s
    spike = -250 * np.exp(-0.5 * ((np.arange(-40, 41) / 1000) / 0.008) ** 2)
    for k in np.arange(1.0, DURATION - 1, 2.1):
        i = int(k * SFREQ)
        for name in ("LA02", "LA03"):
            out[name][i - 40:i + 41] += spike
    # seizure on LA01-LA04 from ONSET: low-voltage fast activity, then 6 Hz rhythm
    sz = t >= ONSET
    ts = t[sz] - ONSET
    fast = 40 * np.sin(2 * np.pi * 25 * ts) * np.clip(ts / 2, 0, 1) * (ts < 6)
    rhythm = 300 * np.sin(2 * np.pi * 6 * ts) * np.clip((ts - 5) / 3, 0, 1)
    for name, gain in (("LA01", 1.0), ("LA02", 0.5), ("LA03", 0.25), ("LA04", 0.1)):
        out[name][sz] += (fast + rhythm) * gain
    out["LB05"] += 400 * np.sin(2 * np.pi * 60 * t)                       # line noise
    out["LB07"] += 2000 * rng.standard_normal(n)                            # noisy contact
    out["LB08"] = np.full(n, 3.0)                                           # flat
    ekg = np.zeros(n)
    for k in np.arange(0.4, DURATION, 0.85):
        i = int(k * SFREQ)
        ekg[i:i + 30] += 1000 * np.hanning(30)
    out["EKG1"] = ekg + 20 * rng.standard_normal(n)
    out["DC1"] = np.where((t % 10) < 0.5, 5000.0, 0.0)
    out["C3"] = background(rng, n, 20.0)
    out["C4"] = background(rng, n, 20.0)
    return out


def write_edf(path: Path, sig: dict[str, np.ndarray], labels: dict[str, str] | None = None,
              annotations=()) -> None:
    from edfio import EdfAnnotation
    signals = []
    for name, x in sig.items():
        lim = float(np.ceil(np.max(np.abs(x)) * 1.05 + 1))
        signals.append(EdfSignal(x, SFREQ, label=(labels or {}).get(name, name), physical_dimension="uV",
                                 physical_range=(-lim, lim), digital_range=(-32767, 32767)))
    path.parent.mkdir(parents=True, exist_ok=True)
    Edf(signals, annotations=[EdfAnnotation(*a) for a in annotations]).write(path)


def tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def coords(name: str) -> tuple[float, float, float]:
    k = int(name[2:])
    x0 = -20.0 if name.startswith("LA") else -40.0
    return (x0 - 3.5 * (k - 1), 10.0, -12.0 if name.startswith("LA") else 5.0)


def make(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    sig = make_signals()
    # --- BIDS
    root = out / "bids"
    d = root / "sub-P01" / "ses-implant1" / "ieeg"
    d.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "synthetic ieeg-prep fixture", "BIDSVersion": "1.10.0", "DatasetType": "raw"}, indent=2))
    stem = "sub-P01_ses-implant1_task-ictal01_run-01"
    bids_sig = {k: v for k, v in sig.items() if k not in ("DC1", "C4")}
    write_edf(d / f"{stem}_ieeg.edf", bids_sig)
    (d / f"{stem}_ieeg.json").write_text(json.dumps({
        "TaskName": "ictal01", "SamplingFrequency": SFREQ, "PowerLineFrequency": 60, "iEEGReference": "LB08",
        "SoftwareFilters": "n/a", "RecordingDuration": DURATION, "RecordingType": "continuous",
        "Manufacturer": "synthetic"}, indent=2))
    rows = []
    for name in bids_sig:
        typ = {"EKG1": "ECG", "C3": "EEG"}.get(name, "SEEG")
        status, desc = ("bad", "broken wire (clinical note)") if name == "LB03" else ("good", "n/a")
        soz = "n/a" if typ != "SEEG" else ("true" if name in ("LA01", "LA02") else "false")
        rows.append([name, typ, "µV", 0, 500, SFREQ, status, desc, soz])
    tsv(d / f"{stem}_channels.tsv", ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency",
                                     "status", "status_description", "soz"], rows)
    tsv(d / f"{stem}_events.tsv", ["onset", "duration", "trial_type"], [[39.0, 15.0, "seizure"],
                                                                        [5.0, 1.0, "artifact"]])
    tsv(d / "sub-P01_ses-implant1_space-MNI152NLin2009aSym_electrodes.tsv", ["name", "x", "y", "z", "size",
                                                                            "resected", "roi"],
        [[c, *coords(c), 0.8, "true" if c.startswith("LA") and int(c[2:]) <= 5 else "false",
          "Left-Hippocampus" if c.startswith("LA") else "ctx-lh-insula"] for c in CONTACTS])
    (d / "sub-P01_ses-implant1_space-MNI152NLin2009aSym_coordsystem.json").write_text(json.dumps(
        {"iEEGCoordinateSystem": "MNI152NLin2009aSym", "iEEGCoordinateUnits": "mm"}, indent=2))
    # --- plain EDF, exporter-style names (POL prefix, -Ref suffix), no sidecars
    plain = {k: v for k, v in sig.items()}
    labels = {k: (f"POL {k[:2]}{int(k[2:])}" if k in CONTACTS else k) for k in plain}
    labels["C3"] = "EEG C3-Ref"
    write_edf(out / "plain" / "patient02.edf", plain, labels, annotations=[(30.0, 2.0, "Seizure")])
    # --- labels.csv: expert onset for the BIDS recording (overrides events.tsv) and SOZ for patient02
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["subject", "recording_id", "channel", "soz", "resected", "onset", "spread", "onset_s"])
        w.writerow(["P01", stem, "", "", "", "", "", "40.5"])
        for c in ("LA01", "LA02"):
            w.writerow(["P01", stem, c, "", "", "true", "true", ""])
        w.writerow(["P01", stem, "LA03", "", "", "", "true", ""])
        w.writerow(["patient02", "", "LA1", "true", "", "", "", ""])
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")
    make(target)
    print(f"fixture written to {target}")
