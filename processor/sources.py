"""
Finding recordings under INPUT_DIR and opening them: BIDS-iEEG (via MNE-BIDS), plain
EDF/EDF+/BDF, BrainVision, FIF (via MNE) and MEF3 sessions (via pymef).

Every reader is wrapped in a `Source` with the same small interface: channel names, the
types the file itself records, sampling rate, length, annotations and
`get(picks, start, stop)` returning µV.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from processor import bids as B
from processor.channels import MNE_TO_BIDS

log = logging.getLogger("processor-ieeg-prep")

EXTENSIONS = {".edf": "edf", ".bdf": "bdf", ".vhdr": "brainvision", ".fif": "fif", ".mefd": "mef3"}


def walk(root: Path) -> tuple[list[Path], list[Path]]:
    """(every file, every *.mefd folder) under root, following symlinks; .mefd folders are not descended."""
    files: list[Path] = []
    mefd: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        keep = []
        for d in sorted(dirnames):
            if d.startswith("."):
                continue
            if d.lower().endswith(".mefd"):
                mefd.append(Path(dirpath) / d)
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                files.append(p)
    return files, mefd


@dataclass
class Candidate:
    path: Path
    fmt: str                      # edf, bdf, brainvision, fif, mef3
    bids: bool
    entities: dict[str, str]
    recording_id: str
    subject: str
    session: str
    task: str
    run: str
    bids_root: Path | None = None
    subject_dir: Path | None = None


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "x"


def describe(path: Path, fmt: str) -> Candidate:
    """Identity of one recording: BIDS entities when the file is in a BIDS layout, else from its name."""
    ents = B.parse_entities(path.name)
    parent = path.parent
    is_bids = bool(ents) and parent.name in ("ieeg", "eeg") and (
        parent.parent.name.startswith("sub-") or parent.parent.parent.name.startswith("sub-"))
    if ents:
        stem = B.stem_of(path)
        sub = ents["sub"]
        sub_dir = None
        for p in path.parents:
            if p.name == f"sub-{sub}":
                sub_dir = p
                break
        return Candidate(path=path, fmt=fmt, bids=is_bids, entities=ents, recording_id=stem, subject=sub,
                         session=ents.get("ses", ""), task=ents.get("task", ""), run=ents.get("run", ""),
                         bids_root=sub_dir.parent if (is_bids and sub_dir) else None, subject_dir=sub_dir)
    name = path.name[:-5] if path.name.lower().endswith(".mefd") else path.stem
    sub = _alnum(name)
    return Candidate(path=path, fmt=fmt, bids=False, entities={}, recording_id=f"sub-{sub}_task-ieeg",
                     subject=sub, session="", task="ieeg", run="")


def find_recordings(input_dir: Path, pattern: str) -> tuple[list[Candidate], list[Path], list[str]]:
    """(recordings matching `pattern`, every file seen, notes). BrainVision is found by its .vhdr."""
    files, mefd = walk(input_dir)
    out: list[Candidate] = []
    notes: list[str] = []
    for p in files:
        name = p.name.lower()
        if name.endswith(".fif.gz"):
            fmt = "fif"
        else:
            fmt = EXTENSIONS.get(p.suffix.lower())
        if fmt is None or fmt == "mef3":
            continue
        if fmt == "fif" and re.search(r"-\d+\.fif(\.gz)?$", name) and "_split-" not in name:
            # continuation file of a split FIF (raw-1.fif): MNE reads it through the first part
            continue
        out.append(describe(p, fmt))
    for d in mefd:
        out.append(describe(d, "mef3"))
    pats = [s.strip() for s in (pattern or "*").split(",") if s.strip()] or ["*"]
    sel = []
    for c in out:
        rel = str(c.path.relative_to(input_dir))
        if any(fnmatch.fnmatch(c.recording_id, p) or fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(c.path.name, p)
               for p in pats):
            sel.append(c)
    if len(sel) < len(out):
        notes.append(f"RECORDINGS={pattern!r} selected {len(sel)} of {len(out)} recording(s)")
    sel.sort(key=lambda c: (c.subject, c.session, c.recording_id))
    return sel, files, notes


@dataclass
class Source:
    ch_names: list[str]
    file_types: dict[str, str]            # BIDS type per channel as recorded in the file / by MNE-BIDS
    types_trusted: bool
    sfreq: float
    n_times: int
    annotations: list[tuple[float, float, str]] = field(default_factory=list)
    line_freq: float | None = None
    meas_date: object = None
    reader: str = ""
    raw: object = None                    # mne Raw (EDF/BV/FIF)
    mef: object = None                    # pymef session
    mef_channels: list[str] = field(default_factory=list)
    mef_scale: dict[str, float] = field(default_factory=dict)
    mef_start: int = 0
    file_bads: list[str] = field(default_factory=list)

    def get(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        """float64 µV, shape (len(picks), stop - start)."""
        if self.raw is not None:
            return self.raw.get_data(picks=picks, start=start, stop=stop) * 1e6
        # By time, not by sample: pymef 1.4.8's read_ts_channels_sample returns samples start+1..stop
        # (and crashes when stop is the channel's length); the time-based read is exact and fills
        # gaps (discontinuities) with NaN, written here as 0.
        n = stop - start
        out = np.zeros((len(picks), n))
        t0 = self.mef_start + int(round(start / self.sfreq * 1e6))
        t1 = self.mef_start + int(round(stop / self.sfreq * 1e6))
        for i, p in enumerate(picks):
            name = self.mef_channels[p]
            x = np.asarray(self.mef.read_ts_channels_uutc(name, [t0, t1]), dtype=np.float64)[:n]
            out[i, : len(x)] = x * self.mef_scale.get(name, 1.0)
        return np.nan_to_num(out, nan=0.0)

    def close(self) -> None:
        try:
            if self.raw is not None:
                self.raw.close()
            if self.mef is not None:
                self.mef.close()
        except Exception:  # noqa: BLE001
            pass


def _mne_types(raw) -> dict[str, str]:
    return {n: MNE_TO_BIDS.get(t, t.upper()) for n, t in zip(raw.ch_names, raw.get_channel_types())}


def _from_raw(raw, reader: str, trusted: bool) -> Source:
    # onsets are relative to meas_date when the annotations carry one, else to the first sample
    ann = [(float(a["onset"] - raw.first_time if raw.annotations.orig_time is not None else a["onset"]),
            float(a["duration"]), str(a["description"])) for a in raw.annotations]
    # MNE pads EDF files whose last data record is incomplete and marks the pad BAD_ACQ_SKIP
    n_times = raw.n_times
    for on, du, de in ann:
        if de.upper() == "BAD_ACQ_SKIP" and abs(on + du - raw.n_times / raw.info["sfreq"]) < 1.0 / raw.info["sfreq"] + 1e-6:
            n_times = min(n_times, int(round(on * raw.info["sfreq"])))
    lf = raw.info.get("line_freq")
    return Source(ch_names=list(raw.ch_names), file_types=_mne_types(raw), types_trusted=trusted,
                  sfreq=float(raw.info["sfreq"]), n_times=int(n_times), annotations=ann,
                  line_freq=float(lf) if lf else None, meas_date=raw.info.get("meas_date"), reader=reader, raw=raw,
                  file_bads=list(raw.info["bads"]))


def _read_mne(path: Path, fmt: str):
    import mne
    if fmt in ("edf",):
        return mne.io.read_raw_edf(path, preload=False, verbose="error", infer_types=False)
    if fmt == "bdf":
        return mne.io.read_raw_bdf(path, preload=False, verbose="error", infer_types=False)
    if fmt == "brainvision":
        return mne.io.read_raw_brainvision(path, preload=False, verbose="error")
    if fmt == "fif":
        return mne.io.read_raw_fif(path, preload=False, verbose="error")
    raise ValueError(f"unsupported format {fmt}")


def _open_mef(path: Path, password: str) -> Source:
    from pymef.mef_session import MefSession
    ms = MefSession(str(path), password or None)
    info = ms.read_ts_channel_basic_info()
    if not info:
        raise ValueError("MEF3 session has no time-series channels")
    fs = [float(np.asarray(i["fsamp"]).ravel()[0]) for i in info]
    common = max(set(fs), key=fs.count)
    names, scale, types, dropped = [], {}, {}, []
    for i, f in zip(info, fs):
        if abs(f - common) > 1e-6:
            dropped.append(i["name"])
            continue
        names.append(i["name"])
        u = np.asarray(i["unit"]).ravel()[0]
        unit = (u.decode("utf-8", "ignore") if isinstance(u, bytes) else str(u)).strip().strip("\x00").lower()
        ufact = float(np.asarray(i["ufact"]).ravel()[0])
        if not np.isfinite(ufact) or ufact == 0:
            ufact = 1.0
        # pymef returns the stored integers; units_conversion_factor turns them into `unit`; express in µV
        scale[i["name"]] = ufact * {"v": 1e6, "volts": 1e6, "mv": 1e3, "millivolts": 1e3, "nv": 1e-3,
                                    "nanovolts": 1e-3}.get(unit, 1.0)
    if dropped:
        log.warning("%s: %d channel(s) at another sampling rate than %g Hz not read: %s", path.name, len(dropped),
                    common, ", ".join(dropped))
    kept = [i for i in info if i["name"] in scale]
    start = min(int(np.asarray(i["start_time"]).ravel()[0]) for i in kept)
    end = max(int(np.asarray(i["end_time"]).ravel()[0]) for i in kept)
    nsamp = min(int(np.asarray(i["nsamp"]).ravel()[0]) for i in kept)
    n = max(nsamp, int(round((end - start) / 1e6 * common)))  # longer than nsamp when there are gaps
    return Source(ch_names=names, file_types=types, types_trusted=False, sfreq=common, n_times=n,
                  reader="pymef", mef=ms, mef_channels=names, mef_scale=scale, mef_start=start)


def open_source(c: Candidate, mef_password: str = "") -> tuple[Source, list[str]]:
    """Open one recording; BIDS ones through MNE-BIDS (types and bads from channels.tsv), else directly."""
    notes: list[str] = []
    if c.fmt == "mef3":
        return _open_mef(c.path, mef_password), notes
    if c.bids and c.bids_root is not None and c.fmt != "mef3":
        try:
            import mne_bids
            bp = mne_bids.get_bids_path_from_fname(c.path)
            bp.update(root=c.bids_root)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne_bids.read_raw_bids(bp, verbose="error")
            return _from_raw(raw, "mne-bids", True), notes
        except Exception as exc:  # noqa: BLE001 - fall back to the plain reader
            notes.append(f"MNE-BIDS could not read it ({type(exc).__name__}: {exc}); read with MNE directly")
    raw = _read_mne(c.path, c.fmt)
    return _from_raw(raw, f"mne-{c.fmt}", c.fmt == "fif"), notes
