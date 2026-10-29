"""
One recording, start to finish: channel selection, bad channels, montage, filters, labels,
coordinates, events, and the BIDS derivative files.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from processor import bids as B
from processor import channels as C
from processor import dsp
from processor import edfout as E
from processor import labels as L
from processor import sources as S

log = logging.getLogger("processor-ieeg-prep")

READ_WINDOW_S = 60.0
MIN_BASELINE_S = 5.0
SNIPPET_S = 10.0
VERIFY_S = 5.0


@dataclass
class Params:
    montage: str = "bipolar"
    line_freq: str = "auto"
    highpass: float = 0.5
    lowpass: float = 0.0
    include_scalp: bool = False
    ieeg_type: str = "SEEG"
    flat_uv: float = 0.5
    bad_z: float = 5.0
    line_noise_max: float = 0.5
    baseline_gap_s: float = 10.0
    baseline_max_s: float = 90.0
    verify: bool = True
    mef_password: str = ""


@dataclass
class OutChannel:
    name: str
    type: str
    contacts: list[str]
    status: str = "good"
    reasons: list[str] = field(default_factory=list)
    labels: dict[str, bool | None] = field(default_factory=dict)
    xyz: tuple[float, float, float] | None = None
    roi: str = ""
    limit: float = 0.0
    step: float = 0.0
    clipped: int = 0


@dataclass
class Result:
    recording_row: dict[str, Any]
    qc_rows: list[dict[str, Any]]
    electrodes: dict[str, dict[str, Any]]       # norm contact -> electrodes.tsv row
    electrode_columns: list[str]
    details: dict[str, Any]
    plot: dict[str, Any]


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def tf(v: bool | None) -> str:
    return "" if v is None else ("true" if v else "false")


def _is_seizure(tt: str) -> bool:
    t = tt.strip().lower()
    if re.search(r"(inter|pre|post|peri)[ _-]?ictal", t):
        return False
    return t in B.SEIZURE_TYPES or t.startswith(("seizure", "sz", "ictal"))


def line_frequency(param: str, sidecar: dict, src: S.Source) -> tuple[float, str]:
    if param.lower() != "auto":
        return float(param), "LINE_FREQ parameter"
    v = sidecar.get("PowerLineFrequency")
    try:
        f = float(v)
        if f > 0:
            return f, "PowerLineFrequency in the BIDS sidecar"
    except (TypeError, ValueError):
        pass
    if src.line_freq:
        return float(src.line_freq), "line frequency recorded in the file"
    return 60.0, "default (no PowerLineFrequency in the source)"


def load(src: S.Source, picks: list[int]) -> np.ndarray:
    """The picked channels as float32 µV, read in windows of all channels (one pass over the file)."""
    out = np.empty((len(picks), src.n_times), dtype=np.float32)
    step = max(int(READ_WINDOW_S * src.sfreq), 1)
    for start in range(0, src.n_times, step):
        stop = min(start + step, src.n_times)
        out[:, start:stop] = src.get(picks, start, stop)
    return out


def process(c: S.Candidate, params: Params, labelset: L.LabelSet, out_root: Path, app_version: str,
            input_dir: Path) -> Result:
    t0 = time.monotonic()
    src, notes = S.open_source(c, params.mef_password)
    try:
        return _process(c, src, notes, params, labelset, out_root, app_version, input_dir, t0)
    finally:
        src.close()


def _process(c, src: S.Source, notes: list[str], params: Params, labelset: L.LabelSet, out_root: Path,
             app_version: str, input_dir: Path, t0: float) -> Result:
    rid = c.recording_id
    sfreq = src.sfreq
    if sfreq < 100:
        raise ValueError(f"sampling rate {sfreq:g} Hz is too low for iEEG")
    if src.n_times < sfreq:
        raise ValueError(f"recording is shorter than 1 s ({src.n_times} samples)")

    # ---- sidecars -----------------------------------------------------------------------------
    ch_tsv = B.read_channels(B.sidecar(c.path, "channels.tsv"))
    ev_path = B.sidecar(c.path, "events.tsv")
    events_in = B.read_events(ev_path)
    js_path = B.sidecar(c.path, "ieeg.json")
    sidecar = B.read_json(js_path) if js_path else {}
    el_path, cs_path = B.find_electrodes(c.path, c.subject_dir)
    elec = B.read_electrodes(el_path, cs_path)

    # ---- channel selection ----------------------------------------------------------------------
    names = src.ch_names
    if ch_tsv:
        missing = [n for n in names if n not in ch_tsv]
        if missing and len(missing) < len(names):
            notes.append(f"{len(missing)} channel(s) not in channels.tsv: {', '.join(missing[:10])}")
    types = {n: (ch_tsv[n].type if n in ch_tsv and ch_tsv[n].type else src.file_types.get(n, "")) for n in names}
    trusted = src.types_trusted or bool(ch_tsv)
    classes = C.classify(names, types, params.ieeg_type, params.include_scalp, trusted)
    kept = [i for i, cl in enumerate(classes) if cl.keep]
    if not kept:
        raise ValueError("no SEEG/ECoG channel (dropped: " +
                         ", ".join(sorted({cl.dropped_type for cl in classes})) + ")")
    out_name: dict[int, str] = {}
    used: set[str] = set()
    for i in kept:
        n = C.clean_name(names[i]) or names[i]
        if n in used:
            n = names[i]
        used.add(n)
        out_name[i] = n

    src_name = {n: names[i] for i, n in out_name.items()}
    line_freq, lf_source = line_frequency(params.line_freq, sidecar, src)
    plan = dsp.plan_filters(sfreq, line_freq, params.highpass, params.lowpass)

    # ---- labels and onset ---------------------------------------------------------------------
    ids = {rid, L.norm_recording(c.path.name), L.norm_recording(c.path.stem)}
    label_rows = labelset.for_recording(c.subject, ids)
    contact_names = [out_name[i] for i in kept]
    rl = L.resolve(label_rows, contact_names, labelset.path.name if labelset.path else "labels.csv")
    if rl.unmatched_channels:
        notes.append(f"labels.csv: {len(rl.unmatched_channels)} channel(s) not in this recording: "
                     + ", ".join(sorted(set(rl.unmatched_channels))[:15]))

    def contact_labels(i: int) -> dict[str, bool | None]:
        k = C.norm_name(out_name[i])
        lab: dict[str, bool | None] = {col: None for col in B.LABEL_COLUMNS}
        for col, v in elec.labels.get(k, {}).items():
            if v is not None:
                lab[col] = v
        row = ch_tsv.get(names[i])
        if row:
            for col, v in row.labels.items():
                if v is not None:
                    lab[col] = v
        for col, v in rl.contacts.get(k, {}).items():
            if v is not None:
                lab[col] = v
        return lab

    seizure_events = [e for e in events_in if _is_seizure(e.trial_type)]
    if not events_in:
        seizure_events = [B.Event(on, du, de, "file annotations") for on, du, de in src.annotations if _is_seizure(de)]
    onset_s, onset_source, sz_end = None, "", None
    if seizure_events:
        e0 = min(seizure_events, key=lambda e: e.onset)
        if e0.duration:
            sz_end = e0.onset + e0.duration
    if rl.onset_s is not None:
        onset_s, onset_source = rl.onset_s, rl.onset_source
    elif seizure_events:
        onset_s = min(e.onset for e in seizure_events)
        onset_source = f"{seizure_events[0].source} ({seizure_events[0].trial_type})"
    duration_s = src.n_times / sfreq
    if onset_s is not None and not (0 <= onset_s < duration_s):
        notes.append(f"seizure onset {onset_s:.2f} s is outside the recording (0–{duration_s:.2f} s); ignored")
        onset_s = None
    task = c.task.lower()
    if onset_s is not None or seizure_events:
        kind = "ictal"
    elif "interictal" in task:
        kind = "interictal"
    elif re.search(r"ictal|seizure", task):
        kind = "ictal"
    else:
        kind = "unknown"
    baseline = None
    if onset_s is not None:
        end = min(onset_s - params.baseline_gap_s, params.baseline_max_s)
        if end >= MIN_BASELINE_S:
            baseline = (0.0, end)
        else:
            notes.append(f"no baseline: onset {onset_s:.1f} s leaves less than {MIN_BASELINE_S:g} s before "
                         f"onset - {params.baseline_gap_s:g} s")

    # ---- load ---------------------------------------------------------------------------------
    t_read = time.monotonic()
    data = load(src, kept)
    t_read = time.monotonic() - t_read

    # ---- bad channels -------------------------------------------------------------------------
    if baseline is not None:
        qs, qe = 0, int(baseline[1] * sfreq)
        qc_window = f"baseline 0–{baseline[1]:.1f} s"
    else:
        qs, qe = 0, src.n_times
        qc_window = "whole recording"
    qcf = dsp.qc_filter(sfreq, line_freq)
    # Metrics are taken against the per-sample median of the channels (a robust common reference),
    # so noise common to every channel through the recording reference does not hide (or create)
    # outliers; flatness and line noise are judged on the channel as recorded.
    raw_mets = [dsp.metrics(data[j, qs:qe], sfreq, line_freq, qcf) for j in range(len(kept))]
    live = [j for j, m in enumerate(raw_mets) if m.sd_raw >= params.flat_uv and m.zero_diff_fraction <= 0.5]
    medref = np.median(data[live, qs:qe], axis=0) if len(live) >= 3 else None
    mets = []
    for j in range(len(kept)):
        if medref is None:
            mets.append(raw_mets[j])
            continue
        m = dsp.metrics(data[j, qs:qe] - medref, sfreq, line_freq, qcf)
        m.sd_raw, m.zero_diff_fraction, m.line_noise_ratio = (raw_mets[j].sd_raw, raw_mets[j].zero_diff_fraction,
                                                               raw_mets[j].line_noise_ratio)
        mets.append(m)
    del medref
    status = ["good"] * len(kept)
    reasons: list[list[str]] = [[] for _ in kept]
    for j, i in enumerate(kept):
        m = mets[j]
        if m.sd_raw < params.flat_uv or m.zero_diff_fraction > 0.5:
            reasons[j].append(f"flat (sd {m.sd_raw:.2g} µV, {100 * m.zero_diff_fraction:.0f}% repeated samples)")
        if m.line_noise_ratio > params.line_noise_max:
            reasons[j].append(f"line noise ({100 * m.line_noise_ratio:.0f}% of 1–250 Hz power at {line_freq:g} Hz "
                              "and harmonics)")
        row = ch_tsv.get(names[i])
        if row and row.status == "bad":
            reasons[j].append("bad in source" + (f": {row.status_description}" if row.status_description else ""))
        elif names[i] in src.file_bads:
            reasons[j].append("bad in source")
    flat = np.array([any(r.startswith("flat") for r in rs) for rs in reasons])
    ll_z = np.full(len(kept), np.nan)
    amp_z = np.full(len(kept), np.nan)
    ktypes = [classes[i].type for i in kept]
    for t in set(ktypes):
        idx = np.array([j for j in range(len(kept)) if ktypes[j] == t and not flat[j]], dtype=int)
        if len(idx) >= 5:
            ll_z[idx] = dsp.robust_z(np.log10([max(mets[j].line_length, 1e-9) for j in idx]), dsp.MIN_LOG_SPREAD)
            amp_z[idx] = dsp.robust_z(np.log10([max(mets[j].amplitude, 1e-9) for j in idx]), dsp.MIN_LOG_SPREAD)
    for j in range(len(kept)):
        if np.isfinite(ll_z[j]) and ll_z[j] > params.bad_z:
            reasons[j].append(f"high line length (robust z {ll_z[j]:.1f})")
        if np.isfinite(amp_z[j]) and amp_z[j] > params.bad_z:
            reasons[j].append(f"high amplitude (robust z {amp_z[j]:.1f})")
        if reasons[j]:
            status[j] = "bad"

    # ---- montage ------------------------------------------------------------------------------
    knames = [out_name[i] for i in kept]
    ieeg_ref = str(sidecar.get("iEEGReference", "") or "")
    already_bipolar = C.looks_bipolar(knames) or "bipolar" in ieeg_ref.lower()
    structured = C.has_electrode_structure(knames)
    mode = params.montage
    if mode in ("bipolar", "auto"):
        if already_bipolar:
            if mode == "bipolar":
                notes.append("channels are already bipolar (" + ("names" if C.looks_bipolar(knames) else
                             f"iEEGReference '{ieeg_ref}'") + "); kept as recorded")
            mode = "as-recorded"
        elif not structured:
            if mode == "bipolar":
                raise ValueError("channel names have no electrode/contact structure (e.g. LA01, LA02), so "
                                 "neighbours are unknown; use MONTAGE=auto, car or as-recorded")
            mode = "as-recorded"
        else:
            mode = "bipolar"

    coords = {}
    for i in kept:
        k = C.norm_name(out_name[i])
        if k in elec.coords and all(math.isfinite(v) for v in elec.coords[k]):
            coords[out_name[i]] = elec.coords[k]
    jidx = {out_name[i]: j for j, i in enumerate(kept)}
    chans: list[OutChannel] = []
    combos: list[tuple[int, int | None]] = []   # (j, j2) -> x_j - x_j2 ; j2 None = x_j (- ref for CAR)
    pair_notes: list[str] = []
    if mode == "bipolar":
        good_names = [n for n in knames if status[jidx[n]] == "good" and ktypes[jidx[n]] != "EEG"]
        pairs, pair_notes = C.bipolar_pairs([n for n in knames if ktypes[jidx[n]] != "EEG"], coords)
        skipped_bad = 0
        for p in pairs:
            if p.a not in good_names or p.b not in good_names:
                skipped_bad += 1
                continue
            ja, jb = jidx[p.a], jidx[p.b]
            chans.append(OutChannel(p.name, ktypes[ja], [p.a, p.b]))
            combos.append((ja, jb))
        unpaired = [n for n in good_names if not any(n in ch.contacts for ch in chans)]
        if skipped_bad:
            pair_notes.append(f"{skipped_bad} pair(s) skipped because a contact is bad")
        if unpaired:
            pair_notes.append(f"{len(unpaired)} good contact(s) in no pair: {', '.join(unpaired[:12])}")
        for n in knames:   # scalp EEG kept with INCLUDE_SCALP: as recorded
            if ktypes[jidx[n]] == "EEG":
                ch = OutChannel(n, "EEG", [n], status[jidx[n]], list(reasons[jidx[n]]))
                chans.append(ch)
                combos.append((jidx[n], None))
        if not chans:
            raise ValueError("no bipolar pair of good neighbouring contacts")
    else:
        for j, n in enumerate(knames):
            chans.append(OutChannel(n, ktypes[j], [n], status[j], list(reasons[j])))
            combos.append((j, None))
    car = None
    if mode == "car":
        good_rows = [j for j in range(len(kept)) if status[j] == "good"]
        if len(good_rows) < 2:
            raise ValueError("common average reference needs at least 2 good channels")
        car = np.zeros(src.n_times, dtype=np.float64)
        for j in good_rows:
            car += data[j]
        car /= len(good_rows)

    # labels and coordinates per output channel
    space = elec.space
    for ch in chans:
        labs = [contact_labels(kept[jidx[n]]) for n in ch.contacts]
        ch.labels = {}
        for col in B.LABEL_COLUMNS:
            vals = [lab.get(col) for lab in labs]
            if any(v is True for v in vals):
                ch.labels[col] = True
            elif vals and all(v is False for v in vals):
                ch.labels[col] = False
            else:
                ch.labels[col] = None
        cs = [coords.get(n) for n in ch.contacts]
        if cs and all(cs):
            ch.xyz = tuple(float(np.mean([c_[k] for c_ in cs])) for k in range(3))  # type: ignore[assignment]
        rois = [elec.roi.get(C.norm_name(n), "") for n in ch.contacts]
        if any(rois):
            ch.roi = rois[0] if len(set(rois)) == 1 else "|".join(r or "n/a" for r in rois)

    # ---- filter, range, encode ----------------------------------------------------------------
    filt = dsp.Filter(plan)
    rec_s, spr = E.record_duration(sfreq)
    n_out = (src.n_times // spr) * spr
    if n_out < src.n_times:
        notes.append(f"last {(src.n_times - n_out) / sfreq:.3f} s dropped to fill whole EDF data records")
    if n_out == 0:
        raise ValueError("recording shorter than one EDF data record")
    snip_start = 0
    if onset_s is not None:
        snip_start = int(max(0.0, onset_s - 2.0) * sfreq)
    snip_len = int(min(SNIPPET_S * sfreq, n_out))
    snip_start = min(snip_start, n_out - snip_len)
    vlen = int(min(VERIFY_S * sfreq, n_out))
    vstarts = sorted({0, max(0, n_out // 2 - vlen // 2), n_out - vlen})
    prefilter = (f"HP:{plan.highpass:g}Hz" if plan.highpass else "HP:DC") + \
                (f" LP:{plan.lowpass:g}Hz" if plan.lowpass else "") + \
                (" N:" + "/".join(f"{f:g}" for f in plan.notch) + "Hz" if plan.notch else "")
    signals = []
    snippets = np.zeros((len(chans), snip_len), dtype=np.float32)
    checks: list[tuple[int, int, np.ndarray]] = []
    for k, (ch, (ja, jb)) in enumerate(zip(chans, combos)):
        x = data[ja].astype(np.float64)
        if jb is not None:
            x -= data[jb]
            sd = float(np.std(x[qs:qe]))
            if sd < params.flat_uv:
                ch.status = "bad"
                ch.reasons.append(f"flat after montage (sd {sd:.2g} µV; contacts shorted?)")
        elif car is not None:
            x -= car
        y = filt(x)[:n_out]
        rng = E.channel_range(y)
        ch.limit, ch.step, ch.clipped = rng.limit, rng.step, rng.clipped
        signals.append(E.make_signal(y, sfreq, ch.name, rng, prefilter, ch.type))
        yc = np.clip(y, -rng.limit, rng.limit)
        snippets[k] = yc[snip_start:snip_start + snip_len]
        if params.verify:
            for vs in vstarts:
                checks.append((k, vs, yc[vs:vs + vlen].copy()))
    del data, car

    # ---- events -------------------------------------------------------------------------------
    ev_rows: list[tuple[float, float | None, str, str]] = []
    if onset_s is not None:
        dur = (sz_end - onset_s) if (sz_end is not None and sz_end > onset_s) else None
        ev_rows.append((onset_s, dur, "seizure", onset_source))
    if baseline is not None:
        ev_rows.append((baseline[0], baseline[1] - baseline[0], "baseline",
                        f"ieeg-prep: recording start to onset - {params.baseline_gap_s:g} s, "
                        f"at most {params.baseline_max_s:g} s"))
    for on, du, de in src.annotations:
        d = de.strip().lower()
        if d == "bad_acq_skip" and on >= n_out / sfreq - 1e-6:
            continue
        if events_in and d != "bad_acq_skip":
            continue  # MNE-BIDS already turned events.tsv into these annotations; taken from events.tsv below
        if d.startswith("bad") or "artifact" in d or "artefact" in d:
            ev_rows.append((on, du, "artifact", f"source annotation '{de}'"))
    for e in events_in:
        d = e.trial_type.lower()
        if "artifact" in d or "artefact" in d:
            ev_rows.append((e.onset, e.duration, "artifact", f"{e.source} ({e.trial_type})"))
    ev_rows.sort(key=lambda r: r[0])

    # ---- write --------------------------------------------------------------------------------
    rel_dir = Path(f"sub-{c.subject}") / (f"ses-{c.session}" if c.session else "") / "ieeg"
    out_dir = out_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    edf_name = f"{rid}_desc-preproc_ieeg.edf"
    startdate, starttime = None, None
    md = src.meas_date
    if isinstance(md, dt.datetime):
        startdate, starttime = md.date(), md.time().replace(microsecond=0, tzinfo=None)
    t_write = time.monotonic()
    E.write_edf(out_dir / edf_name, signals,
                [(on, du, tt) for on, du, tt, _ in ev_rows if tt in ("seizure", "baseline")],
                startdate, starttime, rec_s)
    del signals
    t_write = time.monotonic() - t_write
    verification = None
    if params.verify:
        verification = E.verify(out_dir / edf_name, checks, [ch.name for ch in chans], [ch.step for ch in chans])
        if verification["max_error_in_steps"] > 0.5 + 1e-3:
            raise ValueError(f"EDF read-back error {verification['max_abs_error_uv']:.4g} µV on "
                             f"{verification['worst_channel']} exceeds half a quantisation step")
    del checks

    ch_rows = []
    for ch in chans:
        # cutoffs of the data as written: ours combined with the source's (channels.tsv) when known
        src_rows = [ch_tsv.get(src_name[n]) for n in ch.contacts]
        lows = [r.low_cutoff for r in src_rows if r and r.low_cutoff is not None]
        highs = [r.high_cutoff for r in src_rows if r and r.high_cutoff is not None and r.high_cutoff > 0]
        lc = max([plan.highpass or 0.0] + lows)
        hc = min([plan.lowpass or sfreq / 2.0, sfreq / 2.0] + highs)
        ch_rows.append({
            "name": ch.name, "type": ch.type, "units": "µV", "low_cutoff": fmt(lc),
            "high_cutoff": fmt(hc), "sampling_frequency": fmt(sfreq), "status": ch.status,
            "status_description": "; ".join(ch.reasons), "contacts": "|".join(ch.contacts),
            **{col: tf(ch.labels.get(col)) for col in B.LABEL_COLUMNS},
            "x": fmt(ch.xyz[0]) if ch.xyz else "", "y": fmt(ch.xyz[1]) if ch.xyz else "",
            "z": fmt(ch.xyz[2]) if ch.xyz else "", "space": space if ch.xyz else "", "roi": ch.roi,
        })
    _write_tsv(out_dir / f"{rid}_desc-preproc_channels.tsv", list(ch_rows[0].keys()), ch_rows)
    _write_tsv(out_dir / f"{rid}_desc-preproc_events.tsv", ["onset", "duration", "trial_type", "source"],
               [{"onset": fmt(on), "duration": "n/a" if du is None else fmt(du), "trial_type": tt, "source": so}
                for on, du, tt, so in ev_rows])
    counts: dict[str, int] = {}
    for ch in chans:
        counts[ch.type] = counts.get(ch.type, 0) + 1
    ref_text = {"bipolar": "bipolar: each channel is the difference of two neighbouring contacts (first minus "
                           "second, see contacts in channels.tsv)",
                "car": "common average of the good channels",
                "as-recorded": ieeg_ref if ieeg_ref and ieeg_ref.lower() != "n/a" else "as recorded (unknown)"}[mode]
    js = {
        "TaskName": c.task or "ieeg",
        "SamplingFrequency": sfreq,
        "PowerLineFrequency": line_freq,
        "iEEGReference": ref_text,
        "SoftwareFilters": {
            **({"Highpass": {"CutoffFrequency": plan.highpass, "FilterType": "Butterworth",
                             "Order": dsp.HIGHPASS_ORDER, "ZeroPhase": True}} if plan.highpass else {}),
            **({"Notch": {"Frequencies": plan.notch, "FilterType": "IIR notch", "Q": dsp.NOTCH_Q,
                          "ZeroPhase": True}} if plan.notch else {}),
            **({"Lowpass": {"CutoffFrequency": plan.lowpass, "FilterType": "Butterworth",
                            "Order": dsp.LOWPASS_ORDER, "ZeroPhase": True}} if plan.lowpass else {}),
        } or "n/a",
        "RecordingDuration": round(n_out / sfreq, 6),
        "RecordingType": "continuous",
        "SEEGChannelCount": counts.get("SEEG", 0), "ECOGChannelCount": counts.get("ECOG", 0),
        "EEGChannelCount": counts.get("EEG", 0), "DBSChannelCount": counts.get("DBS", 0),
        "ECGChannelCount": 0, "EMGChannelCount": 0, "EOGChannelCount": 0, "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        **{k: sidecar[k] for k in ("Manufacturer", "ManufacturersModelName", "InstitutionName",
                                   "iEEGPlacementScheme", "iEEGGround", "ElectrodeManufacturer",
                                   "HardwareFilters", "RecordingType") if k in sidecar and k != "RecordingType"},
        "Sources": [c.origin],
        "PennsieveIeegPrep": {
            "montage": mode, "montageRequested": params.montage, "lineFrequencySource": lf_source,
            "sourceFormat": source_format(c), "reader": src.reader, "qcWindow": qc_window,
            "edfPhysicalRange": "per channel, symmetric ±1.1 × min(max|x|, 3 × p99.99|x|), at least 1 µV; "
                                "digital ±32767",
            "onsetSource": onset_source or None,
        },
    }
    import json
    with open(out_dir / f"{rid}_desc-preproc_ieeg.json", "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=2, ensure_ascii=False)

    # ---- electrodes (per subject/session, merged by the caller) -------------------------------
    el_rows: dict[str, dict[str, Any]] = {}
    for i in kept:
        n = out_name[i]
        k = C.norm_name(n)
        lab = contact_labels(i)
        xyz = coords.get(n)
        if not (xyz or any(lab.get(col) is not None for col in ("soz", "resected")) or k in elec.roi):
            continue
        el_rows[k] = {"name": n, "x": fmt(xyz[0]) if xyz else "", "y": fmt(xyz[1]) if xyz else "",
                      "z": fmt(xyz[2]) if xyz else "", "size": "n/a", "space": space if xyz else "",
                      "roi": elec.roi.get(k, ""), "soz": tf(lab.get("soz")), "resected": tf(lab.get("resected")),
                      **elec.extra.get(k, {})}

    # ---- qc rows ------------------------------------------------------------------------------
    qc_rows = []
    jmap = {i: j for j, i in enumerate(kept)}
    for i, cl in enumerate(classes):
        if not cl.keep:
            qc_rows.append({"recording_id": rid, "channel": names[i], "status": "dropped", "reason": cl.reason,
                            "line_length_z": "", "amplitude_z": "", "line_noise_ratio": "",
                            "dropped_type": cl.dropped_type, "amplitude_uv": "", "line_length_uv": ""})
            continue
        j = jmap[i]
        qc_rows.append({"recording_id": rid, "channel": out_name[i], "status": status[j],
                        "reason": "; ".join(reasons[j]), "line_length_z": fmt(ll_z[j], 2),
                        "amplitude_z": fmt(amp_z[j], 2), "line_noise_ratio": fmt(mets[j].line_noise_ratio, 4),
                        "dropped_type": "", "amplitude_uv": fmt(mets[j].amplitude, 2),
                        "line_length_uv": fmt(mets[j].line_length, 3)})
    for ch in chans:
        if ch.status == "bad" and len(ch.contacts) > 1:
            qc_rows.append({"recording_id": rid, "channel": ch.name, "status": "bad",
                            "reason": "; ".join(ch.reasons), "line_length_z": "", "amplitude_z": "",
                            "line_noise_ratio": "", "dropped_type": "", "amplitude_uv": "", "line_length_uv": ""})

    n_good = sum(1 for ch in chans if ch.status == "good")
    seconds = time.monotonic() - t0
    rec_row = {
        "recording_id": rid, "subject": c.subject, "session": c.session, "task": c.task, "run": c.run,
        "path": str(rel_dir / edf_name), "sfreq": fmt(sfreq), "n_channels": len(chans), "n_good": n_good,
        "duration_s": fmt(n_out / sfreq, 3), "montage": mode, "kind": kind,
        "onset_s": fmt(onset_s, 3) if onset_s is not None else "",
        "baseline_start_s": fmt(baseline[0], 3) if baseline else "",
        "baseline_end_s": fmt(baseline[1], 3) if baseline else "",
        "source_format": source_format(c),
    }
    dropped = {}
    for cl in classes:
        if not cl.keep:
            dropped[cl.dropped_type] = dropped.get(cl.dropped_type, 0) + 1
    details = {
        "input": c.origin, "reader": src.reader, "sfreq": sfreq,
        "source_channels": len(names), "kept_contacts": len(kept), "dropped": dropped,
        "bad_contacts": int(sum(1 for s_ in status if s_ == "bad")), "montage": mode,
        "output_channels": len(chans), "good_channels": n_good, "line_freq": line_freq,
        "line_freq_source": lf_source, "filters": plan.describe(), "onset_s": onset_s, "onset_source": onset_source,
        "kind": kind, "baseline": baseline, "qc_window": qc_window,
        "sidecars": {k: (_rel(v, input_dir) if v else None) for k, v in
                     (("channels", B.sidecar(c.path, "channels.tsv")), ("events", ev_path), ("ieeg", js_path),
                      ("electrodes", el_path), ("coordsystem", cs_path))},
        "labels_csv_rows": len(label_rows), "labels_known": sorted(
            col for col in B.LABEL_COLUMNS if any(ch.labels.get(col) is not None for ch in chans)),
        "clipped_samples": int(sum(ch.clipped for ch in chans)),
        "quantisation_step_uv": {"median": float(np.median([ch.step for ch in chans])),
                                 "max": float(max(ch.step for ch in chans))},
        "verification": verification, "notes": notes + pair_notes,
        "seconds": {"total": round(seconds, 2), "read": round(t_read, 2), "write": round(t_write, 2)},
    }
    plot = {"recording_id": rid, "sfreq": sfreq, "snippet": snippets, "snippet_start_s": snip_start / sfreq,
            "names": [ch.name for ch in chans], "status": [ch.status for ch in chans],
            "contact_status": [(row["channel"], row["status"]) for row in qc_rows[:len(classes)]],
            "onset_s": onset_s, "montage": mode, "kind": kind}
    return Result(rec_row, qc_rows, el_rows, ["name", "x", "y", "z", "size", "space", "roi", "soz", "resected"]
                  + elec.extra_columns, details, plot)


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def source_format(c: S.Candidate) -> str:
    base = {"edf": "edf", "bdf": "bdf", "brainvision": "brainvision", "fif": "fif", "mef3": "mef3"}[c.fmt]
    return f"bids-{base}" if c.bids else base


def _write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
