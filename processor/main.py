"""
processor-ieeg-prep: the first step of the epilepsy iEEG workflows.

Finds every recording under INPUT_DIR (BIDS-iEEG, EDF/EDF+/BDF, BrainVision, FIF, MEF3) and
an optional labels.csv, and writes one BIDS-iEEG derivative:

  OUTPUT_DIR/ieeg/dataset_description.json
  OUTPUT_DIR/ieeg/recordings.csv
  OUTPUT_DIR/ieeg/sub-<label>/[ses-<label>/]ieeg/<stem>_desc-preproc_{ieeg.edf,ieeg.json,channels.tsv,events.tsv}
  OUTPUT_DIR/ieeg/sub-<label>/[ses-<label>/]ieeg/sub-<label>[_ses-<label>]_desc-preproc_electrodes.tsv
  OUTPUT_DIR/ieeg/provenance.json
  OUTPUT_DIR/qc/qc.csv, OUTPUT_DIR/qc/<recording_id>.png, OUTPUT_DIR/qc/provenance.json

One recording at a time; a recording that fails is logged and skipped.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import resource
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from processor import labels as L
from processor import plots
from processor import prep as P
from processor import sources as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("processor-ieeg-prep")
log.setLevel(logging.INFO)
for noisy in ("mne", "mne_bids", "matplotlib", "pymef"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

APP_NAME = "processor-ieeg-prep"
REPO_URL = "https://github.com/pennsieve-apps/processor-ieeg-prep"
REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING_COLUMNS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
                     "duration_s", "montage", "kind", "onset_s", "baseline_start_s", "baseline_end_s",
                     "source_format"]
QC_COLUMNS = ["recording_id", "channel", "status", "reason", "line_length_z", "amplitude_z", "line_noise_ratio",
              "dropped_type", "amplitude_uv", "line_length_uv"]


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("mne", "mne-bids", "edfio", "pymef", "numpy", "scipy", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def env_params() -> dict[str, str]:
    return {
        "RECORDINGS": env("RECORDINGS", "*"),
        "LABELS_FILE": env("LABELS_FILE", "labels.csv"),
        "MONTAGE": env("MONTAGE", "bipolar").lower(),
        "LINE_FREQ": env("LINE_FREQ", "auto").lower(),
        "HIGHPASS": env("HIGHPASS", "0.5"),
        "LOWPASS": env("LOWPASS", "0"),
        "INCLUDE_SCALP": env("INCLUDE_SCALP", "false").lower(),
        "IEEG_TYPE": env("IEEG_TYPE", "seeg").lower(),
        "FLAT_UV": env("FLAT_UV", "0.5"),
        "BAD_Z": env("BAD_Z", "5"),
        "LINE_NOISE_MAX": env("LINE_NOISE_MAX", "0.5"),
        "BASELINE_GAP_S": env("BASELINE_GAP_S", "10"),
        "BASELINE_MAX_S": env("BASELINE_MAX_S", "90"),
        "VERIFY_EDF": env("VERIFY_EDF", "true").lower(),
    }


def parse_params(raw: dict[str, str]) -> P.Params:
    def num(k: str, lo: float | None = None) -> float:
        try:
            v = float(raw[k])
        except ValueError:
            raise ValueError(f"{k} must be a number (got {raw[k]!r})") from None
        if lo is not None and v < lo:
            raise ValueError(f"{k} must be >= {lo:g} (got {v:g})")
        return v

    def flag(k: str) -> bool:
        if raw[k] not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"{k} must be true or false (got {raw[k]!r})")
        return raw[k] in ("true", "1", "yes")

    if raw["MONTAGE"] not in ("bipolar", "car", "as-recorded", "auto"):
        raise ValueError(f"MONTAGE must be bipolar, car, as-recorded or auto (got {raw['MONTAGE']!r})")
    if raw["LINE_FREQ"] != "auto":
        if num("LINE_FREQ", 0) not in (0.0, 50.0, 60.0):
            log.warning("LINE_FREQ=%s is neither 50 nor 60 Hz", raw["LINE_FREQ"])
    if raw["IEEG_TYPE"] not in ("seeg", "ecog"):
        raise ValueError(f"IEEG_TYPE must be seeg or ecog (got {raw['IEEG_TYPE']!r})")
    return P.Params(
        montage=raw["MONTAGE"], line_freq=raw["LINE_FREQ"], highpass=num("HIGHPASS", 0), lowpass=num("LOWPASS", 0),
        include_scalp=flag("INCLUDE_SCALP"), ieeg_type=raw["IEEG_TYPE"].upper(), flat_uv=num("FLAT_UV", 0),
        bad_z=num("BAD_Z", 0), line_noise_max=num("LINE_NOISE_MAX", 0), baseline_gap_s=num("BASELINE_GAP_S", 0),
        baseline_max_s=num("BASELINE_MAX_S", 0), verify=flag("VERIFY_EDF"),
        mef_password=os.environ.get("MEF_PASSWORD", ""),
    )


def write_csv(path: Path, header: list[str], rows: list[dict], delimiter: str = ",") -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", lineterminator="\n", delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)


def find_labels(files: list[Path], name: str, input_dir: Path) -> Path | None:
    if not name:
        return None
    hits = [p for p in files if p.name == name]
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.relative_to(input_dir).parts), str(p)))
    if len(hits) > 1:
        log.warning("%d files named %s; using %s", len(hits), name, hits[0].relative_to(input_dir))
    return hits[0]


def source_datasets(files: list[Path]) -> list[dict]:
    out = []
    for p in files:
        if p.name == "dataset_description.json":
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if d.get("DatasetType") == "derivative" and any(g.get("Name") == "ieeg-prep" for g in
                                                             d.get("GeneratedBy", []) if isinstance(g, dict)):
                continue
            entry = {k: d[k] for k in ("Name", "DatasetDOI") if d.get(k)}
            if d.get("DatasetDOI"):
                doi = str(d["DatasetDOI"])
                entry["DOI"] = doi if doi.startswith("doi:") else f"doi:{doi.split('doi.org/')[-1]}"
            if entry:
                out.append(entry)
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    raw_params = env_params()
    try:
        params = parse_params(raw_params)
    except ValueError as exc:
        log.error("bad parameter: %s", exc)
        return 2

    ieeg_dir, qc_dir = output_dir / "ieeg", output_dir / "qc"
    ieeg_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    app_version = version_of(REPO_ROOT)
    rel = lambda p: str(p.relative_to(input_dir))  # noqa: E731

    candidates, files, notes = S.find_recordings(input_dir, raw_params["RECORDINGS"])
    for n in notes:
        log.info(n)
    labels_path = find_labels(files, raw_params["LABELS_FILE"], input_dir)
    try:
        labelset = L.read_labels(labels_path)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        log.error("cannot use %s: %s", labels_path, exc)
        return 1
    log.info("input: %d recording(s) (%s), labels: %s", len(candidates),
             ", ".join(f"{k} {v}" for k, v in Counter(c.fmt for c in candidates).items()) or "none",
             f"{rel(labels_path)} ({len(labelset.rows)} rows)" if labels_path else "none")
    if not candidates:
        listing = [rel(p) for p in files]
        log.error("no recording (EDF, BDF, BrainVision .vhdr, FIF, MEF3 .mefd) under INPUT_DIR; it holds %d file(s): %s",
                  len(listing), ", ".join(listing[:40]) + (" …" if len(listing) > 40 else ""))
        return 1

    ids = Counter(c.recording_id for c in candidates)
    skipped: list[dict] = []
    for c in candidates:
        if ids[c.recording_id] > 1:
            log.error("%s: recording id %s is used by %d inputs, skipped", c.origin, c.recording_id,
                      ids[c.recording_id])
            skipped.append({"input": c.origin, "recording_id": c.recording_id, "reason": "duplicate recording id"})
    candidates = [c for c in candidates if ids[c.recording_id] == 1]

    rec_rows: list[dict] = []
    qc_rows: list[dict] = []
    electrodes: dict[tuple[str, str], dict[str, Any]] = {}
    el_cols: dict[tuple[str, str], list[str]] = {}
    details: dict[str, Any] = {}
    prov_recs: list[dict] = []
    for c in candidates:
        try:
            res = P.process(c, params, labelset, ieeg_dir, app_version, input_dir)
        except Exception as exc:  # noqa: BLE001 - one recording's failure must not sink the run
            log.error("%s: FAILED: %s\n%s", c.origin, exc, traceback.format_exc())
            skipped.append({"input": c.origin, "recording_id": c.recording_id,
                            "reason": f"{type(exc).__name__}: {exc}"})
            prov_recs.append({"recording_id": c.recording_id, "input": c.origin, "status": "failed",
                              "error": str(exc)})
            continue
        rec_rows.append(res.recording_row)
        qc_rows.extend(res.qc_rows)
        key = (c.subject, c.session)
        electrodes.setdefault(key, {}).update(res.electrodes)
        cols = el_cols.setdefault(key, [])
        cols.extend(x for x in res.electrode_columns if x not in cols)
        d = res.details
        details[c.recording_id] = d
        try:
            plots.qc_figure(res.plot, qc_dir / f"{c.recording_id}.png")
        except Exception as exc:  # noqa: BLE001 - a plot must not fail the recording
            log.warning("%s: QC figure failed: %s", c.recording_id, exc)
        v = d.get("verification") or {}
        prov_recs.append({"recording_id": c.recording_id, "input": c.origin, "status": "ok",
                          "output": res.recording_row["path"], "seconds": d["seconds"]["total"]})
        log.info("%s: %d source ch -> %d kept (%d bad, dropped %s) -> %s %d ch (%d good); %s; onset %s; "
                 "line %g Hz; EDF read-back max error %.3g µV (%.2f steps); %.1fs (read %.1fs, write %.1fs); "
                 "peak RSS %.0f MB",
                 c.recording_id, d["source_channels"], d["kept_contacts"], d["bad_contacts"],
                 dict(d["dropped"]) or "none", d["montage"], d["output_channels"], d["good_channels"], d["kind"],
                 "n/a" if d["onset_s"] is None else f"{d['onset_s']:.2f} s", d["line_freq"],
                 v.get("max_abs_error_uv", float("nan")), v.get("max_error_in_steps", float("nan")),
                 d["seconds"]["total"], d["seconds"]["read"], d["seconds"]["write"], peak_rss_mb())
        for n in d["notes"]:
            log.info("%s: %s", c.recording_id, n)

    # electrodes.tsv per subject/session
    for (sub, ses), rows in electrodes.items():
        if not rows:
            continue
        stem = f"sub-{sub}" + (f"_ses-{ses}" if ses else "")
        d = ieeg_dir / f"sub-{sub}" / (f"ses-{ses}" if ses else "") / "ieeg"
        d.mkdir(parents=True, exist_ok=True)
        write_csv(d / f"{stem}_desc-preproc_electrodes.tsv", el_cols[(sub, ses)], list(rows.values()), "\t")

    rec_rows.sort(key=lambda r: r["recording_id"])
    write_csv(ieeg_dir / "recordings.csv", RECORDING_COLUMNS, rec_rows)
    write_csv(qc_dir / "qc.csv", QC_COLUMNS, qc_rows)
    finished = datetime.now(timezone.utc)
    dd = {
        "Name": "ieeg-prep derivative",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "ieeg-prep", "Version": app_version, "CodeURL": REPO_URL,
                         "Description": "channel selection, bad channels, montage, notch/high-pass filters, labels"}],
        "SourceDatasets": source_datasets(files) or [{"URL": "n/a"}],
    }
    with open(ieeg_dir / "dataset_description.json", "w", encoding="utf-8") as fh:
        json.dump(dd, fh, indent=2, ensure_ascii=False)
    provenance = {
        "app": APP_NAME, "appVersion": app_version, "tools": tool_versions(),
        "startedAt": started.isoformat(timespec="seconds"), "finishedAt": finished.isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t_start, 2), "peakRssMB": round(peak_rss_mb(), 1),
        "parameters": raw_params,
        "inputsFound": {"recordings": [c.origin for c in candidates] + [s["input"] for s in skipped
                                                                          if s["reason"] == "duplicate recording id"],
                        "labels": rel(labels_path) if labels_path else None},
        "recordings": prov_recs, "skipped": skipped, "details": details,
    }
    for d in (ieeg_dir, qc_dir):
        with open(d / "provenance.json", "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2, ensure_ascii=False, default=str)
    log.info("%d of %d recording(s) prepared, %d skipped, %.1fs, peak RSS %.0f MB; wrote %s and %s", len(rec_rows),
             len(rec_rows) + len(skipped), len(skipped), time.monotonic() - t_start, peak_rss_mb(), ieeg_dir, qc_dir)
    if not rec_rows:
        log.error("every recording failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
