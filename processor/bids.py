"""
BIDS sidecars of a source recording: _channels.tsv, _events.tsv, _ieeg.json,
_electrodes.tsv / _coordsystem.json (with the label columns soz, resected, onset,
spread), found next to the recording or up the BIDS inheritance chain.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from processor.channels import norm_name

log = logging.getLogger("processor-ieeg-prep")

ENTITY_ORDER = ("sub", "ses", "task", "acq", "run", "proc", "space", "rec", "split")
LABEL_COLUMNS = ("soz", "resected", "onset", "spread")
_TRUE = {"true", "1", "yes", "y", "t", "x", "soz", "resected", "onset", "spread"}
_FALSE = {"false", "0", "no", "n", "f", ""}
NA = {"n/a", "na", "nan", "none", ""}


def parse_entities(name: str) -> dict[str, str]:
    """BIDS entities of a file name: {'sub': 'HUP224', 'ses': 'postimplant', ...} (empty when not BIDS)."""
    stem = name.split(".")[0]
    parts = stem.split("_")
    ents: dict[str, str] = {}
    for p in parts:
        if "-" in p:
            k, v = p.split("-", 1)
            if re.fullmatch(r"[a-z]+", k) and v:
                ents[k] = v
    return ents if "sub" in ents else {}


def stem_of(path: Path) -> str:
    """The entity chain before the suffix: sub-X_ses-Y_task-Z_run-01 for sub-X_ses-Y_task-Z_run-01_ieeg.edf."""
    name = path.name
    if name.endswith(".mefd"):
        name = name[:-5]
    base = name.split(".")[0]
    if "_" in base and base.rsplit("_", 1)[1] in ("ieeg", "eeg", "meg"):
        base = base.rsplit("_", 1)[0]
    return base


def parse_bool(v: str | None) -> bool | None:
    """'true'/'1'/'yes' -> True, 'false'/'0'/'no' -> False, n/a/empty -> None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("n/a", "na", "nan", "none", ""):
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    try:
        return float(s) != 0
    except ValueError:
        return None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]


def read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("%s: unreadable JSON (%s)", path, exc)
        return {}


def sidecar(rec_path: Path, suffix_ext: str, all_files: list[Path] | None = None) -> Path | None:
    """
    The sidecar `<stem>_<suffix_ext>` of a recording, or one up the inheritance chain (fewer
    entities, same directory or a parent up to the subject folder).
    """
    stem = stem_of(rec_path)
    d = rec_path.parent
    exact = d / f"{stem}_{suffix_ext}"
    if exact.is_file():
        return exact
    ents = parse_entities(rec_path.name)
    candidates: list[Path] = []
    for folder in (d, d.parent, d.parent.parent):
        if not folder.is_dir():
            continue
        for f in folder.glob(f"*_{suffix_ext}"):
            fe = parse_entities(f.name)
            if fe and all(ents.get(k) == v for k, v in fe.items() if k not in ("space",)):
                candidates.append(f)
        if candidates:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda p: -len(parse_entities(p.name)))
    return candidates[0]


@dataclass
class ChannelRow:
    name: str
    type: str = ""
    status: str = ""
    status_description: str = ""
    units: str = ""
    low_cutoff: float | None = None
    high_cutoff: float | None = None
    labels: dict[str, bool | None] = field(default_factory=dict)


def _num(v: str | None) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def read_channels(path: Path | None) -> dict[str, ChannelRow]:
    if path is None:
        return {}
    out = {}
    for r in read_tsv(path):
        name = r.get("name", "")
        if not name:
            continue
        out[name] = ChannelRow(
            name=name, type=r.get("type", "").upper(), status=r.get("status", "").lower(),
            status_description="" if r.get("status_description", "").lower() in NA else r.get("status_description", ""),
            units=r.get("units", ""), low_cutoff=_num(r.get("low_cutoff")), high_cutoff=_num(r.get("high_cutoff")),
            labels={c: parse_bool(r.get(c)) for c in LABEL_COLUMNS if c in r},
        )
    return out


@dataclass
class Event:
    onset: float
    duration: float | None
    trial_type: str
    source: str


SEIZURE_TYPES = {"ictal", "seizure", "sz", "seizure onset", "seizure_onset", "onset", "eeg onset", "ueo"}


def read_events(path: Path | None) -> list[Event]:
    if path is None:
        return []
    out = []
    for r in read_tsv(path):
        try:
            onset = float(r.get("onset", ""))
        except ValueError:
            continue
        try:
            dur = float(r.get("duration", ""))
        except ValueError:
            dur = None
        tt = r.get("trial_type") or r.get("value") or ""
        out.append(Event(onset, dur, tt, path.name))
    return out


@dataclass
class Electrodes:
    path: Path | None = None
    space: str = ""
    coords: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # norm name -> xyz
    roi: dict[str, str] = field(default_factory=dict)                             # norm name -> roi
    labels: dict[str, dict[str, bool | None]] = field(default_factory=dict)      # norm name -> {soz: ...}
    names: dict[str, str] = field(default_factory=dict)                          # norm name -> name as written
    extra_columns: list[str] = field(default_factory=list)                       # other columns, carried through
    extra: dict[str, dict[str, str]] = field(default_factory=dict)               # norm name -> {column: value}


def find_electrodes(rec_path: Path, subject_dir: Path | None) -> tuple[Path | None, Path | None]:
    """
    (electrodes.tsv, coordsystem.json) for a recording: the BIDS sidecar when there is one;
    otherwise any *electrodes.tsv under the subject folder (some datasets, e.g. Discover 530,
    keep it in sub-X/derivatives/). An MNI space is preferred when several exist.
    """
    p = sidecar(rec_path, "electrodes.tsv")
    found: list[Path] = [p] if p else []
    if not found and subject_dir is not None and subject_dir.is_dir():
        found = sorted(subject_dir.rglob("*electrodes.tsv"))
    if not found:
        return None, None
    found.sort(key=lambda f: (0 if "mni" in f.name.lower() else 1, len(f.parts), str(f)))
    el = found[0]
    cs = None
    ents = parse_entities(el.name)
    for f in sorted(el.parent.glob("*coordsystem.json")):
        fe = parse_entities(f.name)
        if not ents or fe.get("space") == ents.get("space"):
            cs = f
            break
    return el, cs


def read_electrodes(path: Path | None, coordsystem: Path | None) -> Electrodes:
    if path is None:
        return Electrodes()
    rows = read_tsv(path)
    if not rows:
        return Electrodes(path=path)
    cols = rows[0].keys()
    name_col = next((c for c in ("name", "label", "channel", "contact") if c in cols), None)
    xyz = None
    for trip in (("x", "y", "z"), ("mni_x", "mni_y", "mni_z"), ("mm_x", "mm_y", "mm_z")):
        if all(c in cols for c in trip):
            xyz = trip
            break
    space = ""
    ents = parse_entities(path.name)
    if ents.get("space"):
        space = ents["space"]
    if coordsystem is not None:
        cs = read_json(coordsystem)
        sys_name = str(cs.get("iEEGCoordinateSystem") or "")
        if sys_name and (sys_name.lower() != "other" or not space):
            space = sys_name
    if not space and xyz == ("mni_x", "mni_y", "mni_z"):
        space = "MNI152"
    if not space and xyz == ("mm_x", "mm_y", "mm_z"):
        space = "native"  # Discover 530: "X coordinate in native space"; see electrodes.json
    roi_col = next((c for c in ("roi", "region", "atlas_label", "label_dkt", "aal") if c in cols and c != name_col), None)
    e = Electrodes(path=path, space=space)
    if name_col is None:
        return e
    skip = {name_col, roi_col, "x", "y", "z", "size", "space", "type", "material", "manufacturer", "group",
            "hemisphere", *LABEL_COLUMNS, *(xyz or ())}
    e.extra_columns = [c for c in cols if c not in skip and not re.search(r"(^|_)(x|y|z)$", c)]
    for r in rows:
        n = r.get(name_col, "")
        if not n:
            continue
        k = norm_name(n)
        e.names[k] = n
        if xyz:
            try:
                e.coords[k] = tuple(float(r[c]) for c in xyz)  # type: ignore[assignment]
            except (ValueError, KeyError):
                pass
        if roi_col and r.get(roi_col, "").lower() not in NA:
            e.roi[k] = r[roi_col]
        if e.extra_columns:
            e.extra[k] = {c: r.get(c, "") for c in e.extra_columns}
        lab = {c: parse_bool(r.get(c)) for c in LABEL_COLUMNS if c in r}
        if lab:
            e.labels[k] = lab
    return e
