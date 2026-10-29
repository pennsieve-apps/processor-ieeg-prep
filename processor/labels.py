"""
labels.csv: expert or clinical labels supplied next to the recordings.

Columns: subject, recording_id (empty = every recording of the subject), channel (a contact
name; empty = a recording-level row), any of soz, resected, onset, spread (true/false), and
onset_s on recording-level rows (seizure onset in seconds from the recording start).

Any other column filled on a recording-level row (channel empty), e.g. `stim_induced` or `lvfa`,
is carried into recordings.csv as a column of the same name (lower-cased), value as written.

A label column that a recording (onset, spread) or subject (soz, resected) has any channel
rows for is complete for it: contacts without a row get `false`. Columns nobody filled stay
unknown (empty).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from processor.bids import LABEL_COLUMNS, parse_bool
from processor.channels import norm_name

log = logging.getLogger("processor-ieeg-prep")

PER_RECORDING = ("onset", "spread")
PER_SUBJECT = ("soz", "resected")
# Columns with a meaning of their own; every other column of a recording-level row is carried
# into recordings.csv as is.
KNOWN_COLUMNS = {"subject", "recording_id", "channel", "onset_s", *LABEL_COLUMNS}
NA = ("", "n/a", "na", "nan")


def norm_subject(s: str) -> str:
    s = s.strip()
    return s[4:] if s.lower().startswith("sub-") else s


def norm_recording(r: str) -> str:
    """A recording id from what a user may write: the stem, a file name, or a path."""
    r = r.strip().replace("\\", "/").split("/")[-1]
    for ext in (".edf", ".edf+", ".bdf", ".vhdr", ".fif", ".mefd", ".fif.gz"):
        if r.lower().endswith(ext):
            r = r[: -len(ext)]
    for suf in ("_ieeg", "_eeg"):
        if r.endswith(suf):
            r = r[: -len(suf)]
    return r


@dataclass
class LabelRow:
    subject: str
    recording: str
    channel: str
    values: dict[str, bool | None]
    onset_s: float | None
    line: int
    extra: dict[str, str] = field(default_factory=dict)  # recording-level rows: other columns


@dataclass
class LabelSet:
    path: Path | None = None
    rows: list[LabelRow] = field(default_factory=list)
    used: set[int] = field(default_factory=set)
    extra_columns: list[str] = field(default_factory=list)  # filled on a recording-level row, in file order

    def for_recording(self, subject: str, recording_ids: set[str]) -> list[LabelRow]:
        sub = norm_subject(subject)
        out = []
        for r in self.rows:
            if r.subject and norm_subject(r.subject) != sub:
                continue
            if r.recording and r.recording not in recording_ids:
                continue
            if not r.subject and not r.recording:
                continue
            out.append(r)
        return out


def read_labels(path: Path | None) -> LabelSet:
    if path is None:
        return LabelSet()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "subject" not in header and "recording_id" not in header:
            raise ValueError(f"{path.name} needs a subject or recording_id column (columns: {', '.join(header)})")
        rows = []
        for i, raw in enumerate(reader, start=2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if k is not None}
            onset = None
            if r.get("onset_s", "") not in ("", "n/a", "NA", "nan"):
                try:
                    onset = float(r["onset_s"])
                except ValueError:
                    log.warning("%s line %d: onset_s %r is not a number; ignored", path.name, i, r["onset_s"])
            channel = r.get("channel", "")
            extra = {} if channel else {k: v for k, v in r.items()
                                        if k and k not in KNOWN_COLUMNS and v.lower() not in NA}
            rows.append(LabelRow(subject=r.get("subject", ""), recording=norm_recording(r.get("recording_id", "")),
                                 channel=channel,
                                 values={c: parse_bool(r.get(c)) for c in LABEL_COLUMNS if c in r},
                                 onset_s=onset, line=i, extra=extra))
    filled = {k for r in rows for k in r.extra}
    return LabelSet(path=path, rows=rows, extra_columns=[h for h in header if h in filled])


@dataclass
class RecordingLabels:
    """Labels resolved for one recording: per contact (normalised name) and the onset."""
    contacts: dict[str, dict[str, bool | None]] = field(default_factory=dict)
    complete: set[str] = field(default_factory=set)  # label columns known for every contact
    onset_s: float | None = None
    onset_source: str = ""
    extra: dict[str, str] = field(default_factory=dict)  # recording-level columns for recordings.csv
    unmatched_channels: list[str] = field(default_factory=list)


def resolve(rows: list[LabelRow], contact_names: list[str], file_name: str) -> RecordingLabels:
    """Apply labels.csv rows to the contacts of one recording."""
    keys = {norm_name(n): n for n in contact_names}
    out = RecordingLabels()
    filled: set[str] = set()
    for r in rows:
        if not r.channel:
            for k, v in r.extra.items():
                if k in out.extra and out.extra[k] != v:
                    log.warning("%s line %d: two %s values for this recording (%r, %r); using the first",
                                file_name, r.line, k, out.extra[k], v)
                else:
                    out.extra[k] = v
            if r.onset_s is not None:
                if out.onset_s is not None and abs(out.onset_s - r.onset_s) > 1e-6:
                    log.warning("%s: two onset_s rows for this recording (%g, %g); using the first",
                                file_name, out.onset_s, r.onset_s)
                else:
                    out.onset_s = r.onset_s
                    out.onset_source = f"{file_name} line {r.line}"
            continue
        k = norm_name(r.channel)
        if k not in keys:
            out.unmatched_channels.append(r.channel)
        for c, v in r.values.items():
            if v is None:
                continue
            out.contacts.setdefault(k, {})[c] = v
            filled.add(c)
    for c in filled:
        out.complete.add(c)
        for k in keys:
            out.contacts.setdefault(k, {}).setdefault(c, False)
    return out
