"""
Channel names and types: which channels are intracranial, which are scalp EEG,
cardiac, muscle, eye, trigger or bookkeeping channels, how contact names split
into an electrode and a contact number, and which contacts are neighbours.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# BIDS channels.tsv `type` values (upper case) -> what we do with them.
IEEG_TYPES = {"SEEG", "ECOG", "DBS"}
SCALP_TYPES = {"EEG"}
DROP_TYPES = {
    "ECG": "ecg", "EKG": "ecg", "EMG": "emg", "EOG": "eog", "VEOG": "eog", "HEOG": "eog",
    "TRIG": "trigger", "STIM": "trigger", "MISC": "misc", "RESP": "misc", "SYSCLOCK": "misc",
    "PPG": "misc", "GSR": "misc", "TEMP": "misc", "AUDIO": "misc", "PD": "misc", "EYEGAZE": "eog",
    "PUPIL": "eog", "REF": "misc", "ADC": "misc", "DAC": "misc", "HLU": "misc", "FITERR": "misc",
    "OTHER": "misc", "BIO": "misc", "CHPI": "misc", "EXCI": "misc", "IAS": "misc", "SYST": "misc",
}
# MNE channel types -> BIDS types
MNE_TO_BIDS = {
    "seeg": "SEEG", "ecog": "ECOG", "dbs": "DBS", "eeg": "EEG", "ecg": "ECG", "emg": "EMG",
    "eog": "EOG", "stim": "TRIG", "misc": "MISC", "resp": "RESP", "bio": "BIO", "syst": "SYST",
    "temperature": "TEMP", "gsr": "GSR",
}

# Name patterns of channels that are never brain signal. Checked case-insensitively against
# the whole name (after stripping "EEG " / "POL " prefixes some exporters add).
_DROP_NAME_PATTERNS: list[tuple[str, str, bool]] = [
    # (pattern, type, strong): strong patterns always drop; weak ones (short names that a depth
    # electrode could also have, e.g. X1, LAT3, E2) only drop when the channel is not one of
    # four or more contacts of an electrode with that name and the source does not type it as iEEG.
    (r"(EKG|ECG)[ _-]*[LR]?\d*[ _-]*([+-]|REF)?", "ecg", True),
    (r"(EMG)[ _-]*[LR]?\d*([+-])?", "emg", True),
    (r"(CHIN|LEG|LLEG|RLEG|LAT|RAT)[ _-]*[LR]?\d*([+-])?", "emg", False),
    (r"(EOG|HEOG|VEOG)[ _-]*\d*", "eog", True),
    (r"(LOC|ROC|LUE|RAE|E[12])", "eog", False),
    (r"(DC|TRIG|TRIGGER|STATUS|EVENT|EVENTS|MARKER|PHOTIC|PHOTIC-REF|IBI|BURSTS|SUPPR|PATIENT EVENT)[ _-]*\d*",
     "trigger", True),
    (r"(STI|STIM|MARK)[ _-]*\d*", "trigger", False),
    (r"(EDF )?ANNOTATIONS?", "misc", True),
    (r"(SPO2|SAO2|OSAT|PULSE|PLETH|AIRFLOW|SNORE|ETCO2|NIBP)[ _-]*\d*", "misc", True),
    (r"(PR|HR|RESP|RR|THOR|ABD|CO2|BP|TEMP|POS|POSITION|LIGHT|SOUND|MIC|X|Y|Z|ACC[XYZ]?|GYRO[XYZ]?)[ _-]*\d*",
     "misc", False),
    (r"(EMPTY|NONE|N/?A|UNUSED|-+|\.+|NOT USED|BLANK)[ _-]*\d*", "empty", True),
    (r"(NC|EXTRA|SPARE|OFF)[ _-]*\d*", "empty", False),
    (r"(REF|GND|GROUND|COM|VREF|CREF|REFERENCE)[ _-]*\d*", "misc", False),
]
_DROP_RES = [(re.compile(r"^(?:" + p + r")$", re.IGNORECASE), t, strong) for p, t, strong in _DROP_NAME_PATTERNS]
_PREFIX_RE = re.compile(r"^(EEG|POL|SEEG|ECOG|IEEG)[ _-]+", re.IGNORECASE)
_REF_SUFFIX_RE = re.compile(r"[ _-]*(-?REF|-?AVG|-?LE|-?AR)$", re.IGNORECASE)

# 10-20 / 10-10 scalp positions often recorded alongside iEEG (HUP: C3, C4, CZ, FZ ...).
SCALP_NAMES = {
    n.upper() for n in (
        "Fp1 Fp2 Fpz F3 F4 F7 F8 Fz C3 C4 Cz P3 P4 P7 P8 Pz O1 O2 Oz T3 T4 T5 T6 T7 T8 A1 A2 M1 M2 "
        "AF3 AF4 AF7 AF8 AFz F1 F2 F5 F6 F9 F10 FC1 FC2 FC3 FC4 FC5 FC6 FCz FT7 FT8 FT9 FT10 C1 C2 C5 C6 "
        "CP1 CP2 CP3 CP4 CP5 CP6 CPz TP7 TP8 TP9 TP10 P1 P2 P5 P6 P9 P10 PO3 PO4 PO7 PO8 POz Iz"
    ).split()
}

# <letters (and ' for left-hemisphere SEEG naming)> <contact number>, e.g. LA01, LA1, A'3, RAH 12, G_10
_CONTACT_RE = re.compile(r"^([A-Za-z][A-Za-z'_ ]*?)[ _]*0*(\d+)$")
_BIPOLAR_RE = re.compile(r"^(.+?)\s*-\s*(.+)$")


def clean_name(name: str) -> str:
    """Name without exporter prefixes ('EEG ', 'POL ') and reference suffixes ('-Ref')."""
    n = _PREFIX_RE.sub("", name.strip())
    return _REF_SUFFIX_RE.sub("", n).strip()


def norm_name(name: str) -> str:
    """Key for matching contact names across files: upper case, no spaces/underscores, no leading zeros."""
    n = clean_name(name).upper().replace(" ", "").replace("_", "")
    m = re.match(r"^(.*?[A-Z'])0*(\d+)$", n)
    if m:
        return m.group(1) + str(int(m.group(2)))
    return n


def split_contact(name: str) -> tuple[str, int] | None:
    """('LA', 1) for 'LA01'; None when the name is not <letters><number>."""
    m = _CONTACT_RE.match(clean_name(name))
    if not m:
        return None
    return m.group(1).strip().upper(), int(m.group(2))


def drop_type_by_name(name: str) -> tuple[str, bool] | None:
    """(non-brain type the name implies, strong?) or None. Types: ecg, emg, eog, trigger, misc, empty."""
    raw = name.strip()
    if not raw:
        return "empty", True
    n = clean_name(raw)
    for rx, t, strong in _DROP_RES:
        if rx.match(n) or rx.match(raw):
            return t, strong
    return None


def scalp_names(names: list[str]) -> set[str]:
    """
    Names that are scalp 10-20/10-10 positions in a recording with intracranial contacts.
    A name like C3 counts as scalp only when its electrode letters (C) have no contacts in the
    file that are not scalp positions, and at most 4 contacts: a depth or strip electrode named
    "C" with contacts C1..C8 keeps its C3 and C4.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        sc = split_contact(n)
        groups[sc[0] if sc else clean_name(n).upper()].append(n)
    out = set()
    for n in names:
        cn = clean_name(n).upper()
        if cn not in SCALP_NAMES:
            continue
        sc = split_contact(n)
        if sc is None:  # Fz, Cz, Pz, Oz, Fpz...
            out.add(n)
            continue
        members = groups[sc[0]]
        if len(members) <= 4 and all(clean_name(m).upper() in SCALP_NAMES for m in members):
            out.add(n)
    return out


@dataclass
class ChannelClass:
    name: str
    type: str            # BIDS type written out (SEEG, ECOG, EEG, ...)
    keep: bool
    dropped_type: str    # '' when kept; else ecg/emg/eog/trigger/misc/empty/scalp/<type>
    reason: str          # why dropped / how the type was decided


def classify(names: list[str], source_types: dict[str, str], default_ieeg: str, include_scalp: bool,
             types_trusted: bool) -> list[ChannelClass]:
    """
    Decide per channel whether it is kept. `source_types` are BIDS type strings from the
    source (channels.tsv, or MNE's types for FIF); `types_trusted` is False for formats that do
    not record types (EDF, BrainVision, MEF3), where every channel arrives as "EEG" and is taken
    to be `default_ieeg` unless its name says otherwise.
    """
    scalp = scalp_names(names)
    group_size: dict[str, int] = defaultdict(int)
    for n in names:
        sc = split_contact(n)
        if sc:
            group_size[sc[0]] += 1
    out = []
    for n in names:
        st = (source_types.get(n) or "").upper()
        hit = drop_type_by_name(n)
        by_name = None
        if hit:
            sc = split_contact(n)
            on_electrode = sc is not None and group_size[sc[0]] >= 4
            if hit[1] or not (on_electrode or (types_trusted and st in IEEG_TYPES)):
                by_name = hit[0]
        if types_trusted and st in DROP_TYPES:
            out.append(ChannelClass(n, st, False, DROP_TYPES[st], f"type {st}"))
            continue
        if by_name:
            out.append(ChannelClass(n, st or "MISC", False, by_name, f"name '{n}' looks like {by_name}"))
            continue
        if n in scalp:
            if include_scalp:
                out.append(ChannelClass(n, "EEG", True, "", "scalp 10-20 name, INCLUDE_SCALP=true"))
            else:
                out.append(ChannelClass(n, "EEG", False, "scalp", f"scalp 10-20 name '{n}' in an iEEG recording"))
            continue
        if types_trusted and st in IEEG_TYPES:
            out.append(ChannelClass(n, st, True, "", f"type {st}"))
        elif types_trusted and st in SCALP_TYPES:
            if include_scalp:
                out.append(ChannelClass(n, "EEG", True, "", "type EEG, INCLUDE_SCALP=true"))
            else:
                out.append(ChannelClass(n, "EEG", False, "scalp", "type EEG (scalp); INCLUDE_SCALP=false"))
        elif types_trusted and st:
            out.append(ChannelClass(n, st, False, st.lower(), f"type {st}"))
        else:
            out.append(ChannelClass(n, default_ieeg, True, "", f"no type in the source; taken as {default_ieeg}"))
    return out


def looks_bipolar(names: list[str]) -> bool:
    """True when most names are already pairs, e.g. 'LA01-LA02'."""
    if not names:
        return False
    hits = 0
    for n in names:
        m = _BIPOLAR_RE.match(n)
        if m and split_contact(m.group(1)) and split_contact(m.group(2)):
            hits += 1
    return hits >= 0.5 * len(names)


GENERIC_PREFIXES = {"CH", "CHAN", "CHANNEL", "C", "E", "EEG", "SEEG", "ECOG", "IEEG", "SIG", "SIGNAL", "INPUT", "IN", "A"}


def has_electrode_structure(names: list[str]) -> bool:
    """
    True when names look like contacts on several electrodes (LA01, LA02, ..., RB05).
    Generic numbering (ch0001 .. ch3431) or a single huge group is not electrode structure.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    parsed = 0
    for n in names:
        sc = split_contact(n)
        if sc:
            parsed += 1
            groups[sc[0]].append(sc[1])
    if parsed < 0.5 * max(len(names), 1):
        return False
    real = {g: v for g, v in groups.items() if g not in GENERIC_PREFIXES and len(v) <= 64}
    covered = sum(len(v) for v in real.values())
    # neighbouring numbers on the same electrode
    pairs = sum(1 for v in real.values() for x in set(v) if x + 1 in set(v))
    return covered >= 0.5 * len(names) and pairs >= 1


@dataclass
class Pair:
    name: str
    a: str
    b: str


def bipolar_pairs(names: list[str], coords: dict[str, tuple[float, float, float]] | None = None,
                  max_gap_factor: float = 2.5) -> tuple[list[Pair], list[str]]:
    """
    Neighbouring contacts on the same electrode, in recording order: (n, n+1) for every
    contact number n whose successor exists. With coordinates, a pair is skipped when its
    contacts are more than `max_gap_factor` × the electrode's median neighbour spacing apart
    (the end of one grid row and the start of the next, G8-G9). Returns (pairs, notes).
    """
    by_elec: dict[str, dict[int, str]] = defaultdict(dict)
    order: list[tuple[str, int]] = []
    for n in names:
        sc = split_contact(n)
        if sc is None:
            continue
        if sc[1] in by_elec[sc[0]]:  # duplicate number (LA1 and LA01): keep the first
            continue
        by_elec[sc[0]][sc[1]] = n
        order.append(sc)
    notes: list[str] = []
    candidates: list[Pair] = []
    for elec, num in order:
        nxt = by_elec[elec].get(num + 1)
        if nxt is not None:
            a = by_elec[elec][num]
            candidates.append(Pair(f"{a}-{nxt}", a, nxt))
    if coords:
        spacing: dict[str, list[float]] = defaultdict(list)
        dists = {}
        for p in candidates:
            ca, cb = coords.get(p.a), coords.get(p.b)
            if ca and cb:
                d = sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5
                dists[p.name] = d
                spacing[split_contact(p.a)[0]].append(d)
        kept = []
        for p in candidates:
            el = split_contact(p.a)[0]
            sp = sorted(spacing.get(el, []))
            if p.name in dists and len(sp) >= 3:
                med = sp[len(sp) // 2]
                if med > 0 and dists[p.name] > max_gap_factor * med:
                    notes.append(f"{p.name}: contacts {dists[p.name]:.1f} mm apart (electrode median {med:.1f} mm); "
                                 "not neighbours, skipped")
                    continue
            kept.append(p)
        candidates = kept
    return candidates, notes
