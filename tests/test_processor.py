"""Unit and end-to-end tests on the synthetic fixture (no Docker)."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import mne
import numpy as np
import pytest

from processor import channels as C
from processor import dsp
from processor import edfout as E
from processor import labels as L
from processor import main as M
from tests import make_fixture as F

RECORDING_COLUMNS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
                     "duration_s", "montage", "kind", "onset_s", "baseline_start_s", "baseline_end_s",
                     "source_format"]
CHANNEL_COLUMNS = ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "status",
                   "status_description", "contacts", "soz", "resected", "onset", "spread", "x", "y", "z", "space",
                   "roi"]
QC_COLUMNS = ["recording_id", "channel", "status", "reason", "line_length_z", "amplitude_z", "line_noise_ratio",
              "dropped_type"]
STEM = "sub-P01_ses-implant1_task-ictal01_run-01"


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def run(monkeypatch, inp: Path, out: Path, **env) -> int:
    defaults = {"INPUT_DIR": str(inp), "OUTPUT_DIR": str(out)}
    for k in M.env_params():
        monkeypatch.delenv(k, raising=False)
    for k, v in {**defaults, **env}.items():
        monkeypatch.setenv(k, v)
    return M.main()


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    return F.make(tmp_path_factory.mktemp("fixture"))


@pytest.fixture(scope="module")
def default_run(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        rc = run(mp, fixture_dir, out)
    finally:
        mp.undo()
    return rc, out


# ---------------------------------------------------------------------------------- channels

def test_contact_names():
    assert C.split_contact("LA01") == ("LA", 1)
    assert C.split_contact("POL RA12") == ("RA", 12)
    assert C.split_contact("A'3") == ("A'", 3)
    assert C.norm_name("EEG LA1-Ref") == C.norm_name("LA01") == "LA1"
    assert C.split_contact("LA01-LA02") is None


def test_classify_drops_non_brain_channels():
    names = ["LA01", "LA02", "LA03", "LA04", "EKG1", "ECG", "C3", "C4", "Cz", "DC1", "EDF Annotations", "EMPTY",
             "X1", "X2", "X3", "X4", "E1"]
    cl = {c.name: c for c in C.classify(names, {}, "SEEG", False, False)}
    assert all(cl[n].keep for n in ("LA01", "X1", "X4"))
    assert cl["EKG1"].dropped_type == cl["ECG"].dropped_type == "ecg"
    assert cl["C3"].dropped_type == cl["Cz"].dropped_type == "scalp"
    assert cl["DC1"].dropped_type == "trigger"
    assert cl["EMPTY"].dropped_type == "empty"
    assert cl["E1"].dropped_type == "eog"
    assert C.classify(["C3", "C4"], {}, "SEEG", True, False)[0].keep
    # a depth electrode named C with contacts C1..C8 keeps its C3 and C4
    depth = [f"C{i}" for i in range(1, 9)]
    assert all(c.keep for c in C.classify(depth, {}, "SEEG", False, False))
    # trusted types win
    cl = {c.name: c for c in C.classify(["LA01", "Z1"], {"LA01": "SEEG", "Z1": "MISC"}, "SEEG", False, True)}
    assert cl["LA01"].keep and not cl["Z1"].keep


def test_bipolar_pairs_and_structure():
    pairs, _ = C.bipolar_pairs(["LA01", "LA02", "LA03", "LB1", "LB3", "LB4"])
    assert [p.name for p in pairs] == ["LA01-LA02", "LA02-LA03", "LB3-LB4"]
    # grid rows: G8 and G9 are not neighbours when coordinates say so
    names = [f"G{i}" for i in range(1, 17)]
    xyz = {f"G{i}": (((i - 1) % 8) * 10.0, ((i - 1) // 8) * 10.0 + (70.0 if i > 8 else 0.0), 0.0) for i in range(1, 17)}
    pairs, notes = C.bipolar_pairs(names, xyz)
    assert "G8-G9" not in [p.name for p in pairs] and notes
    assert C.has_electrode_structure(["LA01", "LA02", "RB1", "RB2"])
    assert not C.has_electrode_structure([f"ch{i:04d}" for i in range(1, 300)])
    assert C.looks_bipolar(["LA01-LA02", "LA02-LA03"])


# ---------------------------------------------------------------------------------- dsp

def test_filters_remove_line_noise_and_keep_high_band():
    fs = 2000.0
    t = np.arange(int(fs * 10)) / fs
    x = 100 * np.sin(2 * np.pi * 60 * t) + 100 * np.sin(2 * np.pi * 180 * t) + 10 * np.sin(2 * np.pi * 210 * t)
    plan = dsp.plan_filters(fs, 60.0, 0.5, 0.0)
    assert plan.notch[:3] == [60.0, 120.0, 180.0] and max(plan.notch) < fs / 2
    y = dsp.Filter(plan)(x)[2000:-2000]
    assert np.std(y - 10 * np.sin(2 * np.pi * 210 * t)[2000:-2000]) < 1.5   # 60/180 Hz gone, 210 Hz (ripple band) kept
    assert dsp.plan_filters(200.0, 60.0, 0.5, 0).notch == [60.0]


def test_robust_z_floor():
    v = np.log10([10, 10.01, 10.02, 9.99, 10.0, 12.0])
    assert dsp.robust_z(v)[-1] > 50                                  # MAD ≈ 0: tiny differences explode
    assert dsp.robust_z(v, dsp.MIN_LOG_SPREAD)[-1] < 5                # 1.2× the median is not an outlier


# ---------------------------------------------------------------------------------- EDF

@pytest.mark.parametrize("scale", [0.3, 87.0, 1500.0, 60000.0])
def test_edf_round_trip(tmp_path, scale):
    rng = np.random.default_rng(1)
    fs = 512.0
    x = scale * rng.standard_normal((3, int(fs * 200)))
    x[2, 100] = 40 * scale                                            # one outlier, clipped by the robust range
    sigs, ranges, kept = [], [], []
    for i in range(3):
        r = E.channel_range(x[i])
        assert r.limit >= 1.0
        sigs.append(E.make_signal(x[i], fs, f"CH{i}", r, "HP:0.5Hz"))
        ranges.append(r)
        kept.append(np.clip(x[i], -r.limit, r.limit))
    assert ranges[2].clipped == 1 and ranges[0].clipped == 0
    p = tmp_path / "rt.edf"
    E.write_edf(p, sigs, [(1.0, 2.0, "seizure")], None, None, 1.0)
    raw = mne.io.read_raw_edf(p, preload=True, verbose="error")
    got = raw.get_data() * 1e6
    for i in range(3):
        err = np.max(np.abs(got[i] - kept[i]))
        assert err <= 0.5 * ranges[i].step * (1 + 1e-3)
        assert ranges[i].step < 0.1 * max(np.std(x[i]), 1.0)          # quantisation far below the signal
    assert [a["description"] for a in raw.annotations] == ["seizure"]
    v = E.verify(p, [(i, 0, kept[i][:512]) for i in range(3)], ["CH0", "CH1", "CH2"], [r.step for r in ranges])
    assert v["max_error_in_steps"] <= 0.5 + 1e-3


def test_symmetric_limit():
    for v in (1077.155, 0.33333, 12345.678, 3.14159265, 1.0):
        lim = E.symmetric_limit(v)
        assert lim >= v
        s = E.make_signal(np.array([0.0, lim, -lim]), 1.0, "x", E.Range(lim, 2 * lim / 65534, 0, lim), "")
        assert s.physical_min == -lim and s.physical_max == lim


# ---------------------------------------------------------------------------------- labels

def test_labels_complete_columns(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("subject,recording_id,channel,soz,onset,onset_s\nsub-P1,r1,,,,12.5\nP1,r1,LA1,,true,\n"
                 "P1,,LA02,true,,\nP2,,LA1,true,,\n")
    ls = L.read_labels(p)
    rows = ls.for_recording("P1", {"r1"})
    res = L.resolve(rows, ["LA01", "LA02", "LA03"], "labels.csv")
    assert res.onset_s == 12.5
    assert res.contacts["LA1"]["onset"] is True and res.contacts["LA3"]["onset"] is False
    assert res.contacts["LA2"]["soz"] is True and res.contacts["LA1"]["soz"] is False
    assert L.norm_recording("sub-X_task-a_ieeg.edf") == "sub-X_task-a"


# ---------------------------------------------------------------------------------- end to end

def test_end_to_end_contract(default_run):
    rc, out = default_run
    assert rc == 0
    ieeg, qc = out / "ieeg", out / "qc"
    recs = read_csv(ieeg / "recordings.csv")
    assert list(recs[0].keys()) == RECORDING_COLUMNS
    by = {r["recording_id"]: r for r in recs}
    assert set(by) == {STEM, "sub-patient02_task-ieeg"}
    r = by[STEM]
    assert r["montage"] == "bipolar" and r["kind"] == "ictal" and r["source_format"] == "bids-edf"
    assert float(r["onset_s"]) == 40.5                                 # labels.csv beats events.tsv (39.0)
    assert (float(r["baseline_start_s"]), float(r["baseline_end_s"])) == (0.0, 30.5)
    assert (ieeg / r["path"]).is_file()
    dd = json.loads((ieeg / "dataset_description.json").read_text())
    assert dd["DatasetType"] == "derivative" and dd["GeneratedBy"][0]["Name"] == "ieeg-prep"

    d = ieeg / "sub-P01" / "ses-implant1" / "ieeg"
    ch = read_csv(d / f"{STEM}_desc-preproc_channels.tsv", "\t")
    assert list(ch[0].keys()) == CHANNEL_COLUMNS
    names = [c["name"] for c in ch]
    # bad contacts LB03 (source), LB05 (line noise), LB07 (noise), LB08 (flat) are in no pair
    assert not any(b in n for n in names for b in ("LB03", "LB05", "LB07", "LB08"))
    assert "LA01-LA02" in names and "LB01-LB02" in names
    c = {x["name"]: x for x in ch}
    assert c["LA01-LA02"]["contacts"] == "LA01|LA02"
    assert c["LA01-LA02"]["onset"] == "true" and c["LA03-LA04"]["onset"] == "false"
    assert c["LA02-LA03"]["soz"] == "true" and c["LA03-LA04"]["soz"] == "false"     # from channels.tsv
    assert c["LA04-LA05"]["resected"] == "true" and c["LA05-LA06"]["resected"] == "true"
    assert c["LA06-LA07"]["resected"] == "false"                                       # from electrodes.tsv
    assert float(c["LA01-LA02"]["x"]) == pytest.approx(-21.75)                         # midpoint
    assert c["LA01-LA02"]["space"] == "MNI152NLin2009aSym" and c["LA01-LA02"]["units"] == "µV"
    ev = read_csv(d / f"{STEM}_desc-preproc_events.tsv", "\t")
    assert [e["trial_type"] for e in ev] == ["baseline", "artifact", "seizure"]
    sz = [e for e in ev if e["trial_type"] == "seizure"][0]
    assert float(sz["duration"]) == pytest.approx(13.5)                                # ends with events.tsv
    js = json.loads((d / f"{STEM}_desc-preproc_ieeg.json").read_text())
    assert js["PowerLineFrequency"] == 60 and js["SamplingFrequency"] == 1000
    assert js["SoftwareFilters"]["Notch"]["Frequencies"][:2] == [60.0, 120.0]
    el = read_csv(d / "sub-P01_ses-implant1_desc-preproc_electrodes.tsv", "\t")
    assert {"name", "x", "y", "z", "soz", "resected"} <= set(el[0])

    raw = mne.io.read_raw_edf(ieeg / r["path"], preload=True, verbose="error")
    assert raw.ch_names == names and raw.info["sfreq"] == 1000
    x = raw.get_data(picks=["LA01-LA02"])[0] * 1e6
    assert np.std(x[45000:50000]) > 2 * np.std(x[5000:20000])                         # the seizure is there

    q = read_csv(qc / "qc.csv")
    assert list(q[0].keys())[: len(QC_COLUMNS)] == QC_COLUMNS
    qs = {(x["recording_id"], x["channel"]): x for x in q}
    assert qs[(STEM, "LB08")]["reason"].startswith("flat")
    assert "line noise" in qs[(STEM, "LB05")]["reason"]
    assert "high" in qs[(STEM, "LB07")]["reason"]
    assert "bad in source" in qs[(STEM, "LB03")]["reason"]
    assert qs[(STEM, "EKG1")]["dropped_type"] == "ecg" and qs[(STEM, "C3")]["dropped_type"] == "scalp"
    assert qs[(STEM, "LA02")]["status"] == "good"                                       # spikes are not "bad"
    assert (qc / f"{STEM}.png").stat().st_size > 10000
    prov = json.loads((ieeg / "provenance.json").read_text())
    assert prov["appVersion"] == "0.1.0" and prov["recordings"][0]["status"] == "ok"
    assert prov["details"][STEM]["verification"]["max_error_in_steps"] <= 0.501


def test_plain_edf(default_run):
    _, out = default_run
    d = out / "ieeg" / "sub-patient02" / "ieeg"
    ch = {x["name"]: x for x in read_csv(d / "sub-patient02_task-ieeg_desc-preproc_channels.tsv", "\t")}
    assert "LA1-LA2" in ch                                       # 'POL LA1' cleaned
    assert ch["LA1-LA2"]["soz"] == "true" and ch["LA2-LA3"]["soz"] == "false"
    q = {x["channel"]: x for x in read_csv(out / "qc" / "qc.csv") if x["recording_id"] == "sub-patient02_task-ieeg"}
    assert q["DC1"]["dropped_type"] == "trigger" and q["EEG C3-Ref"]["dropped_type"] == "scalp"
    r = [x for x in read_csv(out / "ieeg" / "recordings.csv") if x["recording_id"] == "sub-patient02_task-ieeg"][0]
    assert r["source_format"] == "edf" and float(r["onset_s"]) == 30.0     # from the EDF+ 'Seizure' annotation


@pytest.mark.parametrize("montage", ["car", "as-recorded"])
def test_other_montages(fixture_dir, tmp_path, monkeypatch, montage):
    assert run(monkeypatch, fixture_dir, tmp_path, MONTAGE=montage, RECORDINGS="sub-P01*", VERIFY_EDF="false") == 0
    ch = read_csv(tmp_path / "ieeg" / "sub-P01" / "ses-implant1" / "ieeg" / f"{STEM}_desc-preproc_channels.tsv",
                  "\t")
    assert len(ch) == 16 and {c["status"] for c in ch} == {"good", "bad"}
    rec = read_csv(tmp_path / "ieeg" / "recordings.csv")[0]
    assert rec["montage"] == montage and rec["n_good"] == "12"
    js = json.loads((tmp_path / "ieeg" / "sub-P01" / "ses-implant1" / "ieeg" /
                     f"{STEM}_desc-preproc_ieeg.json").read_text())
    assert ("common average" in js["iEEGReference"]) == (montage == "car")


def test_auto_montage_generic_names(tmp_path, monkeypatch):
    """Atlas-style ch0001 names have no electrode structure: auto keeps them; bipolar refuses."""
    sig = F.make_signals()
    gen = {f"ch{i + 1:04d}": sig[n] for i, n in enumerate(F.CONTACTS[:6])}
    F.write_edf(tmp_path / "in" / "atlas.edf", gen)
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o1", MONTAGE="auto") == 0
    assert read_csv(tmp_path / "o1" / "ieeg" / "recordings.csv")[0]["montage"] == "as-recorded"
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o2", MONTAGE="bipolar") == 1     # the only recording fails


def test_fif_and_failure_isolation(fixture_dir, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    inp.mkdir()
    raw = mne.io.read_raw_edf(fixture_dir / "plain" / "patient02.edf", preload=True, verbose="error")
    raw.set_channel_types({n: "seeg" for n in raw.ch_names if n.startswith("POL")} | {"EKG1": "ecg"},
                          verbose="error")
    raw.info["line_freq"] = 50.0
    raw.save(inp / "patient03_raw.fif", verbose="error")
    (inp / "broken.edf").write_bytes(b"0" * 1000)
    assert run(monkeypatch, inp, tmp_path / "out", VERIFY_EDF="false") == 0
    recs = read_csv(tmp_path / "out" / "ieeg" / "recordings.csv")
    assert [r["source_format"] for r in recs] == ["fif"]
    prov = json.loads((tmp_path / "out" / "ieeg" / "provenance.json").read_text())
    assert [s["input"] for s in prov["skipped"]] == ["broken.edf"]
    js = next((tmp_path / "out" / "ieeg").rglob("*_ieeg.json"))
    assert json.loads(js.read_text())["PowerLineFrequency"] == 50.0


def test_mef3(tmp_path, monkeypatch):
    from pymef.mef_session import MefSession
    sig = F.make_signals()
    path = tmp_path / "in" / "patient04.mefd"
    path.parent.mkdir()
    ms = MefSession(str(path), None, new_session=True)
    n = 20000
    start = 946684800000000
    for name in F.CONTACTS[:6] + ["EKG1"]:
        data = np.round(sig[name][:n] * 10).astype(np.int32)              # stored in 0.1 µV units
        ms.write_mef_ts_segment_metadata(name, 0, None, None, start, start + int(n / F.SFREQ * 1e6),
                                         {"sampling_frequency": F.SFREQ, "units_conversion_factor": 0.1,
                                          "units_description": "uV", "recording_duration": int(n / F.SFREQ * 1e6)},
                                         {})
        ms.write_mef_ts_segment_data(name, 0, None, None, 1000, data)
    cwd = os.getcwd()
    try:
        assert run(monkeypatch, tmp_path / "in", tmp_path / "out", MONTAGE="as-recorded") == 0
    finally:
        os.chdir(cwd)
    rec = read_csv(tmp_path / "out" / "ieeg" / "recordings.csv")[0]
    assert rec["source_format"] == "mef3" and rec["n_channels"] == "6" and float(rec["duration_s"]) == 20.0
    raw = mne.io.read_raw_edf(tmp_path / "out" / "ieeg" / rec["path"], preload=True, verbose="error")
    y = raw.get_data(picks=["LA01"])[0] * 1e6
    ref = dsp.Filter(dsp.plan_filters(F.SFREQ, 60.0, 0.5, 0))(np.round(sig["LA01"][:n] * 10) / 10)
    assert np.max(np.abs(y[2000:-2000] - ref[2000:-2000])) < 0.5                       # sample-exact, in µV


def test_no_input(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("x")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1


def test_bad_parameter(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir, tmp_path, MONTAGE="laplacian") == 2
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_mef3_zip(tmp_path, monkeypatch):
    """A MEF3 session delivered as one compressed file is unpacked under /tmp and read."""
    import zipfile
    from pymef.mef_session import MefSession
    sig = F.make_signals()
    sess = tmp_path / "build" / "patient05.mefd"
    sess.parent.mkdir()
    ms = MefSession(str(sess), None, new_session=True)
    n, start = 10000, 946684800000000
    for name in F.CONTACTS[:4]:
        ms.write_mef_ts_segment_metadata(name, 0, None, None, start, start + int(n / F.SFREQ * 1e6),
                                         {"sampling_frequency": F.SFREQ, "units_conversion_factor": 1.0,
                                          "units_description": "uV"}, {})
        ms.write_mef_ts_segment_data(name, 0, None, None, 1000, np.round(sig[name][:n]).astype(np.int32))
    (tmp_path / "in").mkdir()
    with zipfile.ZipFile(tmp_path / "in" / "patient05.mefd.zip", "w") as z:
        for f in sess.rglob("*"):
            z.write(f, f.relative_to(sess))                       # contents without the .mefd folder
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", VERIFY_EDF="false") == 0
    rec = read_csv(tmp_path / "out" / "ieeg" / "recordings.csv")[0]
    assert rec["recording_id"] == "sub-patient05_task-ieeg" and rec["montage"] == "bipolar"
    prov = json.loads((tmp_path / "out" / "ieeg" / "provenance.json").read_text())
    assert prov["recordings"][0]["input"] == "patient05.mefd.zip"
