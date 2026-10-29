# processor-ieeg-prep

The first step of the two epilepsy iEEG workflows (Interictal Biomarker Mapping; Seizure Onset
Localisation). It takes intracranial EEG as it comes, a BIDS-iEEG dataset or loose EDF,
BrainVision, FIF or MEF3 files, with or without clinical labels, and turns it into one
**BIDS-iEEG derivative** that every later step (spikes, band power, HFOs, normative mapping,
DynaSD, epileptogenicity index, fragility, consensus, report) reads without glue:

- **only brain channels**: SEEG and ECoG contacts are kept; ECG/EKG, EMG, EOG, trigger/DC,
  annotation, empty and scalp EEG channels (C3, C4, Cz… recorded alongside) are dropped and
  listed in the QC table;
- **bad channels found and explained**: flat contacts, contacts dominated by mains noise, and
  contacts far noisier or larger than the rest of the implant are marked `bad` with the
  reason; channels the source already marked bad stay bad;
- **one montage**: bipolar by default (neighbouring contacts on the same electrode, `LA01-LA02`,
  never across a bad contact), or common average, or as recorded;
- **clean but full-band**: mains frequency and its harmonics notched, a 0.5 Hz high-pass, no
  low-pass, so ripples and fast ripples are still there for the HFO step;
- **labels and anatomy travel with the data**: seizure-onset zone, resected, onset and spread
  labels and electrode coordinates (bipolar channels at the midpoint of their contacts) are in
  `_channels.tsv`; the expert seizure onset and a pre-ictal baseline interval are in
  `_events.tsv` and `recordings.csv`.

**For the epileptologist**: the output is what you would review in a bipolar montage, with the
broken and noisy contacts set aside (each with a reason you can check in `qc.csv` and the QC
figure), and with your labels (SOZ, resection, onset and spread channels) attached so every
later result can be compared to them. **For the researcher**: every recording comes out in the
same shape (EDF+ in µV, BIDS sidecars, one index table), the reading code is MNE-Python /
MNE-BIDS, and every decision is recorded in `provenance.json`.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `ieeg` | input | subfolder (`INPUT_DIR/ieeg/`; files at the root of `INPUT_DIR` are found too) | A BIDS-iEEG dataset (`sub-*/[ses-*/]ieeg/*_ieeg.{edf,vhdr,fif}` with `_channels.tsv`, `_events.tsv`, `_electrodes.tsv`, `_coordsystem.json`, `_ieeg.json`), or plain `*.edf`, `*.bdf`, `*.vhdr`, `*.fif`, MEF3 sessions (`*.mefd/`, or `*.mefd.zip/.tar/.tar.gz`), anywhere in the input (folders and symlinks followed); plus an optional `labels.csv`. |
| `ieeg` | output | `OUTPUT_DIR/ieeg/` | The BIDS-iEEG derivative (below) and `provenance.json`. |
| `qc` | output | `OUTPUT_DIR/qc/` | `qc.csv`, one `<recording_id>.png` per recording, `provenance.json`. |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `RECORDINGS` | `*` | Comma-separated globs choosing recordings, matched against the recording id (`sub-HUP224_ses-postimplant_task-ictal*`), the file name and its path in the input. |
| `MONTAGE` | `bipolar` | `bipolar`: pairs of neighbouring contacts (see *Montage*). `car`: common average of the good channels. `as-recorded`: no re-referencing. `auto`: `as-recorded` when names already look bipolar (`LA01-LA02`) or have no electrode/contact structure (`ch0001`), else `bipolar`. With `bipolar`, data that are already bipolar (names, or `iEEGReference` in the sidecar) are kept as they are; data without contact structure fail with a message pointing to `auto`. |
| `LINE_FREQ` | `auto` | Mains frequency (Hz). `auto`: `PowerLineFrequency` from the BIDS sidecar (or FIF file), else 60. `0` turns the notch off. |
| `HIGHPASS` | `0.5` | High-pass cutoff (Hz); `0` turns it off. |
| `LOWPASS` | `0` | Low-pass cutoff (Hz); `0` keeps the full band (default: HFO detection needs it). |
| `INCLUDE_SCALP` | `false` | Keep scalp EEG channels (type `EEG`, or 10-20 names in an iEEG file). In a bipolar montage they are written as recorded. |
| `IEEG_TYPE` | `seeg` | Type given to brain channels of files that do not record types (EDF, BrainVision and MEF3 outside BIDS): `seeg` or `ecog`. |
| `LABELS_FILE` | `labels.csv` | Name of the labels file, looked for anywhere in the input (the shallowest wins). |
| `FLAT_UV` | `0.5` | Flat when the detrended standard deviation is below this many µV, or more than half the samples repeat the previous one. |
| `BAD_Z` | `5` | Outlier threshold on the robust z-score of log line length and log amplitude (see *Bad channels*). |
| `LINE_NOISE_MAX` | `0.5` | Line-noise dominated when more than this fraction of the 1–250 Hz power lies within ±1 Hz of the line frequency and its harmonics. |
| `BASELINE_GAP_S` | `10` | The baseline ends this many seconds before the seizure onset. |
| `BASELINE_MAX_S` | `90` | The baseline starts at the recording start and lasts at most this long. |
| `VERIFY_EDF` | `true` | Read every EDF back with MNE and fail the recording when a value differs from what was meant by more than half a quantisation step. |
| `MEF_PASSWORD` | (unset) | Password of encrypted MEF3 sessions (environment variable; not in `app.yml`). |

## What it does, step by step

1. **Finds** the recordings and `labels.csv`. A file in a BIDS layout (`sub-X/[ses-Y/]ieeg/`)
   is read with **MNE-BIDS** (`read_raw_bids`: channel types and bad channels from
   `_channels.tsv`); if that fails, or the file is not in a BIDS layout, it is read with
   MNE-Python's reader for its format; MEF3 with **pymef**. Sidecars are found next to the
   recording or up the BIDS inheritance chain; an `electrodes.tsv` elsewhere in the subject's
   folder (Discover 530 keeps it in `sub-X/derivatives/`) is used when there is no BIDS one.
   The recording id is the BIDS stem (`sub-HUP224_ses-postimplant_task-ictal339135_run-00`);
   for a non-BIDS file it is `sub-<file name without punctuation>_task-ieeg`. Two inputs with
   the same id are both skipped.
2. **Channel types.** Types from `_channels.tsv` (or the FIF file) are trusted; ECG/EKG, EMG,
   EOG, TRIG/STIM, MISC… are dropped. Names are checked as well: `EKG1`, `ECG`, `EMG`, `EOG`,
   `DC1`, `TRIG`, `EDF Annotations`, `EMPTY`, `SpO2`… are always dropped; short names a depth
   electrode could also have (`X1`, `LAT3`, `E2`, `REF`) only when they are not one of four or
   more contacts of an electrode with that name. Scalp 10-20/10-10 names (`C3`, `C4`, `Cz`,
   `Fp1`…) are scalp EEG unless their electrode letters have other, non-10-20 contacts or more
   than four (a depth electrode `C` with `C1`…`C8` keeps its `C3`). Exporter prefixes and
   suffixes are removed from names (`POL LA1`, `EEG LA1-Ref` → `LA1`).
3. **Bad channels** (contacts, before the montage), measured on the **baseline** when a seizure
   onset is known (so ictal activity does not count against the seizure-onset contacts), else
   on the whole recording:
   - *flat*: detrended SD < `FLAT_UV`, or > 50 % repeated samples;
   - *line noise*: power ratio > `LINE_NOISE_MAX` (on the channel as recorded);
   - *outlier*: on each channel minus the per-sample median of all channels (a robust common
     reference, so noise every contact shares through the recording reference neither hides nor
     creates outliers), after a 1 Hz high-pass and the notch: **line length** = median |Δx| per
     sample and **amplitude** = 1.4826 × MAD. Medians make both insensitive to sparse transients
     (interictal spikes, stimulation pulses: a channel full of spikes is not bad). Each is
     log10-transformed and turned into a robust z-score across the channels of the same type,
     (x − median) / max(1.4826 × MAD, 0.05 decades); z > `BAD_Z` is bad. Only the upper tail
     counts. The 0.05-decade floor keeps implants whose contacts are almost identical (MAD → 0)
     from flagging channels that differ by a few percent: at `BAD_Z=5` a channel needs at least
     1.8× the median to be an outlier;
   - *bad in source*: `status=bad` in `_channels.tsv` (with its `status_description`) or MNE's
     bad list.
   All reasons go into `status_description` and `qc.csv`.
4. **Montage** (see below), then **filters**, per channel, zero-phase (forward-backward): IIR
   notch (Q = 30, −3 dB width 2 Hz at 60 Hz) at the line frequency and every harmonic more than
   1 Hz below Nyquist; Butterworth high-pass (order 4, applied twice) at `HIGHPASS`; optional
   low-pass at `LOWPASS`.
5. **Labels.** Per contact, in increasing priority: `_electrodes.tsv` columns, `_channels.tsv`
   columns, `labels.csv`. Names match ignoring case, spaces and leading zeros (`LA1` = `LA01`).
   A bipolar channel is `true` when either contact is, `false` when both are `false`, unknown
   otherwise. A `labels.csv` column that a recording (`onset`, `spread`) or subject (`soz`,
   `resected`) has any rows for is complete for it: contacts without a row are `false`.
   Any other column filled on a recording-level row of `labels.csv` (`channel` empty), e.g.
   `stim_induced` or `lvfa`, is copied into `recordings.csv` as a column of the same name
   (lower-cased; value as written; empty for recordings without such a row). A column named
   like one ieeg-prep computes (`kind`, `baseline_start_s`, ...) is ignored with a warning.
6. **Seizure onset and baseline.** Onset: `labels.csv` `onset_s`, else the first `_events.tsv`
   row whose `trial_type` is `seizure`/`ictal`/`sz`…, else such an EDF+/FIF annotation. Its end
   (for the seizure event's duration) comes from the source event when there is one. Baseline:
   from the recording start to `onset − BASELINE_GAP_S`, at most `BASELINE_MAX_S` long, and only
   when that leaves ≥ 5 s. `kind` is `ictal` with an onset or a seizure event (or a task named
   `*ictal*`/`*seizure*`), `interictal` for tasks named `*interictal*`, else `unknown`.
7. **Writes** the EDF+, its sidecars, and the QC figure; then reads the EDF back with MNE.

Recordings are processed one at a time; one that fails is logged and skipped, and the run fails
only when every recording failed or none was found.

### Montage

Bipolar pairs are made from names `<letters><number>` (`LA01`, `LA1`, `A'3`, `RAH 12`): contact
*n* and *n + 1* of the same electrode, in recording order, named `<first>-<second>`
(`LA01-LA02`, first minus second). A pair is skipped when either contact is bad; when the
coordinates say the two are more than 2.5× the electrode's median spacing apart (the end of
one grid row and the start of the next, `G8-G9`); and contacts with no good neighbour drop out
(both are logged). The `contacts` column says which contacts made each channel (`LA01|LA02`).
After the montage a pair whose difference is flat (shorted contacts) is marked bad.

Common average (`car`): every kept channel minus the mean of the good ones; bad channels are
written too, with `status=bad`. `as-recorded` keeps every kept channel, bad ones marked.

## Outputs

**`ieeg/recordings.csv`** (one row per recording): `recording_id, subject, session, task, run,
path` (EDF, relative to the port), `sfreq, n_channels, n_good, duration_s, montage` (`bipolar`,
`car`, `as-recorded`), `kind` (`ictal`, `interictal`, `unknown`), `onset_s, baseline_start_s,
baseline_end_s` (empty when unknown), `source_format` (`bids-edf`, `edf`, `bdf`,
`brainvision`, `fif`, `mef3`, with `bids-` for BIDS inputs), then any extra recording-level
columns of `labels.csv` (e.g. `stim_induced, lvfa`; see Labels above).

**`ieeg/sub-<label>/[ses-<label>/]ieeg/<stem>_desc-preproc_ieeg.edf`**: EDF+ (16-bit), µV, the
output channels in recording order, data records of 1 s (the last incomplete second, < 1 s, is
dropped), seizure and baseline as EDF+ annotations. Each channel has its own symmetric physical
range ±L, L = 1.1 × min(max |x|, 3 × the 99.99th percentile of |x|), at least 1 µV, stored
exactly in the 8-character header field, and digital range ±32767 (0 µV is digital 0). So
without extreme outliers nothing is clipped, and one saturated artefact cannot coarsen a whole
channel; samples beyond ±L are clipped and counted (`provenance.json`, `clipped_samples`). On the
demo the median quantisation step is 0.01 µV.

**`_channels.tsv`**: `name, type` (`SEEG`, `ECOG`, `EEG`), `units` (`µV`), `low_cutoff,
high_cutoff` (this app's filters combined with the source's), `sampling_frequency, status`
(`good`/`bad`), `status_description` (why bad), `contacts`, `soz, resected, onset, spread`
(`true`/`false`, empty = unknown), `x, y, z, space` (midpoint for bipolar channels; space from
the electrodes file name or `_coordsystem.json`), `roi` (atlas region from `_electrodes.tsv`;
`A|B` when the two contacts of a pair differ).

**`_events.tsv`**: `onset, duration, trial_type, source`; `trial_type` is `seizure` (onset =
expert/clinical onset), `baseline`, or `artifact` (source events or annotations named
artifact/bad, and gaps in discontinuous recordings).

**`_ieeg.json`**: `TaskName, SamplingFrequency, PowerLineFrequency, iEEGReference` (what the
montage did), `SoftwareFilters` (notch frequencies, high-pass, low-pass), `RecordingDuration`,
channel counts, the source's `Manufacturer` and similar fields, `Sources`, and
`PennsieveIeegPrep` (montage chosen and requested, where the line frequency and onset came from,
the source format and reader, the QC window, the EDF range rule).

**`sub-<label>[_ses-<label>]_desc-preproc_electrodes.tsv`** (per subject and session, when
coordinates, regions or SOZ/resected labels exist): `name, x, y, z, size, space, roi, soz,
resected` and the source's other columns (e.g. `spike_24h` of the atlas).

**`dataset_description.json`**: BIDS derivative, `GeneratedBy` ieeg-prep with version,
`SourceDatasets` from the input's `dataset_description.json`.

**`qc/qc.csv`** (one row per source channel, plus bipolar channels that went bad after the
montage): `recording_id, channel, status` (`good`, `bad`, `dropped`), `reason, line_length_z,
amplitude_z, line_noise_ratio, dropped_type` (`ecg`, `emg`, `eog`, `trigger`, `misc`, `empty`,
`scalp`), `amplitude_uv, line_length_uv`.

**`qc/<recording_id>.png`**: 10 s of the output traces (from 2 s before the onset when there is
one, else from the start; up to 48 channels, bad ones in red and named) and a strip with the
status of every source channel (good / bad / dropped, the names of those not good below it).

**`provenance.json`** (both folders): app version, tool versions, parameters, inputs found, per
recording the reader, channel counts, dropped types, filters, onset source, QC window,
sidecars used, labels found, clipped samples, quantisation step (median and max), the EDF
read-back result, notes and timings; failed recordings with the error.

## Demo results

On the 24 seizure clips of Discover 530 (6 patients, SEEG, 1024 Hz, 94–230 contacts,
4.2–7.7 min each) with the expert-consensus `labels.csv`: all 24 prepared; 72–204 bipolar
channels per clip (median 105); 0–43 bad contacts per clip, 162 in total (104 line-noise
dominated, 67 high line length, 9 high amplitude; some have two reasons), most of them outer
contacts; HUP238 has two clips with 34 and 43, its left electrodes' outer contacts dominated by
60 Hz. 8 ECG and 4 scalp channels dropped. Of the expert onset channels only one contact in one
clip came out bad (HUP257 stim8576 RB04, next to the stimulated pair). The expert onset
(`onset_s`) lies −1.49 to +12.18 s from the clinically marked approximate onset at 120 s
(median +1.42 s). Median quantisation step 0.012 µV (largest 2.4 µV, on a channel with ±80 mV
stimulation artefacts); 512 samples clipped in total; EDF read-back error ≤ 0.5 step everywhere.
On the 5 atlas patients (`scripts/atlas-to-bids.py`; 200 Hz, already bipolar): kept as
recorded, no bad channels, SOZ/resected labels and MNI coordinates carried through.

## Runtime

`cpu: 2048` (2 vCPU), `memory: 4096` MiB, `standard` compute, timeout 2 h.

Measured with the image (`docker run --memory 3g`, linux/amd64 under emulation on an Apple
M-series Mac, which is several times slower than native amd64) on the 24 demo clips
(94–230 contacts, 1024 Hz, 4.2–7.7 min; 3.3 GB of EDF): **peak memory 1.04 GB**
(`memory.peak` of the container's cgroup, page cache included; peak RSS 0.93 GB), **wall time
9 min 16 s** (≈ 23 s per clip; 4 min 18 s natively on the Mac). Memory grows with one
recording's size: the kept channels are held as float32 (4 bytes × channels × samples) plus
the int16 output (2 bytes × output channels × samples) and a few float64 channels at a time.
A synthetic 1 h, 128-contact, 1024 Hz EDF (0.94 GB; no onset, so the QC metrics use the whole
hour) peaks at **2.3 GB RSS** and takes 3 min 42 s natively on the Mac (the Docker measurement
of it could not be completed: the shared 8 GB Docker VM was nearly full with other builders'
containers and the kernel killed the run at 2.5 GB). 4 GB covers recordings up to about
1.5 h × 128 channels at 1024 Hz; longer or denser recordings need proportionally more (or
should be split).

## Limitations

- **Bad-channel detection is statistical**, relative to the other contacts of the same
  recording. A recording where most contacts are bad, or a contact that is only intermittently
  bad, is not caught. Outer contacts in white matter or outside the brain often come out as
  line-noise dominated or high line length; check `qc.csv` before excluding a region.
  Stimulation-induced seizures carry stimulation pulses in their baseline; the robust measures
  ignore isolated pulses, but contacts next to the stimulated pair can still be flagged.
- **Without a known onset** the metrics are taken over the whole recording, so a seizure inside
  an `unknown` recording can make seizing contacts look like outliers.
- **Bipolar pairing is by name**: it needs `<letters><number>` names and assumes consecutive
  numbers are neighbours. Grids without coordinates may pair the end of one row with the start
  of the next; contacts named without numbers are left out.
- **The montage changes what a channel is**: a bipolar channel reflects activity near both
  contacts; its label is `true` if either contact is labelled.
- **No artefact rejection in time** beyond what the source annotates; no resampling (any rate
  ≥ 100 Hz is written as is; HFO work needs ≥ 1000 Hz).
- The EDF format stores 16 bits per sample; values beyond a channel's range are clipped (rare by
  construction, counted in `provenance.json`). The last < 1 s of a recording that does not fill
  a whole 1 s data record is dropped.
- MEF3 channels at a sampling rate other than the most common one are not read.
- EDF labels hold 16 characters: a channel name longer than that (rare; bipolar names of long
  contact names) is truncated in the EDF while `_channels.tsv` keeps the full name.

## Licences

Our code: Apache-2.0 (`LICENSE`). Bundled components and their licences: `THIRD_PARTY.md`
(MNE-Python, MNE-BIDS, NumPy, SciPy: BSD-3-Clause; edfio, pymef: Apache-2.0; Matplotlib: PSF-based;
no GPL code). Demo data: Discover 530 is CC BY 4.0, Discover 179 is ODC-By 1.0.

## Citations

- Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python. *Front
  Neurosci* 2013;7:267. doi:10.3389/fnins.2013.00267. PMID 24431986.
- Appelhoff S, Sanderson M, Brooks T, et al. MNE-BIDS: Organizing electrophysiological data into
  the BIDS format and facilitating their analysis. *J Open Source Softw* 2019;4:1896.
  doi:10.21105/joss.01896. PMID 35990374.
- Holdgraf C, Appelhoff S, Bickel S, et al. iEEG-BIDS, extending the Brain Imaging Data Structure
  specification to human intracranial electrophysiology. *Sci Data* 2019;6:102.
  doi:10.1038/s41597-019-0105-7. PMID 31239438.
- Bigdely-Shamlo N, Mullen T, Kothe C, Su KM, Robbins KA. The PREP pipeline: standardized
  preprocessing for large-scale EEG analysis. *Front Neuroinform* 2015;9:16.
  doi:10.3389/fninf.2015.00016. PMID 26150785. (Robust z-score outlier criterion for bad
  channels.)
- Bernabei JM, Sinha N, Arnold TC, et al. Normative intracranial EEG maps epileptogenic tissues
  in focal epilepsy. *Brain* 2022;145:1949–1961. doi:10.1093/brain/awab480. PMID 35640886.
  (The atlas used as demo data.)
- Ojemann WK, Zhou D, Armstrong C, Sinha N, Litt B, Conrad E. Multi-Expert Seizure Annotation
  (version 3) [dataset]. Pennsieve Discover; 2026. doi:10.26275/dely-fzwx.
- Bernabei J, Litt B. HUP Interictal iEEG Atlas (version 1) [dataset]. Pennsieve Discover; 2021.
  doi:10.26275/lzhy-ox20.
- Software without a paper: edfio, https://github.com/the-siesta-group/edfio; pymef,
  https://github.com/msel-source/pymef (Mayo Systems Electrophysiology Laboratory).

All DOIs were checked against Crossref (articles) and DataCite (datasets); `CITATION.cff`
lists the works with the authors exactly as those registries give them (Crossref lists only
the first author of the MNE-Python paper).
