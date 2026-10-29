# Sample input

The app needs real recordings; none are committed here. Three ways to get input:

## 1. The demo seizure clips (BIDS-iEEG, with expert labels)

From the workflow repository (`pennsieve-apps/workflow-epilepsy-ieeg`):

```sh
scripts/fetch-sample-data.sh data                 # Discover 530 (24 SEEG clips, 6 patients) and 179 (atlas)
python3 scripts/seizures-to-labels.py              # writes data/seizures/labels.csv (expert consensus onsets)
```

Use `data/seizures/` as the input: it holds the BIDS dataset (`DATA/sub-*/ses-postimplant/ieeg/`)
and `labels.csv` side by side, so the expert unequivocal-onset time replaces the approximate
onset of `_events.tsv`, and the consensus onset/spread channels are carried into `_channels.tsv`.

```sh
docker run --rm --memory 2g --env-file dev.env -e RECORDINGS='sub-HUP224*ictal339135*' \
  -v "$PWD/../workflow-epilepsy-ieeg/data/seizures":/data/input:ro -v "$PWD/data/out":/data/output \
  processor-ieeg-prep:dev
```

## 2. The interictal atlas as BIDS

```sh
python3 scripts/atlas-to-bids.py                   # 5 patients of HUP_atlas_final.mat -> data/atlas-bids/
```

Each patient becomes `sub-HUPatlas<NNN>/ieeg/` with a 60 s, 200 Hz bipolar recording, MNI
coordinates and SOZ/resected labels in `_electrodes.tsv`. The channels are already bipolar
(`iEEGReference`), so the app keeps them as recorded.

## 3. Your own recordings

Anything under the input folder is found: BIDS-iEEG datasets, `*.edf`, `*.bdf`, `*.vhdr` (with
its `.eeg`/`.vmrk`), `*.fif`, MEF3 sessions (`*.mefd/` folders, or zipped/tarred). Add a
`labels.csv` to carry clinical labels:

```csv
subject,recording_id,channel,soz,resected,onset,spread,onset_s
P07,,LA1,true,true,,,
P07,,LA2,true,false,,,
P07,sub-P07_task-ieeg,,,,,,312.4
P07,sub-P07_task-ieeg,LA1,,,true,,
```

For a file that is not in a BIDS layout the recording id is `sub-<file name without
punctuation>_task-ieeg` (e.g. `P07.edf` → `sub-P07_task-ieeg`); `recording_id` may also be the
file name. A column a recording (onset, spread) or subject (soz, resected) has any rows for is
taken as complete: contacts without a row are `false`.
