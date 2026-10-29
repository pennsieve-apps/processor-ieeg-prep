# Third-party components

The image is `python:3.12.16-slim-bookworm` (Debian 12 and CPython 3.12, PSF-2.0 and the Debian
package licences) plus these Python packages, installed as binary wheels from PyPI at the
versions pinned in `requirements.txt`. Nothing GPL-licensed is linked into the process.

| Package | Version | Licence | Used for |
|---|---|---|---|
| [MNE-Python](https://github.com/mne-tools/mne-python) | 1.13.2 | BSD-3-Clause | reading EDF/BDF/BrainVision/FIF, reading the written EDF back |
| [MNE-BIDS](https://github.com/mne-tools/mne-bids) | 0.20.0 | BSD-3-Clause | reading BIDS-iEEG recordings (types and bad channels from channels.tsv) |
| [edfio](https://github.com/the-siesta-group/edfio) | 0.4.18 | Apache-2.0 | writing EDF+ with a physical range per channel |
| [pymef](https://github.com/msel-source/pymef) | 1.4.8 | Apache-2.0 (Mayo Foundation; bundles the MEF 3.0 C library, same licence) | reading MEF3 sessions |
| NumPy | 2.5.4 | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | arrays |
| SciPy | 1.18.1 | BSD-3-Clause | filters, Welch spectra |
| Matplotlib | 3.11.2 | Matplotlib licence (PSF-based) | QC figures |
| contourpy, cycler, kiwisolver | 1.4.0, 0.12.1, 1.5.1 | BSD-3-Clause | Matplotlib dependencies |
| fonttools, pyparsing | 4.68.0, 3.3.3 | MIT | Matplotlib dependencies |
| Pillow | 13.0.0 | MIT-CMU | Matplotlib dependency (PNG) |
| python-dateutil | 3.0.0 | Apache-2.0 OR BSD-3-Clause | Matplotlib dependency |
| six | 1.17.0 | MIT | python-dateutil dependency |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | MNE/Matplotlib dependency |
| decorator | 5.3.1 | BSD-2-Clause | MNE dependency |
| lazy-loader | 0.6 | BSD-3-Clause | MNE dependency |
| Jinja2, MarkupSafe | 3.1.6, 3.0.4 | BSD-3-Clause | MNE dependency (not used at run time) |
| pooch, platformdirs | 1.9.0, 4.14.0 | BSD-3-Clause, MIT | MNE dependency (downloads; never used: no network at run time) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | MNE dependency |
| requests, urllib3, idna, charset-normalizer, certifi | 2.34.2, 2.8.0, 3.20, 3.5.2, 2026.10.23 | Apache-2.0, MIT, BSD-3-Clause, MIT, MPL-2.0 | pooch dependencies (unused) |
| tzdata | 2026.6 | Apache-2.0 | time zones |

MPL-2.0 (certifi, tqdm) is a file-level copyleft licence: the files are distributed unmodified
in the image; their source is on PyPI.

pymef asks that academic work facilitated by it acknowledge the Mayo Systems
Electrophysiology Laboratory, Rochester, MN.

## Demo data (not in the image)

- Pennsieve Discover 530 v3, "Multi-Expert Seizure Annotation" (Ojemann WK, Zhou D, Armstrong C,
  Sinha N, Litt B, Conrad E), doi:10.26275/dely-fzwx, CC BY 4.0.
- Pennsieve Discover 179 v1, "HUP Interictal iEEG Atlas" (Bernabei J, Litt B),
  doi:10.26275/lzhy-ox20, ODC-By 1.0.
