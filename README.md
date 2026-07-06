# Data-Driven Relaxed Polytopic H-infinity Synthesis With A Posteriori Fixed-Gain Certification

This repository contains the reproducibility code and cached numerical
results for the paper on data-driven relaxed polytopic H-infinity state
feedback synthesis with a posteriori fixed-gain certification.

The code implements the DCCVR synthesis workflow, fixed-gain audit
diagnostics, F-8 MIMO pitch-tracking simulations, Monte Carlo tests, and
figure generation used in the manuscript.

## Repository layout

- `acc_f8_formation_16d.py`: main synthesis, audit, and simulation driver.
- `f8_benchmark_utils.py`: F-8 benchmark model and utility routines.
- `matrix_hull.py`: barycentric matrix-hull diagnostic.
- `sanity_check.py`: consistency checks against cached result files.
- `_generate_tracking_data.py`: tracking-history export helper.
- `acc_f8_formation_results/`: cached CSV, NPZ, and per-scenario outputs.
- `figures/`: paper figure PDFs and the script used to regenerate them.

## Requirements

Python 3.8 or newer is recommended. Install the Python dependencies with

```bash
pip install -r requirements.txt
```

Full optimization runs require MOSEK and a valid MOSEK license. Set the
license location before running synthesis, for example

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

On Windows PowerShell, use

```powershell
$env:MOSEKLM_LICENSE_FILE = ".\mosek.lic"
```

The included cached result files allow figure regeneration and sanity
checks without rerunning the full SDP pipeline.

## Quick start

From the repository root, regenerate the paper figures and check the
cached numerical outputs with

```bash
python figures/make_figures.py
python sanity_check.py
```

The generated PDFs are written to `figures/`.

## Full reproduction

To rerun the main pipeline from scratch, use

```bash
python acc_f8_formation_16d.py
python _generate_tracking_data.py
python figures/make_figures.py
python sanity_check.py
```

The full synthesis and audit run solves several semidefinite programs
and can take a substantial amount of time, depending on hardware and
MOSEK settings.

## Notes on method names

Some internal result keys retain legacy names used during numerical
experiments. In the manuscript, the proposed controller is referred to
as DCCVR, and the no-relaxation ablation is DCCVR (`beta=0`).

## License

This repository is released under the Apache License 2.0.
