"""
Generate paper figures from simulation outputs.

Reads:
  - acc_f8_formation_results/monte_carlo/F8_monte_carlo_trials.csv

Produces:
  - figures/fig_mc_boxplot.pdf       (Monte Carlo RMSE boxplot)
  - figures/fig_tracking.pdf         (tracking time histories)

Style: Wong (2011) colour-blind-friendly palette, identical to the
in-simulation plotter (acc_f8_formation_16d.py). Type-1 fonts so the
PDF embeds editable text in the camera-ready submission.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def read_csv_columns(path: Path) -> Dict[str, List[str]]:
    """Read CSV into a dict mapping column name -> list of cell strings."""
    cols: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols[k].append(v)
    return cols


def to_float(strs: List[str]) -> np.ndarray:
    return np.array([float(s) for s in strs], dtype=float)

# Wong (2011) palette, matched to acc_f8_formation_16d._F8_COLOR_*.
# Keys here MUST match the controller names produced by
# acc_f8_formation_16d.synthesize_f8_controllers (which is what the
# Monte-Carlo CSV column "Controller" and the tracking_histories.npz
# field prefix actually use); LABEL maps those internal keys to the
# display strings used in the paper.
COLOR = {
    "Proposed":               "#0072B2",
    "NoRelax-ProposedActive": "#D55E00",
    "QS-Hinf":                "#009E73",
    "PDL-Hinf":               "#CC79A7",
    "RobustLQR":              "#E69F00",
}
ORDER = ["Proposed", "NoRelax-ProposedActive", "QS-Hinf",
         "PDL-Hinf", "RobustLQR"]
LABEL = {
    "Proposed":               "DCCVR",
    "NoRelax-ProposedActive": r"DCCVR ($\beta\!=\!0$)",
    "QS-Hinf":                r"QS $\mathrm{H}_{\infty}$",
    "PDL-Hinf":               r"PDL $\mathrm{H}_{\infty}$",
    "RobustLQR":              "RobustLQR",
}


def setup_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# -----------------------------------------------------------------
# Monte Carlo RMSE boxplot.
# -----------------------------------------------------------------
def make_mc_boxplot(mc_csv: Path, out_pdf: Path) -> None:
    cols = read_csv_columns(mc_csv)
    ctrl = np.asarray(cols["Controller"])
    rmse = to_float(cols["RMSE_theta"])
    rmse_deg: Dict[str, np.ndarray] = {}
    for c in ORDER:
        rmse_deg[c] = np.rad2deg(rmse[ctrl == c])

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    pos = np.arange(1, len(ORDER) + 1)
    box = ax.boxplot(
        [rmse_deg[c] for c in ORDER],
        positions=pos,
        widths=0.55,
        whis=(5, 95),
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none",
                        markeredgewidth=0.5, alpha=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(linewidth=0.7),
        whiskerprops=dict(linewidth=0.6, color="black"),
        capprops=dict(linewidth=0.6, color="black"),
        patch_artist=True,
    )
    for patch, c in zip(box["boxes"], ORDER):
        patch.set_facecolor(COLOR[c])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    ax.set_xticks(pos)
    ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=15, ha="right")
    ax.set_ylabel(r"Pitch tracking RMSE  [$^\circ$]")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6, zorder=-10)
    ax.set_axisbelow(True)

    n_trials = int(min(len(rmse_deg[c]) for c in ORDER))
    ax.text(0.98, 0.97, f"n = {n_trials} per controller\nwhiskers: 5/95%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig 1] wrote {out_pdf}")


# -----------------------------------------------------------------
# Pitch-tracking time history (nominal + hard-core synthesis
# stress + adversarial plant outside the audited matrix hull).
# -----------------------------------------------------------------
def make_tracking_plot(tracking_npz: Path, out_pdf: Path,
                       ac_idx: int = 0) -> None:
    """Three-panel pitch-tracking time history on AC{ac_idx+1}:

      (a) nominal plant                  -- linear y, autoscaled; all
          five controllers track within ~0.3 rad.
      (b) hard core synthesis stress (s_i=0) -- linear y clipped to +/-0.5
          rad. Unstable traces (DCCVR with beta=0,
          PDL H-infinity) leave the window; their out of window peaks are
          listed in the in-figure footnote.
      (c) adversarial worst (outside audited matrix hull) -- linear y clipped
          to +/-0.5 rad. Same footnote convention.

    Data source: acc_f8_formation_results/tracking_histories.npz, which
    must contain ``nominal_<ctrl>_theta``, ``hc_worst_<ctrl>_theta`` and
    ``worst_<ctrl>_theta`` per controller.
    """
    with np.load(tracking_npz, allow_pickle=True) as D:
        t          = np.asarray(D["t"], dtype=float)
        theta_ref  = np.asarray(D["theta_ref"], dtype=float)
        n_ref      = theta_ref.shape[1]
        theta_hist: dict = {}
        for plant_key in ("nominal", "hc_worst", "worst"):
            for c in ORDER:
                arr = np.asarray(D[f"{plant_key}_{c}_theta"], dtype=float)
                theta_hist[(plant_key, c)] = arr

    t_ref = t[:n_ref]
    ac = int(ac_idx)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.5, 2.8),
        gridspec_kw=dict(hspace=0.50),
    )

    REF_STYLE = dict(color="black", linewidth=0.8, linestyle=":",
                     label="reference")

    # ------- (a) Nominal plant -------
    for c in ORDER:
        tr = theta_hist[("nominal", c)][ac, :]
        ax1.plot(t[: len(tr)], tr, color=COLOR[c], linewidth=1.0,
                 label=LABEL[c])
    ax1.plot(t_ref, theta_ref[ac, :], **REF_STYLE)
    ax1.set_title("(a) nominal plant")
    ax1.set_ylabel(fr"$\theta_{{{ac+1}}}$ [rad]")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0.0, float(t[-1]))

    # Panels (b) and (c) share a tight +/-0.5 rad window so that the
    # stable traces remain legible while unstable traces exit the window
    # (their out of window peaks are listed in the in-figure footnote).
    CLIP_HARD = 0.5

    # ------- (b) Hard-core synthesis stress -------
    for c in ORDER:
        tr = theta_hist[("hc_worst", c)][ac, :]
        ax2.plot(t[: len(tr)], tr, color=COLOR[c], linewidth=1.0,
                 label=LABEL[c])
    ax2.plot(t_ref, theta_ref[ac, :], **REF_STYLE)
    ax2.set_title(r"(b) hard core synthesis stress  ($s_i=0$)")
    ax2.set_ylabel(fr"$\theta_{{{ac+1}}}$ [rad]")
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.0, float(t[-1]))
    ax2.set_ylim(-CLIP_HARD, CLIP_HARD)

    # ------- (c) Adversarial worst (outside audited matrix hull) -------
    for c in ORDER:
        tr = theta_hist[("worst", c)][ac, :]
        ax3.plot(t[: len(tr)], tr, color=COLOR[c], linewidth=1.0,
                 label=LABEL[c])
    ax3.plot(t_ref, theta_ref[ac, :], **REF_STYLE)
    ax3.set_title("(c) adversarial worst  (outside audited matrix hull)")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel(fr"$\theta_{{{ac+1}}}$ [rad]")
    ax3.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax3.set_axisbelow(True)
    ax3.set_xlim(0.0, float(t[-1]))
    ax3.set_ylim(-CLIP_HARD, CLIP_HARD)

    # Shared legend (single row) below the time axis label.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=6,
               frameon=False, columnspacing=0.7,
               handletextpad=0.4, handlelength=1.2)

    # In-figure footnote: out of window peak magnitudes for the
    # diverging traces. Two compact lines, kept inside the saved PDF.
    footnote = (
        r"Out of window peaks: (b) DCCVR ($\beta=0$) $\sim$7 rad, PDL $\mathrm{H}_{\infty}$ $\sim$200 rad;"
        "\n"
        r"(c) DCCVR $\sim$25 rad, DCCVR ($\beta=0$) $\sim$1.2$\times$10$^{4}$ rad, "
        r"PDL $\mathrm{H}_{\infty}$ $\sim$2.4$\times$10$^{4}$ rad."
    )
    fig.text(0.5, -0.08, footnote,
             ha="center", va="top",
             fontsize=6.5, style="italic", color="black")

    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[tracking] wrote {out_pdf}")


# -----------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    sim_root = repo_root / "acc_f8_formation_results"

    tracking_npz = sim_root / "tracking_histories.npz"
    mc_csv = sim_root / "monte_carlo" / "F8_monte_carlo_trials.csv"


    if not mc_csv.exists():
        sys.exit(f"missing {mc_csv}")

    setup_ieee_style()

    make_mc_boxplot(mc_csv, here / "fig_mc_boxplot.pdf")
    if tracking_npz.exists():
        make_tracking_plot(tracking_npz, here / "fig_tracking.pdf")
    else:
        print(f"[fig 4] skipped (missing {tracking_npz});"
              f" run paper_code/_generate_tracking_data.py first")
    print("done.")


if __name__ == "__main__":
    main()
