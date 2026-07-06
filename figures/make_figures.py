"""
Generate paper figures from simulation outputs.

Reads:
  - acc_f8_formation_results/acc_f8_formation_results.npz
      (vertex scores, parameters, active-set mask, data-plant location)
  - acc_f8_formation_results/monte_carlo/F8_monte_carlo_trials.csv
  - acc_f8_formation_results/pdl_hinf_decay_ablation/F8_pdl_decay_ablation.csv

Produces:
  - figures/fig_score_heatmap.pdf    (parameter-space vertex score map)
  - figures/fig_mc_boxplot.pdf       (Monte Carlo RMSE boxplot)
  - figures/fig_pdl_ablation.pdf     (PDL-Hinf decay-rate ablation)

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
# Fig 1: Parameter-space vertex consistency-score map.
#
# 2x3 grid of contour-filled subplots, one per pair of the four plant
# parameters (Z_alpha V^{-1}, M_alpha, M_q, M_delta_e). Each panel is
# a 2D slice through the data-generating plant p_star; the out-of-plane
# parameters are held at p_star so s(p_star) = 0 sits in every panel.
#
# Style follows keee-st.py Exp1 geometry visualisation: filled
# contour with RdBu_r colormap, hard-core dashed contour at s = 0.05,
# black rectangle for the prior box, yellow star for p_star, red rings
# for the ICE active set, circle / square markers for full / reduced
# gain replicas. All six panels share a single colour scale.
# -----------------------------------------------------------------
def make_score_heatmap(npz_path: Path, out_pdf: Path) -> None:
    """Render Fig 1 as a 2x3 subplot grid of contour-filled
    consistency-score fields, one per pair of plant parameters."""
    with np.load(npz_path, allow_pickle=True) as data:
        keys = [str(k) for k in data["vertex_param_keys"]]
        params = np.asarray(data["vertex_params"], dtype=float)
        scores = np.asarray(data["vertex_scores"], dtype=float)
        act_sc = np.asarray(data["vertex_act_scales"], dtype=float)
        active = np.asarray(data["vertex_active_mask"], dtype=bool)
        p_star = np.asarray(data["p_data_source"], dtype=float)
        pair_index = [str(s) for s in data["heatmap_pair_index"]]
        pair_data = {}
        for s in pair_index:
            x_name, y_name = s.split("|")
            tag = f"pair_{x_name}_{y_name}"
            pair_data[(x_name, y_name)] = (
                np.asarray(data[f"{tag}_X"], dtype=float),
                np.asarray(data[f"{tag}_Y"], dtype=float),
                np.asarray(data[f"{tag}_Z"], dtype=float),
                np.asarray(data[f"{tag}_xb"], dtype=float),
                np.asarray(data[f"{tag}_yb"], dtype=float),
            )

    idx_of = {k: i for i, k in enumerate(keys)}
    label_map = {
        "za_v":    r"$Z_\alpha V^{-1}$",
        "m_alpha": r"$M_\alpha$",
        "m_q":     r"$M_q$",
        "m_de":    r"$M_{\delta_e}$",
    }
    # Show only two representative panels: one informative direction
    # (contains the identifiable parameter Z_alpha V^{-1}) and one
    # uninformative direction (purely within the rank-deficient
    # (M_alpha, M_q, M_delta_e) block). The remaining four pairs are
    # qualitatively identical to one of these two (see paper caption).
    pairs_to_show = [("za_v", "m_de"), ("m_alpha", "m_q")]

    # Single global colour scale shared across panels (same vmax as the
    # full six-panel view) so the uninformative panel visibly collapses
    # to the hard-core end of the scale.
    vmax_global = max(float(Z.max()) for (_, _, Z, _, _) in pair_data.values())
    vmax_global = max(vmax_global, 1e-3)
    levels = np.linspace(0.0, vmax_global, 20)

    fig, axes = plt.subplots(
        1, 2, figsize=(3.5, 1.95),
        gridspec_kw=dict(wspace=0.40),
    )
    panel_letters = ["(a) informative", "(b) uninformative"]
    cf_handle = None
    for ax, (x_name, y_name), letter in zip(axes.flat, pairs_to_show,
                                            panel_letters):
        X, Y, Z, xb, yb = pair_data[(x_name, y_name)]
        cf_handle = ax.contourf(
            X, Y, Z, levels=levels, cmap="RdBu_r", alpha=0.90, extend="max",
        )
        # Hard-core contour at s = 0.05 (only when the field crosses it).
        if Z.max() > 0.05:
            try:
                ax.contour(X, Y, Z, levels=[0.05], colors="#2E7D32",
                           linewidths=0.9, linestyles="--")
            except Exception:
                pass

        # Prior parameter box.
        ax.add_patch(Rectangle(
            (xb[0], yb[0]), xb[1] - xb[0], yb[1] - yb[0],
            linewidth=1.0, edgecolor="black", facecolor="none",
            linestyle="-", zorder=4,
        ))

        xs = params[:, idx_of[x_name]]
        ys = params[:, idx_of[y_name]]
        sel_full = np.isclose(act_sc, 1.0)
        sel_red  = ~sel_full
        ax.scatter(xs[sel_full], ys[sel_full],
                   c=scores[sel_full], cmap="RdBu_r",
                   vmin=0.0, vmax=vmax_global,
                   marker="o", s=22, edgecolors="black", linewidths=0.5,
                   zorder=5)
        ax.scatter(xs[sel_red], ys[sel_red],
                   c=scores[sel_red], cmap="RdBu_r",
                   vmin=0.0, vmax=vmax_global,
                   marker="s", s=18, edgecolors="#222222", linewidths=0.4,
                   zorder=5)
        if np.any(active):
            ax.scatter(xs[active], ys[active],
                       facecolors="none", edgecolors="#D62728",
                       marker="o", s=58, linewidths=0.9, zorder=6)
        ax.scatter([p_star[idx_of[x_name]]], [p_star[idx_of[y_name]]],
                   marker="*", s=110, facecolor="#FFD700",
                   edgecolors="black", linewidths=0.7, zorder=7)

        ax.set_title(f"{letter} {label_map[x_name]} vs {label_map[y_name]}",
                     pad=2.5)
        ax.set_xlabel(label_map[x_name])
        ax.set_ylabel(label_map[y_name])
        ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.55)
        ax.set_axisbelow(True)
        margin_x = 0.1 * (xb[1] - xb[0])
        margin_y = 0.1 * (yb[1] - yb[0])
        ax.set_xlim(xb[0] - margin_x, xb[1] + margin_x)
        ax.set_ylim(yb[0] - margin_y, yb[1] + margin_y)

    # Single shared colorbar on the right.
    fig.subplots_adjust(left=0.10, right=0.83, bottom=0.38, top=0.90)
    cax = fig.add_axes([0.86, 0.40, 0.025, 0.48])
    cbar = fig.colorbar(cf_handle, cax=cax)
    cbar.set_label(r"$s_i$")

    # Single shared legend below the two panels (3 columns x 2 rows).
    legend_items = [
        plt.Line2D([], [], marker="o", linestyle="None", color="black",
                   markerfacecolor="#BBBBBB", markeredgecolor="black",
                   markersize=5.5, label="full gain"),
        plt.Line2D([], [], marker="s", linestyle="None", color="#222222",
                   markerfacecolor="#BBBBBB", markeredgecolor="#222222",
                   markersize=5.0, label="reduced gain"),
        plt.Line2D([], [], marker="o", linestyle="None",
                   markerfacecolor="none", markeredgecolor="#D62728",
                   markersize=8, markeredgewidth=1.0, label="ICE active"),
        plt.Line2D([], [], marker="*", linestyle="None",
                   markerfacecolor="#FFD700", markeredgecolor="black",
                   markersize=9, label=r"$p^{\star}$"),
        plt.Line2D([], [], color="black", linestyle="-", linewidth=1.0,
                   marker="", label=r"$\mathcal{V}_{\mathrm{param}}$ box"),
        plt.Line2D([], [], color="#2E7D32", linestyle="--", linewidth=1.0,
                   marker="", label=r"$s_i\!=\!0.05$"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               bbox_to_anchor=(0.46, -0.03), ncol=3,
               frameon=False, columnspacing=1.0,
               handletextpad=0.4, handlelength=1.4)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig 1] wrote {out_pdf}")


# -----------------------------------------------------------------
# Fig 2: Monte Carlo RMSE boxplot (previously Fig 1)
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
# Fig 3: PDL-Hinf decay-rate ablation
# -----------------------------------------------------------------
def make_pdl_ablation(abl_csv: Path, out_pdf: Path) -> None:
    cols = read_csv_columns(abl_csv)
    ctrl = np.asarray(cols["Controller"])
    decay = to_float(cols["decay_rate"])
    gamma_col = "gamma_syn_diag" if "gamma_syn_diag" in cols else "gamma"
    gamma = to_float(cols[gamma_col])
    knorm = to_float(cols["K_norm"])
    cunst = to_float(cols["corner_unstable"])

    pdl_mask = np.array([c.startswith("PDL-Hinf") for c in ctrl])
    pdl_decay = decay[pdl_mask]
    pdl_gamma = gamma[pdl_mask]
    pdl_knorm = knorm[pdl_mask]
    pdl_cunst = cunst[pdl_mask]
    order = np.argsort(-pdl_decay)  # descending: 0.95, 0.90, 0.85
    pdl_decay = pdl_decay[order]
    pdl_gamma = pdl_gamma[order]
    pdl_knorm = pdl_knorm[order]
    pdl_cunst = pdl_cunst[order]

    prop_idx = np.where(ctrl == "Proposed")[0][0]
    proposed_gamma = float(gamma[prop_idx])

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                             gridspec_kw=dict(hspace=0.22))

    # -- Top panel: gamma and ||K||_F
    ax1 = axes[0]
    ax1.plot(pdl_decay, pdl_gamma, "-o", color=COLOR["PDL-Hinf"],
             markersize=4, label=r"PDL $\mathrm{H}_{\infty}$  $\gamma_{\rm syn}$", linewidth=1.1)
    ax1.set_ylabel(r"$\gamma_{\rm syn}$ (solver diagnostic)")
    ax1.axhline(proposed_gamma, color=COLOR["Proposed"], linestyle=":",
                linewidth=1.0,
                label=fr"DCCVR $\gamma_{{\rm syn}}={proposed_gamma:.2f}$ (diag.)")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=COLOR["PDL-Hinf"])

    ax1b = ax1.twinx()
    ax1b.plot(pdl_decay, pdl_knorm, "-^", color="#444444",
              markersize=3.5, linewidth=0.9,
              label=r"PDL $\mathrm{H}_{\infty}$  $\|K\|_F$")
    ax1b.set_ylabel(r"$\|K\|_F$", color="#444444")
    ax1b.tick_params(axis="y", labelcolor="#444444", direction="in")
    ax1b.spines["top"].set_visible(False)

    # combined legend (placed in upper-left to avoid the high-gamma
    # data points which sit in the upper right at tighter decays)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5, zorder=-10)
    ax1.set_axisbelow(True)

    # -- Bottom panel: corner instability percentage
    ax2 = axes[1]
    pct = 100.0 * pdl_cunst
    ax2.bar(pdl_decay, pct,
            width=0.018, color=COLOR["PDL-Hinf"], alpha=0.7,
            edgecolor="black", linewidth=0.5,
            label=r"PDL $\mathrm{H}_{\infty}$ corner unstable")
    ax2.axhline(0.0, color=COLOR["Proposed"], linestyle=":", linewidth=1.0,
                label="DCCVR corner unstable = 0%")
    for x, y in zip(pdl_decay, pct):
        ax2.text(x, y + 0.7, f"{y:.1f}%", ha="center", va="bottom",
                 fontsize=7, color="black")
    ax2.set_ylabel("Corner unstable rate  [%]")
    ax2.set_xlabel(r"PDL $\mathrm{H}_{\infty}$ decay rate  $\alpha^2$")
    ax2.set_ylim(0, max(pct) * 1.35)
    # natural left-to-right increase: 0.85 -> 0.90 -> 0.95
    ax2.set_xlim(0.835, 0.965)
    ax2.set_xticks([0.85, 0.90, 0.95])
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5, zorder=-10)
    ax2.set_axisbelow(True)
    ax2.annotate("", xy=(0.86, -0.30), xytext=(0.94, -0.30),
                 xycoords=("data", "axes fraction"),
                 textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="<-", color="#666666", lw=0.6))
    ax2.text(0.90, -0.36, "tighter decay (harder)",
             transform=ax2.get_xaxis_transform(),
             ha="center", va="top", fontsize=7, color="#666666")

    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[fig 3] wrote {out_pdf}")


# -----------------------------------------------------------------
# Fig 4: Pitch-tracking time history (nominal + hard-core synthesis
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
    print(f"[fig 4] wrote {out_pdf}")


# -----------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    sim_root = repo_root / "acc_f8_formation_results"

    npz_path = sim_root / "acc_f8_formation_results.npz"
    tracking_npz = sim_root / "tracking_histories.npz"
    mc_csv = sim_root / "monte_carlo" / "F8_monte_carlo_trials.csv"
    abl_csv = sim_root / "pdl_hinf_decay_ablation" / "F8_pdl_decay_ablation.csv"

    if not npz_path.exists():
        sys.exit(f"missing {npz_path}")
    if not mc_csv.exists():
        sys.exit(f"missing {mc_csv}")
    if not abl_csv.exists():
        sys.exit(f"missing {abl_csv}")

    setup_ieee_style()
    with np.load(npz_path, allow_pickle=True) as data:
        has_heatmap_cache = "heatmap_pair_index" in data.files
    if has_heatmap_cache:
        make_score_heatmap(npz_path, here / "fig_score_heatmap.pdf")
    else:
        existing = here / "fig_score_heatmap.pdf"
        if existing.exists():
            print(f"[fig 1] kept existing {existing} "
                  "(heatmap cache not present in NPZ)")
        else:
            print("[fig 1] skipped "
                  "(run _augment_npz_scores.py to build heatmap cache)")
    make_mc_boxplot(mc_csv, here / "fig_mc_boxplot.pdf")
    make_pdl_ablation(abl_csv, here / "fig_pdl_ablation.pdf")
    if tracking_npz.exists():
        make_tracking_plot(tracking_npz, here / "fig_tracking.pdf")
    else:
        print(f"[fig 4] skipped (missing {tracking_npz});"
              f" run paper_code/_generate_tracking_data.py first")
    print("done.")


if __name__ == "__main__":
    main()
