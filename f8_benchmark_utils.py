# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import itertools
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Any
import numpy as np
from numpy.linalg import eigvalsh
import scipy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# =========================================================
# Plotting functions
# =========================================================
def set_publication_style(
        *,
        context: str = "paper",
        column: str = "single",
        font_family: Optional[List[str]] = None,
        use_tex: bool = False,
) -> Dict[str, Tuple[float, float]]:
    mm_to_in = 1.0 / 25.4
    fig_w_single = 89.0 * mm_to_in
    fig_w_double = 183.0 * mm_to_in
    fig_h_single = 65.0 * mm_to_in
    fig_h_double = 70.0 * mm_to_in

    sizes = {
        "single": (fig_w_single, fig_h_single),
        "double": (fig_w_double, fig_h_double),
    }

    if font_family is None:
        font_family = ["Arial", "Helvetica", "DejaVu Sans"]

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

    base = {
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,

        "font.family": "sans-serif",
        "font.sans-serif": font_family,
        "text.usetex": bool(use_tex),
        "mathtext.fontset": "dejavusans",

        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    mpl.rcParams.update(base)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)

    _ = column
    return sizes

# File export helper
def save_pub_figure(
        fig: mpl.figure.Figure,
        out_dir: str,
        stem: str,
        *,
        formats: Tuple[str, ...] = ("pdf", "png"),
        dpi_png: int = 600,
        transparent: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fmt_l = fmt.lower()
        path = os.path.join(out_dir, f"{stem}.{fmt_l}")
        if fmt_l in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(path, dpi=dpi_png, transparent=transparent)
        else:
            fig.savefig(path, transparent=transparent)


def _setup_axes(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=False, right=False, which="both")
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.25)


def _add_gust_shading(ax: plt.Axes, gusts: List[Dict[str, float]]) -> None:
    for g in gusts:
        ax.axvspan(g["t0"], g["t1"], alpha=0.10, linewidth=0)


# =========================================================
# MOSEK license handling
# =========================================================
# Resolution order:
#   1. Caller-provided license_path argument (highest priority);
#   2. MOSEKLM_LICENSE_FILE environment variable;
#   3. Otherwise raise FileNotFoundError with a helpful message.
# A release tree must not silently depend on the developer's local
# Windows install path, so there is no hard-coded fallback.
def configure_mosek_license(
        license_path: Optional[str] = None,
        search_dir: Optional[str] = None,
        verbose: bool = True,
) -> str:
    if license_path is None:
        env_path = os.environ.get("MOSEKLM_LICENSE_FILE")
        if env_path:
            # Defensive strip: Windows users sometimes end up with a
            # trailing newline / whitespace in MOSEKLM_LICENSE_FILE
            # after `setx` or copy-paste from a text file;
            # the MOSEK C runtime caches the polluted value at first
            # `import mosek` and even resetting os.environ later cannot
            # un-cache it. Stripping here defends against that case
            # before any mosek module touches the env var.
            license_path = env_path.strip().strip(";")
        else:
            raise FileNotFoundError(
                "MOSEK license not configured.\n"
                "Set MOSEKLM_LICENSE_FILE to your mosek.lic path "
                "(or '<port>@<server>' for a floating license), or "
                "pass license_path= to configure_mosek_license(). "
                "Free academic license: "
                "https://www.mosek.com/products/academic-licenses/"
            )
    if search_dir is None:
        search_dir = os.path.dirname(license_path) or "."
    # Always normalise the env var to the cleaned path so subsequent
    # mosek imports in this process see a sane value (this fixes the
    # current process even if MOSEKLM_LICENSE_FILE was polluted at
    # process start; system-level env var still needs `setx` to be
    # cured permanently).
    if license_path:
        os.environ["MOSEKLM_LICENSE_FILE"] = license_path
    if license_path and ("@" in license_path):
        os.environ["MOSEKLM_LICENSE_FILE"] = license_path
        if verbose:
            print(f"[MOSEK] Using license server: {license_path}")
        return license_path

    if license_path and os.path.isfile(license_path):
        os.environ["MOSEKLM_LICENSE_FILE"] = license_path
        if verbose:
            print(f"[MOSEK] Using license file: {license_path}")
        return license_path

    candidates = []
    if license_path:
        candidates.extend([license_path, license_path + ".txt"])
    for c in candidates:
        if os.path.isfile(c):
            os.environ["MOSEKLM_LICENSE_FILE"] = c
            if verbose:
                print(f"[MOSEK] Using license file: {c}")
            return c

    found = []
    if search_dir and os.path.isdir(search_dir):
        found = glob.glob(os.path.join(search_dir, "**", "*.lic"), recursive=True)

    if found:
        found.sort()
        lic = found[0]
        os.environ["MOSEKLM_LICENSE_FILE"] = lic
        if verbose:
            print(f"[MOSEK] Auto-found license file: {lic}")
        return lic

    env_val = os.environ.get("MOSEKLM_LICENSE_FILE", "")
    raise FileNotFoundError(
        "MOSEK license file was not found.\n"
        f"Tried license_path: {license_path}\n"
        f"Searched directory: {search_dir}\n"
        f"Current MOSEKLM_LICENSE_FILE='{env_val}'\n"
        "Please set MOSEKLM_LICENSE_FILE to a valid mosek.lic file, "
        "use '<port>@<server>' for a floating license, or pass "
        "license_path= to configure_mosek_license()."
    )


# =========================================================
# Benchmark configuration and vertex hooks
# =========================================================
@dataclass(frozen=True)
class SynthesisParams:
    Ts: float = 0.1
    g: float = 9.81

    d_max: float = 2.0
    du_max: float = 3.5

    u_abs_min: Tuple[float, float, float, float] = (-6.0, -4.0, -4.0, -4.0)
    u_abs_max: Tuple[float, float, float, float] = (6.0, 4.0, 4.0, 4.0)
    du_max_vec: Optional[Tuple[float, float, float, float]] = None
    sat_tol: float = 0.99

    decay_rate: float = 0.95
    w_gamma: float = 1.0
    w_mu: float = 0.1
    w_beta: float = 1e-3
    beta_score_tol: float = 1e-4
    gamma2_scale: float = 1.0
    mosek_pfeas: float = 1e-6
    mosek_dfeas: float = 1e-6
    lmi_margin: float = 0.0
    mosek_solve_form: str = "free"

    Qx_perf: Tuple[float, ...] = (20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1)
    Rd_perf: Tuple[float, ...] = (5.0, 10.0, 10.0, 10.0)
    Qx_lqr: Tuple[float, ...] = (20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)
    Ru_lqr: Tuple[float, ...] = (5, 10, 10, 10)

    enforce_perf_all_vertices: bool = True
    seed: int = 26

    fig_context: str = "paper"
    fig_column: str = "double"
    fig_formats: Tuple[str, ...] = ("pdf", "png")
    fig_dpi_png: int = 600
    fig_transparent: bool = False
    fig_show_titles: bool = True
    fig_out_dir: str = "acc_f8_formation_results"



# The F-8 driver installs its plant-specific matrix builder after importing
# this module. Keeping this hook explicit avoids carrying unused plant models
# in the public utility file.
def build_vertex_matrices(
        p: Dict[str, float],
        Ts: float,
        g: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("build_vertex_matrices must be supplied by the benchmark driver")


# =========================================================
# Performance output matrices
# =========================================================
def build_performance_matrices(syn: SynthesisParams) -> Tuple[np.ndarray, np.ndarray]:
    Qx_diag = np.array(syn.Qx_perf, dtype=float)
    Rd_diag = np.array(syn.Rd_perf, dtype=float)

    Cc = np.zeros((16, 16))
    dim_q = len(Qx_diag)
    Cc[0:dim_q, 0:dim_q] = np.diag(np.sqrt(Qx_diag))

    Dc = np.zeros((16, 4))
    Dc[12:16, 0:4] = np.diag(np.sqrt(Rd_diag))
    return Cc, Dc


# =========================================================
# Utilities
# =========================================================
def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = la.solve_discrete_are(A, B, Q, R)
    K = -la.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def saturate_norm(u: np.ndarray, umax: float) -> np.ndarray:
    n = float(np.linalg.norm(u))
    if n <= umax or n < 1e-12:
        return u
    return u * (umax / n)


def clip_vec(u: np.ndarray, umin: np.ndarray, umax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, umin), umax)


def colored_noise(dim: int, steps: int, alpha: float, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((dim, steps))
    for k in range(1, steps):
        x[:, k] = alpha * x[:, k - 1] + (1 - alpha) * rng.standard_normal(dim) * scale
    return x


def spectral_radius(M: np.ndarray) -> float:
    w = la.eigvals(M)
    return float(np.max(np.abs(w)))

#
def trapz_sum(y: np.ndarray, Ts: float) -> float:
    return float(np.sum(y) * Ts)


# =========================================================
# Data-consistency ellipsoid construction
# =========================================================
def simulate_batch_data(
        p_gen: Dict[str, float],
        syn: SynthesisParams,
        L: int = 1200,
        excite_scale: float = 1.2,
        meas_noise_std: float = 0.0005,
        *,
        K_fb: Optional[np.ndarray] = None,
        p_lqr: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Data-generation logic:
      - plant: p_gen determines the state update
        x_{k+1} = A_gen x_k + B_gen u_k + S_gen d_k.
      - controller: use K_fb when supplied; otherwise compute an LQR
        gain from the p_lqr model when available.
    """
    Ts, g = syn.Ts, syn.g
    d_max = syn.d_max
    du_max = syn.du_max
    rng = np.random.default_rng(syn.seed)


    A_gen, B_gen, S_gen = build_vertex_matrices(p_gen, Ts, g)

    Cc, Dc = build_performance_matrices(syn)

    Qlqr = np.diag(syn.Qx_lqr)
    Rlqr = np.diag(syn.Ru_lqr)

    if K_fb is not None:
        K0 = np.asarray(K_fb, dtype=float)
    else:
        p_for_lqr = p_lqr if p_lqr is not None else p_gen
        A_lqr, B_lqr, _ = build_vertex_matrices(p_for_lqr, Ts, g)
        K0 = dlqr(A_lqr, B_lqr, Qlqr, Rlqr)

    # Disturbance
    u_dither = colored_noise(4, L, alpha=0.92, scale=excite_scale, seed=syn.seed + 10)
    d_col = colored_noise(6, L, alpha=0.97, scale=1.0, seed=syn.seed + 20)

    # Simulation buffers
    x = np.zeros((16, L + 1))
    u_c = np.zeros((4, L))
    dbar = np.zeros((6, L))
    z = np.zeros((16, L))

    # Initial condition
    x0 = np.zeros(16)
    x0[0:3] = np.array([0.2, -0.2, 0.1])
    x0[6:9] = np.deg2rad([2.0, -2.0, 1.0])
    x[:, 0] = x0

    # Rollout
    for k in range(L):

        dbar[:, k] = saturate_norm(d_col[:, k], 1.0) * d_max


        uk = (K0 @ x[:, k]) + u_dither[:, k]
        uk = saturate_norm(uk, du_max)

        u_c[:, k] = uk
        z[:, k] = Cc @ x[:, k] + Dc @ uk


        x[:, k + 1] = A_gen @ x[:, k] + B_gen @ uk + S_gen @ dbar[:, k]
        x[:, k + 1] += rng.standard_normal(16) * meas_noise_std

    return dict(
        X_t=x[:, 0:L], X_tp1=x[:, 1:L + 1], U_t=u_c, Z_t=z, Dbar_t=dbar,
        # A/B/S_c are the matrices of the data-generating plant.
        A=A_gen, B=B_gen, S_c=S_gen, Cc=Cc, Dc=Dc, K0=K0
    )
def build_psi_data(batch: Dict[str, np.ndarray], syn: SynthesisParams, d_max_override: Optional[float] = None) -> np.ndarray:
    X_t = batch["X_t"]
    X_tp1 = batch["X_tp1"]
    U_t = batch["U_t"]
    Z_t = batch["Z_t"]
    S_c = batch["S_c"]

    L = X_t.shape[1]
    V = np.vstack([X_t, U_t])
    Xbreve = np.vstack([X_tp1, Z_t])

    d_max = d_max_override if d_max_override is not None else syn.d_max
    tilde_dbar = L * (d_max ** 2) * np.eye(6)
    top = S_c @ tilde_dbar @ S_c.T
    Dtilde = la.block_diag(top, np.zeros((16, 16)))

    Psi = np.block([
        [V @ V.T, -(V @ Xbreve.T)],
        [-(Xbreve @ V.T), (Xbreve @ Xbreve.T - Dtilde)]
    ])
    return 0.5 * (Psi + Psi.T)


def estimate_dmax_from_batch(
        batch: Dict[str, np.ndarray],
        A_nom: np.ndarray,
        B_nom: np.ndarray,
        S_for_est: np.ndarray,
        *,
        d_max_prev: float = 2.0,
        alpha: float = 0.01,
        delta: float = 1e-3,
        eta: float = 0.2,
        dmin: float = 0.1,
        dcap: float = 20.0,
        method: str = "quantile+dkw",
        fast_up: float = 1.2,
        slow_down: float = 0.95,
        cross_fit: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    X_t = batch["X_t"]
    X_tp1 = batch["X_tp1"]
    U_t = batch["U_t"]
    L = X_t.shape[1]

    if cross_fit and L >= 20:
        half = L // 2
        idx_est = np.arange(0, half)
        idx_build = np.arange(half, L)
        cross_fit_used = True
    else:
        idx_est = np.arange(L)
        idx_build = np.arange(L)
        cross_fit_used = False

    # Residual
    Res = X_tp1[:, idx_est] - (A_nom @ X_t[:, idx_est] + B_nom @ U_t[:, idx_est])

    # Disturbance-channel projection
    S_pinv = np.linalg.pinv(S_for_est)
    d_est = S_pinv @ Res
    d_norms = np.linalg.norm(d_est, axis=0)
    L_est = len(idx_est)

    # Printed diagnostics
    r_proj = S_for_est @ d_est
    r_unexpl = Res - r_proj
    unexpl_norms = np.linalg.norm(r_unexpl, axis=0)
    res_norms = np.linalg.norm(Res, axis=0) + 1e-12
    unexpl_ratios = unexpl_norms / res_norms
    unexpl_p50 = float(np.quantile(unexpl_ratios, 0.5))
    unexpl_p90 = float(np.quantile(unexpl_ratios, 0.9))
    sc_mismatch = unexpl_p90 > 0.5
    if sc_mismatch:
        print(f"  [AdaptDmax] WARNING: unexplained_ratio p90={unexpl_p90:.3f} > 0.5, "
              f"S_c channel structure may be mismatched!")

    # --- DKW confidence correction ---
    # DKW confidence correction for the optional adaptive d_max estimate.
    if method == "quantile+dkw" and L_est >= 2:
        eps_dkw = np.sqrt(np.log(2.0 / delta) / (2.0 * L_est))
        q_level = min(1.0 - alpha + eps_dkw, 0.999)
    else:
        q_level = min(1.0 - alpha, 0.999)

    dmax_est = float(np.quantile(d_norms, q_level))


    dmax_target = float(np.clip(dmax_est * (1.0 + eta), dmin, dcap))


    if dmax_target > d_max_prev:
        dmax_new = min(dmax_target, fast_up * d_max_prev)
    else:
        dmax_new = max(dmax_target, slow_down * d_max_prev)


    hit_cap = dmax_new >= dcap
    hit_floor = dmax_new <= dmin
    dmax_new = float(np.clip(dmax_new, dmin, dcap))


    hard_ratio = float(np.mean(d_norms <= dmax_new))

    diag = dict(
        dmax_est_raw=dmax_est,
        dmax_target=dmax_target,
        dmax_new=dmax_new,
        d_max_prev=d_max_prev,
        q_level=q_level,
        L_est=L_est,
        d_norms_max=float(np.max(d_norms)),
        d_norms_p99=float(np.quantile(d_norms, 0.99)),
        d_norms_median=float(np.median(d_norms)),
        hard_consistent_ratio=hard_ratio,
        hit_cap=hit_cap,
        hit_floor=hit_floor,
        unexplained_ratio_p50=unexpl_p50,
        unexplained_ratio_p90=unexpl_p90,
        sc_mismatch_warning=sc_mismatch,
        cross_fit_used=cross_fit_used,
        idx_build=idx_build,
    )

    print(f"  [AdaptDmax] d_max: {d_max_prev:.4f} -> {dmax_new:.4f} "
          f"(est_raw={dmax_est:.4f}, target={dmax_target:.4f}, q={q_level:.4f}, eta={eta})")
    print(f"  [AdaptDmax] hard_consistent_ratio={hard_ratio:.4f}, "
          f"d_norms: max={diag['d_norms_max']:.4f}, p99={diag['d_norms_p99']:.4f}")
    print(f"  [AdaptDmax] unexplained_ratio: p50={unexpl_p50:.4f}, p90={unexpl_p90:.4f}"
          f"{'  *** S_c MISMATCH ***' if sc_mismatch else ''}")
    if hit_cap:
        print(f"  [AdaptDmax] WARNING: d_max hit cap={dcap}")
    if hit_floor:
        print(f"  [AdaptDmax] WARNING: d_max hit floor={dmin}")

    return dmax_new, diag


# =========================================================
# Vertices + scores
# =========================================================
def enumerate_vertices(bounds: Any) -> List[Dict[str, float]]:
    keys = list(bounds.__dataclass_fields__.keys())
    vertices: List[Dict[str, float]] = []
    for bits in itertools.product([0, 1], repeat=len(keys)):
        p: Dict[str, float] = {}
        for key, bit in zip(keys, bits):
            lo, hi = getattr(bounds, key)
            p[key] = float(hi if bit else lo)
        vertices.append(p)
    return vertices


def farthest_point_select_vertices(
        all_vertices: List[Dict[str, object]],
        K_budget: int,
        bounds: Any,
        seed: int = 0,
) -> List[int]:
    """
    Select K_budget vertices by farthest-point sampling in parameter
    space. The selection uses parameter distances only.
    """
    keys = list(bounds.__dataclass_fields__.keys())
    # Normalize parameters to [0, 1].
    scales = []
    for k in keys:
        lo, hi = getattr(bounds, k)
        scales.append((lo, hi - lo if hi > lo else 1.0))

    N = len(all_vertices)
    coords = np.zeros((N, len(keys)))
    for i, v in enumerate(all_vertices):
        for j, k in enumerate(keys):
            coords[i, j] = (float(v["p"][k]) - scales[j][0]) / scales[j][1]

    rng = np.random.default_rng(seed)
    selected = [rng.integers(N)]
    min_dist = np.full(N, np.inf)

    for _ in range(K_budget - 1):
        last = coords[selected[-1]]
        d = np.sum((coords - last) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -1.0
        selected.append(int(np.argmax(min_dist)))

    return sorted(selected)

# Raw vertex-score scalar used before the soft-score normalization.
def compute_vertex_score_scalar(
        Delta_i: np.ndarray,
        Psi_data: np.ndarray,
        mode: str = "lambda_max",
) -> float:
    M = np.hstack([Delta_i, np.eye(32)])
    S = M @ Psi_data @ M.T
    S = 0.5 * (S + S.T)

    ev = eigvalsh(S)

    if mode == "trace":
        return float(-np.trace(S))
    elif mode == "lambda_max":
        return float(ev[-1])
    elif mode == "pos_eig_sum":
        return float(-np.sum(np.maximum(ev, 0.0)))
    elif mode == "min_eig":
        return float(ev[0])
    elif mode == "mean_eig":
        return float(np.mean(ev))
    else:
        raise ValueError("mode must be one of: trace / lambda_max / pos_eig_sum / min_eig / mean_eig")


def build_all_vertices_and_scores(
        bounds: Any,
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        score_mode: str = "lambda_max",
        max_vertices: Optional[int] = None,
        seed: int = 123,
) -> List[Dict[str, object]]:
    Ts, g = syn.Ts, syn.g
    Cc, Dc = build_performance_matrices(syn)

    verts = enumerate_vertices(bounds)
    if max_vertices is not None and max_vertices < len(verts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(verts), size=max_vertices, replace=False)
        verts = [verts[i] for i in idx]

    out = []
    for p in verts:
        A, B, S_c = build_vertex_matrices(p, Ts, g)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        s_i = compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode)
        out.append(dict(A=A, B=B, S=S_c, C=Cc, D=Dc, Delta=Delta, s=s_i, p=p))
    return out
def selfcheck_print_param_sraw(
        p: Dict[str, float],
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        *,
        label: str = "p_true",
        score_mode: str = "lambda_max",
        compare_scores: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the raw score for a parameter dictionary p, using the same
    definition as build_all_vertices_and_scores. If compare_scores is
    provided, also report the percentile within that score pool.
    """
    Ts, g = syn.Ts, syn.g
    A, B, _ = build_vertex_matrices(p, Ts, g)
    Cc, Dc = build_performance_matrices(syn)
    Delta = np.block([[A, B],
                      [Cc, Dc]])

    s_raw = compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode)

    msg = f"[SelfCheck] {label}: s_raw={s_raw:.6e} (mode={score_mode})"

    if compare_scores is not None and len(compare_scores) > 0:
        cs = np.asarray(compare_scores, dtype=float)
        cs_min = float(np.min(cs))
        cs_max = float(np.max(cs))
        rank_le = int(np.sum(cs <= s_raw))
        pct = 100.0 * rank_le / float(len(cs))   # Smaller means closer to the minimum score.
        msg += (f" | among pool: min={cs_min:.6e}, max={cs_max:.6e}, "
                f"percentile={pct:.2f}% (<=count {rank_le}/{len(cs)}), "
                f"gap_to_min={s_raw - cs_min:.3e}")

    print(msg)
    return float(s_raw)


def compute_si_from_vi(
        vertices: List[Dict[str, object]],
        L: int,
        *,
        n_hc: int = 0,
        tau: float = 0.0,
        q_scale: float = 0.9,
        eps: float = 1e-12,
        use_len_norm: bool = True,
        soft_fallback: bool = True,
        rho: float = 0.1,
        hard_ratio_min: float = 0.01,
        sigma_degenerate_thr: float = 1e-10,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    """Map raw residual scores r_i (vertex['s']) to soft scores s_i in
    [0, 1] per paper Eq. (4):
        s_i = clip( 1 - exp( -max(0, (r_i - tau_D) / L) / sigma ), 0, 1 ),
    with the convention s_i = 0 on the hard core
        I_hard = { i : r_i <= tau_D }.

    n_hc > 0 (preferred, used by the F-8 pipeline): tau_D is the n_hc-th
        order statistic of {r_i}, so |I_hard| = n_hc exactly. This is a
        direct implementation of paper Eq. (4); no post-promotion in
        downstream ICE is required.

    n_hc = 0 (legacy): keep tau_D = tau (default 0); if fewer than
        hard_ratio_min vertices satisfy r_i <= tau, fall back to the
        rho-quantile of {r_i}. Retained for backward compatibility with
        scripts that previously combined this routine with the ICE flag
        force_top_n_hard_core.
    """
    raw_s = np.array([float(v["s"]) for v in vertices], dtype=float)
    n = len(raw_s)
    soft_mode = False

    if int(n_hc) > 0:
        # --- Paper Eq. (4): tau_D = n_hc-th order statistic of {r_i};
        # I_hard = the n_hc lowest-residual vertices. No fallback. ---
        n_hc_eff = min(int(n_hc), n)
        order = np.argsort(raw_s)
        tau_effective = float(raw_s[order[n_hc_eff - 1]])
        hard_consistent_mask = np.zeros(n, dtype=bool)
        hard_consistent_mask[order[:n_hc_eff]] = True
        hard_ratio = float(n_hc_eff) / float(n)
        print(f"  [ScorePipeline] paper Eq.(4) mode: n_hc={n_hc_eff}, "
              f"tau_D={tau_effective:.6e}")
    else:
        # --- Legacy soft-fallback path (no n_hc supplied). ---
        hard_consistent_mask = raw_s <= tau
        hard_ratio = float(np.mean(hard_consistent_mask))
        tau_effective = tau

        if soft_fallback and hard_ratio < hard_ratio_min:
            tau_effective = float(np.quantile(raw_s, rho))
            hard_consistent_mask = raw_s <= tau_effective
            hard_ratio = float(np.mean(hard_consistent_mask))
            soft_mode = True
            print(f"  [ScorePipeline] SOFT-CONSISTENCY mode: "
                  f"hard_ratio(tau=0)={float(np.mean(raw_s <= tau)):.4f} < {hard_ratio_min}")
            print(f"  [ScorePipeline]   tau_effective={tau_effective:.6e} "
                  f"(rho={rho} quantile of raw scores)")

    p = np.maximum(0.0, raw_s - tau_effective)

    # Normalize the scale using the data length.
    if use_len_norm and L > 0:
        p = p / (float(L) + eps)

    # Quantile-based scaling and fallback check.
    p_pos = p[p > eps]
    if len(p_pos) == 0:
        sigma = 0.0
    else:
        sigma = float(np.quantile(p_pos, q_scale))

    uninformative = sigma < sigma_degenerate_thr

    if uninformative:
        s_final = np.zeros(n)
        print(f"  [ScorePipeline] UNINFORMATIVE: sigma={sigma:.3e} < {sigma_degenerate_thr:.1e}, "
              f"all s_i set to 0 (prior-only fallback)")
    else:
        r = p / (sigma + eps)
        s_final = 1.0 - np.exp(-r)

    # Map to [0, 1].
    s_final = np.clip(s_final, 0.0, 1.0)

    # Output.
    out = []
    for i, v in enumerate(vertices):
        vv = dict(v)
        vv["s_raw"] = float(v["s"])
        vv["s"] = float(s_final[i])
        out.append(vv)

    diag = dict(
        tau_effective=tau_effective,
        soft_mode=soft_mode,
        hard_ratio=hard_ratio,
        sigma=sigma,
        uninformative=uninformative,
        n_consistent=int(np.sum(s_final == 0.0)),
        n_total=n,
        s_min=float(np.min(s_final)),
        s_max=float(np.max(s_final)),
        s_mean=float(np.mean(s_final)),
    )

    print(f"  [ScorePipeline] tau_eff={tau_effective:.3e}, sigma={sigma:.3e}, "
          f"soft_mode={soft_mode}, uninformative={uninformative}")
    print(f"  [ScorePipeline] s: n_zero={diag['n_consistent']}/{n}, "
          f"min={diag['s_min']:.4f}, max={diag['s_max']:.4f}, mean={diag['s_mean']:.4f}")

    return out, diag


def sample_consistent_models(
        bounds: Any,
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        *,
        tau: float,
        score_mode: str = "lambda_max",
        n_samples: int = 8000,
        seed: int = 0,
) -> List[Dict[str, object]]:
    Ts, g = syn.Ts, syn.g
    Cc, Dc = build_performance_matrices(syn)
    keys = list(bounds.__dataclass_fields__.keys())
    rng = np.random.default_rng(int(seed))

    out = []
    for _ in range(int(n_samples)):
        p = {}
        for k in keys:
            lo, hi = getattr(bounds, k)
            p[k] = float(rng.uniform(lo, hi))

        A, B, S_c = build_vertex_matrices(p, Ts, g)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        s_raw = float(compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode))
        if s_raw <= float(tau):
            out.append(dict(A=A, B=B, S=S_c, C=Cc, D=Dc, Delta=Delta, s=s_raw, p=p))
    return out

# =========================================================
# Vertex subset selection
# =========================================================
def _param_vec_from_bounds(p: Dict[str, float], bounds: Any) -> np.ndarray:
    keys = list(bounds.__dataclass_fields__.keys())
    v = []
    for k in keys:
        lo, hi = getattr(bounds, k)
        den = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        v.append((float(p[k]) - lo) / den)
    return np.array(v, dtype=float)
def _delta_flat(v: Dict[str, object]) -> np.ndarray:
    """Flatten Delta to 1D vector."""
    return np.asarray(v["Delta"], dtype=float).reshape(-1)

# Whitening helper.
def _affine_whiten_from_columns(V: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an invertible affine scaling to column vectors:
    x -> (x - mu) / sig. V has shape (m, K).
    """
    mu = np.mean(V, axis=1, keepdims=True)
    sig = np.std(V, axis=1, keepdims=True)
    sig = np.where(sig < eps, 1.0, sig)
    Vw = (V - mu) / sig
    return Vw, mu, sig


def select_vertices_farthest_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        initial_index: int = 0,
) -> List[Dict[str, object]]:
    """
    Perform max-min farthest-point sampling in Delta space using
    Euclidean distance after an invertible affine whitening.
    """
    if len(vertices) == 0:
        return []
    if K >= len(vertices):
        return list(vertices)

    # X: (n, m)
    X = np.vstack([_delta_flat(v) for v in vertices])  # (n, m)
    # whiten per-coordinate (affine invertible)
    mu = np.mean(X, axis=0, keepdims=True)
    sig = np.std(X, axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xw = (X - mu) / sig

    n = Xw.shape[0]
    initial_index = int(np.clip(initial_index, 0, n - 1))

    selected = [initial_index]
    dist2 = np.sum((Xw - Xw[initial_index]) ** 2, axis=1)

    for _ in range(1, K):
        j = int(np.argmax(dist2))
        selected.append(j)
        dist2 = np.minimum(dist2, np.sum((Xw - Xw[j]) ** 2, axis=1))

    # Remove duplicates while preserving order.
    out, seen = [], set()
    for i in selected:
        if i not in seen:
            out.append(vertices[i])
            seen.add(i)
    return out

def select_vertices_support_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        n_dirs: int = 300,
        seed: int = 0,
        initial_index: int = 0,
) -> List[Dict[str, object]]:
    """
    Select support points in Delta space by taking extrema along random
    directions. This approximates an outer convex hull more directly
    than pure farthest-point sampling.
    """
    if len(vertices) == 0:
        return []
    if K >= len(vertices):
        return list(vertices)

    # X: (n, m)
    X = np.vstack([_delta_flat(v) for v in vertices])  # (n, m)

    # Invertible affine whitening improves conditioning and preserves hull membership.
    mu = np.mean(X, axis=0, keepdims=True)
    sig = np.std(X, axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xw = (X - mu) / sig

    rng = np.random.default_rng(int(seed))
    m = Xw.shape[1]
    dirs = rng.standard_normal((int(n_dirs), m))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    idxs = set()
    for d in dirs:
        proj = Xw @ d
        idxs.add(int(np.argmax(proj)))
        idxs.add(int(np.argmin(proj)))
        if len(idxs) >= 5 * K:
            break

    cand = [vertices[i] for i in sorted(idxs)]

    # Too many candidates: reduce to K points with farthest sampling.
    if len(cand) > K:
        cand = select_vertices_farthest_delta(cand, K=K, initial_index=int(initial_index))
    # Too few candidates: fill from the remaining vertices.
    elif len(cand) < K:
        rest = [vertices[i] for i in range(len(vertices)) if i not in idxs]
        if len(rest) > 0:
            extra = select_vertices_farthest_delta(rest, K=min(K - len(cand), len(rest)), initial_index=0)
            cand = cand + extra

    return cand[:K]

def select_vertices_stratified_farthest_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        seed: int = 0,
) -> List[Dict[str, object]]:
    """
    Split vertices into low, middle, and high score tiers, then apply
    Delta-space farthest-point sampling within each tier.
    """
    if len(vertices) == 0:
        return []
    if K >= len(vertices):
        return list(vertices)

    ratios = np.array(ratios, dtype=float)
    if ratios.size != 3 or np.any(ratios < 0) or float(np.sum(ratios)) <= 1e-12:
        ratios = np.array([0.4, 0.2, 0.4], dtype=float)
    ratios = ratios / float(np.sum(ratios))

    k_low = int(round(K * float(ratios[0])))
    k_mid = int(round(K * float(ratios[1])))
    k_high = int(K - k_low - k_mid)
    k_low = max(1, min(k_low, K - 2))
    k_mid = max(0, min(k_mid, K - k_low - 1))
    k_high = int(K - k_low - k_mid)

    s = np.array([float(v["s"]) for v in vertices], dtype=float)
    order = np.argsort(s)
    verts_sorted = [vertices[i] for i in order]
    n = len(verts_sorted)

    cut1 = int(np.floor(n / 3))
    cut2 = int(np.floor(2 * n / 3))
    low = verts_sorted[:max(1, cut1)]
    mid = verts_sorted[max(1, cut1):max(cut2, cut1 + 1)]
    high = verts_sorted[max(cut2, cut1 + 1):]

    rng = np.random.default_rng(int(seed))

    sel_low = select_vertices_farthest_delta(low, K=min(k_low, len(low)), initial_index=0)

    sel_mid = []
    if k_mid > 0 and len(mid) > 0:
        init = int(rng.integers(0, len(mid)))
        sel_mid = select_vertices_farthest_delta(mid, K=min(k_mid, len(mid)), initial_index=init)

    sel_high = []
    if k_high > 0 and len(high) > 0:
        init = int(rng.integers(0, len(high)))
        sel_high = select_vertices_farthest_delta(high, K=min(k_high, len(high)), initial_index=init)

    out = sel_low + sel_mid + sel_high

    if len(out) < K:
        seen = {id(v) for v in out}
        rest = [v for v in verts_sorted if id(v) not in seen]
        if len(rest) > 0:
            need = K - len(out)
            extra = select_vertices_farthest_delta(rest, K=min(need, len(rest)), initial_index=0)
            out.extend(extra)

    return out[:K]


def select_vertices_farthest(
        vertices: List[Dict[str, object]],
        bounds: Any,
        K: int,
        seed: int = 0,
        initial_index: int = 0,
) -> List[Dict[str, object]]:
    if K >= len(vertices):
        return list(vertices)
    if len(vertices) == 0:
        return []

    X = np.vstack([_param_vec_from_bounds(v["p"], bounds) for v in vertices])
    initial_index = int(np.clip(initial_index, 0, len(vertices) - 1))

    selected = [initial_index]
    dist2 = np.sum((X - X[initial_index]) ** 2, axis=1)

    for _ in range(1, K):
        j = int(np.argmax(dist2))
        selected.append(j)
        dist2 = np.minimum(dist2, np.sum((X - X[j]) ** 2, axis=1))

    seen = set()
    out = []
    for i in selected:
        if i not in seen:
            out.append(vertices[i])
            seen.add(i)
    return out


def select_vertices_stratified_farthest(
        vertices: List[Dict[str, object]],
        bounds: Any,
        K: int = 20,
        *,
        ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        seed: int = 0,
) -> List[Dict[str, object]]:
    if K >= len(vertices):
        return list(vertices)
    if len(vertices) == 0:
        return []

    ratios = np.array(ratios, dtype=float)
    if ratios.size != 3 or np.any(ratios < 0) or float(np.sum(ratios)) <= 1e-12:
        ratios = np.array([0.4, 0.2, 0.4], dtype=float)
    ratios = ratios / float(np.sum(ratios))

    k_low = int(round(K * float(ratios[0])))
    k_mid = int(round(K * float(ratios[1])))
    k_high = int(K - k_low - k_mid)
    k_low = max(1, min(k_low, K - 2))
    k_mid = max(0, min(k_mid, K - k_low - 1))
    k_high = int(K - k_low - k_mid)

    s = np.array([float(v["s"]) for v in vertices], dtype=float)
    order = np.argsort(s)
    verts_sorted = [vertices[i] for i in order]
    n = len(verts_sorted)

    cut1 = int(np.floor(n / 3))
    cut2 = int(np.floor(2 * n / 3))
    low = verts_sorted[:max(1, cut1)]
    mid = verts_sorted[max(1, cut1):max(cut2, cut1 + 1)]
    high = verts_sorted[max(cut2, cut1 + 1):]

    rng = np.random.default_rng(int(seed))

    sel_low = select_vertices_farthest(low, bounds, K=min(k_low, len(low)), seed=int(seed) + 11, initial_index=0)

    sel_mid = []
    if k_mid > 0 and len(mid) > 0:
        init = int(rng.integers(0, len(mid)))
        sel_mid = select_vertices_farthest(mid, bounds, K=min(k_mid, len(mid)), seed=int(seed) + 22, initial_index=init)

    sel_high = []
    if k_high > 0 and len(high) > 0:
        init = int(rng.integers(0, len(high)))
        sel_high = select_vertices_farthest(high, bounds, K=min(k_high, len(high)), seed=int(seed) + 33,
                                            initial_index=init)

    out = sel_low + sel_mid + sel_high

    if len(out) < K:
        seen = {id(v) for v in out}
        rest = [v for v in verts_sorted if id(v) not in seen]
        if len(rest) > 0:
            need = K - len(out)
            extra = select_vertices_farthest(rest, bounds, K=min(need, len(rest)), seed=int(seed) + 44, initial_index=0)
            out.extend(extra)

    return out[:K]


# =========================================================
# Robust synthesis (MOSEK)
# =========================================================
def _level_to_float(x) -> float:
    arr = np.array(x).reshape(-1)
    return float(arr[0])


def _pick_order_by_symmetry(mat_c: np.ndarray, mat_f: np.ndarray) -> str:
    sym_c = float(np.linalg.norm(mat_c - mat_c.T))
    sym_f = float(np.linalg.norm(mat_f - mat_f.T))
    return "C" if sym_c <= sym_f else "F"


def _level_to_matrix(var, shape: Tuple[int, int], prefer_order: Optional[str] = None) -> np.ndarray:
    arr = np.array(var.level(), dtype=float).reshape(-1)
    if arr.size != shape[0] * shape[1]:
        raise ValueError(f"Unexpected level size {arr.size} for shape {shape}")

    mat_c = arr.reshape(shape, order="C")
    mat_f = arr.reshape(shape, order="F")

    if prefer_order is None:
        if shape[0] == shape[1]:
            order = _pick_order_by_symmetry(mat_c, mat_f)
        else:
            order = "C"
    else:
        order = prefer_order

    return mat_c if order == "C" else mat_f


def solve_vertex_fusion_sdp_mosek(
        vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        verbose: bool = True,
        eps_Q: float = 1e-6,
        active_indices: Optional[Sequence[int]] = None,
        num_threads: int = 0,
        max_iters: int = 50,
        rel_gap: float = 1e-4,
        time_limit_sec: Optional[int] = 900,
        beta_lb: float = 0.0,
        decay_rate: Optional[float] = None,
        w_gamma: float = 1.0,
        w_mu: float = 0.1,
        enforce_perf_all_vertices: bool = True,
        x0_feas: Optional[np.ndarray] = None,
        gamma2_ub: Optional[float] = None,
        fixed_beta_values: Optional[object] = None,
) -> Dict[str, np.ndarray]:
    # MOSEK license: configure_mosek_license() honours MOSEKLM_LICENSE_FILE
    # and caller-provided paths. This call contains no hard-coded path.
    configure_mosek_license(verbose=True)
    import mosek.fusion as mf

    du_max = syn.du_max
    d_max = syn.d_max
    if decay_rate is None:
        decay_rate = getattr(syn, "decay_rate", 0.98)

    if active_indices is None:
        active_indices = list(range(len(vertices)))
    else:
        active_indices = list(active_indices)

    M = mf.Model("vertex_fusion")
    try:
        return _solve_vertex_fusion_inner(M, vertices, syn, verbose, eps_Q,
            active_indices, num_threads, max_iters, rel_gap, time_limit_sec,
            beta_lb, decay_rate, w_gamma, w_mu, enforce_perf_all_vertices, x0_feas, gamma2_ub, fixed_beta_values)
    finally:
        M.dispose()


def _solve_vertex_fusion_inner(M, vertices, syn, verbose, eps_Q,
        active_indices, num_threads, max_iters, rel_gap, time_limit_sec,
        beta_lb, decay_rate, w_gamma, w_mu, enforce_perf_all_vertices, x0_feas, gamma2_ub, fixed_beta_values):
    import mosek.fusion as mf
    du_max = syn.du_max
    d_max = syn.d_max

    if verbose:
        M.setLogHandler(sys.stdout)

    try:
        if num_threads is None or num_threads <= 0:
            ncpu = os.cpu_count() or 1
            num_threads = int(min(8, max(1, ncpu)))
        M.setSolverParam("numThreads", int(num_threads))
    except Exception:
        pass

    try:
        M.setSolverParam("intpntMaxIterations", int(max_iters))
        M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
        M.setSolverParam("intpntCoTolPfeas", float(getattr(syn, "mosek_pfeas", 1e-6)))
        M.setSolverParam("intpntCoTolDfeas", float(getattr(syn, "mosek_dfeas", 1e-6)))
        solve_form = str(getattr(syn, "mosek_solve_form", "free")).lower()
        if solve_form in ("free", "primal", "dual"):
            M.setSolverParam("intpntSolveForm", solve_form)
        if time_limit_sec is not None and time_limit_sec > 0:
            M.setSolverParam("optimizerMaxTime", float(time_limit_sec))
    except Exception:
        pass

    Q = M.variable("Q", mf.Domain.inPSDCone(16))
    Y = M.variable("Y", [4, 16], mf.Domain.unbounded())
    beta_ub = float(getattr(syn, "beta_ub", 1000.0))
    beta_score_tol = float(getattr(syn, "beta_score_tol", 1e-9))
    fixed_beta_arr = None
    if fixed_beta_values is not None:
        fixed_beta_arr = np.asarray(fixed_beta_values, dtype=float).reshape(-1)
        if fixed_beta_arr.size == 1:
            fixed_beta_arr = np.full(len(active_indices), float(fixed_beta_arr[0]), dtype=float)
        if fixed_beta_arr.size != len(active_indices):
            raise ValueError("fixed_beta_values must be scalar or match active vertex count")
    per_vertex_beta = bool(getattr(syn, "per_vertex_beta", False)) and fixed_beta_arr is None
    n_beta = len(active_indices) if per_vertex_beta else 1
    if fixed_beta_arr is not None:
        pass
    elif per_vertex_beta:
        beta_vars = []
        beta_scales = []
        for j, vidx in enumerate(active_indices):
            s_j = max(float(vertices[vidx]["s"]), 0.0)
            if s_j <= beta_score_tol:
                beta_vars.append(None)
                beta_scales.append(0.0)
            else:
                beta_vars.append(M.variable(f"eta_{j}", mf.Domain.inRange(float(beta_lb) * s_j, beta_ub * s_j)))
                beta_scales.append(s_j)
    else:
        beta = M.variable("beta", mf.Domain.inRange(float(beta_lb), beta_ub))

    gamma2_scale = float(getattr(syn, "gamma2_scale", 1.0))
    if not np.isfinite(gamma2_scale) or gamma2_scale <= 0.0:
        raise ValueError("gamma2_scale must be a finite positive scalar")
    gamma2_sqrt_scale = float(np.sqrt(gamma2_scale))
    gamma2 = M.variable("gamma2_scaled", mf.Domain.greaterThan(0.0))
    mu = M.variable("mu", mf.Domain.greaterThan(1e-9))
    if gamma2_ub is not None and np.isfinite(float(gamma2_ub)):
        M.constraint("gamma2_ub", gamma2, mf.Domain.lessThan(float(gamma2_ub) / gamma2_scale))

    I16d = mf.Matrix.dense(np.eye(16))
    I6d = mf.Matrix.dense(np.eye(6))
    I4d = mf.Matrix.dense(np.eye(4))

    I6e = mf.Expr.constTerm(I6d)
    I4e = mf.Expr.constTerm(I4d)
    epsI16e = mf.Expr.constTerm(mf.Matrix.dense(eps_Q * np.eye(16)))
    lmi_margin = float(getattr(syn, "lmi_margin", 0.0))
    if not np.isfinite(lmi_margin) or lmi_margin < 0.0:
        raise ValueError("lmi_margin must be a finite nonnegative scalar")
    lmi_margin_e = mf.Expr.constTerm(mf.Matrix.dense(lmi_margin * np.eye(54)))

    _Z_cache: Dict[Tuple[int, int], mf.Expression] = {}

    def Z(r: int, c: int):
        key = (r, c)
        if key not in _Z_cache:
            _Z_cache[key] = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((r, c))))
        return _Z_cache[key]

    M.constraint("Q_pd", mf.Expr.sub(Q, epsI16e), mf.Domain.inPSDCone(16))

    if x0_feas is not None:
        x0_feas = np.asarray(x0_feas, dtype=float).reshape(-1)
        assert x0_feas.shape[0] == 16
        one = mf.Matrix.dense([[1.0]])
        xrow = mf.Matrix.dense(x0_feas.reshape(1, 16))
        xcol = mf.Matrix.dense(x0_feas.reshape(16, 1))
        row1 = mf.Expr.hstack([mf.Expr.constTerm(one), mf.Expr.constTerm(xrow)])
        row2 = mf.Expr.hstack([mf.Expr.constTerm(xcol), Q])
        Feas = mf.Expr.vstack([row1, row2])
        M.constraint("feas_x0", Feas, mf.Domain.inPSDCone(17))

    for j, vidx in enumerate(active_indices):
        v = vertices[vidx]
        Ai = v["A"];
        Bi = v["B"];
        Ci = v["C"];
        Di = v["D"]
        Sci = v["S"] * d_max
        Sci_lmi = Sci / gamma2_sqrt_scale
        s_i = float(v["s"])

        Ai_m = mf.Matrix.dense(Ai)
        Bi_m = mf.Matrix.dense(Bi)
        Ci_m = mf.Matrix.dense(Ci)
        Di_m = mf.Matrix.dense(Di)
        Sci_m = mf.Matrix.dense(Sci_lmi)
        Sci_e = mf.Expr.constTerm(Sci_m)

        AQ_BY = mf.Expr.add(mf.Expr.mul(Ai_m, Q), mf.Expr.mul(Bi_m, Y))
        CQ_DY = mf.Expr.add(mf.Expr.mul(Ci_m, Q), mf.Expr.mul(Di_m, Y))

        if fixed_beta_arr is not None:
            beta_term = mf.Expr.constTerm(mf.Matrix.dense((-float(fixed_beta_arr[j]) * s_i) * np.eye(16)))
        else:
            if per_vertex_beta:
                eta_j = beta_vars[j]
                if eta_j is None:
                    beta_term = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((16, 16))))
                else:
                    beta_term = mf.Expr.mul(mf.Matrix.dense(-np.eye(16)), eta_j)
            else:
                beta_term = mf.Expr.mul(mf.Matrix.dense((-1.0 * s_i) * np.eye(16)), beta)
        tl = mf.Expr.add(mf.Expr.mul(-float(decay_rate), Q), beta_term)

        row1 = mf.Expr.hstack([tl, Z(16, 6), mf.Expr.transpose(AQ_BY), mf.Expr.transpose(CQ_DY)])
        row3 = mf.Expr.hstack([AQ_BY, Sci_e, mf.Expr.neg(Q), Z(16, 16)])
        I16e = mf.Expr.constTerm(I16d)

        gamW = mf.Expr.mul(I6d, gamma2)
        row2 = mf.Expr.hstack([Z(6, 16), mf.Expr.neg(gamW), mf.Expr.transpose(Sci_e), Z(6, 16)])

        row4 = mf.Expr.hstack([CQ_DY, Z(16, 6), Z(16, 16), mf.Expr.neg(I16e)])

        LMI = mf.Expr.vstack([row1, row2, row3, row4])
        M.constraint(f"vertex_lmi_{j}", mf.Expr.sub(mf.Expr.neg(LMI), lmi_margin_e), mf.Domain.inPSDCone(54))

        if enforce_perf_all_vertices:
            muI = mf.Expr.mul(I16d, mu)
            rowp1 = mf.Expr.hstack([Q, mf.Expr.transpose(CQ_DY)])
            rowp2 = mf.Expr.hstack([CQ_DY, muI])
            PERF = mf.Expr.vstack([rowp1, rowp2])
            M.constraint(f"perf_lmi_{j}", PERF, mf.Domain.inPSDCone(32))

    if (not enforce_perf_all_vertices) and (len(active_indices) > 0):
        v0 = vertices[active_indices[0]]
        Ci = v0["C"];
        Di = v0["D"]
        Ci_m = mf.Matrix.dense(Ci)
        Di_m = mf.Matrix.dense(Di)
        CQ_DY = mf.Expr.add(mf.Expr.mul(Ci_m, Q), mf.Expr.mul(Di_m, Y))
        muI = mf.Expr.mul(I16d, mu)
        rowp1 = mf.Expr.hstack([Q, mf.Expr.transpose(CQ_DY)])
        rowp2 = mf.Expr.hstack([CQ_DY, muI])
        PERF = mf.Expr.vstack([rowp1, rowp2])
        M.constraint("perf_lmi_nominal", PERF, mf.Domain.inPSDCone(32))

    rowu1 = mf.Expr.hstack([mf.Expr.mul(float(du_max ** 2), Q), mf.Expr.transpose(Y)])
    rowu2 = mf.Expr.hstack([Y, I4e])
    UINC = mf.Expr.vstack([rowu1, rowu2])
    M.constraint("input_inc", UINC, mf.Domain.inPSDCone(20))

    term_main = mf.Expr.add(mf.Expr.mul(float(w_gamma) * gamma2_scale, gamma2), mf.Expr.mul(float(w_mu), mu))
    if fixed_beta_arr is not None:
        beta_penalty = mf.Expr.constTerm(float(np.mean(fixed_beta_arr)))
    elif per_vertex_beta:
        beta_penalty = mf.Expr.constTerm(0.0)
        for bj, sj in zip(beta_vars, beta_scales):
            if bj is not None:
                beta_penalty = mf.Expr.add(beta_penalty, mf.Expr.mul(1.0 / sj, bj))
        beta_penalty = mf.Expr.mul(1.0 / float(max(1, n_beta)), beta_penalty)
    else:
        beta_penalty = beta
    w_beta = float(getattr(syn, "w_beta", 1e-3))
    obj = mf.Expr.add(term_main, mf.Expr.mul(w_beta, beta_penalty))
    M.objective("min_obj", mf.ObjectiveSense.Minimize, obj)
    try:
        M.solve()

        sol_status = M.getProblemStatus()
        primal_status = M.getPrimalSolutionStatus()
        dual_status = M.getDualSolutionStatus()
        solver_optimal = (
            sol_status == mf.ProblemStatus.PrimalAndDualFeasible
            and primal_status == mf.SolutionStatus.Optimal
            and dual_status == mf.SolutionStatus.Optimal
        )
        if sol_status != mf.ProblemStatus.PrimalAndDualFeasible:
            print(f"  Warning: solver status {sol_status}, result may be suboptimal")
        if not solver_optimal:
            print(f"  Warning: solution status primal={primal_status}, dual={dual_status}")

        Qc = _level_to_matrix(Q, (16, 16), prefer_order=None)
        Qf = _level_to_matrix(Q, (16, 16), prefer_order="F")
        order = _pick_order_by_symmetry(Qc, Qf)
        Qv = _level_to_matrix(Q, (16, 16), prefer_order=order)
        Yv = _level_to_matrix(Y, (4, 16), prefer_order="C")
        Qv = 0.5 * (Qv + Qv.T)

        if fixed_beta_arr is not None:
            betav = fixed_beta_arr.copy()
        elif per_vertex_beta:
            beta_items = []
            eta_items = []
            for bj, sj in zip(beta_vars, beta_scales):
                if bj is None:
                    eta_items.append(0.0)
                    beta_items.append(0.0)
                else:
                    eta_v = max(_level_to_float(bj.level()), 0.0)
                    eta_items.append(eta_v)
                    beta_items.append(eta_v / sj)
            betav = np.array(beta_items, dtype=float)
            etav = np.array(eta_items, dtype=float)
        else:
            betav = np.array([max(_level_to_float(beta.level()), 0.0)], dtype=float)
            etav = betav.copy()
        g2_internal = _level_to_float(gamma2.level())
        g2v = gamma2_scale * g2_internal
        muv = _level_to_float(mu.level())

        if np.isnan(g2v) or np.isinf(g2v):
            raise ValueError("Solver returned NaN/Inf metrics")

        Kv = (la.solve(Qv.T, Yv.T)).T
        active_vertices = [vertices[int(ii)] for ii in active_indices]
        lmi_max_violation = float(np.max(check_all_vertex_violations(
            active_vertices, Kv, Qv, g2v, betav, float(decay_rate), d_max
        )))

        return dict(
            Q=Qv,
            Y=Yv,
            K=Kv,
            beta=betav,
            beta_s=etav if per_vertex_beta else betav.copy(),
            beta_mean=np.array([float(np.mean(betav))]),
            beta_max=np.array([float(np.max(betav))]),
            per_vertex_beta=np.array([1 if (per_vertex_beta or fixed_beta_arr is not None) else 0]),
            fixed_beta=np.array([1 if fixed_beta_arr is not None else 0]),
            active_indices=np.asarray(active_indices, dtype=int),
            gamma2=np.array([g2v]),
            mu=np.array([muv]),
            decay_rate=np.array([float(decay_rate)]),
            solver_threads=np.array([num_threads]),
            solver_max_iters=np.array([max_iters]),
            solver_rel_gap=np.array([rel_gap]),
            solver_time_limit=np.array([time_limit_sec]),
            beta_lb=np.array([beta_lb]),
            gamma2_ub=np.array([float(gamma2_ub) if gamma2_ub is not None else np.nan]),
            gamma2_scale=np.array([gamma2_scale]),
            gamma2_internal=np.array([g2_internal]),
            lmi_margin=np.array([lmi_margin]),
            w_gamma=np.array([w_gamma]),
            w_mu=np.array([w_mu]),
            w_beta=np.array([w_beta]),
            enforce_perf_all_vertices=np.array([1 if enforce_perf_all_vertices else 0]),
            solver_problem_status=np.array([str(sol_status)]),
            solver_primal_status=np.array([str(primal_status)]),
            solver_dual_status=np.array([str(dual_status)]),
            lmi_max_violation=np.array([lmi_max_violation]),
            success=bool(solver_optimal)
        )

    except Exception as e:
        print(f"  Optimization failed: {str(e)}")
        print("  Returning dummy solution")

        return dict(
            Q=np.eye(16),
            Y=np.zeros((4, 16)),
            K=np.zeros((4, 16)),
            beta=np.zeros(int(len(active_indices) if fixed_beta_arr is not None else max(1, n_beta)), dtype=float),
            beta_mean=np.array([0.0]),
            beta_max=np.array([0.0]),
            per_vertex_beta=np.array([1 if (per_vertex_beta or fixed_beta_arr is not None) else 0]),
            fixed_beta=np.array([1 if fixed_beta_arr is not None else 0]),
            active_indices=np.asarray(active_indices, dtype=int),
            gamma2=np.array([1e9]),
            mu=np.array([1e9]),
            decay_rate=np.array([float(decay_rate)]),
            solver_threads=np.array([num_threads]),
            solver_max_iters=np.array([max_iters]),
            solver_rel_gap=np.array([rel_gap]),
            solver_time_limit=np.array([time_limit_sec]),
            beta_lb=np.array([beta_lb]),
            gamma2_ub=np.array([float(gamma2_ub) if gamma2_ub is not None else np.nan]),
            w_gamma=np.array([w_gamma]),
            w_mu=np.array([w_mu]),
            w_beta=np.array([float(getattr(syn, "w_beta", 1e-3))]),
            enforce_perf_all_vertices=np.array([0]),
            success=False
        )


# =========================================================
# Iterative Constraint Exchange (Cutting-Plane style)
# =========================================================
def eval_vertex_lmi_violation(
        v: Dict[str, object],
        K: np.ndarray,
        Q: np.ndarray,
        gamma2_val: float,
        beta_val: float,
        decay_rate: float,
        d_max: float,
) -> float:
    """
    Compute the LMI residual for a single vertex given
    (K, Q, gamma2, beta). Values above zero indicate violation.
    """
    Ai = v["A"]; Bi = v["B"]; Ci = v["C"]; Di = v["D"]
    Sci = v["S"] * d_max
    s_i = float(v["s"])

    Y = K @ Q
    AQ_BY = Ai @ Q + Bi @ Y
    CQ_DY = Ci @ Q + Di @ Y

    n = 16; nw = 6
    tl = -decay_rate * Q - beta_val * s_i * np.eye(n)

    LMI = np.zeros((n + nw + n + n, n + nw + n + n))
    # row/col 0:16, 16:22, 22:38, 38:54
    # Block (1,1)
    LMI[0:n, 0:n] = tl
    # Block (1,3)
    LMI[0:n, n+nw:n+nw+n] = AQ_BY.T
    # Block (1,4)
    LMI[0:n, n+nw+n:] = CQ_DY.T
    # Block (2,2)
    LMI[n:n+nw, n:n+nw] = -gamma2_val * np.eye(nw)
    # Block (2,3)
    LMI[n:n+nw, n+nw:n+nw+n] = Sci.T
    # Block (3,1)
    LMI[n+nw:n+nw+n, 0:n] = AQ_BY
    # Block (3,2)
    LMI[n+nw:n+nw+n, n:n+nw] = Sci
    # Block (3,3)
    LMI[n+nw:n+nw+n, n+nw:n+nw+n] = -Q
    # Block (4,1)
    LMI[n+nw+n:, 0:n] = CQ_DY
    # Block (4,4)
    LMI[n+nw+n:, n+nw+n:] = -np.eye(n)

    LMI = 0.5 * (LMI + LMI.T)
    # LMI <= 0 requires all eigenvalues <= 0; the residual is the maximum eigenvalue.
    return float(eigvalsh(LMI)[-1])


def check_all_vertex_violations(
        all_vertices: List[Dict[str, object]],
        K: np.ndarray,
        Q: np.ndarray,
        gamma2_val: float,
        beta_val: object,
        decay_rate: float,
        d_max: float,
) -> np.ndarray:
    """
    Evaluate the LMI residual over all vertices. Positive entries
    indicate violation.
    """
    violations = np.zeros(len(all_vertices))
    beta_arr = np.asarray(beta_val, dtype=float).reshape(-1)
    for i, v in enumerate(all_vertices):
        beta_i = float(beta_arr[i]) if beta_arr.size == len(all_vertices) else float(beta_arr[0])
        violations[i] = eval_vertex_lmi_violation(
            v, K, Q, gamma2_val, beta_i, decay_rate, d_max
        )
    return violations


def iterative_constraint_exchange(
        all_vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        K_budget: int = 20,
        max_rounds: int = 8,
        viol_tol: float = 1e-4,
        seed: int = 42,
        decay_rate: Optional[float] = None,
        w_gamma: float = 1.0,
        w_mu: float = 0.1,
        enforce_perf_all_vertices: bool = True,
        verbose: bool = True,
        bounds: Optional[Any] = None,
        stratified_ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        m_incon_override: Optional[int] = None,
        x0_feas: Optional[np.ndarray] = None,
        force_include_hard_core: bool = False,
        hard_core_s_threshold: float = 1e-9,
        force_top_n_hard_core: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """
    Iterative Constraint Exchange (ICE) -- stratified version.

    Reference: paper Algorithm 1 + Eq. (4) (hard-core definition).
    The hard core in the paper is
        I_hard = { i : r_i <= tau_D },  tau_D = n_hc-th order statistic of {r_i}
    i.e. the n_hc lowest-residual vertices.

    The preferred F-8 pipeline materialises the hard core upstream via
    compute_si_from_vi(n_hc=...), which sets s_i = 0 on the n_hc
    lowest-residual vertices (paper Eq. (4), 1:1). In that mode, this
    routine only protects the existing hard-core vertices when
    force_include_hard_core=True. The optional force_top_n_hard_core
    argument is retained only for legacy scripts and should normally be
    left at zero.

    Implementation notes:
    - The initial active set is selected by stratified farthest sampling.
    - Swaps preserve tier quotas and keep at least m_incon vertices with
      s > 0 in the active set.
    - If the worst violator is already active, the next-largest violator
      is considered.
    - best_sol is selected using (max_violation, n_violated, gamma).
    - m_incon_override replaces the lower bound on m_incon, for example
      when all scores are zero in a no-relaxation run.
    - If force_include_hard_core is True, all hard-core vertices with
      s_i <= hard_core_s_threshold are included initially, provided the
      hard core fits in K_budget. Hard-core vertices are not swapped out.
    """
    if decay_rate is None:
        decay_rate = getattr(syn, "decay_rate", 0.98)

    N = len(all_vertices)
    K_budget = min(K_budget, N)

    s_all = np.array([float(v["s"]) for v in all_vertices], dtype=float)

    # --- Hard-core selection per paper Eq. (4): I_hard = the n_hc
    # lowest-residual vertices, i.e. the n_hc-th order statistic of {r_i}.
    # Equivalent here to the n_hc lowest-s_i vertices because
    # compute_si_from_vi() above is monotone in r_i. We materialise this
    # by forcing s_i = 0 on these n_hc = force_top_n_hard_core vertices,
    # which makes the slack term s_i*beta*I in (5) vanish (unrelaxed LMI)
    # while leaving the remaining s_i untouched for the soft shell. ---
    if int(force_top_n_hard_core) > 0:
        n_force = min(int(force_top_n_hard_core), N)
        promote = np.argsort(s_all)[:n_force]
        for i in promote.tolist():
            all_vertices[i]["s"] = 0.0
            s_all[i] = 0.0
        if verbose:
            print(f"[ICE] force_top_n_hard_core={n_force}: "
                  f"promoted indices {sorted(promote.tolist())} to s=0")

    # --- Tier classification: low(0) / mid(1) / high(2) by s rank ---
    order = np.argsort(s_all)
    tier = np.zeros(N, dtype=int)
    n3 = max(1, N // 3)
    for rank, idx in enumerate(order):
        if rank < n3:
            tier[idx] = 0
        elif rank < 2 * n3:
            tier[idx] = 1
        else:
            tier[idx] = 2

    if m_incon_override is not None:
        m_incon = int(m_incon_override)
    else:
        m_incon = max(1, int(np.ceil(0.2 * K_budget)))

    # --- Hard-core indices (s_i <= threshold). Always identified, but only
    # forced into the active set when force_include_hard_core is True. ---
    hard_core_idx = {i for i in range(N) if s_all[i] <= float(hard_core_s_threshold)}
    n_hard = len(hard_core_idx)
    use_force_hc = bool(force_include_hard_core) and n_hard > 0 and n_hard <= K_budget

    # --- Initial selection: stratified farthest-point sampling, with optional
    # hard-core enforcement (initial active set must contain every hard-core
    # vertex; remaining budget is filled by stratified farthest-point on the
    # soft complement). ---
    if bounds is not None:
        if use_force_hc:
            active_idx = set(hard_core_idx)
            soft_verts = [all_vertices[i] for i in range(N) if i not in hard_core_idx]
            need = K_budget - n_hard
            if need > 0 and len(soft_verts) > 0:
                sel_soft = select_vertices_stratified_farthest(
                    soft_verts, bounds, need,
                    ratios=stratified_ratios, seed=seed,
                )
                id_to_idx = {id(v): i for i, v in enumerate(all_vertices)}
                for v in sel_soft:
                    vid = id(v)
                    if vid in id_to_idx:
                        active_idx.add(id_to_idx[vid])
            if len(active_idx) < K_budget:
                remaining = [i for i in range(N) if i not in active_idx]
                for i in remaining[:K_budget - len(active_idx)]:
                    active_idx.add(i)
        else:
            sel_verts = select_vertices_stratified_farthest(
                all_vertices, bounds, K_budget,
                ratios=stratified_ratios, seed=seed,
            )
            id_to_idx = {id(v): i for i, v in enumerate(all_vertices)}
            active_idx = set()
            for v in sel_verts:
                vid = id(v)
                if vid in id_to_idx:
                    active_idx.add(id_to_idx[vid])
            if len(active_idx) < K_budget:
                remaining = [i for i in range(N) if i not in active_idx]
                for i in remaining[:K_budget - len(active_idx)]:
                    active_idx.add(i)
    else:
        sorted_idx = sorted(range(N), key=lambda i: s_all[i])
        active_idx = set(sorted_idx[:K_budget])
        if use_force_hc:
            active_idx |= hard_core_idx
            # If forcing made active_idx larger than K_budget (only possible
            # when stratified path skipped), trim by removing softest extras
            # while never removing a hard-core vertex.
            while len(active_idx) > K_budget:
                removable = [i for i in active_idx if i not in hard_core_idx]
                if not removable:
                    break
                active_idx.discard(max(removable, key=lambda i: s_all[i]))

    if verbose:
        n_s_pos = sum(1 for i in active_idx if s_all[i] > 1e-12)
        print(f"[IterExchange] Init: K={K_budget}, m_incon={m_incon}, "
              f"s>0 in active={n_s_pos}, stratified={'yes' if bounds else 'no'}")

    best_sol = None
    best_metric = (float("inf"), N, float("inf"))
    best_active_list = sorted(active_idx)

    def _beta_for_all_vertices(sol: Dict[str, np.ndarray], active_list_local: List[int]) -> np.ndarray:
        b = np.asarray(sol.get("beta", np.array([0.0])), dtype=float).reshape(-1)
        if b.size == len(active_list_local):
            fill = float(np.max(b)) if b.size > 0 else 0.0
            out = np.full(N, fill, dtype=float)
            for jj, ii in enumerate(active_list_local):
                out[int(ii)] = float(b[jj])
            return out
        return np.full(N, float(b[0]) if b.size > 0 else 0.0, dtype=float)

    for rnd in range(max_rounds):
        active_list = sorted(active_idx)
        active_verts = [all_vertices[i] for i in active_list]

        if verbose:
            s_vals = [float(all_vertices[i]["s"]) for i in active_list]
            n_nonzero = sum(1 for sv in s_vals if sv > 1e-12)
            print(f"\n[Round {rnd+1}/{max_rounds}] Active: {len(active_list)}, "
                  f"s range: [{min(s_vals):.3e}, {max(s_vals):.3e}], s>0: {n_nonzero}")

        sol = solve_vertex_fusion_sdp_mosek(
            active_verts, syn,
            verbose=False,
            num_threads=0,
            max_iters=50,
            rel_gap=1e-4,
            time_limit_sec=1800,
            beta_lb=0.0,
            decay_rate=decay_rate,
            w_gamma=w_gamma,
            w_mu=w_mu,
            enforce_perf_all_vertices=enforce_perf_all_vertices,
            x0_feas=np.zeros(16) if x0_feas is None else x0_feas,
        )

        if not sol.get("success", False):
            print(f"  [Round {rnd+1}] SDP infeasible, stopping.")
            break

        K_ctrl = sol["K"]
        Q_val = sol["Q"]
        gamma2_val = float(sol["gamma2"][0])
        beta_vec_all = _beta_for_all_vertices(sol, active_list)
        sol["beta_all"] = beta_vec_all.copy()
        beta_val = float(np.mean(beta_vec_all))
        beta_max_val = float(np.max(beta_vec_all))
        gamma_val = float(np.sqrt(max(gamma2_val, 0.0)))

        if verbose:
            print(f"  SDP solved: gamma_syn_diag={gamma_val:.4f}, beta_mean={beta_val:.4f}, beta_max={beta_max_val:.4f}")

        violations = check_all_vertex_violations(
            all_vertices, K_ctrl, Q_val, gamma2_val, beta_vec_all, decay_rate, syn.d_max
        )
        sol["vertex_violations"] = violations.copy()

        max_viol_idx = int(np.argmax(violations))
        max_viol_val = float(violations[max_viol_idx])
        n_violated = int(np.sum(violations > viol_tol))
        n_consistent_violated = sum(
            1 for i in range(N)
            if violations[i] > viol_tol and float(all_vertices[i]["s"]) < 1e-6
        )

        if verbose:
            print(f"  Verification: max_violation={max_viol_val:.6f} (vertex {max_viol_idx}), "
                  f"n_violated={n_violated}/{N}, consistent_violated={n_consistent_violated}")

        current_metric = (max_viol_val, n_violated, gamma_val)
        sol["max_violation"] = np.array([max_viol_val])
        sol["n_violated"] = np.array([n_violated])
        sol["certified_all"] = np.array([1 if max_viol_val <= viol_tol else 0])
        sol["viol_tol"] = np.array([viol_tol])
        if current_metric < best_metric:
            best_metric = current_metric
            best_sol = sol
            best_active_list = list(active_list)

        if max_viol_val <= viol_tol:
            if verbose:
                print(f"  *** Converged: all vertices satisfy LMI (tol={viol_tol:.1e}) ***")
            best_metric = current_metric
            best_sol = sol
            best_active_list = list(active_list)
            break

        # --- Find swap-in: first violator outside active set ---
        viol_order = np.argsort(-violations)
        swap_in_idx = None
        for cand in viol_order:
            cand = int(cand)
            if violations[cand] <= viol_tol:
                break
            if cand not in active_idx:
                swap_in_idx = cand
                break

        if swap_in_idx is None:
            if verbose:
                print(f"  All violators already in active set, stopping.")
            break

        swap_in_viol = float(violations[swap_in_idx])
        swap_in_tier = int(tier[swap_in_idx])
        will_add_nonzero = s_all[swap_in_idx] > 1e-12

        # --- Find swap-out: most redundant, respecting tier & m_incon.
        # When force_include_hard_core is on, hard-core vertices are
        # protected from removal so the unrelaxed-LMI subset is enforced
        # by the SDP throughout all rounds. ---
        active_violations = {i: violations[i] for i in active_idx}
        if use_force_hc:
            removable_pool = [i for i in active_violations.keys() if i not in hard_core_idx]
        else:
            removable_pool = list(active_violations.keys())
        candidates_remove = sorted(removable_pool, key=lambda i: active_violations[i])

        active_s_nonzero_count = sum(1 for i in active_idx if s_all[i] > 1e-12)

        def _can_remove(i: int) -> bool:
            removing_nonzero = s_all[i] > 1e-12
            new_count = active_s_nonzero_count - (1 if removing_nonzero else 0) + (1 if will_add_nonzero else 0)
            return new_count >= m_incon

        swap_out_idx = None
        for i in candidates_remove:
            if tier[i] == swap_in_tier and _can_remove(i):
                swap_out_idx = i
                break
        if swap_out_idx is None:
            for i in candidates_remove:
                if _can_remove(i):
                    swap_out_idx = i
                    break

        if swap_out_idx is None:
            if verbose:
                print(f"  Cannot find removable vertex (m_incon={m_incon} constraint), stopping.")
            break

        if verbose:
            s_new = float(all_vertices[swap_in_idx]["s"])
            s_old = float(all_vertices[swap_out_idx]["s"])
            print(f"  Swap: remove vertex {swap_out_idx} "
                  f"(viol={active_violations[swap_out_idx]:.6f}, s={s_old:.3e}, tier={tier[swap_out_idx]})"
                  f" -> add vertex {swap_in_idx} "
                  f"(viol={swap_in_viol:.6f}, s={s_new:.3e}, tier={swap_in_tier})")

        active_idx.discard(swap_out_idx)
        active_idx.add(swap_in_idx)

    if best_sol is None:
        raise RuntimeError("iterative_constraint_exchange: no feasible solution found")

    final_verts = [all_vertices[i] for i in best_active_list]
    best_gamma = best_metric[2]
    if verbose:
        print(f"\n[IterExchange] Final: {len(final_verts)} vertices, "
              f"gamma_syn_diag={best_gamma:.4f}, max_viol={best_metric[0]:.6f}, n_violated={best_metric[1]}")

    return final_verts, best_sol


# =========================================================
# Simulation with actuator constraints
# =========================================================
def apply_increment_limits(
        uc: np.ndarray,
        syn: SynthesisParams,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    flags = dict(rate_sat=False, norm_sat=False)
    uc2 = uc.copy()

    if syn.du_max_vec is not None:
        umax = np.array(syn.du_max_vec, dtype=float).reshape(4)
        before = uc2.copy()
        uc2 = clip_vec(uc2, -umax, umax)
        if np.any(np.abs(uc2 - before) > 1e-12):
            flags["rate_sat"] = True

    before = uc2.copy()
    uc2 = saturate_norm(uc2, syn.du_max)
    if np.linalg.norm(uc2) > syn.du_max * (1 - 1e-12) and np.linalg.norm(before) > syn.du_max + 1e-12:
        flags["norm_sat"] = True

    return uc2, flags


def apply_absolute_actuator_limits(
        u_abs: np.ndarray,
        uc: np.ndarray,
        syn: SynthesisParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
    flags = dict(abs_sat=False)

    umin = np.array(syn.u_abs_min, dtype=float).reshape(4)
    umax = np.array(syn.u_abs_max, dtype=float).reshape(4)

    u_next_raw = u_abs + uc
    u_next = clip_vec(u_next_raw, umin, umax)
    if np.any(np.abs(u_next - u_next_raw) > 1e-12):
        flags["abs_sat"] = True

    uc_eff = u_next - u_abs
    return uc_eff, u_next, flags
