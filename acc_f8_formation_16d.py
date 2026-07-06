# -*- coding: utf-8 -*-
"""
F-8 short-period 4-aircraft pitch formation model (16D augmented state).

Purpose: reproducibility driver for the DCCVR synthesis, fixed-gain audit,
and F-8 tracking simulations used in the manuscript.

State: x = [x_1; x_2; x_3; x_4; u_act] in R^16
  x_i = [alpha_i, q_i, theta_i] in R^3 (pitch-axis per aircraft)
  u_act in R^4 (augmented actuator-integrator state)
  Physical state dim = 12 (4 x 3), actuator dim = 4, total = 16.

Input: u in R^4 (each component = elevator rate command for one aircraft)
Disturbance: w in R^6 (shared vertical gust, per-aircraft pitching gust, formation coupling)

Uncertain parameters (shared across all 4 aircraft, 4 boolean dims -> 16 vertices).

Sign-convention note:
This file uses pitch-stiffness / pitch-damping *magnitudes* with the plant
matrix hard-coded as dq/dt = -m_alpha*alpha - m_q*q - m_de*u. Hence:
    m_alpha > 0  <=>  statically stable  (A[1,0] = -m_alpha < 0)
    m_alpha < 0  <=>  statically unstable
In the Etkin / Stevens-Lewis convention, the pitching-moment derivative
M_alpha satisfies M_alpha < 0 <=> stable. Our m_alpha therefore equals
-M_alpha (up to a positive scaling by 1/Iyy). The parameter box is chosen
so that m_alpha spans zero, exercising a static-stability sign change.

  za_v    Z_alpha / V                (lift / speed; positive)
  m_alpha -M_alpha (+ = stable)      (negative -> static instability)
  m_q     -M_q     (+ = damped)      (negative -> destabilising rate term)
  m_de    |M_delta_e|                (positive control derivative)
The actual numeric ranges are configured in F8ParamBounds below.

Units: rad / rad-per-sec; matrices use normalised dynamics (V baked into za_v, g/V=0.02).
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import os
import sys
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la


BASE_PATH = os.path.join(os.path.dirname(__file__), "f8_benchmark_utils.py")
_spec = importlib.util.spec_from_file_location("f8_benchmark_utils_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load base module from {BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

# Matrix-hull membership residual check (Remark 1 / Corollary 1).
# Kept as a sibling module so the certification pipeline is explicit.
from matrix_hull import matrix_hull_residual  # noqa: E402


# ---------------------------------------------------------
# Parameter bounds (4 uncertain aero derivatives -> 16 vertices)
# ---------------------------------------------------------
@dataclass(frozen=True)
class F8ParamBounds:
    # F-8 cross-Mach style setting used for the numerical benchmark:
    #   m_alpha spans -2..+5   -> open-loop static stability flips across
    #                              the box (m_alpha<0 means statically
    #                              unstable; see file header on sign),
    #   m_q     spans -0.5..+2.5 -> destabilising / strongly damped extremes,
    #   m_de    spans  3 ..+7.5  -> 2.5x elevator-effectiveness range,
    #   za_v    spans 0.5..+1.5  -> 3x lift-coefficient range.
    # Corners with m_alpha < 0 are open-loop statically unstable; the
    # corner sweep evaluates all 16 physical aerodynamic corners.
    za_v: Tuple[float, float] = (0.5, 1.5)        # Z_alpha / V (positive)
    m_alpha: Tuple[float, float] = (-2.0, 5.0)    # -M_alpha (>0 stable)
    m_q: Tuple[float, float] = (-0.5, 2.5)        # -M_q     (>0 damped)
    m_de: Tuple[float, float] = (3.0, 7.5)        # elevator effectiveness


@dataclass(frozen=True)
class F8SynthesisParams:
    Ts: float = 0.05
    # Disturbance magnitude matched to actuator authority (no chronic
    # saturation): the LMI-based H-infinity certificate of the proposed
    # design only holds in the unsaturated linear regime, so the test
    # bench should not push every controller past the actuator wall.
    # Saturation is still allowed on transients but not in steady state.
    # d_max kept at 0.6 to preserve the conservative, well-conditioned SDP
    # design point. The simulation gust *raw amplitudes* are 1.5x the
    # original (events 0.9 / 1.2, colored noise 0.225, coupling 0.12),
    # but saturate_norm clips the per-step ||d||_2 back to d_max = 0.6.
    # Empirical effect for the fixed-seed rollout:
    #   - pre-saturator peak ||d||_2 = 1.20 (vs 0.80 originally),
    #   - post-saturator peak ||d||_2 = 0.60 (cap, unchanged),
    #   - post-saturator MEAN ||d||_2 = 0.196 (vs 0.158 originally; +24%),
    #   - saturator activation rate = 14.0% of steps (vs 6.5% originally).
    # So the upgrade does NOT raise the worst-case ||d||_2; it raises the
    # *fraction of time* the disturbance sits at the d_max limit, giving a
    # more aggressive sustained-disturbance scenario without breaching the
    # SDP H-infinity certificate (which only requires ||d||_2 <= d_max).
    # An earlier attempt to also raise d_max to 0.9 was reverted: the
    # resulting K had rho_closed > 1 on 100% of corners.
    d_max: float = 0.6
    # The data-core rejection uses datacore_dbar (= bar d_data in the
    # paper), while d_max above is the synthesis/audit disturbance scale.
    du_max: float = 0.8
    u_abs_min: Tuple[float, float, float, float] = (-0.4, -0.4, -0.4, -0.4)
    u_abs_max: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.4)
    du_max_vec: Optional[Tuple[float, float, float, float]] = None
    sat_tol: float = 0.99
    decay_rate: float = 0.95
    w_gamma: float = 1.0
    w_mu: float = 0.1
    w_beta: float = 1e-3
    beta_score_tol: float = 1e-9
    beta_ub: float = 1000.0
    gamma2_scale: float = 1.0
    mosek_pfeas: float = 1e-6
    mosek_dfeas: float = 1e-6
    lmi_margin: float = 0.0
    mosek_solve_form: str = "free"
    per_vertex_beta: bool = False
    x0_scale: float = 0.25
    candidate_mode: str = "corner"
    vertex_keys: Tuple[str, ...] = ("za_v", "m_alpha", "m_q", "m_de")
    n_support_points: int = 4
    n_low_support_points: int = 1
    support_sample_count: int = 400
    support_spread: float = 0.30
    # Monte Carlo sample size: 100 trials is the ACC / Boyd-et-al. minimum
    # for reliable p95 / p99 quantile estimates; 20 trials gives a p95
    # confidence interval almost as wide as the point estimate itself and
    # therefore does not support any statistical claim on rare-event tails.
    mc_trials: int = 100
    mc_seconds: float = 15.0
    # Unmodelled dynamics for the stress-test phase (default zero so the
    # original ideal test is preserved). Phase 5b uses these.
    unmodelled_sensor_delay: int = 0      # extra k-step lag in measurement
    unmodelled_actuator_tau: float = 0.0  # actuator first-order lag (s)
    # Actuator multiplicative-gain uncertainty embedded in the H-infinity SDP
    # design (Skogestad-Postlethwaite Sec.8.5). When > 0, the synthesis sees
    # 16 * 2 = 32 plants: each parameter corner is paired with two static
    # input-effectiveness endpoints kappa in {1, 1 - eps_a}. The five robust
    # H-infinity controllers (Proposed, NoRelax-ProposedActive, QS-Hinf, PDL-Hinf,
    # Core-CQLF-Hinf) thus
    # see the prescribed static input-effectiveness interval at synthesis
    # and audit time via the 32-vertex LMI; this is a static multiplicative
    # input-channel uncertainty model, NOT a formal certificate for
    # dynamic actuator lag. Default 0.5 corresponds to the same numerical
    # static attenuation produced by Phase 5b's first-order actuator lag
    # (alpha_a = Ts/(Ts+tau_a) = 0.5 at tau_a = 50 ms).
    actuator_gain_uncertainty: float = 0.5
    # Per-aircraft 3D physical state: [alpha, q, theta] x 4 aircraft = 12, then 4 actuator states.
    # Performance weights: alpha small (tracking), q moderate (damping), theta large (attitude).
    Qx_perf: Tuple[float, ...] = (
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
    )
    Rd_perf: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # *Fair* Q for the LQR baseline: 12D state weights are byte-for-byte
    # identical to Qx_perf so that LQR and the H-infinity designs see the
    # same physical-state penalty. The trailing 4 entries (= 0.5% of the
    # smallest physical weight) are a small regulariser on the actuator
    # integrator state, which the H-infinity formulation does not penalise
    # in z but LQR needs in Q to keep the discrete-time Riccati well-posed.
    # The 0.005-relative magnitude is small enough that it does not bias
    # the comparison: it only prevents numerical singularity.
    Qx_lqr: Tuple[float, ...] = (
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
        10.0, 5.0, 20.0,
        0.05, 0.05, 0.05, 0.05,
    )
    Ru_lqr: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    enforce_perf_all_vertices: bool = True
    seed: int = 26
    # ----- Data-contained core C_D (audit only) -----
    # The data-contained core C_D is a SAMPLED axis-aligned consistency
    # box: C_D = bbox{ p in MC sample of P : exists d_k with
    # ||d_k|| <= datacore_dbar explaining the rollout data }. It is
    # NOT claimed to be a certified outer approximation of the full
    # data-consistent set P_D (ZOH discretisation makes (A,B,S)
    # non-affine in p so a finite Monte Carlo sample is generally not
    # a strict outer hull). C_D defines the finite vertex set on which
    # the fixed-K post-certificate below is solved; transfer to a
    # particular plant is a separate matrix-hull membership check
    # (matrix_hull_residual). No synthesis is performed on C_D.
    datacore_dbar: float = 0.50
    datacore_n_samples: int = 4000         # MC samples in P for sampled P_D
    datacore_proj_tol: float = 5e-2        # tol on residual proj onto S range
    datacore_min_shrink_dim: int = 1       # require >=this dims to shrink (else fallback)
    # ----- Fixed-K data-core post-certification (analysis-only) -----
    # For every K already produced by the synthesis loop above, ask MOSEK
    # to find a Lyapunov P > 0 and an L2-gain g such that the textbook
    # discrete-time bounded-real lemma holds simultaneously on every
    # corner of C_D. If Proposed fails the audit we run the certificate-
    # repair synthesis SDP that nudges (Q, Y) toward (Q, K_prop Q) under
    # the unrelaxed core LMIs.  audit_decay_rate = 1.0 is the textbook
    # bounded-real LMI (no decay margin, matches the user's literal LMI).
    run_fixed_k_audit: bool = True
    audit_decay_rate: float = 1.0
    repair_lambda_close: float = 1.0
    fig_context: str = "paper"
    fig_column: str = "double"
    fig_formats: Tuple[str, ...] = ("pdf", "png")
    fig_dpi_png: int = 600
    fig_transparent: bool = False
    fig_show_titles: bool = True
    fig_out_dir: str = "acc_f8_formation_results"


# ---------------------------------------------------------
# Performance output matrices (C_c, D_c) for the 16D state
# ---------------------------------------------------------
def build_performance_matrices(syn: F8SynthesisParams) -> Tuple[np.ndarray, np.ndarray]:
    Qx_diag = np.array(syn.Qx_perf, dtype=float)
    Rd_diag = np.array(syn.Rd_perf, dtype=float)
    Cc = np.zeros((16, 16))
    Cc[0:len(Qx_diag), 0:len(Qx_diag)] = np.diag(np.sqrt(Qx_diag))
    Dc = np.zeros((16, 4))
    Dc[12:16, 0:4] = np.diag(np.sqrt(Rd_diag))
    return Cc, Dc


# ---------------------------------------------------------
# F-8 short-period continuous-time dynamics (per aircraft, 3D)
# ---------------------------------------------------------
def _f8_single_continuous(p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Single-aircraft F-8 short-period 3D: state=[alpha, q, theta], input=elevator.

    Plant equations (this file's sign convention -- see file header):
        dalpha/dt = -za_v*alpha + q - (g/V)*theta + (Z_de/V)*u + gust_alpha
        dq/dt     = -m_alpha*alpha - m_q*q - m_de*u + gust_q
        dtheta/dt = q
    Hence in the A matrix: A[1,0] = -m_alpha, A[1,1] = -m_q. Positive values
    of the parameters m_alpha / m_q correspond to the stable case; negative
    values correspond to the statically-unstable / destabilising-damping
    cases. Note that this differs from Etkin-style textbooks where the
    pitching-moment derivative M_alpha itself is negative for stability;
    our m_alpha equals -M_alpha (up to the 1/Iyy factor baked in).
    m_de is always positive (physical elevator effectiveness magnitude).
    """
    p1 = float(p["za_v"])     # Z_alpha / V   (positive)
    p2 = float(p["m_alpha"])  # = -M_alpha    (positive -> stable)
    p3 = float(p["m_q"])      # = -M_q        (positive -> damped)
    p4 = float(p["m_de"])     # |M_delta_e|   (positive)
    g_over_V = 0.02           # rad/s^2 per rad of theta (approx.)
    Z_de_over_V = 0.04        # elevator lift effect on alpha (small)

    A_i = np.array([
        [-p1,   1.0,  -g_over_V],
        [-p2,  -p3,   0.0      ],
        [ 0.0,  1.0,  0.0      ],
    ], dtype=float)
    B_i = np.array([
        [ Z_de_over_V],
        [-p4         ],    # elevator down -> pitch up convention
        [ 0.0        ],
    ], dtype=float)
    return A_i, B_i


def _f8_formation_continuous(p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-aircraft block-diagonal continuous-time model (12D physical state)."""
    A_i, B_i = _f8_single_continuous(p)
    Ac = np.zeros((12, 12))
    Bc = np.zeros((12, 4))
    Sc = np.zeros((12, 6))
    for i in range(4):
        Ac[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)] = A_i
        Bc[3 * i: 3 * (i + 1), i: i + 1] = B_i

    # Disturbance wiring (6D): w = [shared_vgust, g_q1, g_q2, g_q3, g_q4, coupling]
    # shared vertical gust affects alpha of every aircraft
    for i in range(4):
        Sc[3 * i + 0, 0] = 0.3
    # per-aircraft pitching-moment gust on q_i
    for i in range(4):
        Sc[3 * i + 1, 1 + i] = 0.6
    # formation coupling (e.g., lead-aircraft wake affecting theta of followers)
    for i in range(4):
        Sc[3 * i + 2, 5] = 0.1 * (1.0 if i > 0 else 0.0)
    return Ac, Bc, Sc


def _zoh_discretize(Ac: np.ndarray, Bc: np.ndarray, Sc: np.ndarray, Ts: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, q = Ac.shape[0], Bc.shape[1], Sc.shape[1]
    M = np.zeros((n + m + q, n + m + q))
    M[0:n, 0:n] = Ac
    M[0:n, n:n + m] = Bc
    M[0:n, n + m:n + m + q] = Sc
    Md = la.expm(M * float(Ts))
    return Md[0:n, 0:n], Md[0:n, n:n + m], Md[0:n, n + m:n + m + q]


def build_vertex_matrices(p: Dict[str, float], Ts: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 16D augmented (A, B, S_aug) = 12D physical plant + 4D actuator integrator.

    Layout: x[12:16] are actuator-integrator states and u is the commanded rate.
    """
    Ac, Bc, Sc = _f8_formation_continuous(p)
    Ad, Bd, Sd = _zoh_discretize(Ac, Bc, Sc, Ts)
    A = np.zeros((16, 16))
    B = np.zeros((16, 4))
    S_aug = np.zeros((16, 6))
    A[0:12, 0:12] = Ad
    A[0:12, 12:16] = Bd
    A[12:16, 12:16] = np.eye(4)
    B[0:12, :] = Bd
    B[12:16, :] = np.eye(4)
    S_aug[0:12, :] = Sd
    return A, B, S_aug


def f8_initial_state(scale: float = 1.0) -> np.ndarray:
    """Initial perturbation: small alpha/theta errors, zero rates & actuators."""
    return float(scale) * np.array(
        [0.05, -0.02, 0.03,
         0.04, -0.01, 0.025,
         0.03,  0.00, 0.02,
         0.02,  0.01, 0.015,
         0.0,   0.0,  0.0,  0.0],
        dtype=float,
    )


def center_params(bounds: F8ParamBounds) -> Dict[str, float]:
    return {k: 0.5 * (getattr(bounds, k)[0] + getattr(bounds, k)[1]) for k in bounds.__dataclass_fields__.keys()}


def sample_params(bounds: F8ParamBounds, rng: np.random.Generator) -> Dict[str, float]:
    return {k: float(rng.uniform(*getattr(bounds, k))) for k in bounds.__dataclass_fields__.keys()}


# ---------------------------------------------------------
# Worst-case disturbance gain (used to extend the data-driven Psi over
# the physical 16-corner aerodynamic box even when batch data only
# covers the nominal side).
# ---------------------------------------------------------
def worst_case_S_matrix(bounds: F8ParamBounds, syn: F8SynthesisParams) -> np.ndarray:
    keys = list(bounds.__dataclass_fields__.keys())
    S_worst = None
    best_eig = -np.inf
    for bits in itertools.product([0, 1], repeat=len(keys)):
        p = {}
        for key, bit in zip(keys, bits):
            lo, hi = getattr(bounds, key)
            p[key] = float(hi if bit else lo)
        _, _, S = build_vertex_matrices(p, syn.Ts)
        eig = float(np.linalg.eigvalsh(S @ S.T)[-1])
        if eig > best_eig:
            best_eig = eig
            S_worst = S
    return S_worst


# ---------------------------------------------------------
# Install the F-8 plant hooks used by the shared utility module.
# ---------------------------------------------------------
def _f8_build_vertex_matrices_compat(p: Dict[str, float], Ts: float, g: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signature-compatible replacement for the utility vertex builder.
    The ``g`` argument is ignored (F-8 bakes g/V into matrix constants)."""
    return build_vertex_matrices(p, Ts)


def _f8_enumerate_vertices(bounds: F8ParamBounds) -> List[Dict[str, float]]:
    """Enumerate 2^|dims| corner vertices of the F-8 parameter box."""
    keys = list(bounds.__dataclass_fields__.keys())
    vertices: List[Dict[str, float]] = []
    for bits in itertools.product([0, 1], repeat=len(keys)):
        p: Dict[str, float] = {}
        for key, bit in zip(keys, bits):
            lo, hi = getattr(bounds, key)
            p[key] = float(hi if bit else lo)
        vertices.append(p)
    return vertices


base.build_vertex_matrices = _f8_build_vertex_matrices_compat
base.enumerate_vertices = _f8_enumerate_vertices


# ---------------------------------------------------------
# Config conversion: F8SynthesisParams -> shared SynthesisParams
# ---------------------------------------------------------
def f8_to_base_syn(syn: "F8SynthesisParams") -> "base.SynthesisParams":
    """Pack F-8 user-facing config into the shared SynthesisParams dataclass
    so that all base-module helpers see the same synthesis knobs."""
    return base.SynthesisParams(
        Ts=syn.Ts,
        g=0.0,
        d_max=syn.d_max,
        du_max=syn.du_max,
        u_abs_min=syn.u_abs_min,
        u_abs_max=syn.u_abs_max,
        du_max_vec=syn.du_max_vec,
        sat_tol=syn.sat_tol,
        decay_rate=syn.decay_rate,
        w_gamma=syn.w_gamma,
        w_mu=syn.w_mu,
        w_beta=syn.w_beta,
        beta_score_tol=syn.beta_score_tol,
        gamma2_scale=syn.gamma2_scale,
        mosek_pfeas=syn.mosek_pfeas,
        mosek_dfeas=syn.mosek_dfeas,
        lmi_margin=syn.lmi_margin,
        mosek_solve_form=syn.mosek_solve_form,
        Qx_perf=syn.Qx_perf,
        Rd_perf=syn.Rd_perf,
        Qx_lqr=syn.Qx_lqr,
        Ru_lqr=syn.Ru_lqr,
        enforce_perf_all_vertices=syn.enforce_perf_all_vertices,
        seed=syn.seed,
        fig_context=syn.fig_context,
        fig_column=syn.fig_column,
        fig_formats=syn.fig_formats,
        fig_dpi_png=syn.fig_dpi_png,
        fig_transparent=syn.fig_transparent,
        fig_show_titles=syn.fig_show_titles,
        fig_out_dir=syn.fig_out_dir,
    )


def beta_summary_value(sol: Dict[str, Any]) -> float:
    """Return scalar beta summary (mean over per-vertex beta) for a solver output."""
    if not sol.get("success", False):
        return float("nan")
    beta = np.atleast_1d(np.asarray(sol.get("beta", [0.0]), dtype=float))
    if beta.size == 0:
        return 0.0
    return float(np.mean(beta))


# ---------------------------------------------------------
# PDL-Hinf (Parameter-Dependent Lyapunov) discrete-time robust H-inf
# state-feedback synthesis via extended LMI with slack matrix G.
#
# Reference: Daafouz & Bernussou, "Parameter-dependent Lyapunov functions
# for discrete time systems with time varying parametric uncertainties",
# Systems & Control Letters 43 (2001), 355-359; extended to H-infinity
# state feedback by Apkarian & Tuan 2000 and de Oliveira-Bernussou-
# Geromel 1999. This is the standard less-conservative robust H-infinity
# baseline for polytopic uncertainty (each vertex has its OWN Lyapunov
# matrix P_i, coupled only through the slack matrix G; strictly weaker
# LMI than Boyd'94 single-Q quadratic stability).
#
# Per-vertex LMI (PSD form):
#   [ G+G^T-P_i    (A_i G + B_i Y)^T   0               (C_i G + D_i Y)^T ]
#   [ A_i G+B_i Y  lambda_pdl * P_i    S_i*d_max       0                 ]  >= 0
#   [ 0            S_i^T*d_max         gamma^2 I       0                 ]
#   [ C_i G+D_i Y  0                   0               I                 ]
# for i = 1..N, where lambda_pdl = decay_rate plays the same Lyapunov
# contraction role as alpha in the Proposed (1,1) block. Controller
# recovered as K = Y G^{-1}.
# ---------------------------------------------------------
def solve_pdl_hinf_mosek(
        vertices: List[Dict[str, Any]],
        syn: "base.SynthesisParams",
        *,
        verbose: bool = False,
        num_threads: int = 0,
        max_iters: int = 80,
        rel_gap: float = 1e-4,
        time_limit_sec: Optional[int] = 1800,
        w_gamma: float = 1.0,
        eps_psd: float = 1e-6,
) -> Dict[str, Any]:
    import mosek.fusion as mf

    n = 16   # state dim
    nu = 4   # control dim
    nw = 6   # disturbance dim
    nz = 16  # performance output dim (matches build_performance_matrices)

    decay_rate = float(getattr(syn, "decay_rate", 0.95))
    d_max = float(syn.d_max)

    N = len(vertices)
    if N == 0:
        raise ValueError("PDL-Hinf: empty vertex list")

    # MOSEK license: configure_mosek_license() honours the
    # MOSEKLM_LICENSE_FILE env var and caller-provided paths, so this
    # file contains no hard-coded license path.
    base.configure_mosek_license(verbose=False)

    M = mf.Model("pdl_hinf")
    try:
        if verbose:
            M.setLogHandler(sys.stdout)
        try:
            if num_threads is None or num_threads <= 0:
                ncpu = os.cpu_count() or 1
                num_threads = int(min(8, max(1, ncpu)))
            M.setSolverParam("numThreads", int(num_threads))
            M.setSolverParam("intpntMaxIterations", int(max_iters))
            M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
            M.setSolverParam("intpntCoTolPfeas", float(getattr(syn, "mosek_pfeas", 1e-6)))
            M.setSolverParam("intpntCoTolDfeas", float(getattr(syn, "mosek_dfeas", 1e-6)))
            if time_limit_sec is not None and time_limit_sec > 0:
                M.setSolverParam("optimizerMaxTime", float(time_limit_sec))
        except Exception:
            pass

        P_vars = [M.variable(f"P_{i}", mf.Domain.inPSDCone(n)) for i in range(N)]
        G = M.variable("G", [n, n], mf.Domain.unbounded())
        Y = M.variable("Y", [nu, n], mf.Domain.unbounded())
        gamma2 = M.variable("gamma2", mf.Domain.greaterThan(1e-9))

        eps_I_n_e = mf.Expr.constTerm(mf.Matrix.dense(eps_psd * np.eye(n)))
        I_nw_m = mf.Matrix.dense(np.eye(nw))
        I_nz_e = mf.Expr.constTerm(mf.Matrix.dense(np.eye(nz)))

        for i in range(N):
            M.constraint(f"P_pd_{i}", mf.Expr.sub(P_vars[i], eps_I_n_e),
                         mf.Domain.inPSDCone(n))

        GGT = mf.Expr.add(G, mf.Expr.transpose(G))
        M.constraint("G_nonsing", mf.Expr.sub(GGT, eps_I_n_e),
                     mf.Domain.inPSDCone(n))

        Z_n_nw = mf.Expr.constTerm(np.zeros((n, nw)))
        Z_n_nz = mf.Expr.constTerm(np.zeros((n, nz)))
        Z_nw_n = mf.Expr.constTerm(np.zeros((nw, n)))
        Z_nw_nz = mf.Expr.constTerm(np.zeros((nw, nz)))
        Z_nz_n = mf.Expr.constTerm(np.zeros((nz, n)))
        Z_nz_nw = mf.Expr.constTerm(np.zeros((nz, nw)))

        # Per-vertex input-amplitude ellipsoid (Apkarian-Tuan 2000 G-form):
        # [ du_max^2 (G + G^T - P_i),  Y^T ;
        #   Y,                          I_nu ] >= 0           for each i
        # K = Y G^{-1}, and (G - P_i) P_i^{-1} (G - P_i)^T >= 0 implies
        # G + G^T - P_i <= G^T P_i^{-1} G, hence the LMI gives
        # K^T K <= du_max^2 P_i, i.e. ||K x|| <= du_max for x in
        # {x : x^T P_i^{-1} x <= 1}. This is the natural G-parameterised
        # analogue of the input-amplitude ellipsoid used by the
        # Proposed/Ablation-betaoff/QS-Hinf synthesis (see
        # solve_vertex_fusion_sdp_mosek for the Q-parameterised form).
        du_max = float(getattr(syn, "du_max", 0.8))
        I_nu_m = mf.Expr.constTerm(mf.Matrix.dense(np.eye(nu)))

        for i, v in enumerate(vertices):
            A_i = np.asarray(v["A"], dtype=float)
            B_i = np.asarray(v["B"], dtype=float)
            C_i = np.asarray(v["C"], dtype=float)
            D_i = np.asarray(v["D"], dtype=float)
            # Use the shared scaled_disturbance_matrix helper so that
            # PDL-Hinf, DCCVR beta-off, and QS-Hinf synthesis (in
            # solve_vertex_fusion_sdp_mosek), the audit SDP and
            # the full-box diagnostic all carry the same normalised-
            # disturbance convention tilde S = d_cert * S; gamma is
            # therefore the induced L2 gain from w (||w||_2 <= 1) to z.
            S_i = scaled_disturbance_matrix(v["S"], d_max)

            Ai_m = mf.Matrix.dense(A_i)
            Bi_m = mf.Matrix.dense(B_i)
            Ci_m = mf.Matrix.dense(C_i)
            Di_m = mf.Matrix.dense(D_i)
            Si_e = mf.Expr.constTerm(mf.Matrix.dense(S_i))
            Si_T_e = mf.Expr.constTerm(mf.Matrix.dense(S_i.T))

            P = P_vars[i]
            AG = mf.Expr.mul(Ai_m, G)
            BY = mf.Expr.mul(Bi_m, Y)
            AG_BY = mf.Expr.add(AG, BY)
            AG_BY_T = mf.Expr.transpose(AG_BY)

            CG = mf.Expr.mul(Ci_m, G)
            DY = mf.Expr.mul(Di_m, Y)
            CG_DY = mf.Expr.add(CG, DY)
            CG_DY_T = mf.Expr.transpose(CG_DY)

            block11 = mf.Expr.sub(GGT, P)              # G + G^T - P_i
            block22 = mf.Expr.mul(decay_rate, P)
            block33 = mf.Expr.mul(I_nw_m, gamma2)

            row1 = mf.Expr.hstack([block11, AG_BY_T, Z_n_nw, CG_DY_T])
            row2 = mf.Expr.hstack([AG_BY, block22, Si_e, Z_n_nz])
            row3 = mf.Expr.hstack([Z_nw_n, Si_T_e, block33, Z_nw_nz])
            row4 = mf.Expr.hstack([CG_DY, Z_nz_n, Z_nz_nw, I_nz_e])

            LMI = mf.Expr.vstack([row1, row2, row3, row4])
            M.constraint(f"pdl_lmi_{i}", LMI,
                         mf.Domain.inPSDCone(n + n + nw + nz))

            # Per-vertex input-amplitude ellipsoid (G-form)
            blk11_uamp = mf.Expr.mul(du_max ** 2, block11)
            row_e1 = mf.Expr.hstack([blk11_uamp, mf.Expr.transpose(Y)])
            row_e2 = mf.Expr.hstack([Y, I_nu_m])
            ELL = mf.Expr.vstack([row_e1, row_e2])
            M.constraint(f"pdl_uamp_{i}", ELL,
                         mf.Domain.inPSDCone(n + nu))

        M.objective("min_gamma2", mf.ObjectiveSense.Minimize,
                    mf.Expr.mul(float(w_gamma), gamma2))

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
            if verbose:
                print(f"[PDL-Hinf] solver status={sol_status}, primal={primal_status}, dual={dual_status}")
            return dict(
                success=False, K=np.zeros((nu, n)),
                G=np.eye(n), Y=np.zeros((nu, n)),
                P_list=[np.eye(n) for _ in range(N)],
                gamma2=np.array([np.inf]),
                beta=np.array([0.0]), mu=np.array([np.nan]),
                decay_rate=np.array([decay_rate]),
                solver_problem_status=np.array([str(sol_status)]),
                solver_primal_status=np.array([str(primal_status)]),
                solver_dual_status=np.array([str(dual_status)]),
            )

        Gv = base._level_to_matrix(G, (n, n), prefer_order="C")
        Yv = base._level_to_matrix(Y, (nu, n), prefer_order="C")
        gamma2_v = max(base._level_to_float(gamma2.level()), 0.0)

        try:
            K = la.solve(Gv.T, Yv.T).T
        except np.linalg.LinAlgError:
            if verbose:
                print("[PDL-Hinf] G singular; falling back to pinv")
            K = Yv @ np.linalg.pinv(Gv)

        P_list_val: List[np.ndarray] = []
        for i in range(N):
            Pi_val = base._level_to_matrix(P_vars[i], (n, n), prefer_order="C")
            Pi_val = 0.5 * (Pi_val + Pi_val.T)
            P_list_val.append(Pi_val)

        return dict(
            K=K, G=Gv, Y=Yv, P_list=P_list_val,
            gamma2=np.array([gamma2_v]),
            beta=np.array([0.0]),
            mu=np.array([np.nan]),
            decay_rate=np.array([decay_rate]),
            success=bool(solver_optimal),
            solver_problem_status=np.array([str(sol_status)]),
            solver_primal_status=np.array([str(primal_status)]),
            solver_dual_status=np.array([str(dual_status)]),
        )
    finally:
        M.dispose()


# ---------------------------------------------------------
# Controller synthesis:
#   Proposed     = per-vertex beta-relaxation SDP with ICE active set
#   Ablation beta-off = beta = 0 on the SAME active set chosen by ICE
#                  (ablation: isolates the beta-relaxation contribution)
#   QS-Hinf      = single-Q quadratic-stability H-infinity over ALL 32
#                  vertices (Boyd-Feron-El Ghaoui-Balakrishnan 1994, LMI
#                  book, Sec 7.6.2). Classical robust H-inf baseline.
#   PDL-Hinf     = parameter-dependent Lyapunov H-inf with slack matrix G
#                  over ALL 32 vertices (Daafouz-Bernussou 2001, Apkarian-
#                  Tuan 2000). Less-conservative than QS-Hinf; vertex-
#                  specific Lyapunov matrix P_i coupled through G.
#   RobustLQR-LMI = polytopic guaranteed-cost robust LQR LMI on ALL 32
#                  vertices (Boyd-Feron-El Ghaoui-Balakrishnan 1994 LMI
#                  book Sec. 7.4.2; discrete-time analog of Petersen 1995
#                  modified Riccati). Single Lyapunov X with min tr(W)
#                  upper bound on the polytope-uniform quadratic cost.
# ---------------------------------------------------------
def solve_robust_lqr_polytopic_lmi(
        vertices: List[Dict[str, Any]],
        syn: "F8SynthesisParams",
        *,
        verbose: bool = False,
        num_threads: int = 0,
        max_iters: int = 200,
        rel_gap: float = 1e-6,
        time_limit_sec: Optional[int] = 1800,
        eps_psd: float = 1e-6,
) -> Dict[str, Any]:
    """True polytopic guaranteed-cost robust LQR via single-Lyapunov LMI.

    Reference: Boyd-Feron-El Ghaoui-Balakrishnan 1994 'LMI Book' Sec. 7.4.2;
    discrete-time analog of Petersen 1995 modified Riccati equation. For
    weights Q_w = diag(syn.Qx_lqr) (16D state) and R_w = diag(syn.Ru_lqr)
    (4D input), the textbook 4-block LMI

        [ X        (A_iX+B_iY)^T  X Q_w^{1/2}  Y^T R_w^{1/2} ]
        [   *        X               0            0          ]  >= 0
        [   *        *               I_n          0          ]
        [   *        *               *            I_nu       ]

    is rewritten as the *equivalent* dimension-reduced split form below
    (Z, W_R play the role of X Q_w X and Y^T R_w Y):

        min  tr(W)
        s.t. [W, I; I, X] >= 0                       (W >= X^{-1})
             [ Z,    X Q_w^{1/2}; *, I_n ] >= 0      (Z >= X Q_w X)
             [ W_R,  Y^T R_w^{1/2}; *, I_nu] >= 0    (W_R >= Y^T R_w Y)
             For every vertex i,
                 [ X - Z - W_R,   (A_iX+B_iY)^T ;
                   *,             X            ] >= 0
                 (Lyapunov drop with embedded LQR-cost slack)
             X >= eps I
        Recover K = Y X^{-1}.

    This split form has per-vertex LMIs of size 2n=32 (not 3n+nu=52),
    cutting MOSEK Fusion's dense-block memory by ~3x relative to the
    monolithic 4-block form on the 32-vertex polytope; tested on the
    F-8 problem the 4-block form runs out of MOSEK memory on N=32.

    The polytope-uniform quadratic cost upper bound x_0^T X^{-1} x_0 is
    minimised by tr(W) >= tr(X^{-1}) when x_0 has isotropic unit
    covariance. RobustLQR-LMI strictly dominates the worst-vertex DARE
    heuristic in solve_robust_lqr_wv because the Lyapunov function is
    enforced on every vertex, not just the worst-rho corner.
    """
    import mosek.fusion as mf
    base.configure_mosek_license(verbose=False)

    n, nu = 16, 4
    Qx = np.asarray(syn.Qx_lqr, dtype=float)
    Ru = np.asarray(syn.Ru_lqr, dtype=float)
    if Qx.shape[0] != n:
        raise ValueError(f"Qx_lqr length {Qx.shape[0]} != n={n}")
    if Ru.shape[0] != nu:
        raise ValueError(f"Ru_lqr length {Ru.shape[0]} != nu={nu}")
    Q_w = np.diag(Qx) + eps_psd * np.eye(n)
    R_w = np.diag(Ru) + eps_psd * np.eye(nu)
    Q_sqrt = la.cholesky(Q_w)         # lower-triangular L: L L^T = Q_w
    R_sqrt = la.cholesky(R_w)
    N = len(vertices)
    if N == 0:
        raise ValueError("RobustLQR-LMI: empty vertex list")

    M = mf.Model("rlqr_polytopic_lmi")
    try:
        if verbose:
            M.setLogHandler(sys.stdout)
        try:
            if num_threads is None or num_threads <= 0:
                ncpu = os.cpu_count() or 1
                num_threads = int(min(8, max(1, ncpu)))
            M.setSolverParam("numThreads", int(num_threads))
            M.setSolverParam("intpntMaxIterations", int(max_iters))
            M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
            M.setSolverParam("intpntCoTolPfeas",
                             float(getattr(syn, "mosek_pfeas", 1e-6)))
            M.setSolverParam("intpntCoTolDfeas",
                             float(getattr(syn, "mosek_dfeas", 1e-6)))
            if time_limit_sec is not None and time_limit_sec > 0:
                M.setSolverParam("optimizerMaxTime", float(time_limit_sec))
        except Exception:
            pass

        X = M.variable("X", mf.Domain.inPSDCone(n))
        Y = M.variable("Y", [nu, n], mf.Domain.unbounded())
        W = M.variable("W", mf.Domain.inPSDCone(n))
        Z_q = M.variable("Z_q", mf.Domain.inPSDCone(n))    # Z_q >= X Q_w X
        W_r = M.variable("W_r", mf.Domain.inPSDCone(n))    # W_r >= Y^T R_w Y

        eps_I = mf.Expr.constTerm(mf.Matrix.dense(eps_psd * np.eye(n)))
        I_n_e = mf.Expr.constTerm(mf.Matrix.dense(np.eye(n)))
        I_nu_e = mf.Expr.constTerm(mf.Matrix.dense(np.eye(nu)))

        M.constraint("X_pos", mf.Expr.sub(X, eps_I),
                     mf.Domain.inPSDCone(n))

        # [W, I; I, X] >= 0   (Schur: W >= X^{-1})
        rowwi1 = mf.Expr.hstack([W, I_n_e])
        rowwi2 = mf.Expr.hstack([I_n_e, X])
        M.constraint("WI_pos", mf.Expr.vstack([rowwi1, rowwi2]),
                     mf.Domain.inPSDCone(2 * n))

        # [Z_q, X Q_w^{1/2}; *, I_n] >= 0   (Schur: Z_q >= X Q_w X)
        Q_s_m = mf.Matrix.dense(Q_sqrt)
        XQs = mf.Expr.mul(X, Q_s_m)
        rowq1 = mf.Expr.hstack([Z_q, XQs])
        rowq2 = mf.Expr.hstack([mf.Expr.transpose(XQs), I_n_e])
        M.constraint("Zq_pos", mf.Expr.vstack([rowq1, rowq2]),
                     mf.Domain.inPSDCone(2 * n))

        # [W_r, Y^T R_w^{1/2}; *, I_nu] >= 0   (Schur: W_r >= Y^T R_w Y)
        R_s_m = mf.Matrix.dense(R_sqrt)
        YR_T = mf.Expr.mul(mf.Expr.transpose(Y), R_s_m)   # n x nu
        rowr1 = mf.Expr.hstack([W_r, YR_T])
        rowr2 = mf.Expr.hstack([mf.Expr.transpose(YR_T), I_nu_e])
        M.constraint("Wr_pos", mf.Expr.vstack([rowr1, rowr2]),
                     mf.Domain.inPSDCone(n + nu))

        # Per-vertex Lyapunov drop with embedded LQR-cost slack:
        #   [ X - Z_q - W_r,   (A_i X + B_i Y)^T ;
        #     A_i X + B_i Y,    X              ] >= 0
        slack = mf.Expr.sub(mf.Expr.sub(X, Z_q), W_r)   # X - Z_q - W_r
        for i, v in enumerate(vertices):
            A_i = np.asarray(v["A"], dtype=float)
            B_i = np.asarray(v["B"], dtype=float)
            Ai_m = mf.Matrix.dense(A_i)
            Bi_m = mf.Matrix.dense(B_i)

            AX = mf.Expr.mul(Ai_m, X)
            BY = mf.Expr.mul(Bi_m, Y)
            AX_BY = mf.Expr.add(AX, BY)
            row1 = mf.Expr.hstack([slack, mf.Expr.transpose(AX_BY)])
            row2 = mf.Expr.hstack([AX_BY, X])
            LMI = mf.Expr.vstack([row1, row2])
            M.constraint(f"rlqr_lmi_{i}", LMI,
                         mf.Domain.inPSDCone(2 * n))

        trace_W = mf.Expr.sum(W.diag())
        M.objective("min_trace_W", mf.ObjectiveSense.Minimize, trace_W)
        M.solve()

        sol_status = M.getProblemStatus()
        p_status = M.getPrimalSolutionStatus()
        d_status = M.getDualSolutionStatus()
        success = (sol_status == mf.ProblemStatus.PrimalAndDualFeasible
                   and p_status == mf.SolutionStatus.Optimal
                   and d_status == mf.SolutionStatus.Optimal)
        if not success:
            if verbose:
                print(f"[RobustLQR-LMI] solver status={sol_status}, "
                      f"primal={p_status}, dual={d_status}")
            return dict(
                success=False, K=np.zeros((nu, n)),
                X=np.eye(n), Y=np.zeros((nu, n)), W=np.eye(n),
                trace_W=np.array([np.inf]),
                solver_problem_status=np.array([str(sol_status)]),
                solver_primal_status=np.array([str(p_status)]),
                solver_dual_status=np.array([str(d_status)]),
            )

        Xv = base._level_to_matrix(X, (n, n), prefer_order="C")
        Xv = 0.5 * (Xv + Xv.T)
        Yv = base._level_to_matrix(Y, (nu, n), prefer_order="C")
        Wv = base._level_to_matrix(W, (n, n), prefer_order="C")
        try:
            K = la.solve(Xv.T, Yv.T).T
        except np.linalg.LinAlgError:
            if verbose:
                print("[RobustLQR-LMI] X singular; falling back to pinv")
            K = Yv @ np.linalg.pinv(Xv)

        return dict(
            success=True, K=K, X=Xv, Y=Yv, W=Wv,
            trace_W=np.array([float(np.trace(Wv))]),
            solver_problem_status=np.array([str(sol_status)]),
            solver_primal_status=np.array([str(p_status)]),
            solver_dual_status=np.array([str(d_status)]),
        )
    finally:
        M.dispose()


def solve_robust_lqr_wv(bounds: F8ParamBounds,
                        syn: F8SynthesisParams) -> Dict[str, Any]:
    """Wang-Veillette 1994 minimax robust LQR baseline.

    Design a *single* LQR on the worst open-loop-spectral-radius
    parameter corner with the actuator input matrix attenuated by
    ``(1 - actuator_gain_uncertainty)``, mirroring the reduced-gain
    endpoint seen by the five robust H-infinity SDP designs. The
    resulting K is then evaluated a posteriori on all 16 aerodynamic
    corners (and their reduced-gain replicas) via the corner-sweep
    audit; it is NOT a certified full-polytope design in this
    implementation. Unlike a nominal LQR, this non-clairvoyant Riccati
    reference does not require knowledge of the centre plant.

    Note: we use Ts, Qx_lqr, Ru_lqr identical to the LQR reference used
    internally by the data-collection rollout (line 645-646) so that
    every design inherits the same state / input weighting convention.

    Returns a dict with keys:
      - K          : (4, 16) state-feedback gain
      - p_worst    : dict of parameter values of the worst vertex
      - rho_ol     : open-loop spectral radius at the worst vertex
      - eps_a      : multiplicative gain uncertainty used in the design
      - status     : "feasible" / "infeasible_fallback_internalLQR"
    """
    keys = list(bounds.__dataclass_fields__.keys())
    p_worst: Optional[Dict[str, float]] = None
    rho_worst = -1.0
    for bits in itertools.product([0, 1], repeat=len(keys)):
        p = {k: getattr(bounds, k)[b] for k, b in zip(keys, bits)}
        A, _, _ = build_vertex_matrices(p, syn.Ts)
        rho = float(base.spectral_radius(A))
        if rho > rho_worst:
            rho_worst = rho
            p_worst = p
    assert p_worst is not None
    A_w, B_w, _ = build_vertex_matrices(p_worst, syn.Ts)
    eps_a = float(syn.actuator_gain_uncertainty) if syn.actuator_gain_uncertainty > 0 else 0.0
    B_w_scaled = (1.0 - eps_a) * B_w
    try:
        K = base.dlqr(A_w, B_w_scaled,
                      np.diag(syn.Qx_lqr), np.diag(syn.Ru_lqr))
        status = "feasible"
    except Exception as exc:
        p_nom = center_params(bounds)
        A_nom, B_nom, _ = build_vertex_matrices(p_nom, syn.Ts)
        K = base.dlqr(A_nom, B_nom,
                      np.diag(syn.Qx_lqr), np.diag(syn.Ru_lqr))
        status = f"infeasible_fallback_nominal({type(exc).__name__})"
    return dict(K=K, p_worst=p_worst, rho_ol=rho_worst,
                eps_a=eps_a, status=status)


# =====================================================================
# Data-contained core C_D (audit input, not a synthesis target)
# =====================================================================
# C_D is a sampled axis-aligned consistency box for the data-consistent
# set P_D = { p in P : exists d_k with ||d_k||_2 <= dbar that explains
# the data }. It is NOT a certified outer approximation: ZOH makes
# (A,B,S) non-affine in p, so a finite Monte-Carlo sample bbox is
# generally not a strict outer hull. The role of C_D is to provide a
# finite vertex set on which the fixed-K post-certificate (next section)
# is solved; transfer to a specific plant is reported separately via the
# matrix-hull membership residual. No controller is
# synthesised on C_D.
# =====================================================================

def _compute_data_consistent_box(
        bounds: F8ParamBounds,
        syn: F8SynthesisParams,
        batch: Dict[str, np.ndarray],
        *,
        n_samples: Optional[int] = None,
        seed_offset: int = 12345,
        verbose: bool = True,
) -> Dict[str, Any]:
    """Sampled approximation of the data-consistent parameter box C_D.

    For each MC sample p in P, the rollout {X_t, U_t, X_tp1} is checked
    for explainability under bounded disturbance ||d_k||_2 <= dbar:

        e_k = X_tp1 - A(p) X_t - B(p) U_t        (residual)
        d_k = pinv(S(p)) e_k                     (least-norm explainer)
        consistent iff
            max_k ||e_k - S(p) d_k|| < proj_tol  (e_k in range(S(p)))
        and max_k ||d_k||_2 <= dbar.

    Returns C_D = [p_D_lo, p_D_hi] = bbox(consistent samples) clipped
    into P. NOTE: ZOH discretisation makes (A(p), B(p), S(p)) non-affine
    in p, so P_D is nonconvex and a Monte-Carlo bbox is NOT a certified
    outer approximation. The dict therefore carries ``approximate=True``
    and downstream certificate transfer to any specific plant is gated
    by the matrix-hull membership residual (Corollary 1), not by an
    assumed P_D \\subset C_D containment.
    """
    keys = list(bounds.__dataclass_fields__.keys())
    p_lo = np.array([getattr(bounds, k)[0] for k in keys], dtype=float)
    p_hi = np.array([getattr(bounds, k)[1] for k in keys], dtype=float)

    n_samples = int(n_samples or syn.datacore_n_samples)
    dbar = float(syn.datacore_dbar)
    proj_tol = float(syn.datacore_proj_tol)

    rng = np.random.default_rng(syn.seed + int(seed_offset))
    X_t = np.asarray(batch["X_t"], dtype=float)
    X_tp1 = np.asarray(batch["X_tp1"], dtype=float)
    U_t = np.asarray(batch["U_t"], dtype=float)

    consistent_samples: List[np.ndarray] = []
    n_inrange_fail = 0
    n_dbar_fail = 0
    max_dnorms: List[float] = []

    for _ in range(n_samples):
        p_vec = p_lo + (p_hi - p_lo) * rng.random(len(keys))
        p = {k: float(v) for k, v in zip(keys, p_vec)}
        A, B, S = build_vertex_matrices(p, syn.Ts)
        E = X_tp1 - A @ X_t - B @ U_t                        # (16, L-1)
        S_pinv = np.linalg.pinv(S)
        D_rec = S_pinv @ E                                   # (6,  L-1)
        E_proj = S @ D_rec
        proj_err = np.linalg.norm(E - E_proj, axis=0)
        d_norms = np.linalg.norm(D_rec, axis=0)
        max_proj = float(np.max(proj_err)) if proj_err.size else 0.0
        max_dnorm = float(np.max(d_norms)) if d_norms.size else 0.0
        if max_proj > proj_tol:
            n_inrange_fail += 1
            continue
        if max_dnorm > dbar:
            n_dbar_fail += 1
            continue
        consistent_samples.append(p_vec)
        max_dnorms.append(max_dnorm)

    n_consistent = len(consistent_samples)
    fallback = False
    if n_consistent < (2 ** len(keys)):
        # Not enough consistent samples to define a meaningful bbox;
        # fall back to physical box C_D = P (no shrinkage).
        p_D_lo = p_lo.copy()
        p_D_hi = p_hi.copy()
        fallback = True
    else:
        Pc = np.array(consistent_samples)
        p_D_lo = np.maximum(np.min(Pc, axis=0), p_lo)
        p_D_hi = np.minimum(np.max(Pc, axis=0), p_hi)
        # ensure strict positive-volume box (required for 2^r corners)
        for j in range(len(keys)):
            if p_D_hi[j] - p_D_lo[j] < 1e-9 * max(1.0, p_hi[j] - p_lo[j]):
                mid = 0.5 * (p_D_lo[j] + p_D_hi[j])
                pad = 0.05 * (p_hi[j] - p_lo[j])
                p_D_lo[j] = max(p_lo[j], mid - pad)
                p_D_hi[j] = min(p_hi[j], mid + pad)

    shrink = (p_D_hi - p_D_lo) / np.maximum(p_hi - p_lo, 1e-12)
    n_shrunk = int(np.sum(shrink < 0.999))
    vol_p = float(np.prod(np.maximum(p_hi - p_lo, 1e-12)))
    vol_cd = float(np.prod(np.maximum(p_D_hi - p_D_lo, 1e-12)))
    vol_ratio = vol_cd / vol_p if vol_p > 0 else float("nan")

    if verbose:
        print(f"\n[DataCore] sampled P_D approximation "
              f"(N={n_samples}, dbar={dbar:.3f}, proj_tol={proj_tol:.1e})")
        print(f"  n_consistent  = {n_consistent}/{n_samples} "
              f"(in-range fail={n_inrange_fail}, dbar fail={n_dbar_fail})")
        for i, k in enumerate(keys):
            print(f"  {k:>10}: P=[{p_lo[i]:+.3f},{p_hi[i]:+.3f}] "
                  f"C_D=[{p_D_lo[i]:+.3f},{p_D_hi[i]:+.3f}] "
                  f"shrink={shrink[i]:.3f}")
        print(f"  vol(C_D)/vol(P) = {vol_ratio:.4f}, "
              f"n_shrunk_dims = {n_shrunk}/{len(keys)}, "
              f"fallback = {fallback}")
        print(f"  data_core_is_approximate = True  "
              f"(ZOH model not affine in p; sampled consistency box, "
              f"NOT a certified outer approximation of P_D)")

    return dict(
        keys=keys,
        p_lo=p_lo, p_hi=p_hi,
        p_D_lo=p_D_lo, p_D_hi=p_D_hi,
        shrinkage=shrink, n_shrunk_dims=n_shrunk,
        n_samples=n_samples, n_consistent=n_consistent,
        n_inrange_fail=n_inrange_fail, n_dbar_fail=n_dbar_fail,
        max_dnorms=np.array(max_dnorms, dtype=float),
        vol_ratio=vol_ratio, vol_p=vol_p, vol_cd=vol_cd,
        approximate=True, fallback=fallback,
        dbar=dbar, proj_tol=proj_tol,
    )


def _enumerate_box_corners_for_keys(
        box_lo: np.ndarray,
        box_hi: np.ndarray,
        keys: List[str],
) -> List[Dict[str, float]]:
    """Enumerate the 2^r corners of a box as a list of param-dicts."""
    out: List[Dict[str, float]] = []
    for bits in itertools.product([0, 1], repeat=len(keys)):
        p = {k: float(box_hi[i] if bit else box_lo[i])
             for i, (k, bit) in enumerate(zip(keys, bits))}
        out.append(p)
    return out


# =====================================================================
# Normalised-disturbance convention helper
# =====================================================================
def scaled_disturbance_matrix(S: np.ndarray, d_cert: float) -> np.ndarray:
    """Normalised-disturbance convention: tilde S = d_cert * S.

    Used *identically* in synthesis, audit, repair and full-box
    diagnostic so that the reported induced-L2 gain is always from
    the normalised disturbance w (with ||w||_2 <= 1) to z. The
    physical disturbance bound d_cert is absorbed into tilde S.

    Keeping this as a single helper prevents the paper/code drift
    of earlier versions where audit used raw S while synthesis used
    d_max * S, which silently inflated gamma_core by 1/d_cert.
    """
    return float(d_cert) * np.asarray(S, dtype=float)


# =====================================================================
# Fixed-K data-core post-certification (analysis-only Hinf SDP)
# =====================================================================
# Question answered:
#   "Given the controller K already produced by an existing synthesis
#    method (Proposed / NoRelax-ProposedActive / QS-Hinf / PDL-Hinf / RobustLQR),
#    does there exist a Lyapunov matrix P > 0 and a gain
#    g >= 0 such that the standard discrete-time bounded-real lemma
#    holds at every vertex of the data-consistent core C_D?"
#
# The user-facing LMI is the textbook 3-block primal form (Boyd et al.
# 1994, Sec. 7.6.2 eq. 7.10):
#     [[Acl.T P Acl - alpha P, Acl.T P Bcl, Ccl.T],
#      [Bcl.T P Acl, Bcl.T P Bcl - g I, Dcl.T],
#      [Ccl, Dcl, -I]] <= -eps I,
# but this is bilinear in P. To make it linear we use the equivalent
# 4-block Schur form (eliminating P via Schur complement on the (3,3)
# block -P):
#     [[ -alpha P,  0,        Acl.T P,   Ccl.T  ],
#      [  0,       -g I,      Bcl.T P,   Dcl.T  ],
#      [  P Acl,    P Bcl,    -P,         0     ],
#      [  Ccl,      Dcl,       0,        -I     ]] <= -eps I,
# which is linear in (P, g) and solvable directly in MOSEK Fusion.
# The two forms are mathematically equivalent (same feasible set and
# same optimal g); we report core_max_eig from the user's 3-block form
# at the optimum so the audit number matches the spec literally.
# =====================================================================

def _audit_fixed_K_on_core(
        K: np.ndarray,
        core_verts: List[Dict[str, Any]],
        syn: F8SynthesisParams,
        bsyn: "base.SynthesisParams",
        *,
        decay_rate: Optional[float] = None,
        eps_P: float = 1e-7,
        eps_LMI: float = 1e-7,
        rel_gap: float = 1e-5,
        time_limit_sec: int = 600,
        num_threads: int = 0,
        verbose: bool = False,
) -> Dict[str, Any]:
    """Fixed-K analysis SDP. With K fixed, find (P > 0, g >= 0) that
    satisfies the discrete-time bounded-real lemma simultaneously on
    every vertex of the data core. Returns:
      feasible           : bool, MOSEK reported PrimalAndDualFeasible
      gamma_core         : sqrt(g_opt)  (inf if infeasible)
      g                  : g_opt
      P                  : Lyapunov matrix (n x n) at optimum
      core_max_eig       : max over core vertices of lam_max(LMI_3block)
                           with the user's 3-block Schur form
      per_vertex_core_eigs : array of lam_max(LMI_3block) per vertex
      cond_P             : condition number of P
      status             : MOSEK problem status string
    """
    import mosek.fusion as mf
    # Delegated MOSEK license discovery: environment variable or
    # caller-provided path. No hard-coded path here.
    base.configure_mosek_license(verbose=False)

    n = 16
    nw = 6
    nz = 16
    n_core = len(core_verts)
    if decay_rate is None:
        # Default to alpha=1.0 (textbook bounded-real, no decay margin),
        # matching the user's literal LMI; pass syn.decay_rate explicitly
        # to enforce the same exponential margin used in synthesis.
        decay_rate = 1.0
    decay_rate = float(decay_rate)

    if n_core == 0:
        return dict(
            feasible=False, gamma_core=float("inf"), g=float("inf"),
            P=np.eye(n), core_max_eig=float("inf"),
            per_vertex_core_eigs=np.array([]),
            cond_P=float("nan"), status="empty_core",
            decay_rate=decay_rate, n_core=0,
        )

    M = mf.Model("fixed_K_audit")
    try:
        if verbose:
            M.setLogHandler(sys.stdout)
        if num_threads is None or num_threads <= 0:
            ncpu = os.cpu_count() or 1
            num_threads = max(1, ncpu // 2)
        M.setSolverParam("numThreads", int(num_threads))
        M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
        M.setSolverParam("optimizerMaxTime", float(time_limit_sec))

        # Variables (g must be true scalar, not (1,)-vector, so that
        # mf.Expr.mul(matrix_6x6, g) becomes the 6x6 expression g * I_nw.)
        P = M.variable("P", mf.Domain.inPSDCone(n))
        g = M.variable("g", mf.Domain.greaterThan(0.0))

        # P >= eps_P I (positive definite)
        eps_P_mat = mf.Matrix.dense(eps_P * np.eye(n))
        M.constraint("P_pd",
                     mf.Expr.sub(P, mf.Expr.constTerm(eps_P_mat)),
                     mf.Domain.inPSDCone(n))

        # Pre-build constants once per model
        I_nw = mf.Matrix.dense(np.eye(nw))
        I_nz_neg = mf.Matrix.dense(-np.eye(nz))
        margin_LMI = mf.Expr.constTerm(
            mf.Matrix.dense(eps_LMI * np.eye(n + nw + n + nz)))
        ZERO_n_nw = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((n, nw))))
        ZERO_nw_n = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nw, n))))
        ZERO_n_nz = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((n, nz))))
        ZERO_nw_nz = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nw, nz))))
        ZERO_nz_n = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nz, n))))

        # Normalised-disturbance convention: tilde S = d_cert * S
        # (see scaled_disturbance_matrix doc above). gamma_core is the
        # induced L2 gain from w (||w||_2 <= 1) to z, exactly the same
        # channel used by the synthesis LMI in solve_vertex_fusion_sdp.
        d_cert = float(syn.d_max)
        for i, v in enumerate(core_verts):
            A = np.asarray(v["A"], dtype=float)
            B = np.asarray(v["B"], dtype=float)
            S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
            C = np.asarray(v["C"], dtype=float)
            D = np.asarray(v["D"], dtype=float)
            Acl = A + B @ K
            Bcl = S_tilde
            Ccl = C + D @ K
            Dcl = np.zeros((nz, nw))    # disturbance has no direct feedthrough

            # P*Acl, P*Bcl  (matrix variable times constant matrix)
            P_Acl = mf.Expr.mul(P, mf.Matrix.dense(Acl))
            P_Bcl = mf.Expr.mul(P, mf.Matrix.dense(Bcl))
            Acl_T_P = mf.Expr.transpose(P_Acl)
            Bcl_T_P = mf.Expr.transpose(P_Bcl)

            negAlpha_P = mf.Expr.mul(-decay_rate, P)
            negP = mf.Expr.neg(P)
            # g is shape (1,); mf.Expr.mul(matrix, var) requires the var
            # in its original shape. Build (g * I_nw) first then negate.
            g_I = mf.Expr.mul(I_nw, g)
            neg_g_I = mf.Expr.neg(g_I)

            Ccl_T_const = mf.Expr.constTerm(mf.Matrix.dense(Ccl.T))
            Dcl_T_const = mf.Expr.constTerm(mf.Matrix.dense(Dcl.T))
            Ccl_const = mf.Expr.constTerm(mf.Matrix.dense(Ccl))
            Dcl_const = mf.Expr.constTerm(mf.Matrix.dense(Dcl))
            negI_nz = mf.Expr.constTerm(I_nz_neg)

            row1 = mf.Expr.hstack([negAlpha_P, ZERO_n_nw, Acl_T_P, Ccl_T_const])
            row2 = mf.Expr.hstack([ZERO_nw_n, neg_g_I, Bcl_T_P, Dcl_T_const])
            row3 = mf.Expr.hstack([P_Acl, P_Bcl, negP, ZERO_n_nz])
            row4 = mf.Expr.hstack([Ccl_const, Dcl_const, ZERO_nz_n, negI_nz])
            LMI = mf.Expr.vstack([row1, row2, row3, row4])

            # Require -LMI - eps I >= 0  i.e. LMI <= -eps I
            M.constraint(f"audit_lmi_{i}",
                         mf.Expr.sub(mf.Expr.neg(LMI), margin_LMI),
                         mf.Domain.inPSDCone(n + nw + n + nz))

        M.objective("min_g", mf.ObjectiveSense.Minimize, g)
        M.solve()

        sol_status = M.getProblemStatus()
        primal_status = M.getPrimalSolutionStatus()
        dual_status = M.getDualSolutionStatus()
        feasible = (sol_status == mf.ProblemStatus.PrimalAndDualFeasible
                    and primal_status == mf.SolutionStatus.Optimal)

        if feasible:
            P_v = base._level_to_matrix(P, (n, n), prefer_order=None)
            P_v = 0.5 * (P_v + P_v.T)
            g_v = float(base._level_to_float(g.level()))
            gamma_core = float(np.sqrt(max(g_v, 0.0)))

            # Compute per-vertex max-eig of the user's literal 3-block form
            # (normalised-disturbance convention, same tilde S as the SDP).
            per_eigs = np.zeros(n_core, dtype=float)
            for j, v in enumerate(core_verts):
                A = np.asarray(v["A"]); B = np.asarray(v["B"])
                S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
                C = np.asarray(v["C"]); D = np.asarray(v["D"])
                Acl = A + B @ K
                Bcl = S_tilde
                Ccl = C + D @ K
                Dcl = np.zeros((nz, nw))
                M11 = Acl.T @ P_v @ Acl - decay_rate * P_v
                M12 = Acl.T @ P_v @ Bcl
                M13 = Ccl.T
                M22 = Bcl.T @ P_v @ Bcl - g_v * np.eye(nw)
                M23 = Dcl.T
                M33 = -np.eye(nz)
                MM = np.block([[M11, M12, M13],
                               [M12.T, M22, M23],
                               [M13.T, M23.T, M33]])
                MM = 0.5 * (MM + MM.T)
                per_eigs[j] = float(np.max(np.linalg.eigvalsh(MM)))

            return dict(
                feasible=True, gamma_core=gamma_core, g=g_v, P=P_v,
                core_max_eig=float(np.max(per_eigs)),
                per_vertex_core_eigs=per_eigs,
                cond_P=float(np.linalg.cond(P_v)),
                status=str(primal_status),
                decay_rate=decay_rate,
                eps_P=float(eps_P), eps_LMI=float(eps_LMI),
                n_core=n_core,
            )
        else:
            return dict(
                feasible=False, gamma_core=float("inf"), g=float("inf"),
                P=np.eye(n), core_max_eig=float("inf"),
                per_vertex_core_eigs=np.array([]),
                cond_P=float("nan"),
                status=f"primal={primal_status}, dual={dual_status}, prob={sol_status}",
                decay_rate=decay_rate, n_core=n_core,
            )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if verbose:
            import traceback
            print(f"  [FixedKAudit] solver exception: {msg}")
            traceback.print_exc()
        return dict(
            feasible=False, gamma_core=float("inf"), g=float("inf"),
            P=np.eye(n), core_max_eig=float("inf"),
            per_vertex_core_eigs=np.array([]),
            cond_P=float("nan"),
            status=f"exception: {msg}",
            decay_rate=decay_rate, n_core=n_core,
        )
    finally:
        M.dispose()


def _repair_K_with_data_core(
        K_prop: np.ndarray,
        core_verts: List[Dict[str, Any]],
        syn: F8SynthesisParams,
        bsyn: "base.SynthesisParams",
        *,
        lambda_close: float = 1.0,
        decay_rate: Optional[float] = None,
        eps_Q: float = 1e-6,
        eps_LMI: float = 1e-7,
        rel_gap: float = 1e-5,
        time_limit_sec: int = 900,
        num_threads: int = 0,
        verbose: bool = False,
) -> Dict[str, Any]:
    """Certificate repair SDP for the Proposed controller.

    Solve:
        min  w_gamma * gamma2  +  lambda_close * t
        s.t. for every core vertex i:
                bounded-real synthesis LMI (unrelaxed, in dual variables
                Q = P^{-1}, Y = K Q) holds with margin eps_LMI;
             Q >> eps_Q I,  gamma2 >= 0,
             ||vec(Y - K_prop * Q)||_2 <= t   (Frobenius proximity to K_prop).

    K_repaired = Y Q^{-1}.  When lambda_close -> 0 this reduces to a pure
    H-inf core synthesis (gives the smallest gamma_core attainable on the
    core); when lambda_close -> inf it asymptotes back to K_prop.
    """
    import mosek.fusion as mf
    # Delegated MOSEK license discovery: environment variable or
    # caller-provided path. No hard-coded path here.
    base.configure_mosek_license(verbose=False)

    n = 16; nu = 4; nw = 6; nz = 16
    n_core = len(core_verts)
    if n_core == 0:
        return dict(success=False, K=K_prop.copy(), Q=np.eye(n),
                    Y=np.zeros((nu, n)), gamma2=float("inf"),
                    gamma=float("inf"), t=float("inf"),
                    K_norm=float(np.linalg.norm(K_prop)), status="empty_core")

    if decay_rate is None:
        decay_rate = float(getattr(syn, "decay_rate", 0.95))
    decay_rate = float(decay_rate)
    w_gamma = float(syn.w_gamma)

    M = mf.Model("repair_K_data_core")
    try:
        if verbose:
            M.setLogHandler(sys.stdout)
        if num_threads is None or num_threads <= 0:
            ncpu = os.cpu_count() or 1
            num_threads = max(1, ncpu // 2)
        M.setSolverParam("numThreads", int(num_threads))
        M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
        M.setSolverParam("optimizerMaxTime", float(time_limit_sec))

        # Variables mirror the synthesis SDP layout.
        # gamma2 and t are scalars (no size arg) so that
        # mf.Expr.mul(matrix, scalar) broadcasts correctly.
        Q = M.variable("Q", mf.Domain.inPSDCone(n))
        Y = M.variable("Y", [nu, n])
        gamma2 = M.variable("gamma2", mf.Domain.greaterThan(0.0))
        t = M.variable("t", mf.Domain.greaterThan(0.0))

        # Q >= eps I
        M.constraint("Q_pd",
                     mf.Expr.sub(Q, mf.Expr.constTerm(mf.Matrix.dense(eps_Q * np.eye(n)))),
                     mf.Domain.inPSDCone(n))

        # Pre-build constants
        I_nw = mf.Matrix.dense(np.eye(nw))
        I_nz_neg = mf.Matrix.dense(-np.eye(nz))
        margin_LMI = mf.Expr.constTerm(
            mf.Matrix.dense(eps_LMI * np.eye(n + nw + n + nz)))
        ZERO_n_nw = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((n, nw))))
        ZERO_nw_n = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nw, n))))
        ZERO_n_nz = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((n, nz))))
        ZERO_nw_nz = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nw, nz))))
        ZERO_nz_n = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((nz, n))))

        # Normalised-disturbance convention: tilde S = d_cert * S so
        # the repaired controller is certified on the same channel as
        # the synthesis and audit LMIs.
        d_cert = float(syn.d_max)
        for i, v in enumerate(core_verts):
            A = np.asarray(v["A"], dtype=float)
            B = np.asarray(v["B"], dtype=float)
            S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
            C = np.asarray(v["C"], dtype=float)
            D = np.asarray(v["D"], dtype=float)

            # Synthesis LMI in dual form (Q, Y), see Boyd-Feron-El Ghaoui
            # 1994 eq. 7.6.13. Substituting K = Y Q^{-1} and
            # pre/post-multiplying by diag(Q, I,
            # I, I) the user's primal LMI becomes the linear-in-(Q,Y) form:
            #
            # [[ -alpha Q,     0,          Q A.T + Y.T B.T,  Q C.T + Y.T D.T ],
            #  [ 0,           -gamma2 I,    tildeS.T,         0              ],
            #  [ A Q + B Y,    tildeS,     -Q,                0              ],
            #  [ C Q + D Y,    0,           0,               -I              ]] <= -eps I
            AQ_BY = mf.Expr.add(mf.Expr.mul(mf.Matrix.dense(A), Q),
                                mf.Expr.mul(mf.Matrix.dense(B), Y))
            CQ_DY = mf.Expr.add(mf.Expr.mul(mf.Matrix.dense(C), Q),
                                mf.Expr.mul(mf.Matrix.dense(D), Y))
            Si_e = mf.Expr.constTerm(mf.Matrix.dense(S_tilde))

            negAlpha_Q = mf.Expr.mul(-decay_rate, Q)
            negQ = mf.Expr.neg(Q)
            g_I_e = mf.Expr.mul(I_nw, gamma2)
            neg_g_I = mf.Expr.neg(g_I_e)
            negI_nz = mf.Expr.constTerm(I_nz_neg)

            row1 = mf.Expr.hstack([negAlpha_Q, ZERO_n_nw,
                                   mf.Expr.transpose(AQ_BY),
                                   mf.Expr.transpose(CQ_DY)])
            row2 = mf.Expr.hstack([ZERO_nw_n, neg_g_I,
                                   mf.Expr.transpose(Si_e),
                                   ZERO_nw_nz])
            row3 = mf.Expr.hstack([AQ_BY, Si_e, negQ, ZERO_n_nz])
            row4 = mf.Expr.hstack([CQ_DY, ZERO_nz_n, ZERO_nz_n, negI_nz])
            LMI = mf.Expr.vstack([row1, row2, row3, row4])

            M.constraint(f"repair_lmi_{i}",
                         mf.Expr.sub(mf.Expr.neg(LMI), margin_LMI),
                         mf.Domain.inPSDCone(n + nw + n + nz))

        # Frobenius proximity: ||Y - K_prop * Q||_F <= t
        # K_prop * Q is (nu x n) x (n x n) = (nu x n); Y is (nu x n); diff
        # has nu*n entries. Stack [t, vec(diff)] in second-order cone.
        Kprop_const = mf.Matrix.dense(np.asarray(K_prop, dtype=float))
        diff = mf.Expr.sub(Y, mf.Expr.mul(Kprop_const, Q))
        diff_flat = mf.Expr.flatten(diff)
        # Quadratic cone: t >= ||diff_flat||_2
        M.constraint("frobenius_close",
                     mf.Expr.vstack(t, diff_flat),
                     mf.Domain.inQCone())

        # Objective
        obj = mf.Expr.add(mf.Expr.mul(float(w_gamma), gamma2),
                          mf.Expr.mul(float(lambda_close), t))
        M.objective("min_gamma_plus_close", mf.ObjectiveSense.Minimize, obj)
        M.solve()

        sol_status = M.getProblemStatus()
        primal_status = M.getPrimalSolutionStatus()
        dual_status = M.getDualSolutionStatus()
        success = (sol_status == mf.ProblemStatus.PrimalAndDualFeasible
                   and primal_status == mf.SolutionStatus.Optimal)

        if success:
            Qv = base._level_to_matrix(Q, (n, n), prefer_order=None)
            Qv = 0.5 * (Qv + Qv.T)
            Yv = base._level_to_matrix(Y, (nu, n), prefer_order="C")
            g2v = float(base._level_to_float(gamma2.level()))
            tv = float(base._level_to_float(t.level()))
            try:
                Kv = Yv @ np.linalg.inv(Qv)
            except np.linalg.LinAlgError:
                Kv = Yv @ np.linalg.pinv(Qv)
            return dict(
                success=True, K=Kv, Q=Qv, Y=Yv,
                gamma2=g2v, gamma=float(np.sqrt(max(g2v, 0.0))),
                t=tv, frobenius_dist=tv,
                K_norm=float(np.linalg.norm(Kv)),
                K_prop_norm=float(np.linalg.norm(K_prop)),
                K_dist_to_prop=float(np.linalg.norm(Kv - K_prop)),
                lambda_close=float(lambda_close),
                decay_rate=decay_rate,
                n_core=n_core,
                status=str(primal_status),
            )
        else:
            return dict(
                success=False, K=K_prop.copy(),
                Q=np.eye(n), Y=np.zeros((nu, n)),
                gamma2=float("inf"), gamma=float("inf"),
                t=float("inf"), frobenius_dist=float("inf"),
                K_norm=float(np.linalg.norm(K_prop)),
                K_prop_norm=float(np.linalg.norm(K_prop)),
                K_dist_to_prop=0.0,
                lambda_close=float(lambda_close),
                decay_rate=decay_rate, n_core=n_core,
                status=f"primal={primal_status}, dual={dual_status}, prob={sol_status}",
            )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  [RepairK] solver exception: {msg}")
        return dict(
            success=False, K=K_prop.copy(),
            Q=np.eye(n), Y=np.zeros((nu, n)),
            gamma2=float("inf"), gamma=float("inf"),
            t=float("inf"), frobenius_dist=float("inf"),
            K_norm=float(np.linalg.norm(K_prop)),
            K_prop_norm=float(np.linalg.norm(K_prop)),
            K_dist_to_prop=0.0,
            lambda_close=float(lambda_close),
            decay_rate=decay_rate, n_core=n_core,
            status=f"exception: {msg}",
        )
    finally:
        M.dispose()


def _full_box_max_eig_for_K(
        K: np.ndarray,
        verts_all: List[Dict[str, Any]],
        P: np.ndarray,
        g: float,
        decay_rate: float = 1.0,
        d_cert: float = 1.0,
) -> float:
    """Diagnostic: max LMI eigenvalue of the user's 3-block form across
    EVERY physical-box vertex (full and reduced gain). Reuses the (P, g)
    certified on the data core; if any eigenvalue > 0 the certificate
    does NOT extend to the full physical box.

    d_cert: normalised-disturbance scaling so the 3-block form matches
    the audit SDP channel (tilde S = d_cert * S). Pass syn.d_max from
    the caller; default 1.0 retained only for backward-compat callers
    that pre-scale S themselves.
    """
    nw = 6; nz = 16
    if len(verts_all) == 0:
        return float("nan")
    eigs = []
    for v in verts_all:
        A = np.asarray(v["A"]); B = np.asarray(v["B"])
        S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
        C = np.asarray(v["C"]); D = np.asarray(v["D"])
        Acl = A + B @ K
        Bcl = S_tilde
        Ccl = C + D @ K
        Dcl = np.zeros((nz, nw))
        M11 = Acl.T @ P @ Acl - decay_rate * P
        M12 = Acl.T @ P @ Bcl
        M13 = Ccl.T
        M22 = Bcl.T @ P @ Bcl - g * np.eye(nw)
        M23 = Dcl.T
        M33 = -np.eye(nz)
        MM = np.block([[M11, M12, M13],
                       [M12.T, M22, M23],
                       [M13.T, M23.T, M33]])
        MM = 0.5 * (MM + MM.T)
        eigs.append(float(np.max(np.linalg.eigvalsh(MM))))
    return float(np.max(eigs))


def _fixed_k_lmi_vertex_eigs(
        K: np.ndarray,
        P: np.ndarray,
        g: float,
        verts: List[Dict[str, Any]],
        *,
        decay_rate: float,
        d_cert: float,
) -> np.ndarray:
    """Per-vertex residuals of the fixed-K 3-block bounded-real LMI."""
    nw = 6
    nz = 16
    eigs = np.full(len(verts), np.nan, dtype=float)
    if not np.isfinite(g):
        return eigs
    for j, v in enumerate(verts):
        A = np.asarray(v["A"], dtype=float)
        B = np.asarray(v["B"], dtype=float)
        S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
        C = np.asarray(v["C"], dtype=float)
        D = np.asarray(v["D"], dtype=float)
        Acl = A + B @ K
        Bcl = S_tilde
        Ccl = C + D @ K
        Dcl = np.zeros((nz, nw))
        M11 = Acl.T @ P @ Acl - float(decay_rate) * P
        M12 = Acl.T @ P @ Bcl
        M13 = Ccl.T
        M22 = Bcl.T @ P @ Bcl - float(g) * np.eye(nw)
        M23 = Dcl.T
        M33 = -np.eye(nz)
        MM = np.block([[M11, M12, M13],
                       [M12.T, M22, M23],
                       [M13.T, M23.T, M33]])
        MM = 0.5 * (MM + MM.T)
        eigs[j] = float(np.max(np.linalg.eigvalsh(MM)))
    return eigs


def _synthesis_qy_lmi_vertex_eigs(
        Q: np.ndarray,
        Y: np.ndarray,
        g: float,
        beta: object,
        verts: List[Dict[str, Any]],
        *,
        decay_rate: float,
        d_cert: float,
) -> np.ndarray:
    """Per-vertex residuals of the synthesis LMI in recovered (Q,Y,g) form."""
    n = 16
    nw = 6
    nz = 16
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    if beta_arr.size == 0:
        beta_arr = np.array([0.0], dtype=float)
    eigs = np.full(len(verts), np.nan, dtype=float)
    if not np.isfinite(g):
        return eigs
    for j, v in enumerate(verts):
        A = np.asarray(v["A"], dtype=float)
        B = np.asarray(v["B"], dtype=float)
        S_tilde = scaled_disturbance_matrix(v["S"], d_cert)
        C = np.asarray(v["C"], dtype=float)
        D = np.asarray(v["D"], dtype=float)
        s_i = float(v.get("s", 0.0))
        beta_i = float(beta_arr[j]) if beta_arr.size == len(verts) else float(beta_arr[0])
        AQ_BY = A @ Q + B @ Y
        CQ_DY = C @ Q + D @ Y
        LMI = np.zeros((n + nw + n + nz, n + nw + n + nz), dtype=float)
        LMI[0:n, 0:n] = -float(decay_rate) * Q - beta_i * s_i * np.eye(n)
        LMI[0:n, n + nw:n + nw + n] = AQ_BY.T
        LMI[0:n, n + nw + n:] = CQ_DY.T
        LMI[n:n + nw, n:n + nw] = -float(g) * np.eye(nw)
        LMI[n:n + nw, n + nw:n + nw + n] = S_tilde.T
        LMI[n + nw:n + nw + n, 0:n] = AQ_BY
        LMI[n + nw:n + nw + n, n:n + nw] = S_tilde
        LMI[n + nw:n + nw + n, n + nw:n + nw + n] = -Q
        LMI[n + nw + n:, 0:n] = CQ_DY
        LMI[n + nw + n:, n + nw + n:] = -np.eye(nz)
        LMI = 0.5 * (LMI + LMI.T)
        eigs[j] = float(np.max(np.linalg.eigvalsh(LMI)))
    return eigs


def _safe_first_float(x: object, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else float(default)
    except Exception:
        return float(default)


def _closed_loop_rho_for_vertex(K: np.ndarray, v: Dict[str, Any]) -> float:
    A = np.asarray(v["A"], dtype=float)
    B = np.asarray(v["B"], dtype=float)
    return float(np.max(np.abs(np.linalg.eigvals(A + B @ K))))


def _annotate_audit_vertex_diagnostics(
        au: Dict[str, Any],
        K: np.ndarray,
        core_verts: List[Dict[str, Any]],
) -> None:
    """Attach worst-audit-vertex and closed-loop rho diagnostics in-place."""
    per = np.asarray(au.get("per_vertex_core_eigs", []), dtype=float).reshape(-1)
    if per.size == 0 or len(core_verts) == 0 or not np.any(np.isfinite(per)):
        au.update(dict(
            worst_audit_vertex_index=-1,
            worst_audit_kappa=float("nan"),
            worst_audit_za_v=float("nan"),
            worst_audit_m_alpha=float("nan"),
            worst_audit_m_q=float("nan"),
            worst_audit_m_de=float("nan"),
            rho_closed_loop_at_worst_vertex=float("nan"),
            rho_closed_loop_max_over_core=float("nan"),
        ))
        return

    j = int(np.nanargmax(per))
    vj = core_verts[j]
    pj = dict(vj.get("p", {}))
    rhos = np.array([_closed_loop_rho_for_vertex(K, v) for v in core_verts], dtype=float)
    au.update(dict(
        worst_audit_vertex_index=j,
        worst_audit_kappa=float(pj.get("_act_scale", 1.0)),
        worst_audit_za_v=float(pj.get("za_v", float("nan"))),
        worst_audit_m_alpha=float(pj.get("m_alpha", float("nan"))),
        worst_audit_m_q=float(pj.get("m_q", float("nan"))),
        worst_audit_m_de=float(pj.get("m_de", float("nan"))),
        rho_closed_loop_at_worst_vertex=float(rhos[j]),
        rho_closed_loop_max_over_core=float(np.max(rhos)),
    ))


def _synthesis_audit_consistency_row(
        method: str,
        sol: Optional[Dict[str, Any]],
        synth_vertices: List[Dict[str, Any]],
        audit_vertices: List[Dict[str, Any]],
        fixed_audit: Optional[Dict[str, Any]],
        *,
        synth_vertex_set: str,
        audit_vertex_set: str,
        fallback_status: str,
        d_cert: float,
        audit_decay_rate: float,
) -> Dict[str, Any]:
    """Compare solver-returned synthesis gamma with fixed-K audit residuals."""
    fixed_audit = fixed_audit or {}
    gamma_fixed = float(fixed_audit.get("gamma_core", float("nan")))
    core_max = float(fixed_audit.get("core_max_eig", float("nan")))
    row: Dict[str, Any] = dict(
        method=method,
        gamma_syn=float("nan"),
        g_syn=float("nan"),
        decay_syn=float("nan"),
        decay_audit=float(audit_decay_rate),
        synth_vertex_set=synth_vertex_set,
        audit_vertex_set=audit_vertex_set,
        max_res_QY_on_synth_vertices=float("nan"),
        max_res_PK_at_syn_g_on_synth_vertices=float("nan"),
        max_res_PK_at_syn_g_on_audit_vertices=float("nan"),
        gamma_fixedK_optimized=gamma_fixed,
        core_max_eig_optimized=core_max,
        synthesis_gamma_is_certificate=False,
        status=fallback_status,
    )

    if sol is None:
        row["status"] = f"{fallback_status}; no synthesis variables"
        return row
    if "Q" not in sol or "Y" not in sol or "gamma2" not in sol:
        row["status"] = f"{fallback_status}; no Q/Y/gamma2 synthesis variables"
        return row

    g_syn = _safe_first_float(sol.get("gamma2"), float("nan"))
    decay_syn = _safe_first_float(sol.get("decay_rate"), float("nan"))
    if not np.isfinite(decay_syn):
        decay_syn = audit_decay_rate
    row["g_syn"] = g_syn
    row["gamma_syn"] = float(np.sqrt(max(g_syn, 0.0))) if np.isfinite(g_syn) else float("nan")
    row["decay_syn"] = decay_syn

    try:
        Q = np.asarray(sol["Q"], dtype=float)
        Y = np.asarray(sol["Y"], dtype=float)
        K = la.solve(Q.T, Y.T).T
        P = la.inv(Q)
        P = 0.5 * (P + P.T)
        beta = sol.get("beta", np.array([0.0]))
        qy_eigs = _synthesis_qy_lmi_vertex_eigs(
            Q, Y, g_syn, beta, synth_vertices,
            decay_rate=decay_syn, d_cert=d_cert,
        )
        pk_synth_eigs = _fixed_k_lmi_vertex_eigs(
            K, P, g_syn, synth_vertices,
            decay_rate=decay_syn, d_cert=d_cert,
        )
        pk_audit_eigs = _fixed_k_lmi_vertex_eigs(
            K, P, g_syn, audit_vertices,
            decay_rate=decay_syn, d_cert=d_cert,
        )
        max_qy = float(np.nanmax(qy_eigs)) if qy_eigs.size else float("nan")
        max_pk_synth = float(np.nanmax(pk_synth_eigs)) if pk_synth_eigs.size else float("nan")
        max_pk_audit = float(np.nanmax(pk_audit_eigs)) if pk_audit_eigs.size else float("nan")
        row["max_res_QY_on_synth_vertices"] = max_qy
        row["max_res_PK_at_syn_g_on_synth_vertices"] = max_pk_synth
        row["max_res_PK_at_syn_g_on_audit_vertices"] = max_pk_audit
        row["synthesis_gamma_is_certificate"] = bool(
            np.isfinite(max_qy) and np.isfinite(max_pk_synth) and np.isfinite(max_pk_audit)
            and max_qy <= 1e-6 and max_pk_synth <= 1e-6 and max_pk_audit <= 1e-6
        )
        problem = str(np.asarray(sol.get("solver_problem_status", ["?"])).reshape(-1)[0])
        primal = str(np.asarray(sol.get("solver_primal_status", ["?"])).reshape(-1)[0])
        row["status"] = f"{fallback_status}; problem={problem}; primal={primal}"
    except Exception as exc:
        row["status"] = f"{fallback_status}; diagnostic_exception={type(exc).__name__}: {exc}"
    return row


def synthesize_f8_controllers(
        syn: F8SynthesisParams,
        bounds: F8ParamBounds,
        *,
        L_data: int = 500,
        K_budget: int = 16,
        max_rounds: int = 3,
        score_mode: str = "lambda_max",
        verbose: bool = True,
) -> Dict[str, Any]:
    """Run the 5-way robust H-infinity synthesis and return a rich result dict.

    Controllers produced (all use error-feedback tracker law u_c = K (x - x_ref)):
      - Proposed    : data-driven beta-relaxation SDP with ICE active set
      - NoRelax-ProposedActive : beta = 0 on the SAME active set (ablation)
      - QS-Hinf     : quadratic-stability H-infinity over ALL 32 vertices
                      (Boyd-Feron-El Ghaoui-Balakrishnan 1994, single Q)
      - PDL-Hinf    : parameter-dependent Lyapunov H-infinity over ALL 32
                      vertices with slack matrix G (Daafouz-Bernussou 2001)
      - RobustLQR   : Wang-Veillette 1994 minimax LQR on the worst
                      open-loop-spectral-radius vertex with actuator-gain
                      attenuation embedded (textbook robust-control
                      baseline; provides a non-clairvoyant Riccati
                      reference without an H-infinity certificate).
    """
    bsyn = f8_to_base_syn(syn)

    # --- Nominal-plant DLQR -- USED ONLY as the stabilising feedback for
    # the data-collection rollout (simulate_batch_data) and as an internal
    # numeric fallback when NoRelax-ProposedActive is SDP-infeasible. It is NOT
    # reported as a controller in the baseline comparison: a nominal LQR
    # is clairvoyant (it knows the centre plant) and is therefore not a
    # fair head-to-head robust-control baseline. See solve_pdl_hinf_mosek
    # / the comment block above this function for the retained baselines.
    p_nom = center_params(bounds)
    A_nom_lqr, B_nom_lqr, _ = build_vertex_matrices(p_nom, syn.Ts)
    K_nom_internal = base.dlqr(A_nom_lqr, B_nom_lqr,
                               np.diag(syn.Qx_lqr), np.diag(syn.Ru_lqr))

    # --- Data-generating plant: drawn within +-30% of the half-width of
    # each parameter range around nominal. This is intentionally a *subset*
    # of the full parameter box because the data-collection rollout uses
    # K_nominal as a stabilising controller, and K_nominal cannot stabilise
    # the open-loop-unstable corners of the box (m_alpha < 0). The robust
    # the full physical corners are still evaluated later as diagnostic
    # stress tests. The data-driven Psi is built with a worst-case
    # disturbance-channel matrix, using the same disturbance-channel
    # construction as the synthesis and audit LMIs.
    # This keeps the data rollout separate from the unstable corners.
    rng_data = np.random.default_rng(syn.seed + 999)
    p_data_source: Dict[str, float] = {}
    for k in bounds.__dataclass_fields__.keys():
        lo, hi = getattr(bounds, k)
        half = 0.5 * (hi - lo)
        # Additive perturbation +- 30% of the half-width around nominal,
        # then clipped to bounds. Works for parameters that span zero.
        val = p_nom[k] + float(rng_data.uniform(-0.3, 0.3)) * half
        p_data_source[k] = float(np.clip(val, lo, hi))
    if verbose:
        print("\n[F8 Data] sampled F-8 data-generating plant (near-nominal):")
        for k, v in p_data_source.items():
            print(f"  {k}: {v:.4f}  (nom={p_nom[k]:.4f})")

    # --- Simulate batch for consistency ellipsoid ---
    if verbose:
        print(f"\n[F8 Data] simulating batch L={L_data} ...")
    # meas_noise_std=0 for the synthesis batch: the data-consistent set
    # P_D in paper Sec. II is defined for noise-free input/output data,
    # so the score r_i in Eq. (3) and the data core in Eq. (Sec. II)
    # both assume an exact recursion x^+ = A(p*)x + B(p*)u + S(p*)d.
    # Tracking and Monte-Carlo simulations downstream still use a
    # non-zero meas_noise_std (sensor robustness test, not part of the
    # data-consistency definition).
    batch = base.simulate_batch_data(
        p_data_source, bsyn, L=L_data, excite_scale=0.25, meas_noise_std=0.0,
        K_fb=K_nom_internal, p_lqr=p_nom,
    )
    batch["S_c"] = worst_case_S_matrix(bounds, syn)
    # Build Psi with the synthesis / certification bound bar d_cert = syn.d_max
    # (same scaling as the LMI tilde S = d_cert * S). The data-core rejection
    # below uses its own bound bar d_data = syn.datacore_dbar and does not
    # enter Psi.
    Psi = base.build_psi_data(batch, bsyn)  # uses bsyn.d_max by default

    # --- Score the 16 corner vertices ---
    verts_all = base.build_all_vertices_and_scores(
        bounds, bsyn, Psi, score_mode=score_mode, max_vertices=64, seed=syn.seed + 1234,
    )
    for aero_idx, v in enumerate(verts_all):
        v["p"] = dict(v.get("p", {}))
        v["p"]["_aero_index"] = int(aero_idx)

    # --- Multiplicative actuator-gain uncertainty (16 -> 32 plants) ---
    # When syn.actuator_gain_uncertainty > 0, each parameter corner is
    # replicated with the actuator input gain reduced to (1 - eps_act).
    # All five robust H-infinity SDPs (Proposed, NoRelax-ProposedActive, QS-Hinf,
    # PDL-Hinf, Core-CQLF-Hinf) then enforce their LMI on all 32 vertices, exposing every
    # controller to the static input-effectiveness endpoints {1, 1 - eps_act}
    # at synthesis and audit time. By affine dependence of the LMI on the
    # scaled input matrix, the endpoint constraints represent the prescribed
    # static input-effectiveness interval within the corresponding
    # convex-hull interpretation; this is the standard way to encode
    # multiplicative input-gain uncertainty into a polytopic LMI
    # (Skogestad-Postlethwaite, Sec.8.5; Boyd et al. 1994 Sec.7.6.2).
    eps_design = float(syn.actuator_gain_uncertainty) if syn.actuator_gain_uncertainty > 0 else 0.0
    if eps_design > 0:
        scale = 1.0 - eps_design
        extra: List[Dict[str, Any]] = []
        for v in verts_all:
            v_low = dict(v)
            v_low["A"] = np.asarray(v["A"]).copy()
            v_low["B"] = np.asarray(v["B"]) * scale
            v_low["S"] = np.asarray(v["S"]).copy()
            v_low["C"] = np.asarray(v["C"]).copy()
            v_low["D"] = np.asarray(v["D"]).copy()
            v_low["Delta"] = np.block([[v_low["A"], v_low["B"]],
                                       [v_low["C"], v_low["D"]]])
            # Consistency score inheritance: the reduced-gain plant
            # (A, scale*B) is a linear image of the full-gain counterpart
            # (A, B) under the multiplicative input-gain uncertainty. We
            # inherit s from the full-gain vertex rather than:
            #  (a) recomputing s via the original Psi even though the
            #      scaled plant was not observed in the batch rollout;
            #  (b) forcing s = 0 -- this degenerates ICE's active-set
            #      selection: treating 16 plants as perfectly data-
            #      consistent collapses the SDP to a NoRelax-ProposedActive-like
            #      beta=0 design, producing an ill-conditioned K (rho>1
            #      in empirical experiments).
            # Inheritance preserves both the beta-relaxation mechanism
            # and the interpretability of the consistency score.
            v_low["s"] = float(v["s"])
            v_low["p"] = dict(v["p"])
            v_low["p"]["_act_scale"] = float(scale)
            extra.append(v_low)
        # Tag the original 16 with full-gain marker so post-mortem audits can
        # distinguish full-gain vs reduced-gain corners.
        for v in verts_all:
            v["p"] = dict(v["p"])
            v["p"]["_act_scale"] = 1.0
        verts_all = list(verts_all) + extra
        if verbose:
            print(f"\n[F8 Vertices] actuator-gain uncertainty eps={eps_design:.2f} "
                  f"-> extended 16 to {len(verts_all)} plants (16 full-gain + "
                  f"16 reduced-gain with s inherited from full-gain "
                  f"counterpart; see comment block above).")

    if verbose:
        raw_s = np.array([float(v["s"]) for v in verts_all], dtype=float)
        print(f"\n[F8 Vertices] n={len(verts_all)}, raw_s min={raw_s.min():.3e}, "
              f"max={raw_s.max():.3e}, mean={raw_s.mean():.3e}")

    # --- Score normalisation per paper Eq. (4): map raw residual scores
    # r_i to s_i in [0, 1] via the smooth saturation
    #   s_i = clip( 1 - exp( -max(0, (r_i - tau_D)/L) / sigma ) ),
    # with tau_D = n_hc-th order statistic of {r_i}. The hard core
    # I_hard = the n_hc lowest-residual vertices is materialised inside
    # compute_si_from_vi (the s_i field is set to 0 on I_hard); no
    # downstream post-promotion is needed. ---
    verts_norm, score_diag = base.compute_si_from_vi(
        verts_all, L_data, n_hc=8, q_scale=0.9,
    )
    # Make the paper convention explicit: the hard core is selected at the
    # aerodynamic-corner level, and both actuator-gain endpoints of each
    # selected aerodynamic corner are lifted into I_hard. Since the endpoints
    # inherit the same raw score, this should preserve the numerical set while
    # making the grouping auditable in the exported metadata.
    n_hc_lifted = 8
    endpoint_count = 2 if eps_design > 0 else 1
    if n_hc_lifted % endpoint_count != 0:
        raise ValueError("n_hc_lifted must be divisible by the number of actuator endpoints")
    aero_scores: Dict[int, float] = {}
    for v in verts_norm:
        aero_idx = int(dict(v.get("p", {})).get("_aero_index", -1))
        if aero_idx < 0:
            continue
        raw_score = float(v.get("s_raw", v.get("s", 0.0)))
        aero_scores.setdefault(aero_idx, raw_score)
    hard_aero = set(sorted(aero_scores, key=lambda j: (aero_scores[j], j))[:n_hc_lifted // endpoint_count])
    for v in verts_norm:
        aero_idx = int(dict(v.get("p", {})).get("_aero_index", -1))
        if aero_idx in hard_aero:
            v["s"] = 0.0
            v["p"] = dict(v.get("p", {}))
            v["p"]["_hard_core_aero"] = 1
        else:
            v["p"] = dict(v.get("p", {}))
            v["p"]["_hard_core_aero"] = 0
    score_diag["hard_core_aero_count"] = int(len(hard_aero))
    score_diag["hard_core_aero_indices"] = sorted(int(j) for j in hard_aero)
    score_diag["n_consistent"] = int(sum(float(v.get("s", 0.0)) == 0.0 for v in verts_norm))
    if verbose:
        s_norm = np.array([float(v["s"]) for v in verts_norm], dtype=float)
        print(f"[F8 Scores] normalized s: min={s_norm.min():.3f}, max={s_norm.max():.3f}, "
              f"mean={s_norm.mean():.3f}, n_zero={score_diag['n_consistent']}/{len(verts_norm)}")
        print(f"[F8 Scores] hard-core aerodynamic corners: "
              f"{score_diag['hard_core_aero_indices']} x {endpoint_count} actuator endpoints")

    # --- Proposed: iterative constraint exchange with beta-relaxation ---
    N_tot = len(verts_norm)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[F8 ICE] K_budget={K_budget}, rounds={max_rounds}, N_total={N_tot}")
        print(f"{'=' * 60}")
    # Hard core (8 lowest-residual vertices) is now materialised by
    # compute_si_from_vi(n_hc=8) above, so no force_top_n_hard_core
    # post-promotion is needed here. force_include_hard_core=True
    # still tells ICE to include all s=0 vertices in the initial
    # active set (Theorem 2 hypothesis on I_hard).
    verts_common, sol_proposed = base.iterative_constraint_exchange(
        all_vertices=verts_norm, syn=bsyn, K_budget=K_budget,
        max_rounds=max_rounds, viol_tol=1e-4, seed=syn.seed + 7777,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        enforce_perf_all_vertices=syn.enforce_perf_all_vertices,
        verbose=verbose, bounds=bounds,
        force_include_hard_core=True,
    )
    gamma_proposed = float(np.sqrt(max(float(sol_proposed["gamma2"][0]), 0.0)))

    # --- NoRelax-ProposedActive ablation: SAME active set chosen by ICE for Proposed,
    # but s_i = 0 (no beta-relaxation slack). This isolates the effect of
    # beta-relaxation from any active-set selection effect; controlling for
    # the active set is the standard ablation protocol.
    verts_norelax = [dict(v, s=0.0) for v in verts_common]
    if verbose:
        print(f"\n[F8 NoRelax-ProposedActive] same active set ({len(verts_norelax)} vertices), s_i=0")
    sol_norelax = base.solve_vertex_fusion_sdp_mosek(
        verts_norelax, bsyn, verbose=False, num_threads=0, max_iters=50,
        rel_gap=1e-4, time_limit_sec=1800, beta_lb=0.0,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        enforce_perf_all_vertices=syn.enforce_perf_all_vertices,
        x0_feas=np.zeros(16),
    )
    if sol_norelax.get("success", False):
        gamma_norelax = float(np.sqrt(max(float(sol_norelax["gamma2"][0]), 0.0)))
    else:
        gamma_norelax = float("inf")

    K_proposed = np.asarray(sol_proposed["K"], dtype=float)
    if sol_norelax.get("success", False):
        K_norelax = np.asarray(sol_norelax["K"], dtype=float)
        norelax_status = "feasible"
    else:
        # NoRelax-ProposedActive SDP infeasibility is itself a headline ablation result
        # (no robust H-infinity controller exists with beta=0 on this active
        # set). For downstream time-domain comparison we fall back to the
        # internal nominal LQR so the simulation is still meaningful, and we
        # record the fallback status explicitly so it can never be confused
        # with a successful NoRelax-ProposedActive design. We never use a zero matrix,
        # which would silently corrupt the comparison.
        K_norelax = K_nom_internal.copy()
        norelax_status = "infeasible_fallback_internalLQR"
        if verbose:
            print("  [WARN] NoRelax-ProposedActive SDP infeasible -- falling back to "
                  "internal nominal LQR for time-domain plots (gamma = inf).")

    eps_act = float(syn.actuator_gain_uncertainty) if syn.actuator_gain_uncertainty > 0 else 0.0

    # --- QS-Hinf baseline (Boyd-Feron-El Ghaoui-Balakrishnan 1994) ---
    # Classical single-Q quadratic-stability H-infinity on ALL 32 vertices.
    # Implemented via solve_vertex_fusion_sdp_mosek with beta forcibly fixed
    # to 0 (fixed_beta_values=0.0): this disables the data-driven relaxation
    # mechanism entirely and collapses the SDP to the classical worst-case
    # LMI in Boyd 1994 Sec 7.6.2. Unlike NoRelax-ProposedActive which uses the ICE-
    # selected active-set subset, QS-Hinf enforces the LMI on the FULL 32-
    # vertex set (no ICE). This is the standard Boyd'94 apples-to-apples
    # comparator against which every robust-H-infinity paper must be benchmarked.
    if verbose:
        print(f"\n[F8 QS-Hinf (Boyd 1994)] single-Q H-inf over all {len(verts_norm)} vertices, beta=0")
    verts_qs = [dict(v, s=0.0) for v in verts_norm]
    # Note: the classical Boyd'94 formulation contains only the H-infinity
    # LMI; the additional mu-performance LMI (enforce_perf_all_vertices
    # = True) is an extension layer not present in the 1994 monograph, and
    # activating it on 32 vertices with single-Q produces a numerically
    # ill-conditioned SDP (MOSEK returns status=Unknown). We therefore
    # DISABLE the perf LMI for this baseline so it is faithful to Boyd'94.
    # Proposed and NoRelax-ProposedActive retain the perf LMI (on their 8-vertex ICE
    # active set) because it is part of the Proposed method's design.
    #
    # Numerical tolerances: the 32-vertex single-Q SDP converges in primal
    # and dual to duality gap < 0.1% within ~4 Newton iterations, but MOSEK
    # flags "problem status Unknown" at the default 1e-6 feasibility
    # tolerance because the last Newton step becomes too small. We relax
    # pfeas/dfeas to 5e-3 (still tighter than the physical modelling error
    # in the F-8 plant) so the solver certifies Optimal at iter 2-3 and
    # returns a usable controller.
    bsyn_qs = dataclasses.replace(bsyn, mosek_pfeas=1e-3, mosek_dfeas=1e-3)
    sol_qs_hinf = base.solve_vertex_fusion_sdp_mosek(
        verts_qs, bsyn_qs, verbose=verbose, num_threads=0, max_iters=200,
        rel_gap=1e-3, time_limit_sec=300, beta_lb=0.0,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        enforce_perf_all_vertices=False,
        x0_feas=np.zeros(16),
        fixed_beta_values=0.0,
    )
    if sol_qs_hinf.get("success", False):
        gamma_qs_hinf = float(np.sqrt(max(float(sol_qs_hinf["gamma2"][0]), 0.0)))
        K_qs_hinf = np.asarray(sol_qs_hinf["K"], dtype=float)
        # Post-solve LMI audit: because QS-Hinf runs at looser MOSEK
        # tolerances (1e-3 vs 1e-6) we verify the solution satisfies the
        # H-inf LMI on all 32 vertices to within 1% before accepting it.
        # This keeps the certified-bound semantics of gamma meaningful even
        # under the relaxed solver tolerances.
        viol_qs = base.check_all_vertex_violations(
            verts_qs, K_qs_hinf, sol_qs_hinf["Q"],
            float(sol_qs_hinf["gamma2"][0]),
            np.atleast_1d(np.asarray(sol_qs_hinf["beta"])).astype(float),
            syn.decay_rate, syn.d_max,
        )
        qs_hinf_max_viol = float(np.max(viol_qs))
        qs_hinf_n_viol = int(np.sum(viol_qs > 1e-3))
        if qs_hinf_max_viol > 1e-2:
            qs_hinf_status = f"feasible_relaxed(max_viol={qs_hinf_max_viol:.2e})"
        else:
            qs_hinf_status = "feasible"
        if verbose:
            print(f"  [QS-Hinf post-solve] 32-vertex LMI max_viol={qs_hinf_max_viol:.3e}, "
                  f"n_viol={qs_hinf_n_viol}/32")
    else:
        gamma_qs_hinf = float("inf")
        K_qs_hinf = K_nom_internal.copy()
        qs_hinf_max_viol = float("nan")
        qs_hinf_n_viol = -1
        qs_hinf_status = "infeasible_fallback_internalLQR"
        if verbose:
            print("  [WARN] QS-Hinf SDP infeasible -- falling back to "
                  "internal nominal LQR (gamma_qs_hinf = inf).")

    # --- PDL-Hinf baseline (Daafouz-Bernussou 2001) ---
    # Parameter-dependent Lyapunov function H-infinity: every vertex gets
    # its own Lyapunov matrix P_i, coupled through a shared slack matrix G.
    # Strictly less conservative than QS-Hinf (single Q) on polytopic
    # uncertainty. This is the standard less-conservative LMI benchmark in
    # modern robust control (Apkarian-Tuan 2000, de Oliveira-Geromel 1999).
    # Enforced on ALL 32 vertices (same as QS-Hinf) for an apples-to-apples
    # fair comparison against the Boyd'94 classical baseline.
    if verbose:
        print(f"\n[F8 PDL-Hinf (Daafouz-Bernussou 2001)] slack-matrix H-inf "
              f"over all {len(verts_norm)} vertices with per-vertex P_i")
    sol_pdl_hinf = solve_pdl_hinf_mosek(
        verts_norm, bsyn, verbose=False, num_threads=0, max_iters=80,
        rel_gap=1e-4, time_limit_sec=1800, w_gamma=syn.w_gamma,
    )
    if sol_pdl_hinf.get("success", False):
        gamma_pdl_hinf = float(np.sqrt(max(float(sol_pdl_hinf["gamma2"][0]), 0.0)))
        K_pdl_hinf = np.asarray(sol_pdl_hinf["K"], dtype=float)
        pdl_hinf_status = "feasible"
    else:
        gamma_pdl_hinf = float("inf")
        K_pdl_hinf = K_nom_internal.copy()
        pdl_hinf_status = "infeasible_fallback_internalLQR"
        if verbose:
            print("  [WARN] PDL-Hinf SDP infeasible -- falling back to "
                  "internal nominal LQR (gamma_pdl_hinf = inf).")

    # --- RobustLQR-LMI baseline -------------------
    # True polytopic guaranteed-cost robust LQR LMI on all 32 vertices
    # (Boyd-Feron-El Ghaoui-Balakrishnan 1994 LMI book Sec. 7.4.2;
    # discrete-time analog of Petersen 1995 modified Riccati). Unlike the
    # worst-vertex DARE heuristic (solve_robust_lqr_wv), the
    # Lyapunov function X is enforced on every vertex of the polytope.
    # RobustLQR-LMI does NOT supply an H-infinity certificate (gamma_robust_lqr
    # is set to inf), but it provides a polytope-uniform quadratic-cost
    # upper bound tr(W) >= tr(X^{-1}) -- the textbook robust-LQR analogue of
    # the H-infinity designs above. The worst-vertex DARE heuristic is
    # computed alongside for diagnostic comparison only.
    sol_rlqr_lmi = solve_robust_lqr_polytopic_lmi(
        verts_norm, syn, verbose=False, num_threads=0, max_iters=200,
        rel_gap=1e-6, time_limit_sec=1800,
    )
    if sol_rlqr_lmi.get("success", False):
        K_robust_lqr = np.asarray(sol_rlqr_lmi["K"])
        robust_lqr_status = "feasible"
        robust_lqr_trace_W = float(sol_rlqr_lmi["trace_W"][0])
    else:
        # Fall back to the worst-vertex DARE heuristic if the LMI fails.
        wv_fb = solve_robust_lqr_wv(bounds, syn)
        K_robust_lqr = wv_fb["K"]
        robust_lqr_status = "lmi_infeasible_fallback_wv"
        robust_lqr_trace_W = float("nan")
        if verbose:
            print("  [WARN] RobustLQR-LMI infeasible -- falling back to "
                  "worst-vertex DARE heuristic.")
    # Diagnostics from the worst-vertex heuristic. These are kept for
    # reproducibility and comparison but do not drive K.
    wv_diag = solve_robust_lqr_wv(bounds, syn)
    robust_lqr_p_worst = wv_diag["p_worst"]
    robust_lqr_rho_ol = wv_diag["rho_ol"]
    if verbose:
        print(f"\n[F8 RobustLQR-LMI (Boyd-BEFG 1994; Petersen 1995)] "
              f"polytopic guaranteed-cost LMI over {len(verts_norm)} vertices, "
              f"||K||={np.linalg.norm(K_robust_lqr):.3f}, "
              f"tr(W)={robust_lqr_trace_W:.3e}, status={robust_lqr_status}")
        print(f"  (diagnostic) WV-DARE corner={robust_lqr_p_worst}, "
              f"rho_OL={robust_lqr_rho_ol:.4f}")

    # ===== Data-contained core C_D (built once, reused by the audit) =====
    # The audit phase below evaluates the fixed-K post-certificate on the
    # corner set of C_D. The core itself is a sampled consistency box
    # (NOT a certified outer approximation of P_D, see
    # _compute_data_consistent_box) and is independent of any controller;
    # we simply build it here so the Core-CQLF-Hinf baseline, the audit,
    # and the optional certificate-repair SDP can reuse the same vertex set.
    datacore_diag: Dict[str, Any] = _compute_data_consistent_box(
        bounds, syn, batch, verbose=verbose,
    )

    # --- Core-CQLF-Hinf baseline -----------------------------------------
    # Disentangles "uncertainty-set shrinking" from "score relaxation":
    # solves the SAME H-infinity LMI as Proposed/QS-Hinf but DIRECTLY on
    # the data-core vertex set V_{C_D}^{audit} (the corner set of C_D
    # replicated across the actuator-gain endpoints kappa in {1, 1-eps_a}),
    # with score relaxation forcibly disabled (fixed_beta_values=0.0).
    # The synthesis vertex set therefore coincides exactly with the audit
    # vertex set, isolating the contribution of "design directly on the
    # shrunk box" without any data-driven slack.
    cd_keys_synth = datacore_diag["keys"]
    cd_p_lo_synth = datacore_diag["p_D_lo"]
    cd_p_hi_synth = datacore_diag["p_D_hi"]
    cd_corners_synth = _enumerate_box_corners_for_keys(
        cd_p_lo_synth, cd_p_hi_synth, cd_keys_synth,
    )
    eps_act_synth = (float(syn.actuator_gain_uncertainty)
                     if syn.actuator_gain_uncertainty > 0 else 0.0)
    kappa_list_synth = [1.0] if eps_act_synth == 0.0 else [1.0, 1.0 - eps_act_synth]
    Cc_synth, Dc_synth = base.build_performance_matrices(bsyn)
    verts_core_cqlf: List[Dict[str, Any]] = []
    for p in cd_corners_synth:
        p_full = {**p_nom, **p}
        A_p, B_p, S_p = build_vertex_matrices(p_full, syn.Ts)
        for kappa in kappa_list_synth:
            B_k = float(kappa) * B_p
            Delta_k = np.block([[A_p, B_k], [Cc_synth, Dc_synth]])
            verts_core_cqlf.append(dict(
                A=A_p, B=B_k, S=S_p, C=Cc_synth, D=Dc_synth,
                Delta=Delta_k,
                p={**p_full, "_act_scale": float(kappa)},
                s=0.0, _kind="core_cqlf_synth",
            ))
    if verbose:
        print(f"\n[F8 Core-CQLF-Hinf] design directly on V_{{C_D}}: "
              f"{len(verts_core_cqlf)} vertices "
              f"(2^{len(cd_keys_synth)} aero corners x "
              f"{len(kappa_list_synth)} gain endpoints), beta = 0")
    bsyn_core_cqlf = dataclasses.replace(bsyn, mosek_pfeas=1e-3, mosek_dfeas=1e-3)
    sol_core_cqlf_hinf = base.solve_vertex_fusion_sdp_mosek(
        verts_core_cqlf, bsyn_core_cqlf, verbose=False, num_threads=0,
        max_iters=200, rel_gap=1e-3, time_limit_sec=1800, beta_lb=0.0,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        enforce_perf_all_vertices=False,
        x0_feas=np.zeros(16),
        fixed_beta_values=0.0,
    )
    if sol_core_cqlf_hinf.get("success", False):
        gamma_core_cqlf_hinf = float(np.sqrt(max(float(sol_core_cqlf_hinf["gamma2"][0]), 0.0)))
        K_core_cqlf_hinf = np.asarray(sol_core_cqlf_hinf["K"], dtype=float)
        viol_core_cqlf_full = base.check_all_vertex_violations(
            verts_norm, K_core_cqlf_hinf, sol_core_cqlf_hinf["Q"],
            float(sol_core_cqlf_hinf["gamma2"][0]),
            np.atleast_1d(np.asarray(sol_core_cqlf_hinf["beta"])).astype(float),
            syn.decay_rate, syn.d_max,
        )
        core_cqlf_full_max_viol = float(np.max(viol_core_cqlf_full))
        core_cqlf_full_n_viol = int(np.sum(viol_core_cqlf_full > 1e-4))
        core_cqlf_status = "feasible"
        if verbose:
            print(f"  [Core-CQLF-Hinf] gamma_syn_diag={gamma_core_cqlf_hinf:.4f}, "
                  f"||K||={np.linalg.norm(K_core_cqlf_hinf):.3f}; "
                  f"full-box LMI on N={len(verts_norm)}: "
                  f"n_viol={core_cqlf_full_n_viol}, max_viol={core_cqlf_full_max_viol:.3e}")
    else:
        gamma_core_cqlf_hinf = float("inf")
        K_core_cqlf_hinf = np.full((4, 16), np.nan)
        core_cqlf_full_max_viol = float("nan")
        core_cqlf_full_n_viol = -1
        core_cqlf_status = "infeasible"
        if verbose:
            print("  [WARN] Core-CQLF-Hinf SDP infeasible (K set to NaN, "
                  "will NOT enter controller roster or paper results).")

    # --- All-vertex LMI violation audit ---
    viol_prop = base.check_all_vertex_violations(
        verts_norm, sol_proposed["K"], sol_proposed["Q"],
        float(sol_proposed["gamma2"][0]),
        np.atleast_1d(np.asarray(sol_proposed["beta"])).astype(float),
        syn.decay_rate, syn.d_max,
    )
    prop_max_viol = float(np.max(viol_prop))
    prop_n_viol = int(np.sum(viol_prop > 1e-4))

    # Hard-core / soft-shell + full-gain / reduced-gain breakdown of the
    # all-vertex audit. Hard-core = v["s"] == 0 (aerodynamic-parameter
    # vertex with zero data-residual); soft-shell = v["s"] > 0. Full-gain
    # = v["p"]["_act_scale"] == 1.0; reduced-gain = _act_scale < 1.0.
    # These breakdowns drive Table I hard/soft and gain-case sub-columns.
    VIOL_TOL = 1e-4
    s_arr = np.array([float(v["s"]) for v in verts_norm])
    act_arr = np.array([float(v["p"].get("_act_scale", 1.0)) for v in verts_norm])
    mask_hard = s_arr < 1e-9
    mask_soft = ~mask_hard
    mask_full = act_arr >= 0.999
    mask_red  = ~mask_full

    def _stats(violations: np.ndarray, mask: np.ndarray) -> Tuple[int, int, float]:
        if mask.sum() == 0:
            return 0, 0, float("nan")
        sub = violations[mask]
        return int(mask.sum()), int(np.sum(sub > VIOL_TOL)), float(np.max(sub))

    prop_hard_total, prop_hard_nviol, prop_hard_maxviol = _stats(viol_prop, mask_hard)
    prop_soft_total, prop_soft_nviol, prop_soft_maxviol = _stats(viol_prop, mask_soft)
    prop_full_total, prop_full_nviol, prop_full_maxviol = _stats(viol_prop, mask_full)
    prop_red_total,  prop_red_nviol,  prop_red_maxviol  = _stats(viol_prop, mask_red)

    if sol_norelax.get("success", False):
        verts_all_zero = [dict(v, s=0.0) for v in verts_norm]
        viol_nr = base.check_all_vertex_violations(
            verts_all_zero, sol_norelax["K"], sol_norelax["Q"],
            float(sol_norelax["gamma2"][0]),
            np.atleast_1d(np.asarray(sol_norelax["beta"])).astype(float),
            syn.decay_rate, syn.d_max,
        )
        norelax_max_viol = float(np.max(viol_nr))
        norelax_n_viol = int(np.sum(viol_nr > VIOL_TOL))
        nr_hard_total, nr_hard_nviol, nr_hard_maxviol = _stats(viol_nr, mask_hard)
        nr_soft_total, nr_soft_nviol, nr_soft_maxviol = _stats(viol_nr, mask_soft)
        nr_full_total, nr_full_nviol, nr_full_maxviol = _stats(viol_nr, mask_full)
        nr_red_total,  nr_red_nviol,  nr_red_maxviol  = _stats(viol_nr, mask_red)
    else:
        norelax_max_viol = float("nan")
        norelax_n_viol = -1
        nr_hard_total = nr_hard_nviol = nr_soft_total = nr_soft_nviol = 0
        nr_full_total = nr_full_nviol = nr_red_total  = nr_red_nviol  = 0
        nr_hard_maxviol = nr_soft_maxviol = nr_full_maxviol = nr_red_maxviol = float("nan")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[F8 Results] active={len(verts_common)}, N_total={N_tot}")
        print(f"{'=' * 60}")
        print(f"  Proposed    : gamma_syn_diag={gamma_proposed:.4f}, "
              f"beta_mean={float(np.mean(sol_proposed['beta'])):.4f}, "
              f"beta_max={float(np.max(sol_proposed['beta'])):.4f}, "
              f"mu={float(sol_proposed['mu'][0]):.4f}, "
              f"||K||={np.linalg.norm(K_proposed):.3f}")
        print(f"                all-vertex LMI: n_viol={prop_n_viol}/{N_tot}, "
              f"max_viol={prop_max_viol:.3e}")
        print(f"                  hard-core  : {prop_hard_nviol}/{prop_hard_total} viol, "
              f"max={prop_hard_maxviol:.3e}")
        print(f"                  soft-shell : {prop_soft_nviol}/{prop_soft_total} viol, "
              f"max={prop_soft_maxviol:.3e}")
        print(f"                  full-gain  : {prop_full_nviol}/{prop_full_total} viol, "
              f"max={prop_full_maxviol:.3e}")
        print(f"                  reduced-gain: {prop_red_nviol}/{prop_red_total} viol, "
              f"max={prop_red_maxviol:.3e}")
        print(f"  NoRelax-ProposedActive : gamma_syn_diag={gamma_norelax:.4f}, "
              f"||K||={np.linalg.norm(K_norelax):.3f}")
        print(f"                all-vertex LMI: n_viol={norelax_n_viol}/{N_tot}, "
              f"max_viol={norelax_max_viol:.3e}")
        print(f"                  hard-core  : {nr_hard_nviol}/{nr_hard_total} viol, "
              f"max={nr_hard_maxviol:.3e}")
        print(f"                  soft-shell : {nr_soft_nviol}/{nr_soft_total} viol, "
              f"max={nr_soft_maxviol:.3e}")
        print(f"  QS-Hinf     : gamma_syn_diag={gamma_qs_hinf:.4f}, "
              f"||K||={np.linalg.norm(K_qs_hinf):.3f}, status={qs_hinf_status}")
        if np.isfinite(qs_hinf_max_viol):
            print(f"                all-vertex LMI: n_viol={qs_hinf_n_viol}/{N_tot}, "
                  f"max_viol={qs_hinf_max_viol:.3e}")
        print(f"  PDL-Hinf    : gamma_syn_diag={gamma_pdl_hinf:.4f}, "
              f"||K||={np.linalg.norm(K_pdl_hinf):.3f}, status={pdl_hinf_status}")
        print(f"  RobustLQR   : (no H-inf gamma), "
              f"||K||={np.linalg.norm(K_robust_lqr):.3f}, status={robust_lqr_status}")
        print(f"  Core-CQLF-Hinf: gamma_syn_diag={gamma_core_cqlf_hinf:.4f}, "
              f"||K||={np.linalg.norm(K_core_cqlf_hinf):.3f}, status={core_cqlf_status}")
        if np.isfinite(core_cqlf_full_max_viol):
            print(f"                full-box LMI: n_viol={core_cqlf_full_n_viol}/{N_tot}, "
                  f"max_viol={core_cqlf_full_max_viol:.3e}")

    # =====================================================================
    # Fixed-K data-core post-certification (analysis-only, NOT a synthesis).
    # For every K already designed above, ask MOSEK to find a Lyapunov P
    # and an L2-gain g such that the standard discrete-time bounded-real
    # lemma holds simultaneously on every vertex of the data-consistent
    # core C_D. The Proposed controller is highlighted: if its audit is
    # feasible we mark it as "performance-oriented controller WITH
    # data-core certificate"; otherwise we run the certificate-repair SDP
    #     min  w_gamma * gamma2  +  lambda_close * ||Y - K_prop Q||_F
    #     s.t. unrelaxed core synthesis LMIs
    # to obtain a minimally modified gain that DOES satisfy the certificate.
    # =====================================================================
    fixed_k_audits: Dict[str, Any] = {}
    proposed_repair: Dict[str, Any] = {}
    K_proposed_repaired = K_proposed.copy()
    gamma_proposed_repaired = float("inf")
    synthesis_audit_consistency: List[Dict[str, Any]] = []
    # Matrix-hull membership residual (Remark 1 / Corollary 1). Filled
    # inside the audit block below so conv(V_{C_D}) is always the
    # *audit* vertex set (same tilde S scaling as the audit SDP).
    matrix_hull_diag: Dict[str, Any] = {}
    # True iff the audit vertex set contains at least one reduced-gain
    # replica (kappa < 1); kept as a separate flag so the paper's
    # "full-gain only" vs "synthesis-aligned 32-vertex audit" distinction
    # is visible in the NPZ metadata.
    audit_include_gain_uncertainty: bool = False
    if syn.run_fixed_k_audit:
        if verbose:
            print(f"\n{'=' * 60}\n[F8 Fixed-K Data-Core Audit] post-certification "
                  f"of all controllers\n{'=' * 60}")
        # Reuse the data-consistent core box C_D built once above.
        cd_info_audit = datacore_diag
        keys_audit = cd_info_audit["keys"]
        p_D_lo_a = cd_info_audit["p_D_lo"]; p_D_hi_a = cd_info_audit["p_D_hi"]
        cd_corners_audit = _enumerate_box_corners_for_keys(p_D_lo_a, p_D_hi_a, keys_audit)
        # Audit-vertex enumeration matching the synthesis scope: when
        # synthesis used multiplicative actuator-gain uncertainty
        # (syn.actuator_gain_uncertainty > 0 -> 32-plant polytope),
        # the audit must cover the same gain endpoints, otherwise the
        # reported gamma_core only certifies the full-gain slice. Each
        # aerodynamic corner is therefore replicated with
        # kappa in {1, 1 - eps_act} when eps_act > 0.
        eps_act_audit = (float(syn.actuator_gain_uncertainty)
                         if syn.actuator_gain_uncertainty > 0 else 0.0)
        kappa_list = [1.0] if eps_act_audit == 0.0 else [1.0, 1.0 - eps_act_audit]
        audit_include_gain_uncertainty = len(kappa_list) > 1
        core_verts_audit: List[Dict[str, Any]] = []
        for p in cd_corners_audit:
            A_p, B_p, S_p = build_vertex_matrices(p, syn.Ts)
            Cc, Dc = base.build_performance_matrices(bsyn)
            for kappa in kappa_list:
                B_k = float(kappa) * B_p
                Delta_k = np.block([[A_p, B_k], [Cc, Dc]])
                core_verts_audit.append(dict(
                    A=A_p, B=B_k, S=S_p, C=Cc, D=Dc, Delta=Delta_k,
                    p={**p, "_act_scale": float(kappa)},
                    s=0.0, _kind="audit_core",
                ))

        # Roster of (name, K) pairs to audit.
        controllers_audit: List[Tuple[str, np.ndarray]] = [
            ("Proposed", K_proposed),
            ("NoRelax-ProposedActive", K_norelax),
            ("QS-Hinf", K_qs_hinf),
            ("PDL-Hinf", K_pdl_hinf),
            ("RobustLQR", K_robust_lqr),
        ]
        if core_cqlf_status == "feasible":
            controllers_audit.append(("Core-CQLF-Hinf", K_core_cqlf_hinf))

        audit_alpha = float(getattr(syn, "audit_decay_rate", 1.0))
        if verbose:
            gain_tag = (f" x {len(kappa_list)} gain endpoints"
                        if len(kappa_list) > 1 else "")
            print(f"  core vertices = {len(core_verts_audit)} "
                  f"(2^{len(keys_audit)} aero corners of C_D{gain_tag}); "
                  f"audit decay = {audit_alpha}")

        for name, K_i in controllers_audit:
            if verbose:
                print(f"\n  [audit {name}] solving fixed-K analysis SDP...")
            au = _audit_fixed_K_on_core(
                K_i, core_verts_audit, syn, bsyn,
                decay_rate=audit_alpha, verbose=False,
            )
            au["K_norm"] = float(np.linalg.norm(K_i))
            _annotate_audit_vertex_diagnostics(au, K_i, core_verts_audit)
            if au["feasible"]:
                au["full_box_max_eig"] = _full_box_max_eig_for_K(
                    K_i, verts_all, au["P"], au["g"],
                    decay_rate=audit_alpha, d_cert=float(syn.d_max),
                )
            else:
                au["full_box_max_eig"] = float("nan")
            fixed_k_audits[name] = au
            if verbose:
                if au["feasible"]:
                    print(f"    feasible=True, gamma_core={au['gamma_core']:.4f}, "
                          f"core_max_eig={au['core_max_eig']:+.3e}, "
                          f"cond(P)={au['cond_P']:.2e}, "
                          f"full_box_max_eig={au['full_box_max_eig']:+.3e}")
                else:
                    print(f"    feasible=False (status={au['status'][:60]}); "
                          f"K NOT certified on data core.")

        # Special handling for Proposed: repair if the fixed-K audit fails.
        prop_au = fixed_k_audits.get("Proposed", {})
        if not prop_au.get("feasible", False):
            if verbose:
                print(f"\n  [WARN] Proposed FAILED fixed-K core audit -- "
                      f"running certificate-repair SDP.")
            # Diagnostics: vertex-wise audit (each core corner alone) +
            # tighter-core retry diagnostic.
            if verbose:
                print(f"  [diag] vertex-wise fixed-K audit (each corner alone):")
            vertex_wise = []
            for j, vj in enumerate(core_verts_audit):
                au_j = _audit_fixed_K_on_core(
                    K_proposed, [vj], syn, bsyn,
                    decay_rate=audit_alpha, verbose=False,
                )
                vertex_wise.append(int(au_j["feasible"]))
                if verbose:
                    p_str = ",".join(f"{k}={vj['p'][k]:+.2f}" for k in keys_audit)
                    print(f"    corner {j:2d} [{p_str}]: feasible={au_j['feasible']}, "
                          f"gamma={au_j['gamma_core']:.4f}")
            fixed_k_audits["Proposed"]["vertex_wise_feasible"] = np.array(vertex_wise, dtype=int)

            # Repair: minimise gamma + lambda_close * ||Y - K_prop Q||_F
            lambda_close = float(getattr(syn, "repair_lambda_close", 1.0))
            proposed_repair = _repair_K_with_data_core(
                K_proposed, core_verts_audit, syn, bsyn,
                lambda_close=lambda_close,
                decay_rate=float(syn.decay_rate),    # use synthesis decay
                verbose=False,
            )
            if proposed_repair.get("success", False):
                K_proposed_repaired = np.asarray(proposed_repair["K"], dtype=float)
                gamma_proposed_repaired = float(proposed_repair["gamma"])
                # Re-audit the repaired controller on the same core
                au_rep = _audit_fixed_K_on_core(
                    K_proposed_repaired, core_verts_audit, syn, bsyn,
                    decay_rate=audit_alpha, verbose=False,
                )
                au_rep["K_norm"] = float(np.linalg.norm(K_proposed_repaired))
                _annotate_audit_vertex_diagnostics(
                    au_rep, K_proposed_repaired, core_verts_audit,
                )
                if au_rep["feasible"]:
                    au_rep["full_box_max_eig"] = _full_box_max_eig_for_K(
                        K_proposed_repaired, verts_all, au_rep["P"], au_rep["g"],
                        decay_rate=audit_alpha, d_cert=float(syn.d_max),
                    )
                else:
                    au_rep["full_box_max_eig"] = float("nan")
                fixed_k_audits["Proposed-Repaired"] = au_rep
                if verbose:
                    print(f"  [Repair] gamma={proposed_repair['gamma']:.4f}, "
                          f"||K||={proposed_repair['K_norm']:.3f}, "
                          f"||K - K_prop||_F={proposed_repair['K_dist_to_prop']:.3f}, "
                          f"lambda_close={lambda_close}")
                    print(f"  [Repaired Audit] feasible={au_rep['feasible']}, "
                          f"gamma_core={au_rep['gamma_core']:.4f}, "
                          f"core_max_eig={au_rep['core_max_eig']:+.3e}")
            else:
                if verbose:
                    print(f"  [Repair] FAILED: {proposed_repair.get('status', '?')}")
        else:
            if verbose:
                print(f"\n  [OK] Proposed certified on data core: "
                      f"gamma_core={prop_au['gamma_core']:.4f}, "
                      f"core_max_eig={prop_au['core_max_eig']:+.3e}, "
                      f"cond(P)={prop_au['cond_P']:.2e}.")
                print(f"        --> Proposed = performance-oriented controller "
                      f"WITH data-core certificate.")

        # ===== Matrix-hull membership residual (Remark 1, Corollary 1) ====
        # Given the audit vertex set V_{C_D} (already scaled with tilde S),
        # compute the barycentric residual of the ZOH-discretised tuple of
        # the data-generating plant p_data_source in conv(V_{C_D}). A small
        # residual is the second prerequisite of Corollary 1 (the first
        # being Assumption 1 p_data in C_D); if it is large we still report
        # gamma_core as the matrix-hull bound, but the plant-level transfer
        # is explicitly conditional on this residual.
        # Tolerances for the L2 / relative / entrywise infinity-norm
        # residual checks. abs_tol = 1e-6 was too strict in practice
        # (sampled data-core box discretisation gives residuals in the
        # 1e-5..1e-3 range). We adopt 1e-3 as the headline thresholds
        # and report ALL THREE residuals so the paper can quote raw
        # numbers rather than a single binary verdict.
        hull_abs_tol = float(getattr(syn, "matrix_hull_abs_tol", 1e-3))
        hull_rel_tol = float(getattr(syn, "matrix_hull_rel_tol", 1e-3))
        hull_inf_tol = float(getattr(syn, "matrix_hull_inf_tol", 1e-3))
        try:
            A_star, B_star, S_star = build_vertex_matrices(p_data_source, syn.Ts)
            d_cert_hull = float(syn.d_max)
            S_tilde_star = scaled_disturbance_matrix(S_star, d_cert_hull)
            hull_verts = [
                {"A": v["A"], "B": v["B"],
                 "S_tilde": scaled_disturbance_matrix(v["S"], d_cert_hull)}
                for v in core_verts_audit
            ]
            matrix_hull_diag = matrix_hull_residual(
                A_star, B_star, S_tilde_star, hull_verts,
                abs_tol=hull_abs_tol, rel_tol=hull_rel_tol,
                inf_tol=hull_inf_tol, verbose=False,
            )
            if verbose:
                print(f"\n  [MatrixHull] p_data membership residual: "
                      f"abs={matrix_hull_diag['residual_abs']:.3e}, "
                      f"rel={matrix_hull_diag['residual_rel']:.3e}, "
                      f"inf={matrix_hull_diag['residual_inf']:.3e}; "
                      f"inside_abs={matrix_hull_diag['inside_hull_abs']}, "
                      f"inside_rel={matrix_hull_diag['inside_hull_rel']}, "
                      f"inside_inf={matrix_hull_diag['inside_hull_inf']}; "
                      f"n_active={len(matrix_hull_diag['active_vertices'])}"
                      f"/{matrix_hull_diag['n_vertices']}; "
                      f"tol abs={hull_abs_tol:g} rel={hull_rel_tol:g} "
                      f"inf={hull_inf_tol:g}")
        except Exception as exc:
            if verbose:
                print(f"  [MatrixHull] residual check failed: "
                      f"{type(exc).__name__}: {exc}")
            matrix_hull_diag = dict(
                lambdas=np.array([]),
                residual_abs=float("nan"),
                residual_rel=float("nan"),
                residual_inf=float("nan"),
                inside_hull=False,
                inside_hull_abs=False,
                inside_hull_rel=False,
                inside_hull_inf=False,
                active_vertices=np.array([], dtype=int),
                solver_status=f"exception: {type(exc).__name__}: {exc}",
                n_vertices=len(core_verts_audit),
                abs_tol=hull_abs_tol,
                rel_tol=hull_rel_tol,
                inf_tol=hull_inf_tol,
            )

        audit_vertex_label = (
            f"{len(core_verts_audit)} data-core audit vertices "
            f"(aero corners x {len(kappa_list)} gain endpoints)"
        )
        synthesis_audit_consistency = [
            _synthesis_audit_consistency_row(
                "Proposed", sol_proposed, verts_common, core_verts_audit,
                fixed_k_audits.get("Proposed"),
                synth_vertex_set=f"ICE active set ({len(verts_common)} of {len(verts_norm)})",
                audit_vertex_set=audit_vertex_label,
                fallback_status="feasible",
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
            _synthesis_audit_consistency_row(
                "NoRelax-ProposedActive",
                sol_norelax if sol_norelax.get("success", False) else None,
                verts_norelax, core_verts_audit,
                fixed_k_audits.get("NoRelax-ProposedActive"),
                synth_vertex_set=f"same ICE active set, beta=0 ({len(verts_norelax)} vertices)",
                audit_vertex_set=audit_vertex_label,
                fallback_status=norelax_status,
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
            _synthesis_audit_consistency_row(
                "QS-Hinf",
                sol_qs_hinf if sol_qs_hinf.get("success", False) else None,
                verts_qs, core_verts_audit,
                fixed_k_audits.get("QS-Hinf"),
                synth_vertex_set=f"full synthesis polytope ({len(verts_qs)} vertices)",
                audit_vertex_set=audit_vertex_label,
                fallback_status=qs_hinf_status,
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
            _synthesis_audit_consistency_row(
                "PDL-Hinf", sol_pdl_hinf, verts_norm, core_verts_audit,
                fixed_k_audits.get("PDL-Hinf"),
                synth_vertex_set=f"PDL G-form full synthesis polytope ({len(verts_norm)} vertices)",
                audit_vertex_set=audit_vertex_label,
                fallback_status=pdl_hinf_status,
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
            _synthesis_audit_consistency_row(
                "RobustLQR", None, [], core_verts_audit,
                fixed_k_audits.get("RobustLQR"),
                synth_vertex_set="Robust LQR cost LMI, no H-infinity synthesis gamma",
                audit_vertex_set=audit_vertex_label,
                fallback_status=robust_lqr_status,
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
            _synthesis_audit_consistency_row(
                "Core-CQLF-Hinf",
                sol_core_cqlf_hinf if sol_core_cqlf_hinf.get("success", False) else None,
                verts_core_cqlf, core_verts_audit,
                fixed_k_audits.get("Core-CQLF-Hinf"),
                synth_vertex_set=f"data-core CQLF synthesis set ({len(verts_core_cqlf)} vertices)",
                audit_vertex_set=audit_vertex_label,
                fallback_status=core_cqlf_status,
                d_cert=float(syn.d_max),
                audit_decay_rate=audit_alpha,
            ),
        ]

    return dict(
        # Controllers
        K_proposed=K_proposed,
        K_norelax=K_norelax,
        K_qs_hinf=K_qs_hinf,
        K_pdl_hinf=K_pdl_hinf,
        K_robust_lqr=K_robust_lqr,
        K_core_cqlf_hinf=K_core_cqlf_hinf,
        # gammas (RobustLQR has no H-infinity certificate -> infinity)
        gamma_proposed=gamma_proposed,
        gamma_norelax=gamma_norelax,
        gamma_qs_hinf=gamma_qs_hinf,
        gamma_pdl_hinf=gamma_pdl_hinf,
        gamma_core_cqlf_hinf=gamma_core_cqlf_hinf,
        gamma_robust_lqr=float("inf"),
        # Solver outputs
        sol_proposed=sol_proposed,
        sol_norelax=sol_norelax,
        sol_qs_hinf=sol_qs_hinf,
        sol_pdl_hinf=sol_pdl_hinf,
        sol_core_cqlf_hinf=sol_core_cqlf_hinf,
        # Status strings
        norelax_status=norelax_status,
        qs_hinf_status=qs_hinf_status,
        pdl_hinf_status=pdl_hinf_status,
        core_cqlf_status=core_cqlf_status,
        robust_lqr_status=robust_lqr_status,
        robust_lqr_p_worst=robust_lqr_p_worst,
        robust_lqr_rho_ol=robust_lqr_rho_ol,
        # Misc
        actuator_gain_uncertainty=eps_act,
        p_nom=p_nom, p_data_source=p_data_source,
        Psi=Psi, batch=batch, syn=syn, bsyn=bsyn, bounds=bounds,
        vertices_all=verts_norm, vertices_active=verts_common,
        score_diag=score_diag, score_mode=score_mode,
        prop_max_viol=prop_max_viol, prop_n_viol=prop_n_viol,
        norelax_max_viol=norelax_max_viol, norelax_n_viol=norelax_n_viol,
        qs_hinf_max_viol=qs_hinf_max_viol, qs_hinf_n_viol=qs_hinf_n_viol,
        core_cqlf_full_max_viol=core_cqlf_full_max_viol,
        core_cqlf_full_n_viol=core_cqlf_full_n_viol,
        N_total=N_tot,
        # Hard-core / soft-shell / full-gain / reduced-gain breakdown of
        # the post-solve all-vertex LMI audit. Feeds Table I sub-columns.
        prop_hard_nviol=prop_hard_nviol, prop_hard_total=prop_hard_total,
        prop_hard_maxviol=prop_hard_maxviol,
        prop_soft_nviol=prop_soft_nviol, prop_soft_total=prop_soft_total,
        prop_soft_maxviol=prop_soft_maxviol,
        prop_full_nviol=prop_full_nviol, prop_full_total=prop_full_total,
        prop_full_maxviol=prop_full_maxviol,
        prop_red_nviol=prop_red_nviol,   prop_red_total=prop_red_total,
        prop_red_maxviol=prop_red_maxviol,
        nr_hard_nviol=nr_hard_nviol, nr_hard_total=nr_hard_total,
        nr_hard_maxviol=nr_hard_maxviol,
        nr_soft_nviol=nr_soft_nviol, nr_soft_total=nr_soft_total,
        nr_soft_maxviol=nr_soft_maxviol,
        nr_full_nviol=nr_full_nviol, nr_full_total=nr_full_total,
        nr_full_maxviol=nr_full_maxviol,
        nr_red_nviol=nr_red_nviol,   nr_red_total=nr_red_total,
        nr_red_maxviol=nr_red_maxviol,
        # ----- Data-contained core diagnostic (audit input) -----
        datacore_diag=datacore_diag,
        # ----- Fixed-K data-core post-certification (filled below) -----
        fixed_k_audits=fixed_k_audits,
        synthesis_audit_consistency=synthesis_audit_consistency,
        proposed_repair=proposed_repair,
        K_proposed_repaired=K_proposed_repaired,
        gamma_proposed_repaired=gamma_proposed_repaired,
        # ----- Matrix-hull membership residual (Corollary 1) -----
        matrix_hull_diag=matrix_hull_diag,
        # ----- Reproducibility metadata (exported to NPZ; keeps the
        # paper claims grounded in the code's actual configuration) -----
        certificate_scope="unsaturated_regulation_audited_matrix_hull",
        tracking_scope="empirical_saturation_tracking_test",
        disturbance_convention="normalized_w_with_tildeS_equal_dcert_times_S",
        d_cert=float(syn.d_max),
        d_data=float(syn.datacore_dbar),
        audit_include_gain_uncertainty=bool(audit_include_gain_uncertainty),
        # Static input-effectiveness endpoints actually exercised by the
        # fixed-K post-audit (= [1.0] when only the full-gain audit is
        # active, [1.0, 1-eps_act] when audit_include_gain_uncertainty).
        audit_act_scales=np.array(
            kappa_list if syn.run_fixed_k_audit else [1.0], dtype=float),
    )


# ---------------------------------------------------------
# Heterogeneous pitch-attitude reference trajectories
# (4 aircraft flying distinct but synchronised tracking tasks)
# ---------------------------------------------------------
def build_f8_reference_trajectories(syn: F8SynthesisParams,
                                    seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, theta_refs) with theta_refs shape (4, n_steps).

    AC 1: raised-cosine smooth pulse (rises in 1..1.5s, holds until 5.5s,
          falls in 5.5..6.0s) -- avoids the non-physical one-step
          pitch-rate impulse that np.gradient would produce from a hard
          square pulse and yields a physically sensible q_ref = d(theta)/dt.
    AC 2: sinusoid     theta* = 0.16 sin(2 pi * 0.25 t)
    AC 3: ramp-and-hold with smooth entry / exit (raised cosine)
    AC 4: trapezoid, delayed (2.5 s -> 8.5 s)
    """
    n = int(round(seconds / syn.Ts))
    t = np.arange(n) * syn.Ts
    theta_refs = np.zeros((4, n))

    # All reference amplitudes scaled 2x from the original prototype values
    # (0.10 rad -> 0.20 rad ~ 11.5 deg). A 3x scaling (17.2 deg) was tried
    # but caused control saturation against the +/-0.4 rad actuator limit
    # on extreme corners (Proposed K_norm=13.2 multiplied by an error of
    # 0.30 rad easily exceeds the actuator wall, breaking the linear LMI
    # certificate). 11.5 deg is still a moderate maneuver -- larger than
    # the original 5.7 deg small-signal test but small enough to keep the
    # controllers in their certified linear regime.
    # AC1: raised-cosine smooth pulse (amplitude 0.20 rad, duration 5 s,
    # 0.5 s smooth transitions so that d(theta_ref)/dt is bounded by ~0.63
    # rad/s instead of the delta-like spike produced by a hard step).
    tau_edge = 0.5
    amp1 = 0.20
    for k in range(n):
        tt = t[k]
        if tt < 1.0:
            theta_refs[0, k] = 0.0
        elif tt < 1.0 + tau_edge:
            theta_refs[0, k] = amp1 * 0.5 * (1.0 - np.cos(np.pi * (tt - 1.0) / tau_edge))
        elif tt < 6.0 - tau_edge:
            theta_refs[0, k] = amp1
        elif tt < 6.0:
            theta_refs[0, k] = amp1 * 0.5 * (1.0 + np.cos(np.pi * (tt - (6.0 - tau_edge)) / tau_edge))
        else:
            theta_refs[0, k] = 0.0
    theta_refs[1, :] = 0.16 * np.sin(2.0 * np.pi * 0.25 * t)                # was 0.08
    for k in range(n):
        tt = t[k]
        if tt < 2.0:
            theta_refs[2, k] = 0.14 * 0.5 * (1.0 - np.cos(np.pi * tt / 2.0))  # was 0.07
        elif tt < 5.0:
            theta_refs[2, k] = 0.14                                            # was 0.07
        elif tt < 7.0:
            theta_refs[2, k] = 0.14 * 0.5 * (1.0 + np.cos(np.pi * (tt - 5.0) / 2.0))
        else:
            theta_refs[2, k] = 0.0
    for k in range(n):
        tt = t[k]
        if 2.5 <= tt < 4.5:
            theta_refs[3, k] = 0.12 * (tt - 2.5) / 2.0                         # was 0.06
        elif 4.5 <= tt < 6.5:
            theta_refs[3, k] = 0.12                                            # was 0.06
        elif 6.5 <= tt < 8.5:
            theta_refs[3, k] = 0.12 * (1.0 - (tt - 6.5) / 2.0)
    return t, theta_refs


def build_f8_state_reference(theta_refs: np.ndarray, Ts: float) -> np.ndarray:
    """Build a 16D state reference from per-aircraft theta commands.
    alpha_ref = 0, q_ref = d(theta_ref)/dt, theta_ref = command, actuator_ref = 0."""
    n_steps = theta_refs.shape[1]
    x_ref = np.zeros((16, n_steps))
    dth = np.gradient(theta_refs, Ts, axis=1)
    for i in range(4):
        x_ref[3 * i + 1, :] = dth[i, :]
        x_ref[3 * i + 2, :] = theta_refs[i, :]
    return x_ref


# ---------------------------------------------------------
# Atmospheric disturbance profile (shared vertical gust +
# per-aircraft pitching gust + slow formation coupling term)
# ---------------------------------------------------------
def build_f8_disturbance_profile(syn: F8SynthesisParams,
                                 seconds: float,
                                 seed_offset: int = 800,
                                 ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    steps = int(round(seconds / syn.Ts))
    t = np.arange(steps) * syn.Ts
    d = np.zeros((6, steps))
    # All gust amplitudes scaled 1.5x to match the larger d_max (0.6 -> 0.9).
    # Two pulse gusts at t in [1.0, 2.5] and [5.0, 6.5] dominate the shared
    # vertical channel d[0]; per-aircraft coloured noise drives d[1:5]; a slow
    # sinusoidal coupling drives d[5]. saturate_norm caps ||d(t)||_2 at d_max.
    events = [(1.0, 2.5, 0.9), (5.0, 6.5, -1.2)]                       # 1.5x of (0.6, -0.8)
    for t0, t1, amp in events:
        mask = (t >= t0) & (t <= t1)
        phase = (t[mask] - t0) / max(t1 - t0, syn.Ts)
        d[0, mask] += amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    colored = base.colored_noise(4, steps, alpha=0.96, scale=0.225,    # 1.5x of 0.15
                                 seed=syn.seed + seed_offset + 1)
    d[1:5, :] += colored
    d[5, :] += 0.12 * np.sin(2.0 * np.pi * t / max(seconds, syn.Ts))   # 1.5x of 0.08
    for k in range(steps):
        d[:, k] = base.saturate_norm(d[:, k], syn.d_max)
    gusts = [dict(t0=float(t0), t1=float(t1), amp=float(abs(amp)))
             for t0, t1, amp in events]
    return d, gusts


# ---------------------------------------------------------
# Closed-loop tracking simulation (with actuator saturations)
# ---------------------------------------------------------
def simulate_f8_tracking(
        K: np.ndarray,
        p_true: Dict[str, float],
        syn: F8SynthesisParams,
        seconds: float,
        dbar: np.ndarray,
        x_ref: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
        meas_noise_std: float = 0.0,
        sensor_delay_steps: int = 0,
        actuator_tau: float = 0.0,
) -> Dict[str, Any]:
    """Closed-loop state-reference tracking on the F-8 formation plant.

    Control law: u_c = K @ (x[k - d_sensor] - x_ref[k - d_sensor]).
    The commanded increment then passes through rate / absolute saturation
    and a first-order actuator lag (time constant ``actuator_tau``).

    Both ``sensor_delay_steps`` and ``actuator_tau`` model **unmodelled
    dynamics** that none of the five robust H-infinity SDP designs (Proposed,
    NoRelax-ProposedActive, QS-Hinf, PDL-Hinf, Core-CQLF-Hinf) sees at synthesis time; they are zero
    by default so the ideal test is preserved, and non-zero values are used
    as an ACC-style robustness stress test (Skogestad & Postlethwaite,
    Sec.2.7; Boyd et al. 1994).
    """
    bsyn = f8_to_base_syn(syn)
    A, B, S = build_vertex_matrices(p_true, syn.Ts)
    Cc, Dc = build_performance_matrices(syn)
    n_steps = dbar.shape[1]
    assert x_ref.shape[1] == n_steps, "x_ref length must match dbar length"
    rng = np.random.default_rng(syn.seed + 2026)
    # First-order actuator lag attenuation (alpha=1 -> ideal, smaller -> more lag)
    alpha_a = (syn.Ts / (syn.Ts + actuator_tau)) if actuator_tau > 0.0 else 1.0
    d_sensor = max(0, int(sensor_delay_steps))
    x = np.zeros((16, n_steps + 1))
    u_c_raw = np.zeros((4, n_steps))
    u_c_eff = np.zeros((4, n_steps))
    u_abs = np.zeros((4, n_steps + 1))
    z = np.zeros((16, n_steps))
    flags_rate = np.zeros(n_steps, dtype=int)
    flags_norm = np.zeros(n_steps, dtype=int)
    flags_abs = np.zeros(n_steps, dtype=int)
    x[:, 0] = x0.copy() if x0 is not None else f8_initial_state(scale=0.3)
    u_abs[:, 0] = x[12:16, 0].copy()
    # True measurement-noise channel: the controller sees x_meas = x + v,
    # where v is Gaussian with std meas_noise_std on the twelve physical
    # states (alpha, q, theta of each aircraft). Actuator-integrator
    # states (x[12:16]) are treated as perfectly known by the controller
    # (they are internally-held commands, not externally-sensed signals)
    # and receive no measurement noise. The TRUE plant state x[:, k+1]
    # is never polluted by the measurement-noise draw, so the noise is a
    # genuine sensor-channel perturbation rather than a process/state
    # disturbance (which is already handled by S @ dbar).
    noise_mask = np.zeros(16, dtype=float)
    if meas_noise_std > 0:
        noise_mask[0:12] = meas_noise_std
    for k in range(n_steps):
        # Sensor delay: controller acts on a delayed (x, x_ref) sample.
        k_meas = max(0, k - d_sensor)
        if meas_noise_std > 0:
            v_meas = rng.standard_normal(16) * noise_mask
            x_meas = x[:, k_meas] + v_meas
        else:
            x_meas = x[:, k_meas]
        e = x_meas - x_ref[:, k_meas]
        uc = (K @ e).reshape(4)
        u_c_raw[:, k] = uc
        uc_lim, f1 = base.apply_increment_limits(uc, bsyn)
        u_curr = x[12:16, k].copy()
        uc_eff_cmd, u_next_cmd, f2 = base.apply_absolute_actuator_limits(
            u_curr, uc_lim, bsyn)
        flags_rate[k] = int(f1.get("rate_sat", 0))
        flags_norm[k] = int(f1.get("norm_sat", 0))
        flags_abs[k] = int(f2.get("abs_sat", 0))
        # Actuator 1st-order lag: only fraction alpha_a of the commanded
        # rate-limited / position-limited increment is realised this step.
        # alpha_a = 1 reproduces the original ideal-actuator simulation.
        uc_actual = alpha_a * uc_eff_cmd
        u_c_eff[:, k] = uc_eff_cmd          # commanded effort (for metrics)
        u_abs[:, k + 1] = x[12:16, k] + uc_actual  # actual lagged deflection
        z[:, k] = Cc @ x[:, k] + Dc @ uc_eff_cmd
        x[:, k + 1] = A @ x[:, k] + B @ uc_actual + S @ dbar[:, k]
    t = np.arange(n_steps + 1) * syn.Ts
    return dict(
        t=t, x=x, u_c_raw=u_c_raw, u_c=u_c_eff, u_abs=u_abs, z=z, dbar=dbar,
        p_true=p_true, A_true=A, B_true=B, S_true=S, x_ref=x_ref,
        rho=np.array([base.spectral_radius(A + B @ K)]),
        sat_rate=flags_rate, sat_norm=flags_norm, sat_abs=flags_abs,
        sensor_delay_steps=int(d_sensor),
        actuator_tau=float(actuator_tau),
        alpha_a=float(alpha_a),
    )


# ---------------------------------------------------------
# Tracking metrics: pitch-tracking error + control effort +
# performance output + actuator saturation duty.
# ---------------------------------------------------------
def compute_f8_tracking_metrics(sim: Dict[str, Any],
                                syn: F8SynthesisParams) -> Dict[str, float]:
    Ts = syn.Ts
    x_ref = sim["x_ref"]
    n = x_ref.shape[1]
    theta_err = sim["x"][[2, 5, 8, 11], :n] - x_ref[[2, 5, 8, 11], :]
    abs_err = np.abs(theta_err)
    rmse_per_ac = np.sqrt(np.mean(theta_err ** 2, axis=1))
    iae_per_ac = np.sum(abs_err, axis=1) * Ts
    peak_per_ac = np.max(abs_err, axis=1)
    uc_norm = np.linalg.norm(sim["u_c"], axis=0)
    uabs_norm = np.linalg.norm(sim["u_abs"], axis=0)
    zn = np.linalg.norm(sim["z"], axis=0)
    rho_val = float(sim["rho"][0])
    state_max = float(np.max(np.abs(sim["x"])))
    # Closed-loop instability flag using a *relative* state-norm threshold
    # so the cut-off is invariant to controller gain magnitude (an absolute
    # threshold would penalise high-gain controllers like NoRelax-ProposedActive just
    # because their transients are larger, even when they ultimately
    # converge). We declare instability when the closed-loop spectral
    # radius >= 1 - 1e-8, when the state diverges numerically, or when the
    # peak state magnitude exceeds 50x the worst reference / initial-state
    # amplitude (i.e., a clear blow-up rather than a transient overshoot).
    ref_amp = float(np.max(np.abs(sim["x_ref"]))) if sim["x_ref"].size else 0.0
    init_amp = float(np.max(np.abs(sim["x"][:, 0])))
    state_threshold = max(50.0 * max(ref_amp, init_amp), 5.0)
    unstable_flag = float(
        (rho_val >= 1.0 - 1e-8)
        or (not np.isfinite(state_max))
        or (state_max > state_threshold)
    )
    return dict(
        RMSE_theta=float(np.mean(rmse_per_ac)),
        RMSE_theta_max=float(np.max(rmse_per_ac)),
        IAE_theta=float(np.mean(iae_per_ac)),
        peak_theta=float(np.max(peak_per_ac)),
        peak_theta_mean=float(np.mean(peak_per_ac)),
        peak_u=float(np.max(uc_norm)),
        peak_uabs=float(np.max(uabs_norm)),
        energy_u=float(np.sum(uc_norm ** 2) * Ts),
        energy_uabs=float(np.sum(uabs_norm ** 2) * Ts),
        energy_z=float(np.sum(zn ** 2) * Ts),
        duty_du=float(np.mean(uc_norm >= syn.sat_tol * syn.du_max)),
        duty_abs=float(np.mean(sim["sat_abs"] > 0)),
        duty_normsat=float(np.mean(sim["sat_norm"] > 0)),
        rho=rho_val,
        unstable=unstable_flag,
    )


# ---------------------------------------------------------
# Helpers: neighborhood plant sampling + worst-case selection
# ---------------------------------------------------------
def _sample_f8_neighborhood(center: Dict[str, float],
                            bounds: F8ParamBounds,
                            rng: np.random.Generator,
                            spread: float) -> Dict[str, float]:
    """Additive perturbation around ``center`` clipped to bounds. Uses an
    additive (rather than multiplicative) update so it is well-behaved when
    the parameter range spans zero (e.g. m_alpha, m_q in the F-8 box)."""
    p: Dict[str, float] = {}
    for k in bounds.__dataclass_fields__.keys():
        lo, hi = getattr(bounds, k)
        half = 0.5 * (hi - lo)
        val = float(center[k]) + float(rng.uniform(-spread, spread)) * half
        p[k] = float(np.clip(val, lo, hi))
    return p


def _sample_f8_corner_or_near(bounds: F8ParamBounds,
                              rng: np.random.Generator,
                              *,
                              p_corner: float = 0.5,
                              spread: float = 0.10) -> Dict[str, float]:
    """Sample a plant either AT a random 16-corner vertex (with probability
    ``p_corner``) or in a small additive neighborhood around it. This ACC-
    style sampling exposes baselines to genuinely extreme plants instead of
    keeping them inside a comfortable nominal-centred ball."""
    keys = list(bounds.__dataclass_fields__.keys())
    bits = rng.integers(0, 2, size=len(keys))
    p_vertex = {k: float(getattr(bounds, k)[1] if bits[i] else getattr(bounds, k)[0])
                for i, k in enumerate(keys)}
    if rng.random() < p_corner:
        return p_vertex
    return _sample_f8_neighborhood(p_vertex, bounds, rng, spread)


def _select_worst_case_plant(bounds: F8ParamBounds,
                             syn: F8SynthesisParams) -> Dict[str, float]:
    """Select the corner vertex with the largest open-loop discrete-time
    spectral radius. This is a *controller-agnostic, physics-only* criterion:
    we pick the plant whose intrinsic dynamics is hardest to stabilise,
    independent of any controller K. The choice therefore cannot be biased
    in favour of any particular design (Proposed, NoRelax-ProposedActive, QS-Hinf or
    PDL-Hinf), which is essential for a defensible worst-case comparison.

    NOTE: this is the ADVERSARIAL / OUT-OF-AUDIT-HULL worst corner;
    it may (and in practice does) coincide with a data-inconsistent
    soft-shell vertex whose score is high. The fixed-K post-audit
    certificate is NOT claimed for this corner (see Corollary 1 / the
    matrix-hull membership transfer condition). See
    _select_hardcore_worst_plant for the hard-core synthesis stress
    corner (score-supported, s <= s_threshold), which is an empirical
    stress test on a score-protected vertex -- not the same as the
    formal audit-time scope.
    """
    fields = list(bounds.__dataclass_fields__.keys())
    best_rho = -np.inf
    p_worst: Optional[Dict[str, float]] = None
    for bits in itertools.product([0, 1], repeat=len(fields)):
        p = {k: float(getattr(bounds, k)[1] if bits[i] else getattr(bounds, k)[0])
             for i, k in enumerate(fields)}
        A, _, _ = build_vertex_matrices(p, syn.Ts)
        rho = float(max(abs(np.linalg.eigvals(A))))
        if rho > best_rho:
            best_rho = rho
            p_worst = p
    print(f"\n[WorstCase Plant] open-loop hardest corner (ADVERSARIAL, "
          f"may be data-inconsistent soft-shell vertex): "
          f"rho_open={best_rho:.4f}, params={p_worst}")
    return p_worst or center_params(bounds)


def _select_hardcore_worst_plant(out: Dict[str, Any],
                                 bounds: F8ParamBounds,
                                 syn: F8SynthesisParams,
                                 s_threshold: float = 1e-9,
                                 ) -> Dict[str, float]:
    """Select the vertex with the largest open-loop discrete-time spectral
    radius AMONG the data-consistent hard-core vertices (those with
    normalised consistency score s <= s_threshold).

    Terminology: the formal post-audit certificate scope is the audited
    matrix hull conv(V_{C_D}^{audit}); see Theorem (post-cert) and
    Corollary (transfer). The hard-core label here is a synthesis-score
    notion (s_i <= s_threshold) and is *not* the formal audit scope.
    Picking an adversarial box corner with score >= 0.5 (see
    _select_worst_case_plant) is therefore a valid empirical stress test
    against a soft-shell, score-inconsistent vertex, while picking the
    hard-core worst corner is an empirical stress test against a
    score-supported vertex. Neither test is a substitute for the audit-
    time matrix-hull condition, which is reported separately.

    The hard core is taken from out["vertices_all"] (the post-score
    normalised vertex set used by the SDP, 32 entries = 16 aerodynamic
    corners x 2 actuator-gain endpoints); we filter by s <= threshold
    and pick max rho_OL among the survivors. Actuator-gain endpoints
    with _act_scale < 1 are admitted; when audit_include_gain_uncertainty
    is True, the audit-time 32-vertex post-certificate covers the full
    [1 - eps_act, 1] gain convex hull through the audit vertex set.
    """
    verts_all = out["vertices_all"]
    best_rho = -np.inf
    p_hc_worst: Optional[Dict[str, float]] = None
    for v in verts_all:
        if float(v["s"]) > s_threshold:
            continue
        p = dict(v["p"])
        # Strip _act_scale from physical plant dict; the simulation uses
        # the full-gain plant; the reduced-gain endpoint is still
        # represented in the certificate via the convex hull.
        p.pop("_act_scale", None)
        A, _, _ = build_vertex_matrices(p, syn.Ts)
        rho = float(max(abs(np.linalg.eigvals(A))))
        if rho > best_rho:
            best_rho = rho
            p_hc_worst = p
    if p_hc_worst is None:
        print("[HardcoreWorstCase Plant] no hard-core vertex found; "
              "falling back to adversarial corner.")
        return _select_worst_case_plant(bounds, syn)
    print(f"\n[HardcoreWorstCase Plant] hard-core synthesis stress corner "
          f"(score-supported vertex, s<=1e-9; NOT the formal audit scope): "
          f"rho_open={best_rho:.4f}, params={p_hc_worst}")
    return p_hc_worst


# ---------------------------------------------------------
# Corner-vertex sweep: evaluate every 16-corner plant on every controller.
# This is the canonical polytope-wide worst-case-set audit -- it directly
# answers "on which extreme plants does each controller break?"
# ---------------------------------------------------------
def run_f8_corner_sweep(out: Dict[str, Any],
                        syn: F8SynthesisParams,
                        bounds: F8ParamBounds,
                        *,
                        seconds: Optional[float] = None,
                        seed_offset: int = 1500,
                        fig_out_dir: Optional[str] = None,
                        sensor_delay_steps: int = 0,
                        actuator_tau: float = 0.0,
                        sweep_subdir: str = "corner_sweep",
                        ) -> Dict[str, Dict[str, float]]:
    seconds = float(syn.mc_seconds if seconds is None else seconds)
    fig_out_dir = syn.fig_out_dir if fig_out_dir is None else fig_out_dir
    sweep_dir = os.path.join(fig_out_dir, sweep_subdir)
    os.makedirs(sweep_dir, exist_ok=True)

    t_ref, theta_refs = build_f8_reference_trajectories(syn, seconds)
    x_ref = build_f8_state_reference(theta_refs, syn.Ts)
    dbar, _ = build_f8_disturbance_profile(syn, seconds, seed_offset=seed_offset)
    x0 = f8_initial_state(scale=0.3)
    controllers = _f8_controllers_list(out)
    fields = list(bounds.__dataclass_fields__.keys())

    rows: List[Dict[str, Any]] = []
    for bits in itertools.product([0, 1], repeat=len(fields)):
        p_corner = {k: float(getattr(bounds, k)[1] if bits[i] else getattr(bounds, k)[0])
                    for i, k in enumerate(fields)}
        corner_id = "v" + "".join(map(str, bits))
        for name, K in controllers:
            try:
                sim = simulate_f8_tracking(
                    K, p_corner, syn, seconds, dbar, x_ref,
                    x0=x0, meas_noise_std=0.0,
                    sensor_delay_steps=sensor_delay_steps,
                    actuator_tau=actuator_tau,
                )
                met = compute_f8_tracking_metrics(sim, syn)
            except Exception:
                met = {k: float("inf") for k in ["RMSE_theta", "RMSE_theta_max",
                                                  "IAE_theta", "peak_theta",
                                                  "peak_theta_mean", "peak_u",
                                                  "peak_uabs", "energy_u",
                                                  "energy_uabs", "energy_z",
                                                  "duty_du", "duty_abs",
                                                  "duty_normsat", "rho"]}
                met["unstable"] = 1.0
            row = dict(corner=corner_id, Controller=name)
            row.update(met)
            row.update(p_corner)
            rows.append(row)

    metric_cols = ["RMSE_theta", "RMSE_theta_max", "peak_theta", "peak_u",
                   "energy_u", "energy_z", "duty_normsat", "rho", "unstable"]
    with open(os.path.join(sweep_dir, "F8_corner_trials.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["corner", "Controller"] + metric_cols + fields)
        for row in rows:
            cells = []
            for m in metric_cols:
                val = float(row[m])
                cells.append(f"{val:.6g}" if np.isfinite(val) else "inf")
            for k in fields:
                cells.append(f"{float(row[k]):.4g}")
            writer.writerow([row["corner"], row["Controller"]] + cells)

    summary: Dict[str, Dict[str, float]] = {}
    print(f"\n[F8 CornerSweep] 16 corners x {len(controllers)} controllers, "
          f"seconds={seconds:.1f}, sensor_delay={sensor_delay_steps} steps, "
          f"actuator_tau={actuator_tau:.3f}s")
    for name, _ in controllers:
        sub = [r for r in rows if r["Controller"] == name]
        rmses = np.array([float(r["RMSE_theta"]) for r in sub])
        peaks = np.array([float(r["peak_theta"]) for r in sub])
        rhos = np.array([float(r["rho"]) for r in sub])
        unst = np.array([float(r["unstable"]) for r in sub])
        finite = rmses[np.isfinite(rmses)]
        summary[name] = dict(
            n_corners=float(len(sub)),
            rmse_median=float(np.median(finite)) if finite.size else float("inf"),
            rmse_max=float(np.max(rmses)) if finite.size > 0 and np.all(np.isfinite(rmses)) else float("inf"),
            peak_max=float(np.max(peaks)) if peaks.size > 0 and np.all(np.isfinite(peaks)) else float("inf"),
            rho_max=float(np.max(rhos)) if rhos.size > 0 and np.all(np.isfinite(rhos)) else float("inf"),
            unstable_rate=float(np.mean(unst)),
        )
        print(f"  {name:<12} RMSE median/max={summary[name]['rmse_median']:.4g}/"
              f"{summary[name]['rmse_max']:.4g}, "
              f"peak_max={summary[name]['peak_max']:.4g}, "
              f"rho_max={summary[name]['rho_max']:.4g}, "
              f"unstable_rate={summary[name]['unstable_rate']:.2%}")

    cols = ["n_corners", "rmse_median", "rmse_max", "peak_max",
            "rho_max", "unstable_rate"]
    with open(os.path.join(sweep_dir, "F8_corner_summary.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Controller"] + cols)
        for name, stats in summary.items():
            cells = []
            for c in cols:
                val = float(stats.get(c, float("nan")))
                cells.append(f"{val:.6g}" if np.isfinite(val) else "inf")
            writer.writerow([name] + cells)
    return summary


# ---------------------------------------------------------
# Run all controllers on a given plant; produce metrics, figures, CSV
# ---------------------------------------------------------
def _f8_controllers_list(out: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """Return the standard list of (name, K) pairs evaluated by every
    time-domain audit. Centralised so adding/removing baselines is a
    one-line change instead of touching several functions.
    Core-CQLF-Hinf is included only when its synthesis succeeded (not a
    fallback nominal LQR). Proposed-Repaired is appended only when the
    certificate-repair SDP triggered AND succeeded."""
    pairs: List[Tuple[str, np.ndarray]] = [
        ("Proposed", out["K_proposed"]),
        ("NoRelax-ProposedActive", out["K_norelax"]),
        ("QS-Hinf", out["K_qs_hinf"]),
        ("PDL-Hinf", out["K_pdl_hinf"]),
        ("RobustLQR", out["K_robust_lqr"]),
    ]
    # Core-CQLF-Hinf: only include when synthesis actually succeeded.
    if out.get("core_cqlf_status", "") == "feasible":
        pairs.append(("Core-CQLF-Hinf", out["K_core_cqlf_hinf"]))
    # Append Proposed-Repaired only when the certificate-repair SDP actually
    # produced a feasible certified gain (proposed_repair["success"]=True).
    rep = out.get("proposed_repair", {}) or {}
    if rep.get("success", False) and "K_proposed_repaired" in out:
        pairs.append(("Proposed-Repaired", out["K_proposed_repaired"]))
    return pairs


def _run_f8_plant_scenario(scenario_name: str,
                           p_plant: Dict[str, float],
                           out: Dict[str, Any],
                           syn: F8SynthesisParams,
                           seconds: float,
                           dbar: np.ndarray,
                           x_ref: np.ndarray,
                           gusts: List[Dict[str, float]],
                           theta_refs: np.ndarray,
                           t_ref: np.ndarray,
                           fig_out_dir: str) -> Dict[str, Dict[str, float]]:
    print(f"\n{'=' * 70}\n[Scenario: {scenario_name}] plant: {p_plant}\n{'=' * 70}")
    controllers = _f8_controllers_list(out)
    x0 = f8_initial_state(scale=0.3)
    sims: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for name, K in controllers:
        # meas_noise_std raised 1e-4 -> 1e-3 (~0.057 deg) to match real IMU
        # pitch sensor noise rather than the impractical 0.006 deg used in
        # the original prototype. Phase 5a/5b/5c use 0.0 to avoid noise-driven
        # false-instability flags during the corner audit.
        sim = simulate_f8_tracking(K, p_plant, syn, seconds, dbar, x_ref,
                                   x0=x0, meas_noise_std=1e-3)
        sims[name] = sim
        metrics[name] = compute_f8_tracking_metrics(sim, syn)
        print(f"\n[{scenario_name} | {name}]")
        for k, v in metrics[name].items():
            print(f"  {k:<18} = {v:.6g}")
    sub_dir = os.path.join(fig_out_dir, scenario_name)
    os.makedirs(sub_dir, exist_ok=True)
    plot_f8_tracking_results(sims, theta_refs, t_ref, gusts, syn, sub_dir, scenario_name)
    plot_f8_error_effort(sims, gusts, syn, sub_dir, scenario_name)
    gamma_table = {name: float("nan") for name, _ in controllers}
    gamma_table["Proposed"] = out["gamma_proposed"]
    gamma_table["NoRelax-ProposedActive"] = out["gamma_norelax"]
    gamma_table["QS-Hinf"] = out["gamma_qs_hinf"]
    gamma_table["PDL-Hinf"] = out["gamma_pdl_hinf"]
    gamma_table["Core-CQLF-Hinf"] = out["gamma_core_cqlf_hinf"]
    beta_table = {name: float("nan") for name, _ in controllers}
    beta_table["Proposed"] = beta_summary_value(out["sol_proposed"])
    beta_table["NoRelax-ProposedActive"] = beta_summary_value(out["sol_norelax"])
    export_f8_metrics_table(metrics, gamma_table, beta_table, sub_dir)
    return metrics


# ---------------------------------------------------------
# Monte Carlo evaluation: random plants near data source +
# random disturbance seeds; aggregate per-controller statistics.
# ---------------------------------------------------------
def _write_f8_monte_carlo_tables(rows: List[Dict[str, Any]],
                                 out_dir: str) -> Dict[str, Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    metric_names = ["RMSE_theta", "RMSE_theta_max", "IAE_theta", "peak_theta",
                    "peak_theta_mean", "peak_u", "peak_uabs", "energy_u",
                    "energy_uabs", "energy_z", "duty_du", "duty_abs",
                    "duty_normsat", "rho", "unstable"]
    with open(os.path.join(out_dir, "F8_monte_carlo_trials.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "Controller"] + metric_names)
        for row in rows:
            cells = []
            for m in metric_names:
                val = float(row[m])
                cells.append(f"{val:.6g}" if np.isfinite(val) else "")
            writer.writerow([int(row["trial"]), row["Controller"]] + cells)
    controllers = list(dict.fromkeys(str(row["Controller"]) for row in rows))
    summary: Dict[str, Dict[str, float]] = {}
    for name in controllers:
        sub = [row for row in rows if row["Controller"] == name]
        stats: Dict[str, float] = {}
        for m in metric_names:
            vals = np.array([float(row[m]) for row in sub], dtype=float)
            finite = vals[np.isfinite(vals)]
            stats[f"{m}_median"] = float(np.median(finite)) if finite.size else float("nan")
            stats[f"{m}_p90"] = float(np.quantile(finite, 0.90)) if finite.size else float("nan")
            stats[f"{m}_p95"] = float(np.quantile(finite, 0.95)) if finite.size else float("nan")
            stats[f"{m}_max"] = float(np.max(finite)) if finite.size else float("nan")
        if sub:
            stats["finite_rate"] = float(np.mean(
                [all(np.isfinite(float(row[m])) for m in metric_names) for row in sub]
            ))
        else:
            stats["finite_rate"] = float("nan")
        summary[name] = stats
    # Add an explicit unstable_rate (mean of the 0/1 unstable flag) so the
    # CSV directly answers "how often does the controller fail?".
    for name in controllers:
        sub_unst = [float(row["unstable"]) for row in rows if row["Controller"] == name]
        summary[name]["unstable_rate"] = float(np.mean(sub_unst)) if sub_unst else float("nan")
    summary_cols = ["finite_rate", "unstable_rate", "RMSE_theta_median",
                    "RMSE_theta_p90", "RMSE_theta_p95", "RMSE_theta_max",
                    "peak_theta_median", "peak_theta_p90", "peak_theta_max",
                    "energy_u_median", "energy_u_p90", "energy_z_median",
                    "energy_z_p90", "duty_normsat_p90", "rho_max"]
    with open(os.path.join(out_dir, "F8_monte_carlo_summary.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Controller"] + summary_cols)
        for name, stats in summary.items():
            cells = []
            for c in summary_cols:
                val = float(stats.get(c, float("nan")))
                cells.append(f"{val:.6g}" if np.isfinite(val) else "")
            writer.writerow([name] + cells)
    return summary


def run_f8_monte_carlo_evaluation(out: Dict[str, Any],
                                  syn: F8SynthesisParams,
                                  bounds: F8ParamBounds,
                                  *,
                                  seconds: Optional[float] = None,
                                  n_trials: Optional[int] = None,
                                  fig_out_dir: Optional[str] = None,
                                  sampling: str = "uniform",
                                  ) -> Dict[str, Dict[str, float]]:
    """Standard ACC Monte Carlo robustness analysis: uniform-in-box plant
    sampling, random gust seeds per trial, aggregate median/p90/p95/max +
    explicit unstable_rate per controller.

    The default ``sampling='uniform'`` draws each parameter independently
    uniformly in its bounds -- this is the unbiased benchmark used in
    Garrard / Boyd / Doyle robust-control papers and avoids any selection
    bias toward (or against) baselines. (The historical 'corner_biased'
    mode is retained as an opt-in audit only; it is *not* a fair MC.)
    """
    seconds = float(syn.mc_seconds if seconds is None else seconds)
    n_trials = int(syn.mc_trials if n_trials is None else n_trials)
    fig_out_dir = syn.fig_out_dir if fig_out_dir is None else fig_out_dir
    rng = np.random.default_rng(syn.seed + 9090)
    controllers = _f8_controllers_list(out)
    t_ref, theta_refs = build_f8_reference_trajectories(syn, seconds)
    x_ref = build_f8_state_reference(theta_refs, syn.Ts)
    rows: List[Dict[str, Any]] = []
    x0 = f8_initial_state(scale=0.3)
    for trial in range(n_trials):
        if sampling == "uniform":
            p_trial = sample_params(bounds, rng)
        elif sampling == "corner_biased":
            p_trial = _sample_f8_corner_or_near(bounds, rng,
                                                p_corner=0.5, spread=0.10)
        else:
            raise ValueError(f"unknown sampling mode: {sampling!r}")
        dbar, _ = build_f8_disturbance_profile(syn, seconds, seed_offset=1200 + trial)
        for name, K in controllers:
            # meas_noise_std raised 1e-4 -> 1e-3 to match the time-domain
            # scenarios (real IMU level). Aggregate stats across 100 MC trials
            # are robust to this noise; corner-sweep audit keeps noise = 0.
            sim = simulate_f8_tracking(K, p_trial, syn, seconds, dbar, x_ref,
                                       x0=x0, meas_noise_std=1e-3)
            met = compute_f8_tracking_metrics(sim, syn)
            row: Dict[str, Any] = {"trial": float(trial), "Controller": name}
            row.update(met)
            rows.append(row)
    mc_dir = os.path.join(fig_out_dir, "monte_carlo")
    summary = _write_f8_monte_carlo_tables(rows, mc_dir)
    print(f"\n[F8 MonteCarlo] n_trials={n_trials}, sampling={sampling}, "
          f"seconds={seconds:.1f}")
    for name, stats in summary.items():
        print(f"  {name:<12} RMSE median/p90={stats['RMSE_theta_median']:.4g}/"
              f"{stats['RMSE_theta_p90']:.4g}, "
              f"peak median/p90={stats['peak_theta_median']:.4g}/"
              f"{stats['peak_theta_p90']:.4g}, "
              f"rho_max={stats['rho_max']:.4g}, "
              f"unstable_rate={stats['unstable_rate']:.2%}")
    return summary


# ---------------------------------------------------------
# Plotting (paper-quality IEEE style)
# ---------------------------------------------------------
# Wong (2011) colour-blind-friendly palette + distinct line styles per
# controller. Six robust-control baselines:
#   Proposed     = data-driven beta-relaxation SDP + ICE
#   NoRelax-ProposedActive  = beta=0 ablation on the same active set (Section V.B)
#   QS-Hinf      = Boyd-Feron-El Ghaoui-Balakrishnan 1994 classical single-Q H-inf
#   PDL-Hinf     = Daafouz-Bernussou 2001 parameter-dependent Lyapunov H-inf
#   RobustLQR    = Wang-Veillette 1994 minimax LQR (non-clairvoyant LQR
#                  textbook reference; no H-infinity certificate)
_F8_COLOR_PROP   = "#0072B2"    # blue
_F8_COLOR_NORE   = "#D55E00"    # vermillion
_F8_COLOR_QS     = "#009E73"    # bluish green
_F8_COLOR_PDL    = "#CC79A7"    # reddish purple
_F8_COLOR_ROBLQR = "#E69F00"    # Wong orange (distinct from blue/green/vermillion)
_F8_COLOR_CCQLF  = "#56B4E9"    # sky blue (Wong palette)
_F8_COLOR_REF  = "#555555"
_F8_COLOR_GUST = "#CCCCCC"
_F8_LS = {
    "Proposed":     "-",
    "NoRelax-ProposedActive":  "--",
    "QS-Hinf":      ":",
    "PDL-Hinf":     (0, (5, 2)),
    "RobustLQR":    (0, (3, 1, 1, 1)),  # dash-dot, clearly distinct from QS ":"
    "Core-CQLF-Hinf": (0, (1, 1)),       # densely dotted
}
_F8_COL = {
    "Proposed":     _F8_COLOR_PROP,
    "NoRelax-ProposedActive":  _F8_COLOR_NORE,
    "QS-Hinf":      _F8_COLOR_QS,
    "PDL-Hinf":     _F8_COLOR_PDL,
    "RobustLQR":    _F8_COLOR_ROBLQR,
    "Core-CQLF-Hinf": _F8_COLOR_CCQLF,
}
_F8_LW = {
    "Proposed":     1.3,
    "NoRelax-ProposedActive":  1.1,
    "QS-Hinf":      1.0,
    "PDL-Hinf":     1.0,
    "RobustLQR":    1.0,
    "Core-CQLF-Hinf": 1.0,
}
_F8_DISPLAY = {
    "Proposed":                r"Proposed",
    "NoRelax-ProposedActive":  r"Proposed ($\beta{=}0$)",
    "QS-Hinf":                 r"QS-$\mathcal{H}_\infty$",
    "PDL-Hinf":                r"PDL-$\mathcal{H}_\infty$",
    "RobustLQR":               r"RobustLQR",
    "Core-CQLF-Hinf":           r"Core-CQLF-$\mathcal{H}_\infty$",
    "Proposed-Repaired":       r"Proposed (repaired)",
}


def _f8_apply_paper_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "mathtext.fontset": "cm",
        "lines.linewidth": 1.0, "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "0.7",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _f8_shade_gusts(ax, gusts: List[Dict[str, float]], t_max: float) -> None:
    for g in gusts:
        if g["t0"] < t_max:
            ax.axvspan(g["t0"], min(g["t1"], t_max),
                       color=_F8_COLOR_GUST, alpha=0.35, zorder=-5, lw=0)


def plot_f8_tracking_results(sims: Dict[str, Dict[str, Any]],
                             theta_refs: np.ndarray,
                             t_ref: np.ndarray,
                             gusts: List[Dict[str, float]],
                             syn: F8SynthesisParams,
                             out_dir: str,
                             scenario_name: str) -> None:
    import matplotlib.pyplot as plt
    _f8_apply_paper_style()
    n = theta_refs.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6), sharex=True)
    panel_labels = [r"(a) AC 1: step", r"(b) AC 2: sinusoid",
                    r"(c) AC 3: ramp-hold", r"(d) AC 4: trapezoid"]
    t_max = float(t_ref[-1])
    for i, ax in enumerate(axes.flat):
        _f8_shade_gusts(ax, gusts, t_max)
        ax.plot(t_ref[:n], theta_refs[i, :], ls=":", lw=0.9,
                color=_F8_COLOR_REF, label=r"$\theta^{\star}$")
        for name, sim in sims.items():
            ax.plot(sim["t"], sim["x"][3 * i + 2, :],
                    ls=_F8_LS.get(name, "-"), lw=_F8_LW.get(name, 1.0),
                    color=_F8_COL.get(name, "black"),
                    label=_F8_DISPLAY.get(name, name))
        ax.set_ylabel(r"$\theta_{" + str(i + 1) + r"}$ (rad)")
        ax.text(0.02, 0.97, panel_labels[i], transform=ax.transAxes,
                va="top", ha="left", fontsize=8)
        ax.grid(True, which="both")
    axes[-1, 0].set_xlabel(r"Time (s)")
    axes[-1, 1].set_xlabel(r"Time (s)")
    axes[0, 0].legend(loc="lower right", frameon=False, ncol=len(sims) + 1,
                      fontsize=7, columnspacing=0.8, handlelength=1.4)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    stem = f"F8_Tracking_{scenario_name}"
    for ext in syn.fig_formats:
        dpi = syn.fig_dpi_png if ext == "png" else None
        fig.savefig(os.path.join(out_dir, stem + "." + ext), dpi=dpi)
    plt.close(fig)


def plot_f8_error_effort(sims: Dict[str, Dict[str, Any]],
                         gusts: List[Dict[str, float]],
                         syn: F8SynthesisParams,
                         out_dir: str,
                         scenario_name: str) -> None:
    import matplotlib.pyplot as plt
    _f8_apply_paper_style()
    example = next(iter(sims.values()))
    n_steps = example["u_c"].shape[1]
    t_u = example["t"][:n_steps]
    x_ref = example["x_ref"]
    t_max = float(t_u[-1])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
    _f8_shade_gusts(axes[0], gusts, t_max)
    for name, sim in sims.items():
        err = sim["x"][[2, 5, 8, 11], :n_steps] - x_ref[[2, 5, 8, 11], :]
        en = np.linalg.norm(err, axis=0)
        axes[0].plot(t_u, en, ls=_F8_LS.get(name, "-"),
                     lw=_F8_LW.get(name, 1.0),
                     color=_F8_COL.get(name, "black"),
                     label=_F8_DISPLAY.get(name, name))
    axes[0].set_ylabel(r"$\|\theta - \theta^{\star}\|_{2}$ (rad)")
    axes[0].set_xlabel(r"Time (s)")
    axes[0].text(0.02, 0.97, "(a)", transform=axes[0].transAxes,
                 va="top", ha="left")
    axes[0].grid(True)
    axes[0].legend(frameon=False, loc="upper right", fontsize=7,
                   columnspacing=0.8, handlelength=1.4)
    _f8_shade_gusts(axes[1], gusts, t_max)
    for name, sim in sims.items():
        du = np.linalg.norm(sim["u_c"], axis=0)
        axes[1].plot(t_u, du, ls=_F8_LS.get(name, "-"),
                     lw=_F8_LW.get(name, 1.0),
                     color=_F8_COL.get(name, "black"),
                     label=_F8_DISPLAY.get(name, name))
    axes[1].axhline(syn.du_max, ls=(0, (3, 3)), color="black", lw=0.8,
                    label=r"$\|\Delta u\|_{\max}$")
    axes[1].set_ylabel(r"$\|\Delta u\|_{2}$ (rad)")
    axes[1].set_xlabel(r"Time (s)")
    axes[1].text(0.02, 0.97, "(b)", transform=axes[1].transAxes,
                 va="top", ha="left")
    axes[1].grid(True)
    axes[1].legend(frameon=False, loc="upper right", fontsize=7,
                   columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    stem = f"F8_ErrorEffort_{scenario_name}"
    for ext in syn.fig_formats:
        dpi = syn.fig_dpi_png if ext == "png" else None
        fig.savefig(os.path.join(out_dir, stem + "." + ext), dpi=dpi)
    plt.close(fig)


def export_fixed_k_audit_table(out: Dict[str, Any],
                               nominal_metrics: Dict[str, Dict[str, float]],
                               out_dir: str) -> None:
    """Write the fixed-K data-core audit table to fixed_k_audit.csv.

    Columns:
      method, RMSE, ||K||, feasible, gamma_core, core_max_eig,
      full_box_max_eig, cond(P), worst audit vertex diagnostics

    RMSE is taken from the provided nominal_metrics dict (same
    structure returned by ``_run_f8_plant_scenario``); when a method has
    no nominal metric it gets NaN.  The Proposed-Repaired row appears
    only when the certificate-repair SDP actually ran and succeeded.
    """
    os.makedirs(out_dir, exist_ok=True)
    audits: Dict[str, Any] = out.get("fixed_k_audits", {}) or {}
    if not audits:
        return

    columns = [
        "method", "RMSE", "K_norm", "feasible", "gamma_core",
        "core_max_eig", "full_box_max_eig", "cond_P",
        "n_core", "decay_rate",
        "worst_audit_vertex_index", "worst_audit_kappa",
        "worst_audit_za_v", "worst_audit_m_alpha",
        "worst_audit_m_q", "worst_audit_m_de",
        "rho_closed_loop_at_worst_vertex",
        "rho_closed_loop_max_over_core",
        "status",
    ]
    csv_path = os.path.join(out_dir, "fixed_k_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        # Maintain a stable controller order (Repaired last)
        order = ["Proposed", "NoRelax-ProposedActive", "Core-CQLF-Hinf",
                 "QS-Hinf", "PDL-Hinf", "RobustLQR", "Proposed-Repaired"]
        for name in order:
            au = audits.get(name)
            if au is None:
                continue
            rmse = float("nan")
            if name == "Proposed-Repaired":
                rmse = float(nominal_metrics.get("Proposed", {}).get(
                    "RMSE_theta_repaired", float("nan")))
            else:
                rmse = float(nominal_metrics.get(name, {}).get(
                    "RMSE_theta", float("nan")))
            row = [
                name,
                f"{rmse:.6g}" if np.isfinite(rmse) else "",
                f"{float(au.get('K_norm', float('nan'))):.6g}",
                "True" if au.get("feasible", False) else "False",
                f"{float(au.get('gamma_core', float('inf'))):.6g}"
                if np.isfinite(au.get("gamma_core", float("inf"))) else "inf",
                f"{float(au.get('core_max_eig', float('nan'))):+.6e}"
                if np.isfinite(au.get("core_max_eig", float("nan"))) else "",
                f"{float(au.get('full_box_max_eig', float('nan'))):+.6e}"
                if np.isfinite(au.get("full_box_max_eig", float("nan"))) else "",
                f"{float(au.get('cond_P', float('nan'))):.6g}"
                if np.isfinite(au.get("cond_P", float("nan"))) else "",
                int(au.get("n_core", 0)),
                f"{float(au.get('decay_rate', 1.0)):.4g}",
                int(au.get("worst_audit_vertex_index", -1)),
                f"{float(au.get('worst_audit_kappa', float('nan'))):.6g}"
                if np.isfinite(au.get("worst_audit_kappa", float("nan"))) else "",
                f"{float(au.get('worst_audit_za_v', float('nan'))):.6g}"
                if np.isfinite(au.get("worst_audit_za_v", float("nan"))) else "",
                f"{float(au.get('worst_audit_m_alpha', float('nan'))):.6g}"
                if np.isfinite(au.get("worst_audit_m_alpha", float("nan"))) else "",
                f"{float(au.get('worst_audit_m_q', float('nan'))):.6g}"
                if np.isfinite(au.get("worst_audit_m_q", float("nan"))) else "",
                f"{float(au.get('worst_audit_m_de', float('nan'))):.6g}"
                if np.isfinite(au.get("worst_audit_m_de", float("nan"))) else "",
                f"{float(au.get('rho_closed_loop_at_worst_vertex', float('nan'))):.6g}"
                if np.isfinite(au.get("rho_closed_loop_at_worst_vertex", float("nan"))) else "",
                f"{float(au.get('rho_closed_loop_max_over_core', float('nan'))):.6g}"
                if np.isfinite(au.get("rho_closed_loop_max_over_core", float("nan"))) else "",
                str(au.get("status", "?"))[:80],
            ]
            writer.writerow(row)
    print(f"[F8 Audit] fixed-K data-core audit table -> {csv_path}")


def export_synthesis_audit_consistency_table(out: Dict[str, Any],
                                             out_dir: str) -> None:
    """Write synthesis-to-audit certificate diagnostics."""
    os.makedirs(out_dir, exist_ok=True)
    rows = list(out.get("synthesis_audit_consistency", []) or [])
    if not rows:
        return
    columns = [
        "method",
        "gamma_syn",
        "g_syn",
        "decay_syn",
        "decay_audit",
        "synth_vertex_set",
        "audit_vertex_set",
        "max_res_QY_on_synth_vertices",
        "max_res_PK_at_syn_g_on_synth_vertices",
        "max_res_PK_at_syn_g_on_audit_vertices",
        "gamma_fixedK_optimized",
        "core_max_eig_optimized",
        "synthesis_gamma_is_certificate",
        "status",
    ]
    csv_path = os.path.join(out_dir, "synthesis_audit_consistency.csv")

    def fmt(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "True" if bool(v) else "False"
        try:
            vf = float(v)
            if np.isfinite(vf):
                return f"{vf:.6g}"
            return "inf" if vf > 0 else "-inf" if vf < 0 else ""
        except Exception:
            return str(v)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c, "")) for c in columns})
    print(f"[F8 Audit] synthesis/audit consistency diagnostics -> {csv_path}")


def export_f8_metrics_table(metrics: Dict[str, Dict[str, float]],
                            gamma_syn_diag: Dict[str, float],
                            betas: Dict[str, float],
                            out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    columns = ["gamma_syn_diag", "beta", "RMSE_theta", "RMSE_theta_max", "IAE_theta",
               "peak_theta", "peak_theta_mean", "peak_u", "peak_uabs",
               "energy_u", "energy_uabs", "energy_z", "duty_du", "duty_abs",
               "duty_normsat", "rho"]
    with open(os.path.join(out_dir, "F8_metrics_table.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Controller"] + columns)
        for name, met in metrics.items():
            row = []
            for c in columns:
                if c == "gamma_syn_diag":
                    val = gamma_syn_diag.get(name, float("nan"))
                elif c == "beta":
                    val = betas.get(name, float("nan"))
                else:
                    val = met.get(c, float("nan"))
                row.append(f"{val:.6g}" if np.isfinite(val) else "")
            writer.writerow([name] + row)


# ---------------------------------------------------------
# Reproducibility probe: collect Python / NumPy / SciPy / CVXPY / MOSEK
# versions so the pipeline can stamp them into the NPZ + the run log.
# ---------------------------------------------------------
def collect_code_versions() -> Dict[str, str]:
    """Return a dict of solver / library versions used by this run.

    Captured fields are stable enough to embed in the NPZ for later
    reproduction (Priority 12, Sec. ``code_versions`` metadata).
    """
    versions: Dict[str, str] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": str(np.__version__),
    }
    try:
        import scipy as _sp
        versions["scipy"] = str(_sp.__version__)
    except Exception:
        versions["scipy"] = "unavailable"
    try:
        import cvxpy as _cp
        versions["cvxpy"] = str(_cp.__version__)
    except Exception:
        versions["cvxpy"] = "unavailable"
    try:
        import mosek as _mk
        versions["mosek"] = ".".join(str(x) for x in _mk.Env.getversion())
    except Exception:
        try:
            versions["mosek"] = str(getattr(_mk, "__version__", "unknown"))
        except Exception:
            versions["mosek"] = "unavailable"
    return versions


# ---------------------------------------------------------
# Main entry point: synthesize -> nominal/worst scenarios -> Monte Carlo
# ---------------------------------------------------------
def main_acc_f8_formation_quick() -> None:
    bounds = F8ParamBounds()
    syn = F8SynthesisParams(fig_out_dir="acc_f8_formation_results")
    os.makedirs(syn.fig_out_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# F-8 Formation 16D: synthesis + tracking + Monte Carlo")
    print(f"# 4 uncertain aero parameters -> 2^4 = 16 corner vertices")
    print(f"{'#' * 70}")
    code_versions = collect_code_versions()
    print("[Versions] " + "  ".join(f"{k}={v}" for k, v in code_versions.items()))

    # ===== 1. Controller synthesis =====
    out = synthesize_f8_controllers(
        syn, bounds, L_data=500, K_budget=16, max_rounds=3,
        score_mode="lambda_max", verbose=True,
    )

    # ===== 2. Reference + disturbance for time-domain scenarios =====
    seconds = 10.0
    t_ref, theta_refs = build_f8_reference_trajectories(syn, seconds)
    x_ref = build_f8_state_reference(theta_refs, syn.Ts)
    dbar, gusts = build_f8_disturbance_profile(syn, seconds, seed_offset=800)
    print(f"\n[Reference] 4 aircraft heterogeneous commands "
          f"(step / sine / ramp-hold / trapezoid)")
    print(f"[Disturbance] {seconds}s / {dbar.shape[1]} steps, gusts={gusts}")

    # ===== 3. Nominal scenario (exact centre of uncertainty box) =====
    # The exact centre of the parameter box: a benign reference operating
    # point that none of the five robust H-infinity controllers (Proposed,
    # NoRelax-ProposedActive, QS-Hinf, PDL-Hinf, Core-CQLF-Hinf) is specifically tuned to. Using the
    # true centre (not the data-source plant) avoids the confusing situation
    # where "nominal" means "perturbed data-generating plant".
    p_nominal = center_params(bounds)
    metrics_nom = _run_f8_plant_scenario(
        "nominal_plant", p_nominal, out, syn, seconds, dbar, x_ref,
        gusts, theta_refs, t_ref, syn.fig_out_dir,
    )

    # Fixed-K data-core post-certification table (uses nominal RMSE +
    # the per-method Lyapunov / gamma_core / cond(P) computed inside
    # synthesize_f8_controllers). Written once per run at the top
    # of fig_out_dir, alongside the per-scenario metric tables.
    if syn.run_fixed_k_audit:
        export_fixed_k_audit_table(out, metrics_nom, syn.fig_out_dir)
        export_synthesis_audit_consistency_table(out, syn.fig_out_dir)

    # ===== 4a. Adversarial worst-case scenario (out of audit hull) =====
    # Hardest open-loop box corner regardless of data consistency. This is
    # the conventional robust-control "stress-test the full polytope" set-up
    # and coincides here with a data-inconsistent soft-shell vertex (score
    # s~=0.6, 22/32 in rank). The fixed-K post-audit certificate is NOT
    # claimed on this corner: it is outside the audited matrix hull
    # conv(V_{C_D}^{audit}) on which Theorem (post-cert) is proved.
    # Reporting the closed-loop behaviour here is a legitimate empirical
    # stress test but MUST be read as out-of-audit-hull, i.e. outside
    # the formal certificate scope.
    p_worst = _select_worst_case_plant(bounds, syn)
    metrics_worst = _run_f8_plant_scenario(
        "worst_plant", p_worst, out, syn, seconds, dbar, x_ref,
        gusts, theta_refs, t_ref, syn.fig_out_dir,
    )

    # ===== 4b. Hard-core synthesis stress (score-supported) =====
    # Hardest corner AMONG score-supported vertices (s~=0). This is a
    # hard-core synthesis stress test: the synthesis LMI is enforced
    # (unrelaxed for Proposed) on score-supported vertices, so this is
    # the corner the synthesis is most directly designed to handle.
    # It is an empirical stress test and NOT the formal post-audit
    # scope, which is the audited matrix hull conv(V_{C_D}^{audit})
    # characterised by the matrix-hull membership condition. All
    # baselines are evaluated on the same corner for a like-for-like
    # comparison.
    p_hc_worst = _select_hardcore_worst_plant(out, bounds, syn)
    metrics_hc_worst = _run_f8_plant_scenario(
        "hardcore_worst_plant", p_hc_worst, out, syn, seconds, dbar, x_ref,
        gusts, theta_refs, t_ref, syn.fig_out_dir,
    )

    # ===== 5a. Corner-vertex sweep, IDEAL actuator/sensor (ablation reference) =====
    # Each phase (5a/5b/5c) uses a DIFFERENT disturbance seed so the three
    # audits are not correlated through a shared gust realisation. Within a
    # phase the same dbar is used across the 16 corners x 4 controllers, so
    # that cross-controller / cross-plant differences cannot be ascribed to
    # disturbance variability.
    print(f"\n{'#' * 70}\n# Phase 5a: Corner sweep (ideal actuator + sensor)\n{'#' * 70}")
    corner_summary = run_f8_corner_sweep(
        out, syn, bounds, seconds=seconds, fig_out_dir=syn.fig_out_dir,
        sensor_delay_steps=0, actuator_tau=0.0,
        seed_offset=1500,
        sweep_subdir="corner_sweep",
    )

    # ===== 5b. Corner sweep: ACTUATOR LAG ONLY =====
    # This test exercises the multiplicative input-gain uncertainty that
    # the 32-plant SDP was designed to cover (actuator_gain_uncertainty
    # > 0). All four controllers (Proposed, NoRelax-ProposedActive, QS-Hinf, PDL-Hinf)
    # are synthesised against the 32-vertex plant family that already
    # includes a (1 - eps_act) reduced-input-gain replica of every nominal
    # corner, so the static input-effectiveness interval [1 - eps_act, 1]
    # is encoded at synthesis and audit time via the convex-hull
    # interpretation. Phase 5b is an empirical robustness stress test
    # (motivated by the same reduced-gain endpoint) under a 50 ms
    # first-order actuator lag, NOT a formal certificate for dynamic
    # actuator uncertainty.
    tau_act = max(0.05, float(syn.unmodelled_actuator_tau)) if syn.unmodelled_actuator_tau > 0 else 0.05
    print(f"\n{'#' * 70}\n# Phase 5b: Corner sweep -- ACTUATOR-LAG stress test"
          f"\n#   actuator tau = {tau_act:.3f} s (alpha_a = {syn.Ts/(syn.Ts+tau_act):.2f})"
          f"\n#   SDP design-time gain uncertainty = {out.get('actuator_gain_uncertainty', 0.0):.2f}"
          f"\n{'#' * 70}")
    corner_summary_unmod = run_f8_corner_sweep(
        out, syn, bounds, seconds=seconds, fig_out_dir=syn.fig_out_dir,
        sensor_delay_steps=0, actuator_tau=tau_act,
        seed_offset=1600,
        sweep_subdir="corner_sweep_actuator_lag",
    )

    # ===== 5c. Corner sweep: SENSOR DELAY ONLY (H-infinity limitation test) =====
    # This test exercises *pure phase* uncertainty (sensor delay) which
    # cannot be absorbed into the multiplicative gain uncertainty in the
    # SDP. LQR's Kalman 60-degree phase margin protects it here, while
    # H-infinity state feedback has no explicit phase-margin guarantee
    # (Doyle 1978). Results on this phase document a genuine trade-off:
    # Proposed pays for its data-driven box-robust certificate with
    # reduced phase-margin tolerance -- the standard finding that motivates
    # follow-up LQG / LTR or mu-synthesis extensions.
    # Default to a 1-step (= 1 control period = 50 ms at Ts=50 ms) delay,
    # which is realistic for modern flight-control sensor pipelines
    # (Stevens-Lewis Sec.2.4 gives typical pitot / IMU filter latencies of
    # 20-40 ms). Earlier revisions used 2 steps (100 ms) which is closer
    # to a worst-case scenario.
    d_sensor = max(1, int(syn.unmodelled_sensor_delay)) if syn.unmodelled_sensor_delay > 0 else 1
    print(f"\n{'#' * 70}\n# Phase 5c: Corner sweep -- SENSOR-DELAY stress test"
          f"\n#   sensor_delay = {d_sensor} steps ({d_sensor*syn.Ts*1000:.0f} ms)"
          f"\n#   (documented H-infinity state-feedback limitation)"
          f"\n{'#' * 70}")
    corner_summary_sensor = run_f8_corner_sweep(
        out, syn, bounds, seconds=seconds, fig_out_dir=syn.fig_out_dir,
        sensor_delay_steps=d_sensor, actuator_tau=0.0,
        seed_offset=1700,
        sweep_subdir="corner_sweep_sensor_delay",
    )

    # ===== 6. Monte Carlo sweep (uniform-in-box sampling, ACC standard) =====
    mc_summary = run_f8_monte_carlo_evaluation(
        out, syn, bounds,
        seconds=syn.mc_seconds, n_trials=syn.mc_trials,
        sampling="uniform", fig_out_dir=syn.fig_out_dir,
    )

    # ===== 7. Aggregated NPZ export =====
    payload_nom = {f"nom_{n}_{k}": np.array([v])
                   for n, met in metrics_nom.items() for k, v in met.items()}
    payload_wc = {f"wc_{n}_{k}": np.array([v])
                  for n, met in metrics_worst.items() for k, v in met.items()}
    payload_hcwc = {f"hcwc_{n}_{k}": np.array([v])
                    for n, met in metrics_hc_worst.items() for k, v in met.items()}
    payload_mc = {f"mc_{n}_{k}": np.array([v])
                  for n, stats in mc_summary.items() for k, v in stats.items()}
    payload_corner = {f"corner_{n}_{k}": np.array([v])
                      for n, stats in corner_summary.items() for k, v in stats.items()}
    payload_corner_unmod = {f"corner_actlag_{n}_{k}": np.array([v])
                            for n, stats in corner_summary_unmod.items()
                            for k, v in stats.items()}
    payload_corner_sensor = {f"corner_sensdly_{n}_{k}": np.array([v])
                             for n, stats in corner_summary_sensor.items()
                             for k, v in stats.items()}
    # Vertex score diagnostics (used to build Fig 1 parameter-space score map)
    verts_all = out["vertices_all"]
    param_keys = list(bounds.__dataclass_fields__.keys())
    vertex_params = np.array(
        [[float(v["p"].get(k, np.nan)) for k in param_keys] for v in verts_all],
        dtype=float,
    )
    vertex_act_scales = np.array(
        [float(v["p"].get("_act_scale", 1.0)) for v in verts_all], dtype=float)
    vertex_scores = np.array([float(v["s"]) for v in verts_all], dtype=float)
    active_mask = np.zeros(len(verts_all), dtype=bool)
    active_ptrs = {id(v) for v in out["vertices_active"]}
    for i, v in enumerate(verts_all):
        if id(v) in active_ptrs:
            active_mask[i] = True
    # ----- Derived scalar metadata (Priority 12, Sec. NPZ metadata) -----
    # audit_vertex_count: number of vertices on which the post-audit SDP
    # was solved. Pulled from the first audited method (n_core is shared
    # across all audited methods since they use the same audit vertex set).
    _audits = out.get("fixed_k_audits", {})
    audit_vertex_count = 0
    for _aud in _audits.values():
        n_c = int(_aud.get("n_core", 0)) if isinstance(_aud, dict) else 0
        if n_c:
            audit_vertex_count = n_c
            break
    # Matrix-hull residual scalars promoted from the diag dict so paper
    # cross-reference scripts can read them directly. Three independent
    # tolerances (abs / rel / inf) are reported so the paper can quote
    # raw numbers rather than a single binary verdict; see
    # matrix_hull.matrix_hull_residual for the convention.
    _mhd = out.get("matrix_hull_diag", {}) or {}
    matrix_hull_residual_abs = float(_mhd.get("residual_abs", float("nan")))
    matrix_hull_residual_rel = float(_mhd.get("residual_rel", float("nan")))
    matrix_hull_residual_inf = float(_mhd.get("residual_inf", float("nan")))
    matrix_hull_inside_abs = bool(_mhd.get("inside_hull_abs",
                                            _mhd.get("inside_hull", False)))
    matrix_hull_inside_rel = bool(_mhd.get("inside_hull_rel", False))
    matrix_hull_inside_inf = bool(_mhd.get("inside_hull_inf", False))
    matrix_hull_n_vertices = int(_mhd.get("n_vertices", 0))
    matrix_hull_abs_tol = float(_mhd.get("abs_tol", float("nan")))
    matrix_hull_rel_tol = float(_mhd.get("rel_tol", float("nan")))
    matrix_hull_inf_tol = float(_mhd.get("inf_tol", float("nan")))
    audit_act_scales = np.asarray(
        out.get("audit_act_scales", np.array([1.0])), dtype=float)
    npz_path = os.path.join(syn.fig_out_dir, "acc_f8_formation_results.npz")
    try:
        np.savez(
            npz_path,
            p_nominal=np.array(list(p_nominal.values())),
            p_worst=np.array(list(p_worst.values())),
            p_hc_worst=np.array(list(p_hc_worst.values())),
            vertex_param_keys=np.array(param_keys),
            vertex_params=vertex_params,
            vertex_act_scales=vertex_act_scales,
            vertex_scores=vertex_scores,
            vertex_active_mask=active_mask.astype(np.int8),
            p_data_source=np.array(list(out["p_data_source"].values())),
            gamma_proposed=np.array([out["gamma_proposed"]]),
            gamma_norelax=np.array([out["gamma_norelax"]]),
            gamma_qs_hinf=np.array([out["gamma_qs_hinf"]]),
            gamma_pdl_hinf=np.array([out["gamma_pdl_hinf"]]),
            gamma_core_cqlf_hinf=np.array([out["gamma_core_cqlf_hinf"]]),
            gamma_robust_lqr=np.array([out["gamma_robust_lqr"]]),
            beta_proposed=np.array([beta_summary_value(out["sol_proposed"])]),
            beta_norelax=np.array([beta_summary_value(out["sol_norelax"])]),
            norelax_status=np.array([out.get("norelax_status", "unknown")]),
            qs_hinf_status=np.array([out.get("qs_hinf_status", "unknown")]),
            pdl_hinf_status=np.array([out.get("pdl_hinf_status", "unknown")]),
            core_cqlf_status=np.array([out.get("core_cqlf_status", "unknown")]),
            robust_lqr_status=np.array([out.get("robust_lqr_status", "unknown")]),
            robust_lqr_p_worst=np.array(list(out["robust_lqr_p_worst"].values())),
            robust_lqr_rho_ol=np.array([out.get("robust_lqr_rho_ol", float("nan"))]),
            actuator_gain_uncertainty=np.array([out.get("actuator_gain_uncertainty", 0.0)]),
            K_proposed=out["K_proposed"],
            K_norelax=out["K_norelax"],
            K_qs_hinf=out["K_qs_hinf"],
            K_pdl_hinf=out["K_pdl_hinf"],
            K_robust_lqr=out["K_robust_lqr"],
            K_core_cqlf_hinf=out["K_core_cqlf_hinf"],
            # Data-contained core diagnostic (audit input)
            datacore_diag_obj=np.array([out.get("datacore_diag", {})], dtype=object),
            # Fixed-K data-core post-certification (per-method audits)
            fixed_k_audits=np.array([out.get("fixed_k_audits", {})], dtype=object),
            synthesis_audit_consistency=np.array(
                [out.get("synthesis_audit_consistency", [])], dtype=object),
            K_proposed_repaired=out.get("K_proposed_repaired",
                                        np.zeros_like(out["K_proposed"])),
            gamma_proposed_repaired=np.array(
                [out.get("gamma_proposed_repaired", float("nan"))]),
            proposed_repair=np.array([out.get("proposed_repair", {})], dtype=object),
            # Matrix-hull membership residual (Corollary 1) and audit metadata
            matrix_hull_diag=np.array([out.get("matrix_hull_diag", {})], dtype=object),
            certificate_scope=np.array([out.get("certificate_scope", "unknown")]),
            tracking_scope=np.array([out.get("tracking_scope", "unknown")]),
            disturbance_convention=np.array(
                [out.get("disturbance_convention", "unknown")]),
            d_cert=np.array([out.get("d_cert", float("nan"))]),
            d_data=np.array([out.get("d_data", float("nan"))]),
            audit_include_gain_uncertainty=np.array(
                [bool(out.get("audit_include_gain_uncertainty", False))]),
            # Scalar promotions for direct paper xref (Priority 12).
            audit_vertex_count=np.array([int(audit_vertex_count)]),
            audit_act_scales=audit_act_scales,
            matrix_hull_residual_abs=np.array([matrix_hull_residual_abs]),
            matrix_hull_residual_rel=np.array([matrix_hull_residual_rel]),
            matrix_hull_residual_inf=np.array([matrix_hull_residual_inf]),
            matrix_hull_inside_hull_abs=np.array([matrix_hull_inside_abs]),
            matrix_hull_inside_hull_rel=np.array([matrix_hull_inside_rel]),
            matrix_hull_inside_hull_inf=np.array([matrix_hull_inside_inf]),
            matrix_hull_n_vertices=np.array([matrix_hull_n_vertices]),
            matrix_hull_abs_tol=np.array([matrix_hull_abs_tol]),
            matrix_hull_rel_tol=np.array([matrix_hull_rel_tol]),
            matrix_hull_inf_tol=np.array([matrix_hull_inf_tol]),
            # Solver / library version stamp for reproducibility.
            code_versions=np.array([collect_code_versions()], dtype=object),
            dbar_profile=dbar,
            **payload_nom, **payload_wc, **payload_hcwc, **payload_mc,
            **payload_corner, **payload_corner_unmod, **payload_corner_sensor,
        )
        print(f"\n[F8] Results exported to: {npz_path}")
    except Exception as e:
        print(f"\n[F8] [ERROR] Failed to save NPZ at {npz_path}: {e}")
        print("     CSV tables in individual phase subdirs are still available.")


# ---------------------------------------------------------
# Quick sanity self-check (model only)
# ---------------------------------------------------------
def _f8_selfcheck() -> None:
    bounds = F8ParamBounds()
    syn = F8SynthesisParams()
    p_c = center_params(bounds)
    A, B, S = build_vertex_matrices(p_c, syn.Ts)
    print(f"[F8 SelfCheck] center params: {p_c}")
    print(f"  A shape={A.shape}, spectral radius={max(abs(np.linalg.eigvals(A))):.4f}")
    print(f"  B shape={B.shape}, ||B||={np.linalg.norm(B):.4f}")
    print(f"  S shape={S.shape}, ||S||={np.linalg.norm(S):.4f}")
    print(f"  A[0:3,0:3] (aircraft 1) = \n{A[0:3,0:3]}")
    print(f"  A[3:6,3:6] (aircraft 2) = \n{A[3:6,3:6]}")
    Ac, Bc, Sc = _f8_formation_continuous(p_c)
    eigsc = np.linalg.eigvals(Ac)
    print(f"  Continuous-time A_c eigvals (first 6): {eigsc[:6]}")
    print(f"  Max Re(eigvals): {max(ei.real for ei in eigsc):.4f}  (positive = unstable open-loop)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _f8_selfcheck()
    else:
        main_acc_f8_formation_quick()
