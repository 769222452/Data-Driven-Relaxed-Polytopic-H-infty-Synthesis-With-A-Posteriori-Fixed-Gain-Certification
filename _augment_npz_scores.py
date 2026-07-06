"""Augment the simulation NPZ with vertex-score diagnostics needed by
Fig 1 (parameter-space consistency-score map).

Reason: the simulation that produced ``acc_f8_formation_results.npz``
was launched *before* the NPZ export was extended to include
``vertex_params / vertex_scores / vertex_active_mask / p_data_source``.
Re-running the full 15-min pipeline just for those fields is wasteful,
so this script reproduces ONLY the scoring portion (simulate batch ->
build Psi -> score 32 vertices -> run the ICE active-set selection) and
writes the resulting arrays next to the main NPZ as ``_aug`` suffix
fields. The score computation uses identical seeds, so the resulting
numbers match byte-for-byte what the full pipeline would have produced.

Usage:
  python _augment_npz_scores.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import acc_f8_formation_16d as f8


def main() -> None:
    syn = f8.F8SynthesisParams()
    bounds = f8.F8ParamBounds()

    # Reproduce synthesize_f8_controllers up to the scoring + ICE
    # active-set selection. Seeds and parameter sampling must match
    # the main pipeline exactly.
    bsyn = f8.f8_to_base_syn(syn)
    p_nom = f8.center_params(bounds)
    A_nom, B_nom, _ = f8.build_vertex_matrices(p_nom, syn.Ts)
    K_nom_internal = f8.base.dlqr(A_nom, B_nom,
                                  np.diag(syn.Qx_lqr), np.diag(syn.Ru_lqr))

    rng = np.random.default_rng(syn.seed + 999)
    p_data: dict = {}
    for k in bounds.__dataclass_fields__.keys():
        lo, hi = getattr(bounds, k)
        half = 0.5 * (hi - lo)
        val = p_nom[k] + float(rng.uniform(-0.3, 0.3)) * half
        p_data[k] = float(np.clip(val, lo, hi))
    print(f"[scoring] p_data = {p_data}")

    # meas_noise_std=0 here matches the synthesis batch in
    # acc_f8_formation_16d.py: the data-consistent set used by the
    # score in paper Eq. (3) is defined for noise-free input/output
    # data. Tracking simulations elsewhere use non-zero noise as a
    # sensor robustness test (separate from data-consistency).
    print("[scoring] running batch rollout (L=500) ...")
    batch = f8.base.simulate_batch_data(
        p_data, bsyn, L=500, excite_scale=0.25, meas_noise_std=0.0,
        K_fb=K_nom_internal, p_lqr=p_nom,
    )
    batch["S_c"] = f8.worst_case_S_matrix(bounds, syn)
    Psi = f8.base.build_psi_data(batch, bsyn)

    verts_all = f8.base.build_all_vertices_and_scores(
        bounds, bsyn, Psi,
        score_mode="lambda_max", max_vertices=64, seed=syn.seed + 1234,
    )
    # Replicate 16 -> 32 if actuator_gain_uncertainty > 0
    eps_design = float(syn.actuator_gain_uncertainty)
    if eps_design > 0:
        scale = 1.0 - eps_design
        extra = []
        for v in verts_all:
            v_low = dict(v)
            v_low["A"] = np.asarray(v["A"]).copy()
            v_low["B"] = np.asarray(v["B"]) * scale
            v_low["S"] = np.asarray(v["S"]).copy()
            v_low["C"] = np.asarray(v["C"]).copy()
            v_low["D"] = np.asarray(v["D"]).copy()
            v_low["Delta"] = np.block([[v_low["A"], v_low["B"]],
                                       [v_low["C"], v_low["D"]]])
            v_low["s"] = float(v["s"])
            v_low["p"] = dict(v["p"])
            v_low["p"]["_act_scale"] = float(scale)
            extra.append(v_low)
        for v in verts_all:
            v["p"] = dict(v["p"])
            v["p"]["_act_scale"] = 1.0
        verts_all = list(verts_all) + extra

    verts_norm, score_diag = f8.base.compute_si_from_vi(
        verts_all, 500, n_hc=8, q_scale=0.9,
    )

    # Reproduce the normalisation sigma used inside compute_si_from_vi so
    # the Fig 1 grid can be mapped to the same [0, 1] scale as the
    # scatter s_i: p = max(0, raw - tau_eff) / L;
    # sigma = 0.9-quantile of p[p>0]; s = 1 - exp(-p/sigma).
    raw_s_all = np.array([float(v["s"]) for v in verts_all], dtype=float)
    tau_eff = float(score_diag.get("tau_effective", 0.0))
    p_arr = np.maximum(0.0, raw_s_all - tau_eff) / 500.0
    p_pos = p_arr[p_arr > 1e-18]
    sigma_norm = float(np.quantile(p_pos, 0.9)) if p_pos.size else 0.0
    print(f"[scoring] normalisation  tau_eff={tau_eff:.3e}  sigma={sigma_norm:.3e}")

    # Run ICE to identify the Proposed active set, with the same settings
    # used by acc_f8_formation_16d.synthesize_f8_controllers:
    # K_budget=16, force_include_hard_core=True, and no downstream
    # force_top_n_hard_core because compute_si_from_vi(n_hc=8) already
    # materialises the hard core upstream. Mismatched settings here would
    # produce an active-set mask that differs from the one underlying the
    # headline Proposed gain in the paper, so Fig. 1 / score-heatmap
    # supplementary would no longer reflect the deployed controller.
    print("[scoring] running ICE active-set selection ...")
    verts_common, sol_prop = f8.base.iterative_constraint_exchange(
        all_vertices=verts_norm, syn=bsyn, K_budget=16,
        max_rounds=3, viol_tol=1e-4, seed=syn.seed + 7777,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        enforce_perf_all_vertices=syn.enforce_perf_all_vertices,
        verbose=False, bounds=bounds,
        force_include_hard_core=True,
    )

    # Gather arrays.
    keys = list(bounds.__dataclass_fields__.keys())
    params = np.array(
        [[float(v["p"].get(k, np.nan)) for k in keys] for v in verts_norm],
        dtype=float,
    )
    act_sc = np.array(
        [float(v["p"].get("_act_scale", 1.0)) for v in verts_norm], dtype=float)
    scores = np.array([float(v["s"]) for v in verts_norm], dtype=float)
    active = np.zeros(len(verts_norm), dtype=bool)
    active_ids = {id(v) for v in verts_common}
    for i, v in enumerate(verts_norm):
        if id(v) in active_ids:
            active[i] = True
    print(f"[scoring] vertex count = {len(verts_norm)}  active = {int(active.sum())}"
          f"  score range = [{scores.min():.4f}, {scores.max():.4f}]")

    # ---- Grid-sampled consistency-score fields for Fig 1 heatmap ----
    # All C(4, 2) = 6 parameter-pair projections are computed; each cut
    # holds the out-of-plane parameters at p_data (the data-generating
    # plant) so that s(p_data) = 0 sits in every panel and the
    # surrounding hard-core / soft-shell structure is directly comparable
    # across cuts. Each grid is 30x30 with a 20% margin around the prior
    # box. Per-cell cost: one A(p), B(p) build + one lambda_max score.
    Cc, Dc = f8.build_performance_matrices(bsyn)
    pair_names = [
        ("za_v",    "m_alpha"),
        ("za_v",    "m_q"),
        ("za_v",    "m_de"),
        ("m_alpha", "m_q"),
        ("m_alpha", "m_de"),
        ("m_q",     "m_de"),
    ]
    heatmap_grids: dict = {}
    for x_name, y_name in pair_names:
        x_lo, x_hi = getattr(bounds, x_name)
        y_lo, y_hi = getattr(bounds, y_name)
        mx, my = 0.2 * (x_hi - x_lo), 0.2 * (y_hi - y_lo)
        xg = np.linspace(x_lo - mx, x_hi + mx, 30)
        yg = np.linspace(y_lo - my, y_hi + my, 30)
        X, Y = np.meshgrid(xg, yg)
        Z = np.zeros_like(X)
        print(f"[scoring] sampling 30x30 grid on ({x_name},{y_name}) ...")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                p_tmp = dict(p_data)
                p_tmp[x_name] = float(X[i, j])
                p_tmp[y_name] = float(Y[i, j])
                A, B, _ = f8.build_vertex_matrices(p_tmp, syn.Ts)
                Delta = np.block([[A, B], [Cc, Dc]])
                Z[i, j] = float(
                    f8.base.compute_vertex_score_scalar(
                        Delta, Psi, mode="lambda_max")
                )
        # Same [0,1] normalisation as the discrete vertex s_i so the
        # grid colour scale and the scatter colour scale agree.
        p_grid = np.maximum(0.0, Z - tau_eff) / 500.0
        if sigma_norm > 0:
            Z = 1.0 - np.exp(-p_grid / sigma_norm)
        else:
            Z = np.zeros_like(Z)
        Z = np.clip(Z, 0.0, 1.0)
        heatmap_grids[(x_name, y_name)] = (X, Y, Z, x_lo, x_hi, y_lo, y_hi)
        print(f"[scoring]   range=[{Z.min():.3f}, {Z.max():.3f}]")

    # Merge into the existing NPZ (overwrite in place so paper / figure
    # generation code can read one unified file).
    npz_path = HERE / "acc_f8_formation_results" / "acc_f8_formation_results.npz"
    with np.load(npz_path, allow_pickle=True) as old:
        merged = {k: old[k] for k in old.files}
    merged["vertex_param_keys"] = np.array(keys)
    merged["vertex_params"] = params
    merged["vertex_act_scales"] = act_sc
    merged["vertex_scores"] = scores
    merged["vertex_active_mask"] = active.astype(np.int8)
    merged["p_data_source"] = np.array(list(p_data.values()), dtype=float)
    # Multi-pair heatmap fields. Naming: pair_<x>_<y>_<X|Y|Z|xb|yb>.
    # Plus an index list of stored pairs so the figure code can iterate.
    pair_index = []
    # Drop legacy single-pair keys from earlier runs to avoid confusion.
    for k in list(merged.keys()):
        if k.startswith("heatmap_") or k.startswith("pair_"):
            del merged[k]
    for (x_name, y_name), (X, Y, Z, x_lo, x_hi, y_lo, y_hi) in heatmap_grids.items():
        tag = f"pair_{x_name}_{y_name}"
        merged[f"{tag}_X"]  = X
        merged[f"{tag}_Y"]  = Y
        merged[f"{tag}_Z"]  = Z
        merged[f"{tag}_xb"] = np.array([x_lo, x_hi], dtype=float)
        merged[f"{tag}_yb"] = np.array([y_lo, y_hi], dtype=float)
        pair_index.append(f"{x_name}|{y_name}")
    merged["heatmap_pair_index"] = np.array(pair_index)

    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **merged)
    os.replace(tmp_path, npz_path)
    print(f"[scoring] augmented NPZ: {npz_path}")


if __name__ == "__main__":
    main()
