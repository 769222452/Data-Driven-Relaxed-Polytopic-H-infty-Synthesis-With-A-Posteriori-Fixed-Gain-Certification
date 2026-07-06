#!/usr/bin/env python3
"""
PDL-Hinf decay-rate ablation study.

Goal
----
The main pipeline (acc_f8_formation_16d.py) runs PDL-Hinf
(Daafouz-Bernussou 2001) with the default decay_rate = 0.95. This
supplementary script checks whether tighter decay-rate choices change
the corner-sweep and Monte Carlo diagnostics.

The script re-synthesises PDL-Hinf for three decay levels and re-runs
the same corner-sweep plus 100-trial Monte Carlo battery. The DCCVR row
is loaded from the main NPZ as an unchanged reference (same seed = 26).

Decay levels swept
------------------
- 0.95  : default in F8SynthesisParams (baseline already in main NPZ)
- 0.90  : moderate tightening
- 0.85  : aggressive tightening

For each decay we report
- gamma_syn_diag   (solver-returned PDL-Hinf synthesis diagnostic)
- ||K||   (Frobenius norm of the controller)
- max|K|  (peak gain element, proxy for actuator-effort risk)
- corner-sweep unstable-rate  (out of 16 corners, no noise, no lag)
- Monte-Carlo RMSE median / p90  (100 uniform-in-box trials, IMU noise)
- Monte-Carlo unstable-rate

The comparison row for DCCVR is loaded from the main NPZ unchanged.

Outputs
-------
- acc_f8_formation_results/pdl_hinf_decay_ablation/F8_pdl_decay_ablation.csv
- prints summary table to stdout.

Usage
-----
    python -X utf8 _pdl_hinf_decay_ablation.py
"""

from __future__ import annotations

import csv
import dataclasses
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Bootstrap: acc_f8_formation_16d already loads keee-st internally.
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import acc_f8_formation_16d as f8  # noqa: E402


# ----------------------------------------------------------------------
# 1. Re-run synthesis ONCE to obtain verts_norm + Proposed K + default
#    PDL-Hinf K. We pin the same seed used by the main pipeline to keep
#    the ablation comparable to the publish-ready run.
# ----------------------------------------------------------------------
def _run_baseline_synthesis() -> Dict[str, Any]:
    bounds = f8.F8ParamBounds()
    syn = f8.F8SynthesisParams(fig_out_dir="acc_f8_formation_results")
    print(f"\n[ABLATION] running baseline synthesis with seed={syn.seed}, "
          f"decay_rate={syn.decay_rate}")
    out = f8.synthesize_f8_controllers(syn, bounds, verbose=True)
    return dict(out=out, syn=syn, bounds=bounds)


# ----------------------------------------------------------------------
# 2. Re-synthesise PDL-Hinf only for a given decay_rate, reusing the
#    32-vertex verts_norm built in step 1 so that all three decay
#    levels see the *exact same* parameter & actuator-gain polytope.
# ----------------------------------------------------------------------
def _resynthesize_pdl_hinf(syn: f8.F8SynthesisParams,
                           verts_norm: List[Dict[str, Any]],
                           decay_rate: float) -> Dict[str, Any]:
    syn_decay = dataclasses.replace(syn, decay_rate=float(decay_rate))
    bsyn = f8.f8_to_base_syn(syn_decay)
    print(f"\n[ABLATION] resynthesising PDL-Hinf at decay_rate={decay_rate:.3f} "
          f"over {len(verts_norm)} vertices ...")
    sol = f8.solve_pdl_hinf_mosek(
        verts_norm, bsyn, verbose=False, num_threads=0, max_iters=80,
        rel_gap=1e-4, time_limit_sec=1800, w_gamma=syn.w_gamma,
    )
    success = bool(sol.get("success", False))
    if not success:
        print(f"  [WARN] PDL-Hinf decay={decay_rate:.3f} infeasible "
              f"(status={sol.get('solver_problem_status', ['?'])})")
        return dict(K=None, gamma=float("inf"), success=False, sol=sol,
                    decay_rate=float(decay_rate))
    K = np.asarray(sol["K"], dtype=float)
    gamma2 = float(np.asarray(sol["gamma2"]).reshape(-1)[0])
    gamma = float(np.sqrt(max(gamma2, 0.0)))
    print(f"  done: gamma_syn_diag={gamma:.4f}, ||K||={np.linalg.norm(K):.3f}, "
          f"max|K|={float(np.max(np.abs(K))):.3f}")
    return dict(K=K, gamma=gamma, success=True, sol=sol,
                decay_rate=float(decay_rate))


# ----------------------------------------------------------------------
# 3. Build a synthetic "out" dict so we can reuse the existing
#    corner-sweep / MC evaluators without modifying main-pipeline code.
#    We monkey-patch _f8_controllers_list so that the standard
#    evaluators iterate over our ablation set instead.
# ----------------------------------------------------------------------
_ABLATION_PAIRS: List[Tuple[str, np.ndarray]] = []


def _patched_controllers_list(out: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    return list(_ABLATION_PAIRS)


def _evaluate_set(syn: f8.F8SynthesisParams,
                  bounds: f8.F8ParamBounds,
                  pairs: List[Tuple[str, np.ndarray]],
                  out_dummy: Dict[str, Any],
                  out_subdir: str) -> Dict[str, Dict[str, Any]]:
    """Run corner sweep (no-noise) + MC (100 uniform trials) on `pairs`
    and return a name -> stats dict for each metric block."""
    global _ABLATION_PAIRS
    _ABLATION_PAIRS = pairs

    # Patch the controllers-list helper used inside corner_sweep / MC.
    original_list = f8._f8_controllers_list
    f8._f8_controllers_list = _patched_controllers_list
    try:
        sweep_dir = os.path.join(syn.fig_out_dir, out_subdir)
        os.makedirs(sweep_dir, exist_ok=True)

        # NOTE: seconds=10.0 mirrors main_acc_f8_formation_quick() Phase 5a
        # so the ablation's corner-sweep numbers are directly comparable to
        # the published ones in acc_f8_formation_results/corner_sweep.
        print(f"\n[ABLATION] {out_subdir}: 16-corner sweep (seconds=10.0) ...")
        corner_summary = f8.run_f8_corner_sweep(
            out_dummy, syn, bounds,
            seconds=10.0,
            sweep_subdir=os.path.join(out_subdir, "corner_sweep"),
        )
        print(f"\n[ABLATION] {out_subdir}: Monte Carlo ({syn.mc_trials} trials) ...")
        mc_summary = f8.run_f8_monte_carlo_evaluation(
            out_dummy, syn, bounds,
            fig_out_dir=os.path.join(syn.fig_out_dir, out_subdir),
        )
    finally:
        f8._f8_controllers_list = original_list

    return dict(corner=corner_summary, mc=mc_summary)


# ----------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------
def main() -> None:
    bounds = f8.F8ParamBounds()
    syn = f8.F8SynthesisParams(fig_out_dir="acc_f8_formation_results")
    cache_path = os.path.join(syn.fig_out_dir, "pdl_hinf_decay_ablation",
                              "_K_cache.npz")

    if os.path.exists(cache_path):
        print(f"[ABLATION] loading K cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=False)
        K_proposed = cache["K_proposed"]
        gamma_proposed_val = float(cache["gamma_proposed"])
        decay_levels = [0.95, 0.90, 0.85]
        pdl_results: Dict[float, Dict[str, Any]] = {}
        for d in decay_levels:
            key = f"K_pdl_d{int(round(d * 100))}"
            pdl_results[d] = dict(
                K=cache[key],
                gamma=float(cache[f"gamma_pdl_d{int(round(d * 100))}"]),
                success=True, sol=None, decay_rate=d,
            )
    else:
        base_run = _run_baseline_synthesis()
        out = base_run["out"]
        # rebind syn / bounds to the ones used inside synthesis (same defaults).
        syn = base_run["syn"]
        bounds = base_run["bounds"]
        verts_norm = out["vertices_all"]

        K_proposed = np.asarray(out["K_proposed"], dtype=float)
        gamma_proposed_val = float(np.asarray(out["gamma_proposed"]).reshape(-1)[0])
        K_pdl_default = np.asarray(out["K_pdl_hinf"], dtype=float)
        gamma_pdl_default = float(np.asarray(out["gamma_pdl_hinf"]).reshape(-1)[0])

        decay_levels = [0.95, 0.90, 0.85]
        pdl_results = {}
        pdl_results[0.95] = dict(
            K=K_pdl_default, gamma=gamma_pdl_default,
            success=bool(out.get("pdl_hinf_status", "") == "feasible"
                         or np.isfinite(gamma_pdl_default)),
            sol=out.get("sol_pdl_hinf"), decay_rate=0.95,
        )
        for d in [0.90, 0.85]:
            pdl_results[d] = _resynthesize_pdl_hinf(syn, verts_norm, d)

        # Save cache so re-running corner_sweep/MC alone is fast.
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_payload = dict(
            K_proposed=K_proposed,
            gamma_proposed=np.array([gamma_proposed_val]),
        )
        for d in decay_levels:
            tag = int(round(d * 100))
            cache_payload[f"K_pdl_d{tag}"] = pdl_results[d]["K"]
            cache_payload[f"gamma_pdl_d{tag}"] = np.array([pdl_results[d]["gamma"]])
        np.savez(cache_path, **cache_payload)
        print(f"[ABLATION] cached synthesised K matrices to {cache_path}")

    # --- Build the controller list for the ablation evaluation.
    eval_pairs: List[Tuple[str, np.ndarray]] = [("Proposed", K_proposed)]
    for d in decay_levels:
        rec = pdl_results[d]
        if not rec["success"] or rec["K"] is None:
            print(f"\n[ABLATION] skipping PDL-Hinf decay={d:.3f} (infeasible)")
            continue
        eval_pairs.append((f"PDL-Hinf(d={d:.2f})", rec["K"]))

    # --- Build a dummy `out` that exposes all required K_* keys.
    out_dummy: Dict[str, Any] = dict(
        K_proposed=K_proposed,
        K_norelax=K_proposed,   # unused (overridden by patched list)
        K_qs_hinf=K_proposed,   # unused
        K_pdl_hinf=K_proposed,  # unused
        syn=syn, bounds=bounds,
    )

    eval_summary = _evaluate_set(
        syn, bounds, eval_pairs, out_dummy,
        out_subdir="pdl_hinf_decay_ablation",
    )
    corner_summary = eval_summary["corner"]
    mc_summary = eval_summary["mc"]

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for name, K in eval_pairs:
        if name.startswith("PDL-Hinf"):
            d = float(name.split("d=")[1].rstrip(")"))
            rec = pdl_results[d]
            gamma = rec["gamma"]
        else:
            d = float("nan")
            gamma = gamma_proposed_val
        K_norm = float(np.linalg.norm(K))
        K_max = float(np.max(np.abs(K)))
        cs = corner_summary.get(name, {})
        mc = mc_summary.get(name, {})
        rows.append(dict(
            Controller=name,
            decay_rate=d,
            gamma_syn_diag=gamma,
            K_norm=K_norm,
            K_max_abs=K_max,
            corner_unstable=float(cs.get("unstable_rate", float("nan"))),
            corner_rmse_median=float(cs.get("rmse_median", float("nan"))),
            corner_rmse_max=float(cs.get("rmse_max", float("nan"))),
            corner_peak_max=float(cs.get("peak_max", float("nan"))),
            mc_rmse_median=float(mc.get("RMSE_theta_median", float("nan"))),
            mc_rmse_p90=float(mc.get("RMSE_theta_p90", float("nan"))),
            mc_unstable=float(mc.get("unstable_rate", float("nan"))),
        ))

    csv_path = os.path.join(syn.fig_out_dir, "pdl_hinf_decay_ablation",
                            "F8_pdl_decay_ablation.csv")
    cols = ["Controller", "decay_rate", "gamma_syn_diag", "K_norm", "K_max_abs",
            "corner_unstable", "corner_rmse_median", "corner_rmse_max",
            "corner_peak_max", "mc_rmse_median", "mc_rmse_p90", "mc_unstable"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(cols)
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, str):
                    cells.append(v)
                elif isinstance(v, float) and not np.isfinite(v):
                    cells.append("inf")
                else:
                    cells.append(f"{float(v):.6g}")
            w.writerow(cells)
    print(f"\n[ABLATION] wrote {csv_path}")

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(" PDL-Hinf decay-rate ablation summary")
    print("=" * 100)
    header = (f"{'Controller':<22} {'decay':>6} {'gamma_syn':>9} "
              f"{'||K||':>8} {'max|K|':>8} "
              f"{'cornUnst':>9} {'cornRMSEm':>10} {'cornPeakMax':>12} "
              f"{'mcRMSEm':>9} {'mcRMSEp90':>10} {'mcUnst':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        d_str = "  -- " if not np.isfinite(r["decay_rate"]) else f"{r['decay_rate']:.2f}"
        print(f"{r['Controller']:<22} {d_str:>6} {r['gamma_syn_diag']:>9.4f} "
              f"{r['K_norm']:>8.3f} {r['K_max_abs']:>8.3f} "
              f"{r['corner_unstable']:>8.2%} {r['corner_rmse_median']:>10.4g} "
              f"{r['corner_peak_max']:>12.4g} "
              f"{r['mc_rmse_median']:>9.4g} {r['mc_rmse_p90']:>10.4g} "
              f"{r['mc_unstable']:>7.2%}")
    print("=" * 100)


if __name__ == "__main__":
    main()
