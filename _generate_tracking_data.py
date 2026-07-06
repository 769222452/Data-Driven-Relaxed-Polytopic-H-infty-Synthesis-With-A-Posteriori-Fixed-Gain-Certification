"""Generate pitch-tracking trajectories for Fig 4 of the paper.

The main simulation pipeline exports only aggregate metrics (RMSE /
peak / rho), not raw state histories, which means the tracking figure
cannot be rendered from ``acc_f8_formation_results.npz`` alone. This
script re-simulates the controllers on:

  (a) the nominal plant            -> key prefix ``nominal``
  (b) the hard-core synthesis stress corner (max rho_OL among the
      score-supported s=0 vertices)  -> key prefix ``hc_worst``
  (c) the adversarial worst corner (max rho_OL over the physical
      aerodynamic corners, regardless of data consistency; out of the
      audit hull)
      -> key prefix ``worst``

using the identical reference and disturbance signals as the main
pipeline (seeds match), and dumps the per-aircraft pitch histories
into a companion NPZ ``tracking_histories.npz`` for the figure
generator.

Outputs:
  acc_f8_formation_results/tracking_histories.npz with fields
    t            : (n+1,)      simulation time [s]
    theta_ref    : (4, n)      pitch reference per aircraft [rad]
    <plant>_<ctrl>_theta : (4, n+1)  realised pitch per aircraft [rad]
  for plant in {nominal, hc_worst, worst} and ctrl in
    {Proposed, NoRelax-ProposedActive, QS-Hinf, PDL-Hinf, RobustLQR}.

Runtime: ~3 s (15 simulations * 10 s * 100 Hz).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import acc_f8_formation_16d as f8


THETA_INDICES = [2, 5, 8, 11]       # per-aircraft pitch rows in x


def main() -> None:
    syn = f8.F8SynthesisParams()
    bounds = f8.F8ParamBounds()
    seconds = 10.0

    # Reference + disturbance: identical seeds to the main pipeline
    # (see run_full_f8_pipeline, seed_offset=800).
    _, theta_refs = f8.build_f8_reference_trajectories(syn, seconds)
    x_ref = f8.build_f8_state_reference(theta_refs, syn.Ts)
    dbar, _ = f8.build_f8_disturbance_profile(syn, seconds, seed_offset=800)
    x0 = f8.f8_initial_state(scale=0.3)

    # Load K matrices + p_nominal / p_worst from the main NPZ.
    npz_path = HERE / "acc_f8_formation_results" / "acc_f8_formation_results.npz"
    D = np.load(npz_path, allow_pickle=True)
    controllers = [
        ("Proposed",    D["K_proposed"]),
        ("NoRelax-ProposedActive", D["K_norelax"]),
        ("QS-Hinf",     D["K_qs_hinf"]),
        ("PDL-Hinf",    D["K_pdl_hinf"]),
        ("RobustLQR",   D["K_robust_lqr"]),
    ]
    # Core-CQLF-Hinf: only include when synthesis succeeded (not a NaN fallback).
    # Support both old (shrinkcd_*) and new (core_cqlf_*) NPZ key names.
    _st_key = "core_cqlf_status" if "core_cqlf_status" in D.files else "shrinkcd_status"
    _k_key  = "K_core_cqlf_hinf" if "K_core_cqlf_hinf" in D.files else "K_shrinkcd_hinf"
    _sc_status = str(np.asarray(D[_st_key]).reshape(-1)[0]) if _st_key in D.files else ""
    if _sc_status == "feasible" and _k_key in D.files:
        controllers.append(("Core-CQLF-Hinf", D[_k_key]))
    p_nominal_arr = np.asarray(D["p_nominal"], dtype=float)
    p_hc_arr      = np.asarray(D["p_hc_worst"], dtype=float)
    p_worst_arr   = np.asarray(D["p_worst"],   dtype=float)
    keys = list(bounds.__dataclass_fields__.keys())
    p_nominal = {k: float(p_nominal_arr[i]) for i, k in enumerate(keys)}
    p_hc      = {k: float(p_hc_arr[i])      for i, k in enumerate(keys)}
    p_worst   = {k: float(p_worst_arr[i])   for i, k in enumerate(keys)}

    print(f"[tracking] p_nominal = {p_nominal}")
    print(f"[tracking] p_hc      = {p_hc}")
    print(f"[tracking] p_worst   = {p_worst}")

    out: dict = {
        "theta_ref": np.asarray(theta_refs, dtype=float),
    }

    for plant_name, p in [("nominal", p_nominal),
                          ("hc_worst", p_hc),
                          ("worst", p_worst)]:
        print(f"\n[tracking] plant={plant_name}")
        for ctrl_name, K in controllers:
            sim = f8.simulate_f8_tracking(
                K, p, syn, seconds, dbar, x_ref,
                x0=x0, meas_noise_std=1e-3,
            )
            x = np.asarray(sim["x"], dtype=float)       # (16, n+1)
            theta_hist = x[THETA_INDICES, :]             # (4, n+1)
            if "t" not in out:
                out["t"] = np.asarray(sim["t"], dtype=float)
            peak = float(np.max(np.abs(theta_hist)))
            print(f"  {ctrl_name:12s}  peak|theta| = {peak:10.3g}")
            out[f"{plant_name}_{ctrl_name}_theta"] = theta_hist

    out_path = HERE / "acc_f8_formation_results" / "tracking_histories.npz"
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    print(f"\n[tracking] wrote {out_path}")


if __name__ == "__main__":
    main()
