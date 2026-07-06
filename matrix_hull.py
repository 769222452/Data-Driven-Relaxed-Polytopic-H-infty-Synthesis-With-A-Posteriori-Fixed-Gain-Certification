"""Barycentric matrix-hull membership residual (Remark 1 / Corollary 1).

Given the audit vertex set

    V_{C_D} = {(A_j, B_j, tilde S_j)}_{j=1..n_V}

and a candidate ZOH-discretised plant tuple

    M_star = (A(p_star), B(p_star), tilde S(p_star))

Corollary 1 requires M_star in conv(V_{C_D}) for the post-audit
certificate gamma_core to transfer to the data-generating plant.
This module solves the small barycentric least-squares problem

    min_{lambda >= 0, 1^T lambda = 1}
        || [vec(A_j); vec(B_j); vec(tilde S_j)] lambda - vec(M_star) ||_2

and reports absolute / relative residual and the argmin lambda.

Interface:
    matrix_hull_residual(A_star, B_star, S_tilde_star, audit_verts)

where `audit_verts` is a list of dicts with keys "A", "B", "S_tilde".
Use `scaled_disturbance_matrix` from acc_f8_formation_16d.py to build
S_tilde entries consistently with the audit SDP (tilde S = d_cert * S).

The SLSQP solver is adequate for small n_V (<= 64) and is pure-SciPy
(no MOSEK round-trip), so this check runs in a few milliseconds.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import scipy.optimize as opt


def _flatten_tuple(A: np.ndarray, B: np.ndarray, S_tilde: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.asarray(A, dtype=float).ravel(),
        np.asarray(B, dtype=float).ravel(),
        np.asarray(S_tilde, dtype=float).ravel(),
    ])


def matrix_hull_residual(
        A_star: np.ndarray,
        B_star: np.ndarray,
        S_tilde_star: np.ndarray,
        audit_verts: List[Dict[str, Any]],
        *,
        tol: float = 1e-6,
        abs_tol: float = None,
        rel_tol: float = None,
        inf_tol: float = None,
        verbose: bool = False,
) -> Dict[str, Any]:
    """Barycentric least-squares hull-membership check (cf. Remark 1).

    Parameters
    ----------
    A_star, B_star, S_tilde_star
        ZOH-discretised matrices of the data-generating plant; S_tilde
        is the *normalised-disturbance* matrix tilde S = d_cert * S.
    audit_verts
        List of dicts with keys "A", "B", "S_tilde" (already scaled
        the same way as the audit SDP -- if you only have raw S the
        caller should pre-multiply by d_cert).
    tol
        Backward-compatible single threshold for the L2 residual; sets
        ``abs_tol`` if the latter is not supplied. Kept so existing
        callers using ``tol=...`` keep working.
    abs_tol, rel_tol, inf_tol
        Independent thresholds for the L2 residual, the relative L2
        residual and the entrywise infinity-norm residual respectively.
        Each defaults to ``tol`` (or 1e-6 if ``tol`` is also unset);
        choose looser tolerances for noisy reproducible runs.

    Returns
    -------
    dict with:
        lambdas             : barycentric weights (n_V,)
        residual_abs        : ||V lambda - vec(M_star)||_2
        residual_rel        : residual_abs / max(1, ||vec(M_star)||_2)
        residual_inf        : ||V lambda - vec(M_star)||_inf  (entrywise)
        inside_hull         : alias of inside_hull_abs (back-compat)
        inside_hull_abs     : residual_abs < abs_tol
        inside_hull_rel     : residual_rel < rel_tol
        inside_hull_inf     : residual_inf < inf_tol
        active_vertices     : indices with lambda > 1e-6
        solver_status       : SciPy message
        n_vertices          : len(audit_verts)
        abs_tol, rel_tol, inf_tol : tolerances used
    """
    abs_tol = float(abs_tol) if abs_tol is not None else float(tol)
    rel_tol = float(rel_tol) if rel_tol is not None else float(tol)
    inf_tol = float(inf_tol) if inf_tol is not None else float(tol)

    if len(audit_verts) == 0:
        return dict(
            lambdas=np.array([]),
            residual_abs=float("inf"),
            residual_rel=float("inf"),
            residual_inf=float("inf"),
            inside_hull=False,
            inside_hull_abs=False,
            inside_hull_rel=False,
            inside_hull_inf=False,
            active_vertices=np.array([], dtype=int),
            solver_status="empty_vertex_set",
            n_vertices=0,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            inf_tol=inf_tol,
        )

    m_star = _flatten_tuple(A_star, B_star, S_tilde_star)
    V = np.stack([
        _flatten_tuple(v["A"], v["B"], v["S_tilde"]) for v in audit_verts
    ], axis=1)  # shape (d, n_V)
    nV = V.shape[1]

    def obj(lam: np.ndarray) -> float:
        r = V @ lam - m_star
        return float(0.5 * r @ r)

    def obj_grad(lam: np.ndarray) -> np.ndarray:
        return V.T @ (V @ lam - m_star)

    # Warm start at centroid (stable for small problems)
    x0 = np.ones(nV) / nV
    res = opt.minimize(
        obj, x0, jac=obj_grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * nV,
        constraints={"type": "eq",
                     "fun": lambda lam: np.sum(lam) - 1.0,
                     "jac": lambda lam: np.ones(nV)},
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    lam = np.clip(res.x, 0.0, None)
    s = float(np.sum(lam))
    if s > 0.0:
        lam = lam / s

    resid = V @ lam - m_star
    resid_abs = float(np.linalg.norm(resid))
    resid_rel = resid_abs / max(1.0, float(np.linalg.norm(m_star)))
    # Entrywise infinity-norm residual: max abs deviation across all
    # entries of (A,B,tilde S). Reported alongside the L2 residual so a
    # single component blow-up is not hidden by an averaged 2-norm.
    resid_inf = float(np.max(np.abs(resid)))
    active = np.where(lam > 1e-6)[0]

    inside_abs = bool(resid_abs < abs_tol)
    inside_rel = bool(resid_rel < rel_tol)
    inside_inf = bool(resid_inf < inf_tol)

    if verbose:
        print(f"[matrix_hull] n_V={nV}, residual_abs={resid_abs:.3e}, "
              f"residual_rel={resid_rel:.3e}, "
              f"residual_inf={resid_inf:.3e}, "
              f"active_vertices={active.tolist()}, "
              f"inside_hull_abs={inside_abs}, "
              f"inside_hull_rel={inside_rel}, "
              f"inside_hull_inf={inside_inf}")

    return dict(
        lambdas=lam,
        residual_abs=resid_abs,
        residual_rel=resid_rel,
        residual_inf=resid_inf,
        # ``inside_hull`` is kept as the L2 alias so existing callers
        # that key on ``inside_hull`` continue to work.
        inside_hull=inside_abs,
        inside_hull_abs=inside_abs,
        inside_hull_rel=inside_rel,
        inside_hull_inf=inside_inf,
        active_vertices=active,
        solver_status=str(res.message),
        n_vertices=nV,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        inf_tol=inf_tol,
    )


if __name__ == "__main__":  # pragma: no cover -- quick smoke test
    rng = np.random.default_rng(0)
    # 4 random 3x3 matrices, M_star = known convex combination
    verts = [
        {"A": rng.standard_normal((3, 3)),
         "B": rng.standard_normal((3, 2)),
         "S_tilde": rng.standard_normal((3, 2))}
        for _ in range(4)
    ]
    lam_true = np.array([0.1, 0.3, 0.5, 0.1])
    A_s = sum(l * v["A"] for l, v in zip(lam_true, verts))
    B_s = sum(l * v["B"] for l, v in zip(lam_true, verts))
    S_s = sum(l * v["S_tilde"] for l, v in zip(lam_true, verts))
    out = matrix_hull_residual(A_s, B_s, S_s, verts, verbose=True)
    print(f"lambda_recovered = {out['lambdas']}")
    print(f"lambda_true      = {lam_true}")
    print(f"inside_hull      = {out['inside_hull']} (expected True)")
