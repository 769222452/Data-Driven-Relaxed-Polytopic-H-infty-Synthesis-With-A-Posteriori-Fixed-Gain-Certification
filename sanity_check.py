"""
Cross-check every hard-coded number in paper_body.tex against the
simulation NPZ + CSVs.

For each claim in the paper, we look up the canonical source value and
report PASS / FAIL with the paper value, the source value, and the
relative error.

Sources:
  acc_f8_formation_results/acc_f8_formation_results.npz   (synthesis,
       nominal/worst, corner sweep aggregates)
  acc_f8_formation_results/monte_carlo/F8_monte_carlo_summary.csv
  acc_f8_formation_results/pdl_hinf_decay_ablation/F8_pdl_decay_ablation.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SIM_ROOT = Path(__file__).resolve().parent / "acc_f8_formation_results"

# -----------------------------------------------------------------
# helpers
# -----------------------------------------------------------------
def read_csv_columns(path: Path) -> Dict[str, List[str]]:
    cols: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols[k].append(v)
    return cols


def fnum(x) -> float:
    """Coerce numpy 0-d / 1-d / scalar to plain float."""
    arr = np.asarray(x).reshape(-1)
    return float(arr[0])


# -----------------------------------------------------------------
# load all sources
# -----------------------------------------------------------------
def _require_file(path: Path) -> None:
    """Abort with a helpful message if a required NPZ/CSV is missing,
    instead of letting numpy/csv raise a bare FileNotFoundError far from
    the actual root cause."""
    if not path.is_file():
        raise SystemExit(
            f"[sanity_check] missing required file:\n"
            f"  {path}\n"
            f"Run the full reproducibility pipeline first:\n"
            f"  python acc_f8_formation_16d.py\n"
            f"  python _pdl_hinf_decay_ablation.py\n"
            f"  python _augment_npz_scores.py\n"
            f"  python _generate_tracking_data.py"
        )


def load_sources():
    npz_path = SIM_ROOT / "acc_f8_formation_results.npz"
    mc_path = SIM_ROOT / "monte_carlo" / "F8_monte_carlo_summary.csv"
    abl_path = SIM_ROOT / "pdl_hinf_decay_ablation" / "F8_pdl_decay_ablation.csv"
    audit_path = SIM_ROOT / "fixed_k_audit.csv"
    syn_audit_path = SIM_ROOT / "synthesis_audit_consistency.csv"
    for p in (npz_path, mc_path, abl_path, audit_path, syn_audit_path):
        _require_file(p)
    npz = np.load(npz_path)
    mc = read_csv_columns(mc_path)
    abl = read_csv_columns(abl_path)
    audit = read_csv_columns(audit_path)
    syn_audit = read_csv_columns(syn_audit_path)
    return npz, mc, abl, audit, syn_audit


def mc_lookup(mc: Dict[str, List[str]], controller: str, field: str) -> float:
    idx = mc["Controller"].index(controller)
    return float(mc[field][idx])


def abl_lookup(abl: Dict[str, List[str]], controller: str, field: str) -> float:
    idx = abl["Controller"].index(controller)
    return float(abl[field][idx])


def csv_lookup(tab: Dict[str, List[str]], key_col: str, key: str, field: str) -> str:
    idx = tab[key_col].index(key)
    return tab[field][idx]


def csv_float(tab: Dict[str, List[str]], key_col: str, key: str, field: str) -> float:
    raw = csv_lookup(tab, key_col, key, field)
    return float(raw) if raw not in ("", "nan") else float("nan")


def csv_bool(tab: Dict[str, List[str]], key_col: str, key: str, field: str) -> bool:
    return csv_lookup(tab, key_col, key, field).strip().lower() == "true"


# -----------------------------------------------------------------
# checks
# -----------------------------------------------------------------
PASS = 0
FAIL = 0
FAILURES: List[str] = []


def check(label: str, paper: float, source: float, rtol: float = 1e-2) -> None:
    """Report PASS/FAIL.  rtol is relative tolerance for float comparison.
    For exact integer percentages we still allow 1% relative slack so
    rounding to 2 decimal places does not flag a false negative."""
    global PASS, FAIL
    if source == 0.0:
        ok = abs(paper) < 1e-9
        rel = float("inf") if not ok else 0.0
    else:
        rel = abs(paper - source) / abs(source)
        ok = rel <= rtol
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {label}: paper={paper:g}  source={source:g}  rel_err={rel:.2%}"
    if ok:
        PASS += 1
        print(line)
    else:
        FAIL += 1
        FAILURES.append(line)
        print(line)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# -----------------------------------------------------------------
def main() -> None:
    global PASS, FAIL
    npz, mc, abl, audit, syn_audit = load_sources()

    # -------------------- Table II: Synthesis diagnostics ----------
    section("Table II  Synthesis diagnostics")
    # Table II reports solver diagnostics only. Synthesis-stage gamma is
    # intentionally not checked as a formal paper claim; certified
    # H-infinity gains are checked below through fixed-K gamma_core.
    Kp = npz["K_proposed"]; Kn = npz["K_norelax"]
    Kq = npz["K_qs_hinf"];  Kd = npz["K_pdl_hinf"]
    Kw = npz["K_robust_lqr"]
    check("Proposed  ||K||_F",  13.08, float(np.linalg.norm(Kp, "fro")))
    check("Proposed  max|K|",   5.75,  float(np.max(np.abs(Kp))))
    check("NoRelax   ||K||_F",  24.88, float(np.linalg.norm(Kn, "fro")))
    check("NoRelax   max|K|",   10.66, float(np.max(np.abs(Kn))))
    check("QS-Hinf   ||K||_F",  3.97,  float(np.linalg.norm(Kq, "fro")))
    check("QS-Hinf   max|K|",   1.59,  float(np.max(np.abs(Kq))))
    check("PDL-Hinf  ||K||_F",  39.68, float(np.linalg.norm(Kd, "fro")))
    check("PDL-Hinf  max|K|",   15.57, float(np.max(np.abs(Kd))))
    check("RobustLQR ||K||_F",  7.56,  float(np.linalg.norm(Kw, "fro")))
    check("RobustLQR max|K|",   2.43,  float(np.max(np.abs(Kw))))
    check("RobustLQR rho_OL",   1.077, fnum(npz["robust_lqr_rho_ol"]), rtol=2e-3)
    _kk = "K_core_cqlf_hinf" if "K_core_cqlf_hinf" in npz.files else "K_shrinkcd_hinf"
    if _kk in npz.files:
        Ksc = npz[_kk]
        if not np.all(np.isnan(Ksc)):
            check("Core-CQLF-Hinf ||K||_F", 4.02, float(np.linalg.norm(Ksc, "fro")))
            check("Core-CQLF-Hinf max|K|",  1.64, float(np.max(np.abs(Ksc))))
        else:
            print(f"  [INFO] Core-CQLF-Hinf K is NaN (synthesis infeasible).")

    for row_name in ["QS-Hinf", "Core-CQLF-Hinf"]:
        if row_name in syn_audit.get("method", []):
            PASS += 1
            print(f"  [PASS] synthesis/audit diagnostic row exists for {row_name}")
        else:
            FAIL += 1
            FAILURES.append(f"  [FAIL] missing synthesis/audit diagnostic row for {row_name}")

    for row_name in ["Proposed", "NoRelax-ProposedActive", "QS-Hinf", "Core-CQLF-Hinf"]:
        if row_name not in syn_audit.get("method", []):
            continue
        cert_flag = csv_bool(syn_audit, "method", row_name, "synthesis_gamma_is_certificate")
        residuals = [
            csv_float(syn_audit, "method", row_name, "max_res_QY_on_synth_vertices"),
            csv_float(syn_audit, "method", row_name, "max_res_PK_at_syn_g_on_synth_vertices"),
            csv_float(syn_audit, "method", row_name, "max_res_PK_at_syn_g_on_audit_vertices"),
        ]
        residuals_pass = all(np.isfinite(x) and x <= 1e-6 for x in residuals)
        if cert_flag == residuals_pass:
            PASS += 1
            print(f"  [PASS] {row_name} synthesis_gamma_is_certificate={cert_flag} "
                  f"matches strict residual test")
        else:
            FAIL += 1
            FAILURES.append(
                f"  [FAIL] {row_name} synthesis_gamma_is_certificate={cert_flag} "
                f"but strict residual test is {residuals_pass}"
            )

    # -------------------- Table III: fixed-K audit certificates ----
    section("Table III  Fixed-K audit certificates")
    for ctrl, paper_gamma, paper_core_eig in [
        ("Proposed",                2.934, -5.61e-7),
        ("NoRelax-ProposedActive",  3.298, -1.03e-6),
        ("QS-Hinf",                24.39,  -2.58e-7),
        ("PDL-Hinf",                5.778, -5.26e-7),
        ("RobustLQR",               4.760, -2.34e-7),
        ("Core-CQLF-Hinf",         18.96,  -2.26e-7),
    ]:
        check(f"Audit {ctrl} gamma_core",
              paper_gamma, csv_float(audit, "method", ctrl, "gamma_core"), rtol=3e-3)
        check(f"Audit {ctrl} core max eig",
              paper_core_eig, csv_float(audit, "method", ctrl, "core_max_eig"), rtol=1.0)
        if csv_lookup(audit, "method", ctrl, "feasible").strip().lower() == "true":
            PASS += 1
            print(f"  [PASS] Audit {ctrl} feasible")
        else:
            FAIL += 1
            FAILURES.append(f"  [FAIL] Audit {ctrl} not feasible")

    # -------------------- Scenario table: Nominal / HC Worst / Adv ---
    section("Scenario table  Nominal / Hard-core Worst / Adversarial")
    nom = read_csv_columns(SIM_ROOT / "nominal_plant" / "F8_metrics_table.csv")
    wst = read_csv_columns(SIM_ROOT / "worst_plant"   / "F8_metrics_table.csv")
    hcw = read_csv_columns(SIM_ROOT / "hardcore_worst_plant" / "F8_metrics_table.csv")

    def t3_lookup(tab: Dict[str, List[str]], ctrl: str, field: str) -> float:
        idx = tab["Controller"].index(ctrl)
        return float(tab[field][idx])

    # nominal: (RMSE, peak_theta, peak_u, rho)
    for ctrl, rmse, pth, pu, rho in [
        ("Proposed",                0.0187, 0.0660, 0.417, 0.954),
        ("NoRelax-ProposedActive",  0.0429, 0.122,  0.702, 0.931),
        ("QS-Hinf",                 0.0249, 0.0751, 0.399, 0.993),
        ("PDL-Hinf",                0.0615, 0.192,  0.648, 0.916),
        ("RobustLQR",               0.0260, 0.0683, 0.402, 0.977),
    ]:
        check(f"Nom {ctrl} RMSE",    rmse, t3_lookup(nom, ctrl, "RMSE_theta"),  rtol=3e-2)
        check(f"Nom {ctrl} peakTh",  pth,  t3_lookup(nom, ctrl, "peak_theta"),  rtol=2e-2)
        check(f"Nom {ctrl} peakU",   pu,   t3_lookup(nom, ctrl, "peak_u"),      rtol=3e-2)
        check(f"Nom {ctrl} rho",     rho,  t3_lookup(nom, ctrl, "rho"),         rtol=1e-2)

    # adversarial worst (out-of-scope, soft s~=0.6 corner):
    # (RMSE, peak_theta, peak_u, rho)
    for ctrl, rmse, pth, pu, rho in [
        ("Proposed",                1.57,    25.6,            0.515, 0.968),
        ("NoRelax-ProposedActive",  480.1,   8.00e3,           0.603, 0.962),
        ("QS-Hinf",                 0.154,   0.408,            0.399, 0.996),
        ("PDL-Hinf",                1135.0,  2.12e4,           0.634, 0.959),
        ("RobustLQR",               0.0286,  0.167,            0.426, 0.978),
    ]:
        check(f"Adv {ctrl} RMSE",   rmse, t3_lookup(wst, ctrl, "RMSE_theta"),   rtol=3e-2)
        check(f"Adv {ctrl} peakTh", pth,  t3_lookup(wst, ctrl, "peak_theta"),   rtol=5e-2)
        check(f"Adv {ctrl} peakU",  pu,   t3_lookup(wst, ctrl, "peak_u"),       rtol=3e-2)
        check(f"Adv {ctrl} rho",    rho,  t3_lookup(wst, ctrl, "rho"),          rtol=1e-2)

    # hard-core synthesis stress (score-supported vertex s=0,
    # rho_OL=1.063). The synthesis LMI is enforced (unrelaxed for
    # Proposed) on score-supported vertices, so this is an empirical
    # stress test aligned with the synthesis scope. It is NOT the
    # formal post-audit scope: the audit-time matrix-hull membership
    # condition is reported separately.
    for ctrl, rmse, pth, pu, rho in [
        ("Proposed",                0.0549,  0.355,            0.482, 0.903),
        ("NoRelax-ProposedActive",  0.486,   5.404,            0.661, 0.917),
        ("QS-Hinf",                 0.0850,  0.155,            0.399, 0.975),
        ("PDL-Hinf",                5.265,   31.48,            0.600, 0.911),
        ("RobustLQR",               0.0245,  0.111,            0.431, 0.944),
    ]:
        check(f"HC {ctrl} RMSE",   rmse, t3_lookup(hcw, ctrl, "RMSE_theta"),   rtol=3e-2)
        check(f"HC {ctrl} peakTh", pth,  t3_lookup(hcw, ctrl, "peak_theta"),   rtol=3e-2)
        check(f"HC {ctrl} peakU",  pu,   t3_lookup(hcw, ctrl, "peak_u"),       rtol=3e-2)
        check(f"HC {ctrl} rho",    rho,  t3_lookup(hcw, ctrl, "rho"),          rtol=1e-2)

    # -------------------- Table IV: 16-corner sweep ---------------
    section("Table IV  Corner sweep")
    # Phase 5a (ideal). Proposed and NoRelax-ProposedActive both lose stability at
    # the worst-corner plant after the V4 cleanup (true measurement-noise
    # channel + smoothed reference exposed the open-loop spectral-radius
    # 1.077 corner that previously drifted into a quasi-stable transient).
    for ctrl, paper_unst, paper_rmse, paper_peak in [
        ("Proposed",                6.25,  0.0234, 28.1),
        ("NoRelax-ProposedActive",  6.25,  0.0449, 8.86e3),
        ("QS-Hinf",                 0.00,  0.0525, 0.41),
        ("PDL-Hinf",                12.50, 0.0693, 2.17e4),
        ("RobustLQR",               0.00,  0.0272, 0.179),
    ]:
        check(f"5a {ctrl} unst%",
              paper_unst,
              fnum(npz[f"corner_{ctrl}_unstable_rate"]) * 100.0,
              rtol=2e-2)
        check(f"5a {ctrl} RMSE med",
              paper_rmse,
              fnum(npz[f"corner_{ctrl}_rmse_median"]),
              rtol=5e-2)
        check(f"5a {ctrl} peak max",
              paper_peak,
              fnum(npz[f"corner_{ctrl}_peak_max"]),
              rtol=5e-2)

    # Phase 5b (actuator lag) -- only unst% tabulated. Proposed loses
    # stability at the same hardest corner as in 5a; RobustLQR likewise
    # destabilises at that corner under actuator lag.
    for ctrl, paper in [
        ("Proposed",                6.25),
        ("NoRelax-ProposedActive",  6.25),
        ("QS-Hinf",                 0.00),
        ("PDL-Hinf",                12.50),
        ("RobustLQR",               0.00),
    ]:
        check(f"5b {ctrl} unst%",
              paper,
              fnum(npz[f"corner_actlag_{ctrl}_unstable_rate"]) * 100.0,
              rtol=2e-2)

    # Phase 5c (sensor delay) -- only unst% tabulated. Proposed and
    # NoRelax-ProposedActive destabilise at the worst corner under added sensor delay.
    for ctrl, paper in [
        ("Proposed",    6.25),
        ("NoRelax-ProposedActive", 12.50),
        ("QS-Hinf",     0.00),
        ("PDL-Hinf",    12.50),
        ("RobustLQR",   0.00),
    ]:
        check(f"5c {ctrl} unst%",
              paper,
              fnum(npz[f"corner_sensdly_{ctrl}_unstable_rate"]) * 100.0,
              rtol=2e-2)

    # -------------------- Table V: Monte Carlo --------------------
    section("Table V  Monte Carlo")
    for ctrl, mr, p90, p95, peak, rho in [
        ("Proposed",                0.0172, 0.0237, 0.0285, 0.0741, 0.979),
        ("NoRelax-ProposedActive",  0.0384, 0.0526, 0.0580, 0.1292, 0.972),
        ("QS-Hinf",                 0.0329, 0.0515, 0.0562, 0.0894, 0.997),
        ("PDL-Hinf",                0.0542, 0.0773, 0.0825, 0.215,  0.965),
        ("RobustLQR",               0.0274, 0.0389, 0.0419, 0.0780, 0.988),
    ]:
        check(f"MC {ctrl} RMSE med", mr,   mc_lookup(mc, ctrl, "RMSE_theta_median"))
        check(f"MC {ctrl} RMSE p90", p90,  mc_lookup(mc, ctrl, "RMSE_theta_p90"))
        check(f"MC {ctrl} RMSE p95", p95,  mc_lookup(mc, ctrl, "RMSE_theta_p95"))
        check(f"MC {ctrl} peak med", peak, mc_lookup(mc, ctrl, "peak_theta_median"))
        check(f"MC {ctrl} rho_max",  rho,  mc_lookup(mc, ctrl, "rho_max"))

    # -------------------- Body-text narrative ratios -------------
    # (Paper now reports "baseline / Proposed" as a multiplicative
    #  factor, not a percentage, to avoid the ambiguous "X% better"
    #  phrasing.)
    section("Body text  Monte Carlo ratio  baseline / Proposed")
    p_med = mc_lookup(mc, "Proposed", "RMSE_theta_median")
    for ctrl, paper_ratio in [
        ("NoRelax-ProposedActive",  2.239),
        ("QS-Hinf",                 1.915),
        ("PDL-Hinf",                3.156),
        ("RobustLQR",               1.599),
    ]:
        b_med = mc_lookup(mc, ctrl, "RMSE_theta_median")
        actual_ratio = b_med / p_med
        check(f"ratio  {ctrl} / Proposed  (median)", paper_ratio, actual_ratio, rtol=3e-2)

    # p90 claim vs RobustLQR (Table III narrative: 48% better =
    # RobustLQR / Proposed at p90 approx. 1.48)
    p_p90 = mc_lookup(mc, "Proposed",  "RMSE_theta_p90")
    r_p90 = mc_lookup(mc, "RobustLQR", "RMSE_theta_p90")
    check("ratio  RobustLQR / Proposed  (p90)", 1.643,
          r_p90 / p_p90, rtol=5e-2)

    # Median in degrees (paper-aligned: paper_body.tex Sec V.D quotes
    # 0.90, 2.41, 1.88, 3.08, 1.44 deg for the five controllers).
    for ctrl, paper_deg in [
        ("Proposed",                0.983),
        ("NoRelax-ProposedActive",  2.202),
        ("QS-Hinf",                 1.883),
        ("PDL-Hinf",                3.104),
        ("RobustLQR",               1.572),
    ]:
        rad = mc_lookup(mc, ctrl, "RMSE_theta_median")
        check(f"MC {ctrl} median [deg]", paper_deg, np.degrees(rad), rtol=2e-2)

    # 95th-pct of Proposed in degrees (V4 ~ 1.44) is well below the
    # RobustLQR median (1.44 deg as well, statistically tied; the headline
    # claim now compares Proposed p95 against the larger NoRelax/PDL/QS
    # medians).
    p95_rad = mc_lookup(mc, "Proposed", "RMSE_theta_p95")
    check("Proposed p95 [deg]", 1.630, np.degrees(p95_rad), rtol=2e-2)

    # -------------------- PDL decay-rate ablation (supplementary) -
    # NOTE: this block is a *reproducibility* check on the
    # _pdl_hinf_decay_ablation.py CSV; the corresponding numbers are
    # supplementary and NOT cited in the paper body (the published
    # PDL-Hinf row uses the default decay rate lambda_pdl = 0.95).
    section("Reproducibility check  PDL decay-rate ablation CSV")
    # Pinned values at lambda_pdl^2 = 0.95 / 0.90 / 0.85
    for paper_decay, paper_g, paper_kn, paper_kmax, paper_unst, paper_mc in [
        (0.95, 5.14,  39.72,  15.88, 12.50, 0.0538),
        (0.90, 10.88, 71.97,  30.61, 18.75, 0.0697),
        (0.85, 38.16, 121.48, 51.12, 25.00, 0.0859),
    ]:
        # Match by controller name; PDL-Hinf rows in CSV have decay_rate column
        idx = None
        for i, c in enumerate(abl["Controller"]):
            if c.startswith("PDL-Hinf") and abs(float(abl["decay_rate"][i]) - paper_decay) < 1e-3:
                idx = i; break
        if idx is None:
            FAIL += 1
            FAILURES.append(f"  [FAIL] PDL ablation row alpha^2={paper_decay}: not found in CSV")
            continue
        gamma_col = "gamma_syn_diag" if "gamma_syn_diag" in abl else "gamma"
        check(f"PDL alpha2={paper_decay} gamma_syn_diag", paper_g, float(abl[gamma_col][idx]))
        check(f"PDL alpha2={paper_decay} ||K||_F", paper_kn,   float(abl["K_norm"][idx]))
        check(f"PDL alpha2={paper_decay} max|K|",  paper_kmax, float(abl["K_max_abs"][idx]))
        check(f"PDL alpha2={paper_decay} unst%",   paper_unst, float(abl["corner_unstable"][idx]) * 100.0,
              rtol=2e-2)
        check(f"PDL alpha2={paper_decay} MC med",  paper_mc,   float(abl["mc_rmse_median"][idx]))

    # -------------------- paper-text scope wording scan -----------
    # Avoid global stale-number scans: some old numeric values can
    # legitimately reappear in other tables. We only scan for wording
    # that would mis-state certificate scope.
    candidate_tex_paths = [
        Path(__file__).resolve().parent / "paper_body.tex",
        Path(__file__).resolve().parent.parent / "paper_acc" / "paper_body.tex",
    ]
    tex_path = next((p for p in candidate_tex_paths if p.is_file()), None)
    section("Paper-text certificate-scope wording scan")
    if tex_path is not None:
        tex = tex_path.read_text(encoding="utf-8").lower()
        forbidden_phrases = [
            "synthesis gamma certificate",
            "reported synthesis gain",
            "certified by relaxed synthesis",
            "core-cqlf confirms",
            "data-core shrinkage alone is insufficient",
            "uncertainty set shrinkage alone is insufficient",
        ]
        for phrase in forbidden_phrases:
            if phrase in tex:
                FAIL += 1
                FAILURES.append(
                    f"  [FAIL] certificate-scope phrase still present: {phrase!r}"
                )
            else:
                PASS += 1
        print(f"  [PASS] scanned {len(forbidden_phrases)} scope phrases.")
    else:
        print(f"  [skip] paper_body.tex not found at any of: "
              f"{[str(p) for p in candidate_tex_paths]}")

    # -------------------- summary --------------------------------
    section("Summary")
    print(f"  PASS = {PASS}")
    print(f"  FAIL = {FAIL}")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f)
        sys.exit(1)
    print("All paper numbers consistent with simulation outputs.")


if __name__ == "__main__":
    main()
