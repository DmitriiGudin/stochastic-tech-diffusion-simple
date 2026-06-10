from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fit_data_utils import build_sssb_fit_data
from sssb_solver import SSSBFitParams, SSSBFitConfig, observed_driven_nll


def unpack_theta(theta: np.ndarray, *, use_covariates: bool, fit_S0: bool) -> SSSBFitParams:
    """
    Optimizer variables are unconstrained.
    Positive parameters are exponentiated.
    """
    theta = np.asarray(theta, dtype=float)

    p = np.exp(theta[0])
    q = np.exp(theta[1])
    gamma_J = np.exp(theta[2])
    k_J = np.exp(theta[3])
    D = np.exp(theta[4])

    idx = 5

    if fit_S0:
        S0 = np.exp(theta[idx])
        idx += 1
    else:
        S0 = 0.0

    r0 = theta[idx]
    idx += 1

    if use_covariates:
        r1 = theta[idx]
        r2 = theta[idx + 1]
    else:
        r1 = 0.0
        r2 = 0.0

    return SSSBFitParams(
        p=p,
        q=q,
        gamma_J=gamma_J,
        k_J=k_J,
        D=D,
        S0=S0,
        r0=r0,
        r1=r1,
        r2=r2,
    )


def initial_theta(*, use_covariates: bool, fit_S0: bool) -> np.ndarray:
    vals = [
        np.log(1e-3),  # p
        np.log(1e-1),  # q
        np.log(1e-1),  # gamma_J
        np.log(5e-1),  # k_J
        np.log(1e-1),  # D
    ]

    if fit_S0:
        vals.append(np.log(1e-6))

    vals.append(-8.0)  # r0

    if use_covariates:
        vals.extend([0.0, 0.0])  # r1, r2

    return np.array(vals, dtype=float)


def random_theta(rng: np.random.Generator, *, use_covariates: bool, fit_S0: bool) -> np.ndarray:
    vals = [
        rng.uniform(np.log(1e-5), np.log(5e-2)),  # p
        rng.uniform(np.log(1e-4), np.log(5.0)),   # q
        rng.uniform(np.log(1e-3), np.log(5.0)),   # gamma_J
        rng.uniform(np.log(1e-3), np.log(5.0)),   # k_J
        rng.uniform(np.log(1e-4), np.log(5.0)),   # D
    ]

    if fit_S0:
        vals.append(rng.uniform(np.log(1e-10), np.log(1e-1)))

    vals.append(rng.uniform(-14.0, -3.0))  # r0

    if use_covariates:
        vals.append(rng.uniform(-3.0, 3.0))  # r1
        vals.append(rng.uniform(-3.0, 3.0))  # r2

    return np.array(vals, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=str)
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--lspv_csv", default="data/raw/uspvdb_v4_0_20260414.csv", type=str)

    parser.add_argument("--dt_years", default=0.05, type=float)
    parser.add_argument("--epsg_project", default=5070, type=int)

    parser.add_argument("--use_covariates", action="store_true")
    parser.add_argument("--fit_S0", action="store_true")

    parser.add_argument("--n_random", default=100, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--out", default="out/fit_sssb/fit_result.json", type=str)

    args = parser.parse_args()

    data = build_sssb_fit_data(
        msh_path=Path(args.mesh),
        node_features_npz=Path(args.features),
        lspv_csv=Path(args.lspv_csv),
        epsg_project=int(args.epsg_project),
    )

    cfg = SSSBFitConfig(
        dt_years=float(args.dt_years),
        use_covariates=bool(args.use_covariates),
        normalize_nll=True,
    )

    print("[FIT] Y shape:", data.Y.shape)
    print("[FIT] years:", data.years)
    print("[FIT] total observed:", float(data.Y.sum()))
    print("[FIT] use_covariates:", cfg.use_covariates)
    print("[FIT] fit_S0:", bool(args.fit_S0))

    def objective(theta: np.ndarray) -> float:
        params = unpack_theta(
            theta,
            use_covariates=cfg.use_covariates,
            fit_S0=bool(args.fit_S0),
        )

        val = observed_driven_nll(
            Y=data.Y,
            years=data.years,
            population=data.population,
            pv_potential=data.pv_potential,
            transmission_distance_km=data.transmission_distance_km,
            L=data.L,
            params=params,
            cfg=cfg,
            return_details=False,
        )

        if not np.isfinite(val):
            return 1e100
        return float(val)

    rng = np.random.default_rng(int(args.seed))

    print("[FIT] random search...")
    best_theta = initial_theta(
        use_covariates=cfg.use_covariates,
        fit_S0=bool(args.fit_S0),
    )
    best_val = objective(best_theta)

    for k in range(int(args.n_random)):
        th = random_theta(
            rng,
            use_covariates=cfg.use_covariates,
            fit_S0=bool(args.fit_S0),
        )
        val = objective(th)

        if val < best_val:
            best_val = val
            best_theta = th
            print(f"[FIT] random {k+1}/{args.n_random}: best NLL = {best_val:.6g}")

    print("[FIT] local optimization...")
    res = minimize(
        objective,
        best_theta,
        method="L-BFGS-B",
        options={
            "maxiter": int(args.maxiter),
            "disp": True,
            "ftol": 1e-10,
            "gtol": 1e-8,
            "maxls": 100,
        },
    )

    best_params = unpack_theta(
        res.x,
        use_covariates=cfg.use_covariates,
        fit_S0=bool(args.fit_S0),
    )

    nll, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=best_params,
        cfg=cfg,
        return_details=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "success": bool(res.success),
        "message": str(res.message),
        "fun": float(res.fun),
        "nll_recomputed": float(nll),
        "theta_unconstrained": res.x.tolist(),
        "params": asdict(best_params),
        "config": asdict(cfg),
        "data": {
            "mesh": str(args.mesh),
            "features": str(args.features),
            "lspv_csv": str(args.lspv_csv),
            "years": data.years.astype(int).tolist(),
            "total_observed": float(data.Y.sum()),
            "n_nodes": int(data.population.size),
            "L_nnz": int(data.L.nnz),
        },
        "diagnostics": {
            "capacity_sum": float(np.sum(details["capacity"])),
            "mu_sum": float(np.sum(details["mu"])),
            "observed_sum": float(np.sum(data.Y)),
            "final_I_mean": float(np.mean(details["final_I"])),
            "final_J_mean": float(np.mean(details["final_J"])),
            "n_substeps_per_year": int(details["n_substeps_per_year"][0]),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[FIT] result:")
    print(json.dumps(payload["params"], indent=2))
    print("[FIT] diagnostics:")
    print(json.dumps(payload["diagnostics"], indent=2))
    print("[FIT] wrote:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())