from __future__ import annotations

import argparse
from pathlib import Path

from fit_data_utils import build_sssb_fit_data
from sssb_solver import SSSBFitParams, SSSBFitConfig, observed_driven_nll


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=str)
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--lspv_csv", default="data/raw/uspvdb_v4_0_20260414.csv", type=str)
    parser.add_argument("--dt_years", default=0.05, type=float)
    args = parser.parse_args()

    data = build_sssb_fit_data(
        msh_path=Path(args.mesh),
        node_features_npz=Path(args.features),
        lspv_csv=Path(args.lspv_csv),
        epsg_project=5070,
    )

    print("Y shape:", data.Y.shape)
    print("years:", data.years)
    print("total adoptions:", data.Y.sum())
    print("n nodes:", data.population.size)
    print("L shape:", data.L.shape)
    print("L nnz:", data.L.nnz)

    params = SSSBFitParams(
        p=0.001,
        q=0.1,
        gamma_J=0.1,
        k_J=0.5,
        D=0.1,
        S0=0.0,
        r0=-8.0,
        r1=0.0,
        r2=0.0,
    )

    cfg = SSSBFitConfig(
        dt_years=float(args.dt_years),
        use_covariates=True,
        normalize_nll=True,
    )

    nll, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=params,
        cfg=cfg,
        return_details=True,
    )

    print("NLL:", nll)
    print("capacity sum:", details["capacity"].sum())
    print("mu sum:", details["mu"].sum())
    print("observed sum:", data.Y.sum())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())