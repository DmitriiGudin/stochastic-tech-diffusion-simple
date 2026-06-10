from __future__ import annotations

import time
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from configs import DEFAULT, CONFIGS

from mesh_utils import (
    MeshBuildConfig,
    build_mesh_for_region,
    make_region_tag,
    mesh_summary,
    plot_mesh_lonlat,
)

from density_utils import (
    build_node_feature_table,
    save_node_features_npz,
    plot_population_comparison,
    plot_population_smoothed,
    map_lspv_adoptions_nearest_node,
    lspv_year_summary,
    lspv_county_year_summary,
    plot_lspv_adoptions_nearest_node,
    map_transmission_distance_to_nodes,
    plot_transmission_distance_nodes,
    map_pvout_to_nodes,
    plot_pv_potential_nodes,
)

from fit_data_utils import build_sssb_fit_data
from sssb_solver import SSSBFitParams, SSSBFitConfig, observed_driven_nll

POSITIVE_PARAMS = {"p", "q", "gamma_J", "k_J", "D", "S0", "FI_a", "FI_b", "FI_c", "r_max"}


def fmt_hhmmss(seconds: float) -> str:
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(name: str) -> dict:
    if name not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config '{name}'. Valid configs: {valid}")
    return deep_update(DEFAULT, CONFIGS[name])


def transformed_names(use_covariates: bool, fit_S0: bool) -> list[str]:
    names = ["p", "q", "gamma_J", "k_J", "D", "r_max"]
    if fit_S0:
        names.append("S0")
    names.append("r0")
    if use_covariates:
        names.extend(["r1", "r2"])

    names.extend(["FI_a", "FI_b", "FI_c"])
    return names


def param_to_theta_value(name: str, value: float) -> float:
    if name in POSITIVE_PARAMS:
        return float(np.log(value))
    return float(value)


def theta_to_param_value(name: str, value: float) -> float:
    if name in POSITIVE_PARAMS:
        return float(np.exp(value))
    return float(value)


def make_bounds(cfg: dict) -> list[tuple[float, float]]:
    use_covariates = bool(cfg["fit"]["use_covariates"])
    fit_S0 = bool(cfg["fit"]["fit_S0"])
    names = transformed_names(use_covariates, fit_S0)

    out = []
    for name in names:
        lo, hi = cfg["param_bounds"][name]
        out.append((param_to_theta_value(name, lo), param_to_theta_value(name, hi)))
    return out


def make_initial_theta(cfg: dict) -> np.ndarray:
    use_covariates = bool(cfg["fit"]["use_covariates"])
    fit_S0 = bool(cfg["fit"]["fit_S0"])
    names = transformed_names(use_covariates, fit_S0)

    vals = []
    for name in names:
        vals.append(param_to_theta_value(name, cfg["initial"][name]))
    return np.array(vals, dtype=float)


def random_theta(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    bounds = make_bounds(cfg)
    return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)


def unpack_theta(theta: np.ndarray, cfg: dict) -> SSSBFitParams:
    use_covariates = bool(cfg["fit"]["use_covariates"])
    fit_S0 = bool(cfg["fit"]["fit_S0"])
    names = transformed_names(use_covariates, fit_S0)

    values = {name: theta_to_param_value(name, val) for name, val in zip(names, theta)}

    return SSSBFitParams(
        p=values["p"],
        q=values["q"],
        gamma_J=values["gamma_J"],
        k_J=values["k_J"],
        D=values["D"],
        S0=values.get("S0", 0.0),
        r_max=values["r_max"],
        r0=values["r0"],
        r1=values.get("r1", 0.0),
        r2=values.get("r2", 0.0),
        FI_a=values["FI_a"],
        FI_b=values["FI_b"],
        FI_c=values["FI_c"],
    )


def make_tag(cfg: dict) -> str:
    states = cfg["region"]["states"]
    counties = cfg["region"]["counties"]
    mesh = cfg["mesh"]

    region_tag = make_region_tag(states, counties)
    return (
        f"{region_tag}"
        f"_h{mesh['h_km']:g}"
        f"_s{mesh['simplify_km']:g}"
        f"_epsg{mesh['epsg_project']}"
    )


def build_mesh_and_features(config_name: str, cfg: dict) -> tuple[Path, Path, dict]:
    paths = cfg["paths"]
    region = cfg["region"]
    mesh_cfg = cfg["mesh"]
    density = cfg["density"]

    out_root = Path("out") / config_name
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    tag = make_tag(cfg)

    mesh_dir = Path("data") / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    msh_path = mesh_dir / f"{tag}.msh"
    features_path = data_dir / f"{tag}_node_features.npz"
    metadata_path = data_dir / f"{tag}_metadata.json"

    admin1_shp = Path(paths["admin1_shp"])
    county_shp = Path(paths["county_shp"])
    pop_csv = Path(paths["pop_csv"])
    lspv_csv = Path(paths["lspv_csv"])
    transmission_shp = Path(paths["transmission_shp"])
    pvout_tif = Path(paths["pvout_tif"])

    mb_cfg = MeshBuildConfig(
        h_km=float(mesh_cfg["h_km"]),
        simplify_km=float(mesh_cfg["simplify_km"]),
        epsg_project=int(mesh_cfg["epsg_project"]),
    )

    if (not msh_path.exists()) or bool(mesh_cfg.get("overwrite_mesh", False)):
        build_mesh_for_region(
            admin1_shp=admin1_shp,
            county_shp=county_shp,
            state_codes=region["states"],
            county_names=region["counties"],
            out_msh=msh_path,
            cfg=mb_cfg,
            verbose=False,
        )
    else:
        print(f"[RUN] using existing mesh: {msh_path}")

    plot_mesh_lonlat(
        msh_path=msh_path,
        out_png=fig_dir / f"{tag}_mesh.png",
        epsg_project=mb_cfg.epsg_project,
    )

    features = build_node_feature_table(
        msh_path=msh_path,
        pop_csv=pop_csv,
        epsg_project=mb_cfg.epsg_project,
        smooth_length_km=float(density["smooth_length_km"]),
        smooth_k_neighbors=int(density["smooth_k_neighbors"]),
        smooth_kernel=str(density["smooth_kernel"]),
    )

    transmission_features = map_transmission_distance_to_nodes(
        msh_path=msh_path,
        transmission_shp=transmission_shp,
        epsg_project=mb_cfg.epsg_project,
        buffer_km=float(density["transmission_buffer_km"]),
    )
    features.update(transmission_features)

    pv_features = map_pvout_to_nodes(
        msh_path=msh_path,
        pvout_tif=pvout_tif,
        epsg_project=mb_cfg.epsg_project,
    )
    features.update(pv_features)

    save_node_features_npz(out_npz=features_path, features=features)

    year = int(density["plot_pop_year"])
    plot_population_comparison(
        msh_path=msh_path,
        nearest_values=features[f"population_nearest_{year}"],
        smooth_values=features[f"population_smooth_{year}"],
        out_png=fig_dir / f"{tag}_population_compare_{year}.png",
        epsg_project=mb_cfg.epsg_project,
        year=year,
    )

    plot_population_smoothed(
        msh_path=msh_path,
        smooth_values=features[f"population_smooth_{year}"],
        out_png=fig_dir / f"{tag}_population_smoothed_{year}.png",
        epsg_project=mb_cfg.epsg_project,
        year=year,
    )

    node_lspv_counts, lspv_inside = map_lspv_adoptions_nearest_node(
        msh_path=msh_path,
        lspv_csv=lspv_csv,
        epsg_project=mb_cfg.epsg_project,
    )

    plot_lspv_adoptions_nearest_node(
        msh_path=msh_path,
        node_counts=node_lspv_counts,
        out_png=fig_dir / f"{tag}_lspv_adoptions_nearest_node.png",
        epsg_project=mb_cfg.epsg_project,
        scale=str(density.get("adoption_plot_scale", "linear")),
    )

    plot_transmission_distance_nodes(
        msh_path=msh_path,
        distances_km=features["transmission_distance_km"],
        out_png=fig_dir / f"{tag}_transmission_distance_km.png",
        epsg_project=mb_cfg.epsg_project,
    )

    plot_pv_potential_nodes(
        msh_path=msh_path,
        pv_values=features["pv_potential"],
        out_png=fig_dir / f"{tag}_pv_potential.png",
        epsg_project=mb_cfg.epsg_project,
    )

    lspv_global_summary = lspv_year_summary(lspv_inside)
    lspv_county_summary = lspv_county_year_summary(
        events_inside=lspv_inside,
        county_shp=county_shp,
        state_codes=region["states"],
        county_names=region["counties"],
    )

    metadata = {
        "config_name": config_name,
        "tag": tag,
        "config": cfg,
        "mesh": mesh_summary(msh_path),
        "node_features_npz": str(features_path),
        "lspv_adoptions": {
            "global": lspv_global_summary,
            "counties": lspv_county_summary,
        },
        "population_mass_checks": {
            "zip_inside_count": int(features["zip_inside_count"][0]),
            "zip_total_count": int(features["zip_total_count"][0]),
            "zip_population_inside_2010": float(features["zip_population_inside_2010"][0]),
            "zip_population_inside_2020": float(features["zip_population_inside_2020"][0]),
            "node_population_smooth_sum_2010": float(features["node_population_smooth_sum_2010"][0]),
            "node_population_smooth_sum_2020": float(features["node_population_smooth_sum_2020"][0]),
        },
        "transmission_lines": {
            "buffer_km": float(density["transmission_buffer_km"]),
            "lines_retained": int(features["transmission_lines_retained"][0]),
            "nodes_zero_distance": int(features["transmission_nodes_zero_count"][0]),
            "distance_min_km": float(features["transmission_distance_min_km"][0]),
            "distance_median_km": float(features["transmission_distance_median_km"][0]),
            "distance_max_km": float(features["transmission_distance_max_km"][0]),
        },
        "pvout": {
            "valid_count": int(features["pv_potential_valid_count"][0]),
            "min": float(features["pv_potential_min"][0]),
            "median": float(features["pv_potential_median"][0]),
            "max": float(features["pv_potential_max"][0]),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return msh_path, features_path, metadata


def fit_model(config_name: str, cfg: dict, msh_path: Path, features_path: Path) -> dict:
    fit = cfg["fit"]
    progress_freq = int(fit.get("progress_freq", 10))
    progress_freq = max(progress_freq, 1)
    paths = cfg["paths"]

    out_root = Path("out") / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    data = build_sssb_fit_data(
        msh_path=msh_path,
        node_features_npz=features_path,
        lspv_csv=Path(paths["lspv_csv"]),
        epsg_project=int(cfg["mesh"]["epsg_project"]),
        population_key=str(fit["population_key"]),
        year_window=fit.get("year_window", None),
    )

    solver_cfg = SSSBFitConfig(
        dt_years=float(fit["dt_years"]),
        use_covariates=bool(fit["use_covariates"]),
        normalize_nll=True,
        capacity_link=str(cfg["capacity"].get("link", "logistic")),
        standardize_covariates=bool(cfg["capacity"].get("standardize_covariates", True)),
    
        condition_on_seed_year=bool(fit.get("condition_on_seed_year", False)),
        seed_year=int(fit.get("seed_year", 2007)),
        include_seed_likelihood=bool(fit.get("include_seed_likelihood", True)),
    )

    def objective(theta: np.ndarray) -> float:
        params = unpack_theta(theta, cfg)
        val = observed_driven_nll(
            Y=data.Y,
            years=data.years,
            population=data.population,
            pv_potential=data.pv_potential,
            transmission_distance_km=data.transmission_distance_km,
            L=data.L,
            params=params,
            cfg=solver_cfg,
            return_details=False,
        )
        return float(val) if np.isfinite(val) else 1e100

    rng = np.random.default_rng(int(fit["seed"]))

    bounds = make_bounds(cfg)
    best_theta = make_initial_theta(cfg)
    best_val = objective(best_theta)

    print("[FIT] random search")
    print("[FIT] initial NLL:", best_val)

    t_random0 = time.perf_counter()

    for k in range(int(fit["n_random"])):
        th = random_theta(rng, cfg)
        val = objective(th)

        if val < best_val:
            best_val = val
            best_theta = th

        if (k + 1) % progress_freq == 0 or (k + 1) == int(fit["n_random"]):
            elapsed = fmt_hhmmss(time.perf_counter() - t_random0)
            print(
                f"[{elapsed}] [FIT random] "
                f"{k + 1}/{fit['n_random']} --- best NLL = {best_val:.6g}"
            )

    print("[FIT] L-BFGS-B")

    t_local0 = time.perf_counter()
    local_iter = {"k": 0, "best": float(best_val)}

    def callback(xk: np.ndarray) -> None:
        local_iter["k"] += 1

        if local_iter["k"] % progress_freq == 0:
            val = objective(xk)
            local_iter["best"] = min(local_iter["best"], float(val))
            elapsed = fmt_hhmmss(time.perf_counter() - t_local0)
            print(
                f"[{elapsed}] [FIT L-BFGS-B] "
                f"iteration {local_iter['k']} --- current NLL = {float(val):.6g}, "
                f"best NLL = {local_iter['best']:.6g}"
            )

    res = minimize(
        objective,
        best_theta,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": int(fit["maxiter"]),
            "ftol": 1e-8,
            "gtol": 1e-5,
            "maxls": 30,
        },
    )

    best_params = unpack_theta(res.x, cfg)

    nll, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=best_params,
        cfg=solver_cfg,
        return_details=True,
    )

    payload = {
        "config_name": config_name,
        "success": bool(res.success),
        "message": str(res.message),
        "fun": float(res.fun),
        "year_window": fit.get("year_window", None),
        "nll_recomputed": float(nll),
        "theta_unconstrained": res.x.tolist(),
        "theta_bounds": bounds,
        "params": asdict(best_params),
        "solver_config": asdict(solver_cfg),
        "config": cfg,
        "data": {
            "mesh": str(msh_path),
            "features": str(features_path),
            "lspv_csv": str(paths["lspv_csv"]),
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

    out_path = Path("out") / config_name / "fit_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[FIT] best params:")
    print(json.dumps(payload["params"], indent=2))
    print("[FIT] diagnostics:")
    print(json.dumps(payload["diagnostics"], indent=2))
    print("[FIT] wrote:", out_path)
    
    print(f"[FIT] random search elapsed: {fmt_hhmmss(time.perf_counter() - t_random0)}")
    print(f"[FIT] local optimization elapsed: {fmt_hhmmss(time.perf_counter() - t_local0)}")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    config_name = str(args.config)
    cfg = load_config(config_name)

    print("[RUN] config:", config_name)

    msh_path, features_path, metadata = build_mesh_and_features(config_name, cfg)
    fit_model(config_name, cfg, msh_path, features_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())