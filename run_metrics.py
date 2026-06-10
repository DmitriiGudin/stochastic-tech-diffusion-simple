from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy.optimize import minimize
from scipy.sparse import identity
from pyproj import Transformer

from configs import DEFAULT, CONFIGS
from fit_data_utils import build_sssb_fit_data
from sssb_solver import SSSBFitParams, SSSBFitConfig, observed_driven_nll


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
        raise ValueError(f"Unknown config {name}. Valid configs: {sorted(CONFIGS)}")
    return deep_update(DEFAULT, CONFIGS[name])


def bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if p <= 0 or q <= 0 or M <= 0:
        return np.full_like(t, np.nan, dtype=float)

    e = np.exp(-(p + q) * t)
    return M * (1.0 - e) / (1.0 + (q / p) * e)


def fit_bass_curve(t: np.ndarray, cum_obs: np.ndarray) -> dict:
    final_obs = float(cum_obs[-1])

    def unpack(theta):
        p = np.exp(theta[0])
        q = np.exp(theta[1])
        M = final_obs + np.exp(theta[2])
        return p, q, M

    def obj(theta):
        p, q, M = unpack(theta)
        pred = bass_cumulative(t, p, q, M)
        return float(np.mean((pred - cum_obs) ** 2))

    theta0 = np.array([np.log(0.01), np.log(0.3), np.log(max(final_obs, 1.0))])
    res = minimize(obj, theta0, method="Nelder-Mead", options={"maxiter": 5000})

    p, q, M = unpack(res.x)
    return {
        "p": float(p),
        "q": float(q),
        "M": float(M),
        "success": bool(res.success),
        "rmse": float(np.sqrt(obj(res.x))),
    }


def poisson_deviance(obs: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, eps)

    term = np.zeros_like(obs, dtype=float)
    positive = obs > 0
    term[positive] = obs[positive] * np.log(obs[positive] / pred[positive])
    return float(2.0 * np.sum(term - (obs - pred)))


def mesh_lonlat(data, epsg_project: int = 5070):
    pts_m = data.mesh_points_km * 1000.0
    tr = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pts_m[:, 0], pts_m[:, 1])
    return np.asarray(lon), np.asarray(lat)


def plot_node_values(
    ax,
    *,
    data,
    values: np.ndarray,
    title: str,
    epsg_project: int,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "log1p",
    n_layers: int = 10,
):
    lon, lat = mesh_lonlat(data, epsg_project=epsg_project)
    tri = data.triangles

    raw_vals = np.asarray(values, dtype=float)

    if scale == "log1p":
        vals = np.log1p(np.clip(raw_vals, 0.0, None))
    elif scale == "linear":
        vals = np.clip(raw_vals, 0.0, None)
    elif scale == "custom_log":
        positive = raw_vals[raw_vals > 0.0]
        if positive.size == 0:
            vals = np.zeros_like(raw_vals, dtype=float)
        else:
            # vmin/vmax are passed in already-transformed log units.
            # For safety, if vmin is unavailable, use max - 8.
            max_val = float(np.nanmax(positive))
            floor_log = np.log(max_val) - 8.0 if vmin is None else float(vmin)
            vals = np.log(np.maximum(raw_vals, np.exp(floor_log)))
    else:
        raise ValueError("scale must be 'linear', 'log1p', or 'custom_log'.")

    finite = np.isfinite(vals)

    if vmin is None:
        vmin = float(np.nanmin(vals[finite])) if np.any(finite) else 0.0
    if vmax is None:
        vmax = float(np.nanquantile(vals[finite], 0.99)) if np.any(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    triang = mtri.Triangulation(lon, lat, tri)
    ax.triplot(triang, linewidth=0.25, color="0.82", alpha=0.55, zorder=1)

    s = max(5.0, 5000.0 / max(np.sqrt(len(vals)), 1.0)) * 0.35

    finite_vals = vals[finite]
    if finite_vals.size == 0:
        layer_edges = np.array([vmin, vmax])
    else:
        qs = np.linspace(0.0, 1.0, int(n_layers) + 1)
        layer_edges = np.nanquantile(finite_vals, qs)
        layer_edges = np.unique(layer_edges)
        if layer_edges.size < 2:
            layer_edges = np.array([vmin, vmax])

    mappable = None

    for ell in range(layer_edges.size - 1):
        lo = layer_edges[ell]
        hi = layer_edges[ell + 1]

        if ell == layer_edges.size - 2:
            mask = finite & (vals >= lo) & (vals <= hi)
        else:
            mask = finite & (vals >= lo) & (vals < hi)

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        # Critical fix: sort within layer from smaller to larger.
        idx = idx[np.argsort(vals[idx])]

        mappable = ax.scatter(
            lon[idx],
            lat[idx],
            c=vals[idx],
            s=s,
            cmap='hot',
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.55 + 0.45 * (ell + 1) / max(layer_edges.size - 1, 1),
            zorder=2 + ell,
        )

    if mappable is None:
        mappable = ax.scatter(
            lon,
            lat,
            c=np.zeros_like(vals),
            s=s,
            cmap='hot',
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.55,
            zorder=2,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    return mappable


def model_monthly_expected_counts(
    *,
    data,
    params: SSSBFitParams,
    cfg: SSSBFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      month_t       shape (n_years*12 + 1,)
      cum_model     cumulative expected total adoptions at monthly grid
    """
    Y = np.asarray(data.Y, dtype=float)
    n_years, n_nodes = Y.shape

    _, details = observed_driven_nll(
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

    annual_mu = details["mu"].sum(axis=1)

    # Monthly interpolation of model annual expected counts.
    # This matches the data convention: annual counts are spread uniformly.
    monthly_counts = np.repeat(annual_mu / 12.0, 12)
    cum_model = np.concatenate([[0.0], np.cumsum(monthly_counts)])

    month_t = np.arange(cum_model.size, dtype=float) / 12.0
    return month_t, cum_model


def observed_monthly_curve(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    annual_counts = Y.sum(axis=1)
    monthly_counts = np.repeat(annual_counts / 12.0, 12)
    cum = np.concatenate([[0.0], np.cumsum(monthly_counts)])
    t = np.arange(cum.size, dtype=float) / 12.0
    return t, cum


def monthly_rate_curve_from_annual_counts(annual_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts annual counts into monthly adoption rates by assuming
    uniform adoption within each year.

    Returns
    -------
    t_month : ndarray
        Monthly time grid in years, length n_years * 12.
    rate_month : ndarray
        Monthly adoption rate in adoptions/year.
        Since annual counts are spread uniformly, each month in year y
        has rate equal to annual_counts[y].
    """
    annual_counts = np.asarray(annual_counts, dtype=float)
    rate_month = np.repeat(annual_counts, 12)
    t_month = (np.arange(rate_month.size, dtype=float) + 0.5) / 12.0
    return t_month, rate_month


def graph_edge_lengths_km(data) -> np.ndarray:
    pts = data.mesh_points_km
    tri = data.triangles

    edges = set()
    for a, b, c in tri:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))

    if not edges:
        return np.array([], dtype=float)

    lengths = []
    for i, j in edges:
        lengths.append(float(np.linalg.norm(pts[i] - pts[j])))

    return np.asarray(lengths, dtype=float)


def adjacency_from_laplacian(L):
    """
    Convert normalized graph Laplacian into unweighted adjacency.
    Off-diagonal nonzeros correspond to graph edges.
    """
    A = L.copy().tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def graph_neighborhood_matrix(L, level: int):
    """
    Boolean sparse matrix B_level where B[i,j]=1 if node j is within
    graph distance <= level from node i.

    level=0 means node itself only.
    level=1 means node + adjacent nodes.
    """
    n = L.shape[0]
    A = adjacency_from_laplacian(L)
    B = identity(n, format="csr", dtype=float)

    frontier = identity(n, format="csr", dtype=float)
    for _ in range(level):
        frontier = frontier @ A
        frontier.data[:] = 1.0
        B = B + frontier
        B.data[:] = 1.0
        B.eliminate_zeros()

    B.data[:] = 1.0
    return B.tocsr()


def neighborhood_aggregate_counts(X: np.ndarray, B) -> np.ndarray:
    """
    Aggregate node-year counts over graph neighborhoods.

    X shape: (n_years, n_nodes)
    B shape: (n_nodes, n_nodes), where B[i,j]=1 if j is in i-neighborhood.

    Returns A[y,i] = sum_{j in neighborhood(i)} X[y,j].
    """
    X = np.asarray(X, dtype=float)
    return (B @ X.T).T


def neighborhood_spatial_metrics(
    *,
    Y_obs: np.ndarray,
    predictions: dict[str, np.ndarray],
    L,
    data,
    max_level: int = 3,
) -> dict:
    """
    Metrics after aggregating counts over graph neighborhoods.

    level=0: original nodewise metrics.
    level=1: node + immediate neighbors.
    level=2: node + neighbors + neighbors-of-neighbors.
    etc.
    """
    out = {}

    edge_lengths = graph_edge_lengths_km(data)
    median_edge_km = float(np.nanmedian(edge_lengths)) if edge_lengths.size else np.nan

    for level in range(max_level + 1):
        B = graph_neighborhood_matrix(L, level)
        Y_smooth = neighborhood_aggregate_counts(Y_obs, B)
        Y_cum_smooth = np.cumsum(Y_smooth, axis=0)

        obs_total_by_node = Y_smooth.sum(axis=0)
        nonzero_mask = obs_total_by_node > 0

        approx_radius_km = (
            float(level * median_edge_km)
            if np.isfinite(median_edge_km)
            else None
        )

        level_key = f"level_{level}"

        out[level_key] = {
            "graph_level": int(level),
            "approx_radius_km": approx_radius_km,
            "median_edge_length_km": median_edge_km,
            "models": {},
        }

        for model_name, pred in predictions.items():
            pred_smooth = neighborhood_aggregate_counts(pred, B)
            pred_cum_smooth = np.cumsum(pred_smooth, axis=0)

            out[level_key]["models"][model_name] = {
                "instantaneous": metric_bundle(
                    Y_smooth.ravel(),
                    pred_smooth.ravel(),
                ),
                "cumulative": metric_bundle(
                    Y_cum_smooth.ravel(),
                    pred_cum_smooth.ravel(),
                ),
            }

            if np.any(nonzero_mask):
                out[level_key]["models"][model_name]["instantaneous_nonzero"] = metric_bundle(
                    Y_smooth[:, nonzero_mask].ravel(),
                    pred_smooth[:, nonzero_mask].ravel(),
                )
                out[level_key]["models"][model_name]["cumulative_nonzero"] = metric_bundle(
                    Y_cum_smooth[:, nonzero_mask].ravel(),
                    pred_cum_smooth[:, nonzero_mask].ravel(),
                )
            else:
                out[level_key]["models"][model_name]["instantaneous_nonzero"] = None
                out[level_key]["models"][model_name]["cumulative_nonzero"] = None

    return out


def yearly_rmse_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def yearly_rmse_cumulative(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.cumsum(obs) - np.cumsum(pred)) ** 2)))


def yearly_rmse_log1p_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(obs) - np.log1p(pred)) ** 2)))


def yearly_rmse_log1p_cumulative(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_cum = np.cumsum(obs)
    pred_cum = np.cumsum(pred)
    return float(np.sqrt(np.mean((np.log1p(obs_cum) - np.log1p(pred_cum)) ** 2)))


def mae_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(obs - pred)))


def rmse_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae_log1p_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log1p(obs) - np.log1p(pred))))


def rmse_log1p_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(obs) - np.log1p(pred)) ** 2)))


def topk_hit_concentration(
    *,
    obs_node_total: np.ndarray,
    pred_node_score: np.ndarray,
    ks: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> dict:
    """
    Fraction of observed adoptions captured by top-k fraction of nodes ranked by prediction.

    obs_node_total:
        Observed cumulative adoptions per node.

    pred_node_score:
        Predicted cumulative mean adoptions per node.

    ks:
        Fractions of nodes to include, e.g. 0.05 = top 5%.
    """
    obs = np.asarray(obs_node_total, dtype=float)
    pred = np.asarray(pred_node_score, dtype=float)

    n = obs.size
    total_obs = float(obs.sum())

    if n == 0 or total_obs <= 0:
        return {f"top_{int(100*k)}pct": None for k in ks}

    order = np.argsort(-pred)

    out = {}
    for k in ks:
        m = max(1, int(np.ceil(k * n)))
        chosen = order[:m]
        captured = float(obs[chosen].sum())
        out[f"top_{int(round(100*k))}pct"] = {
            "node_fraction": float(k),
            "n_nodes": int(m),
            "observed_captured": captured,
            "observed_total": total_obs,
            "capture_fraction": captured / total_obs,
        }

    return out


def calibration_by_risk_bin(
    *,
    obs_node_total: np.ndarray,
    pred_node_total: np.ndarray,
    n_bins: int = 10,
    nonfinite_policy: str = "error",  # "error" or "zero"
) -> dict:
    """
    Bin nodes by predicted cumulative mean count and compare observed vs predicted totals.

    The bin totals should add up to the post-seed observed and predicted totals
    unless non-finite predictions are present and nonfinite_policy="zero" is used.

    nonfinite_policy:
        "error": raise if obs/pred contain non-finite values.
        "zero": replace non-finite predictions with 0 and keep all nodes.
    """
    obs = np.asarray(obs_node_total, dtype=float)
    pred = np.asarray(pred_node_total, dtype=float)

    if obs.shape != pred.shape:
        raise ValueError(f"obs and pred must have same shape, got {obs.shape} vs {pred.shape}")

    n_nodes_input = int(obs.size)
    obs_total_input = float(np.nansum(obs))
    pred_total_input = float(np.nansum(pred))

    finite_obs = np.isfinite(obs)
    finite_pred = np.isfinite(pred)

    if not np.all(finite_obs):
        bad = int(np.sum(~finite_obs))
        raise ValueError(f"obs_node_total contains {bad} non-finite entries.")

    if not np.all(finite_pred):
        bad = int(np.sum(~finite_pred))
        if nonfinite_policy == "error":
            raise ValueError(
                f"pred_node_total contains {bad} non-finite entries. "
                "This usually means the SSSB expected-count solver produced NaN/inf values."
            )
        if nonfinite_policy == "zero":
            pred = np.where(finite_pred, pred, 0.0)
        else:
            raise ValueError("nonfinite_policy must be 'error' or 'zero'.")

    if obs.size == 0:
        return {
            "summary": {
                "n_nodes_input": n_nodes_input,
                "obs_total_input": obs_total_input,
                "pred_total_input": pred_total_input,
                "obs_total_binned": 0.0,
                "pred_total_binned": 0.0,
            },
            "bins": {},
        }

    # Sort by predicted risk.
    order = np.argsort(pred)
    obs_sorted = obs[order]
    pred_sorted = pred[order]

    bins = np.array_split(np.arange(obs_sorted.size), n_bins)

    out_bins = {}
    obs_total_binned = 0.0
    pred_total_binned = 0.0

    for b, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue

        obs_sum = float(obs_sorted[idx].sum())
        pred_sum = float(pred_sorted[idx].sum())

        obs_total_binned += obs_sum
        pred_total_binned += pred_sum

        out_bins[f"bin_{b:02d}"] = {
            "n_nodes": int(idx.size),
            "pred_min": float(pred_sorted[idx].min()),
            "pred_max": float(pred_sorted[idx].max()),
            "pred_mean": float(pred_sorted[idx].mean()),
            "obs_total": obs_sum,
            "pred_total": pred_sum,
            "obs_minus_pred": obs_sum - pred_sum,
            "obs_over_pred": float(obs_sum / pred_sum) if pred_sum > 0 else None,
        }

    return {
        "summary": {
            "n_nodes_input": n_nodes_input,
            "obs_total_input": obs_total_input,
            "pred_total_input": pred_total_input,
            "obs_total_binned": float(obs_total_binned),
            "pred_total_binned": float(pred_total_binned),
            "obs_binning_error": float(obs_total_binned - obs_total_input),
            "pred_binning_error": float(pred_total_binned - pred_total_input),
        },
        "bins": out_bins,
    }


def metric_bundle(obs: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "RMSE": rmse_counts(obs, pred),
        "MAE": mae_counts(obs, pred),
        "RMSE_log1p": rmse_log1p_counts(obs, pred),
        "MAE_log1p": mae_log1p_counts(obs, pred),
    }


def make_bass_node_baselines(
    *,
    annual_bass: np.ndarray,      # (n_years,)
    population: np.ndarray,       # (n_nodes,)
    n_nodes: int,
) -> dict[str, np.ndarray]:
    """
    Returns annual node-level predicted counts for two spatially naive Bass baselines.

    uniform:
        annual_bass[y] / n_nodes

    population:
        annual_bass[y] * population_i / sum_i population_i
    """
    annual_bass = np.asarray(annual_bass, dtype=float)
    population = np.asarray(population, dtype=float)
    population = np.clip(population, 0.0, None)

    uniform_weights = np.full(n_nodes, 1.0 / max(n_nodes, 1), dtype=float)

    if population.sum() > 0:
        pop_weights = population / population.sum()
    else:
        pop_weights = uniform_weights.copy()

    return {
        "bass_uniform": annual_bass[:, None] * uniform_weights[None, :],
        "bass_population": annual_bass[:, None] * pop_weights[None, :],
    }


def condition_node_predictions_on_seed(
    *,
    predictions: dict[str, np.ndarray],
    Y_obs: np.ndarray,
    years: np.ndarray,
    seed_year: int,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Overwrite predicted annual node counts through seed_year, inclusive,
    with observed annual node counts.

    This makes deterministic expected-count comparisons fair with
    seed-conditioned simulations.
    """
    matches = np.where(years == int(seed_year))[0]
    if matches.size == 0:
        raise ValueError(f"seed_year={seed_year} not found in years.")

    seed_idx = int(matches[0])

    out = {}
    for name, pred in predictions.items():
        arr = np.asarray(pred, dtype=float).copy()
        arr[: seed_idx + 1, :] = np.asarray(Y_obs[: seed_idx + 1, :], dtype=float)
        out[name] = arr

    return out, seed_idx


def nodewise_spatial_metrics(
    *,
    Y_obs: np.ndarray,                  # (n_years, n_nodes)
    predictions: dict[str, np.ndarray], # same shape
) -> dict:
    """
    Computes nodewise instantaneous and cumulative metrics.

    Two domains:
      all nodes
      nonzero nodes: nodes with total observed adoption > 0
    """
    Y_obs = np.asarray(Y_obs, dtype=float)
    obs_cum = np.cumsum(Y_obs, axis=0)

    total_obs_by_node = Y_obs.sum(axis=0)
    nonzero_mask = total_obs_by_node > 0

    out = {}

    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        pred_cum = np.cumsum(pred, axis=0)

        out[name] = {
            "instantaneous": metric_bundle(Y_obs.ravel(), pred.ravel()),
            "cumulative": metric_bundle(obs_cum.ravel(), pred_cum.ravel()),
        }

        if np.any(nonzero_mask):
            out[name]["instantaneous_nonzero"] = metric_bundle(
                Y_obs[:, nonzero_mask].ravel(),
                pred[:, nonzero_mask].ravel(),
            )
            out[name]["cumulative_nonzero"] = metric_bundle(
                obs_cum[:, nonzero_mask].ravel(),
                pred_cum[:, nonzero_mask].ravel(),
            )
        else:
            out[name]["instantaneous_nonzero"] = None
            out[name]["cumulative_nonzero"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fit_json", default=None, type=str)
    args = parser.parse_args()

    cfg_named = load_config(args.config)

    fit_json = Path(args.fit_json) if args.fit_json else Path("out") / args.config / "fit_result.json"
    if not fit_json.exists():
        raise FileNotFoundError(fit_json)

    with open(fit_json, "r", encoding="utf-8") as f:
        fit_payload = json.load(f)

    mesh_path = Path(fit_payload["data"]["mesh"])
    features_path = Path(fit_payload["data"]["features"])
    lspv_csv = Path(fit_payload["data"]["lspv_csv"])

    data = build_sssb_fit_data(
        msh_path=mesh_path,
        node_features_npz=features_path,
        lspv_csv=lspv_csv,
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        population_key=str(cfg_named["fit"]["population_key"]),
        year_window=cfg_named["fit"].get("year_window", None),
    )

    params = SSSBFitParams(**fit_payload["params"])
    solver_cfg = SSSBFitConfig(**fit_payload["solver_config"])

    nll, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=params,
        cfg=solver_cfg,
        return_details=True,
    )

    annual_obs = data.Y.sum(axis=1)
    annual_model = details["mu"].sum(axis=1)

    t_obs_month, cum_obs_month = observed_monthly_curve(data.Y)

    # Bass fit to monthly cumulative observed curve.
    bass = fit_bass_curve(t_obs_month, cum_obs_month)
    
    print("[CLASSIC BASS FIT]")
    print("Uniform classic Bass model:")
    print(f"  p = {bass['p']:.12g}")
    print(f"  q = {bass['q']:.12g}")
    print(f"  M = {bass['M']:.12g}")
    print(f"  fit RMSE = {bass['rmse']:.12g}")
    print(f"  optimizer success = {bass['success']}")
    
    print("Population-informed classic Bass model:")
    print(f"  p = {bass['p']:.12g}")
    print(f"  q = {bass['q']:.12g}")
    print(f"  M = {bass['M']:.12g}")
    print(f"  fit RMSE = {bass['rmse']:.12g}")
    print(f"  optimizer success = {bass['success']}")
    print(
        "Note: both Bass baselines share the same aggregate Bass fit; "
        "they differ only in spatial allocation."
    )
    
    cum_bass = bass_cumulative(t_obs_month, bass["p"], bass["q"], bass["M"])
    
    # Convert monthly Bass cumulative curve to annual increments.
    # month index 12*k corresponds to end of year k.
    bass_cum_year_end = cum_bass[12::12]
    bass_cum_year_start = np.concatenate([[0.0], bass_cum_year_end[:-1]])
    annual_bass = bass_cum_year_end - bass_cum_year_start
    
    t_rate_obs, rate_obs = monthly_rate_curve_from_annual_counts(annual_obs)
    calendar_rate_months = data.years[0] + t_rate_obs
    
    bass_node_baselines = make_bass_node_baselines(
        annual_bass=annual_bass,
        population=data.population,
        n_nodes=data.Y.shape[1],
    )
    
    node_predictions_raw = {
        "sssb": details["mu"],
        **bass_node_baselines,
    }
    
    if getattr(solver_cfg, "condition_on_seed_year", False):
        node_predictions, seed_idx = condition_node_predictions_on_seed(
            predictions=node_predictions_raw,
            Y_obs=data.Y,
            years=data.years,
            seed_year=int(solver_cfg.seed_year),
        )
        eval_start = seed_idx + 1
    else:
        node_predictions = node_predictions_raw
        seed_idx = -1
        eval_start = 0
    
    if eval_start >= data.Y.shape[0]:
        raise ValueError(
            f"No post-seed years available: seed_year={solver_cfg.seed_year}, "
            f"observed years={data.years[0]}-{data.years[-1]}."
        )
    
    # Post-seed arrays: use these for instantaneous/yearly-new metrics.
    Y_eval = data.Y[eval_start:, :]
    node_predictions_eval = {
        name: pred[eval_start:, :]
        for name, pred in node_predictions.items()
    }
    
    annual_obs_eval = annual_obs[eval_start:]
    annual_model_eval = annual_model[eval_start:]
    annual_bass_eval = annual_bass[eval_start:]
    
    # Full-window arrays: use these for cumulative spatial metrics.
    # node_predictions has already been seed-conditioned, so all models match
    # observed annual node counts through seed_year.
    obs_node_total_cum = data.Y.sum(axis=0)
    
    ranking_metrics = {}
    calibration_metrics = {}
    
    for model_name, pred_yi_full in node_predictions.items():
        pred_node_total_cum = np.asarray(pred_yi_full, dtype=float).sum(axis=0)
    
        ranking_metrics[model_name] = topk_hit_concentration(
            obs_node_total=obs_node_total_cum,
            pred_node_score=pred_node_total_cum,
            ks=(0.01, 0.05, 0.10, 0.20),
        )
    
        calibration_metrics[model_name] = calibration_by_risk_bin(
            obs_node_total=obs_node_total_cum,
            pred_node_total=pred_node_total_cum,
            n_bins=10,
        )
        
    annual_model_conditioned = np.asarray(node_predictions["sssb"], dtype=float).sum(axis=1)
    annual_bass_uniform_conditioned = np.asarray(node_predictions["bass_uniform"], dtype=float).sum(axis=1)
    annual_bass_conditioned = annual_bass_uniform_conditioned
    
    # Seed-conditioned curves for plotting cumulative totals and adoption rates.
    # These match the convention used by cumulative metrics:
    #   - observed annual node counts through seed_year;
    #   - model-predicted annual counts after seed_year.
    monthly_model_conditioned = np.repeat(annual_model_conditioned / 12.0, 12)
    cum_model_month = np.concatenate([[0.0], np.cumsum(monthly_model_conditioned)])
    
    monthly_bass_conditioned = np.repeat(annual_bass_conditioned / 12.0, 12)
    cum_bass_conditioned = np.concatenate([[0.0], np.cumsum(monthly_bass_conditioned)])
    
    _, rate_model = monthly_rate_curve_from_annual_counts(annual_model_conditioned)
    _, rate_bass = monthly_rate_curve_from_annual_counts(annual_bass_conditioned)
    
    rmse_yearly = yearly_rmse_counts(annual_obs_eval, annual_model_eval)
    rmse_yearly_log1p = yearly_rmse_log1p_counts(annual_obs_eval, annual_model_eval)
    mae_yearly = mae_counts(annual_obs_eval, annual_model_eval)
    mae_yearly_log1p = mae_log1p_counts(annual_obs_eval, annual_model_eval)
    
    rmse_yearly_bass = yearly_rmse_counts(annual_obs_eval, annual_bass_eval)
    rmse_yearly_log1p_bass = yearly_rmse_log1p_counts(annual_obs_eval, annual_bass_eval)
    mae_yearly_bass = mae_counts(annual_obs_eval, annual_bass_eval)
    mae_yearly_log1p_bass = mae_log1p_counts(annual_obs_eval, annual_bass_eval)
    
    rmse_cum = yearly_rmse_cumulative(annual_obs, annual_model_conditioned)
    rmse_cum_log1p = yearly_rmse_log1p_cumulative(annual_obs, annual_model_conditioned)
    mae_cum = mae_counts(np.cumsum(annual_obs), np.cumsum(annual_model_conditioned))
    mae_cum_log1p = mae_log1p_counts(np.cumsum(annual_obs), np.cumsum(annual_model_conditioned))
    
    rmse_cum_bass = yearly_rmse_cumulative(annual_obs, annual_bass_conditioned)
    rmse_cum_log1p_bass = yearly_rmse_log1p_cumulative(annual_obs, annual_bass_conditioned)
    mae_cum_bass = mae_counts(np.cumsum(annual_obs), np.cumsum(annual_bass_conditioned))
    mae_cum_log1p_bass = mae_log1p_counts(np.cumsum(annual_obs), np.cumsum(annual_bass_conditioned))
    
    deviance_sssb = poisson_deviance(annual_obs_eval, annual_model_eval)
    deviance_bass = poisson_deviance(annual_obs_eval, annual_bass_eval)
    
    spatial_node_metrics_instantaneous = nodewise_spatial_metrics(
        Y_obs=Y_eval,
        predictions=node_predictions_eval,
    )
    
    spatial_node_metrics_cumulative_full = nodewise_spatial_metrics(
        Y_obs=data.Y,
        predictions=node_predictions,
    )
    
    spatial_node_metrics = {}
    
    for model_name in node_predictions:
        spatial_node_metrics[model_name] = {
            "instantaneous": spatial_node_metrics_instantaneous[model_name]["instantaneous"],
            "instantaneous_nonzero": spatial_node_metrics_instantaneous[model_name]["instantaneous_nonzero"],
            "cumulative": spatial_node_metrics_cumulative_full[model_name]["cumulative"],
            "cumulative_nonzero": spatial_node_metrics_cumulative_full[model_name]["cumulative_nonzero"],
        }
        
    neighborhood_metrics_instantaneous = neighborhood_spatial_metrics(
        Y_obs=Y_eval,
        predictions=node_predictions_eval,
        L=data.L,
        data=data,
        max_level=3,
    )
    
    neighborhood_metrics_cumulative_full = neighborhood_spatial_metrics(
        Y_obs=data.Y,
        predictions=node_predictions,
        L=data.L,
        data=data,
        max_level=3,
    )
    
    neighborhood_metrics = {}
    
    for level_key in neighborhood_metrics_cumulative_full:
        neighborhood_metrics[level_key] = {
            **neighborhood_metrics_cumulative_full[level_key],
            "models": {},
        }
    
        for model_name in node_predictions:
            neighborhood_metrics[level_key]["models"][model_name] = {
                "instantaneous": neighborhood_metrics_instantaneous[level_key]["models"][model_name]["instantaneous"],
                "instantaneous_nonzero": neighborhood_metrics_instantaneous[level_key]["models"][model_name]["instantaneous_nonzero"],
                "cumulative": neighborhood_metrics_cumulative_full[level_key]["models"][model_name]["cumulative"],
                "cumulative_nonzero": neighborhood_metrics_cumulative_full[level_key]["models"][model_name]["cumulative_nonzero"],
            }

    out_dir = Path("out") / args.config / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    calendar_months = data.years[0] + t_obs_month

    ax.plot(calendar_months, cum_obs_month, label="Observed data")
    ax.plot(calendar_months, cum_bass_conditioned, label="Classic Bass fit, seed-conditioned")
    ax.plot(calendar_months, cum_model_month, label="SSSB expected cumulative, seed-conditioned")

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative LSPV adoptions")
    ax.set_title("Cumulative adoption curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_path = out_dir / "cumulative_adoption_curve.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(calendar_months, np.log1p(cum_obs_month), label="Observed data")
    ax.plot(calendar_months, np.log1p(cum_bass_conditioned), label="Classic Bass fit, seed-conditioned")
    ax.plot(calendar_months, np.log1p(cum_model_month), label="SSSB expected cumulative, seed-conditioned")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("log1p cumulative LSPV adoptions")
    ax.set_title("Cumulative adoption curve, log1p scale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig_path_log = out_dir / "cumulative_adoption_curve_log1p.png"
    fig.savefig(fig_path_log, dpi=200)
    plt.close(fig)
    
    field_plot_q = float(cfg_named["density"].get("field_plot_quantile", 1.0))
    field_plot_q = min(max(field_plot_q, 0.0), 1.0)
    
    field_values = {
        "U": details["cum_mu_U"],
        "V": details["cum_mu_V"],
        "I": details["final_I"],
        "J": details["final_J"],
    }
    
    print("[FIELD PLOT]")
    print(f"  field_plot_quantile = {field_plot_q}")
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.ravel()
    
    for ax, (name, vals) in zip(axes, field_values.items()):
        vals = np.asarray(vals, dtype=float)
        log_vals = np.log1p(np.clip(vals, 0.0, None))
        finite_log_vals = log_vals[np.isfinite(log_vals)]
    
        vmin_i = 0.0
    
        if finite_log_vals.size == 0:
            vmax_i = 1.0
        else:
            vmax_i = float(np.nanquantile(finite_log_vals, field_plot_q))
    
        if vmax_i <= vmin_i:
            vmax_i = 1.0
    
        print(
            f"  {name}: "
            f"raw max={float(np.nanmax(vals)):.6g}, "
            f"log1p vmax used={vmax_i:.6g}"
        )
    
        sc = plot_node_values(
            ax,
            data=data,
            values=vals,
            title=f"{name} at end of data period",
            epsg_project=int(cfg_named["mesh"]["epsg_project"]),
            vmin=vmin_i,
            vmax=vmax_i,
        )
    
        fig.colorbar(
            sc,
            ax=ax,
            fraction=0.045,
            pad=0.02,
            label=f"ln(1 + {name})",
        )
    
    fig_path_fields = out_dir / "final_fields_UV_IJ_log1p.png"
    fig.savefig(fig_path_fields, dpi=200)
    plt.close(fig)
    
    actual_cum_node = data.Y.sum(axis=0)
    pred_cum_node = node_predictions["sssb"].sum(axis=0)
    
    adoption_scale = str(cfg_named["density"].get("adoption_plot_scale", "log1p"))
    
    if adoption_scale == "linear":
        actual_scale = "linear"
        pred_scale = "linear"
        actual_vals_for_scale = actual_cum_node
        pred_vals_for_scale = pred_cum_node
        actual_cbar_label = "Observed cumulative count"
        pred_cbar_label = "Predicted cumulative mean"
        shared_cbar_label = "Cumulative count"
    
    elif adoption_scale == "log1p":
        actual_scale = "log1p"
        pred_scale = "log1p"
        actual_vals_for_scale = np.log1p(actual_cum_node)
        pred_vals_for_scale = np.log1p(pred_cum_node)
        actual_cbar_label = "ln(1 + observed cumulative count)"
        pred_cbar_label = "ln(1 + predicted cumulative mean)"
        shared_cbar_label = "ln(1 + cumulative count)"
    
    elif adoption_scale == "mixed":
        actual_scale = "linear"
        pred_scale = "custom_log"
    
        actual_vals_for_scale = actual_cum_node
    
        mixed_log_range = float(cfg_named["density"].get("mixed_log_range", 8.0))
        vmax_pred = float(np.nanmax(pred_cum_node))
    
        if vmax_pred <= 0:
            pred_floor = -mixed_log_range
            pred_vals_for_scale = np.zeros_like(pred_cum_node)
        else:
            pred_floor = np.log(vmax_pred) - mixed_log_range
            pred_vals_for_scale = np.log(
                np.maximum(pred_cum_node, np.exp(pred_floor))
            )
    
        actual_cbar_label = "Observed cumulative count"
        pred_cbar_label = "ln(predicted cumulative mean count)"
    
    else:
        raise ValueError("adoption_plot_scale must be 'linear', 'log1p', or 'mixed'.")
        
    shared_colorbar = bool(cfg_named["density"].get("adoption_shared_colorbar", False))
    if adoption_scale == "mixed":
        shared_colorbar = False
    
    plot_q = float(cfg_named["density"].get("adoption_plot_quantile", 1.0))
    plot_q = min(max(plot_q, 0.0), 1.0)
    
    if shared_colorbar:
        combined_vals_for_scale = np.concatenate([
            np.asarray(actual_vals_for_scale, dtype=float).ravel(),
            np.asarray(pred_vals_for_scale, dtype=float).ravel(),
        ])
    
        shared_vmin = 0.0
        shared_vmax = float(np.nanquantile(combined_vals_for_scale, plot_q))
        if shared_vmax <= shared_vmin:
            shared_vmax = 1.0
    
        actual_vmin = pred_vmin = shared_vmin
        actual_vmax = pred_vmax = shared_vmax
    
    else:
        actual_vmin = 0.0
        actual_vmax = float(np.nanquantile(actual_vals_for_scale, plot_q))
        if actual_vmax <= actual_vmin:
            actual_vmax = 1.0
    
        if adoption_scale == "mixed":
            pred_vmin = float(np.nanmin(pred_vals_for_scale))
            pred_vmax = float(np.nanmax(pred_vals_for_scale))
        else:
            pred_vmin = 0.0
            pred_vmax = float(np.nanquantile(pred_vals_for_scale, plot_q))
    
        if pred_vmax <= pred_vmin:
            pred_vmax = pred_vmin + 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    sc0 = plot_node_values(
        axes[0],
        data=data,
        values=actual_cum_node,
        title="Observed cumulative adoptions",
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        vmin=actual_vmin,
        vmax=actual_vmax,
        scale=actual_scale,
    )
    
    sc1 = plot_node_values(
        axes[1],
        data=data,
        values=pred_cum_node,
        title="Predicted cumulative mean counts",
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        vmin=pred_vmin,
        vmax=pred_vmax,
        scale=pred_scale,
    )
    
    if shared_colorbar:
        fig.colorbar(
            sc1,
            ax=axes.ravel().tolist(),
            fraction=0.035,
            pad=0.02,
            label=actual_cbar_label if actual_cbar_label == pred_cbar_label else "Cumulative count",
        )
    else:
        fig.colorbar(
            sc0,
            ax=axes[0],
            fraction=0.045,
            pad=0.02,
            label=actual_cbar_label,
        )
    
        fig.colorbar(
            sc1,
            ax=axes[1],
            fraction=0.045,
            pad=0.02,
            label=pred_cbar_label,
        )
    
    fig_path_spatial = out_dir / f"observed_vs_predicted_cumulative_{adoption_scale}.png"
    fig.savefig(fig_path_spatial, dpi=200)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(calendar_rate_months, rate_obs, label="Observed data")
    ax.plot(calendar_rate_months, rate_bass, label="Classic Bass fit, seed-conditioned")
    ax.plot(calendar_rate_months, rate_model, label="SSSB expected rate, seed-conditioned")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Adoption rate, adoptions/year")
    ax.set_title("Adoption rate curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig_path_rate = out_dir / "adoption_rate_curve.png"
    fig.savefig(fig_path_rate, dpi=200)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(calendar_rate_months, np.log1p(rate_obs), label="Observed data")
    ax.plot(calendar_rate_months, np.log1p(rate_bass), label="Classic Bass fit, seed-conditioned")
    ax.plot(calendar_rate_months, np.log1p(rate_model), label="SSSB expected rate, seed-conditioned")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("ln(1 + adoption rate)")
    ax.set_title("Adoption rate curve, log1p scale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig_path_rate_log = out_dir / "adoption_rate_curve_log1p.png"
    fig.savefig(fig_path_rate_log, dpi=200)
    plt.close(fig)
    
    metrics = {
        "config": args.config,
        "fit_json": str(fit_json),
        "nll": float(nll),
        "years": data.years.astype(int).tolist(),
        "observed_total_full_window": float(annual_obs.sum()),
        "model_expected_total_full_window": float(annual_model.sum()),
        "observed_total_instantaneous_metric_window": float(annual_obs_eval.sum()),
        "model_expected_total_instantaneous_metric_window": float(annual_model_eval.sum()),
        "metric_window_start_year": int(data.years[eval_start]),
        "metric_window_end_year": int(data.years[-1]),
        "aggregate_metrics": {
            "yearly_counts": {
                "sssb": {
                    "RMSE": rmse_yearly,
                    "MAE": mae_yearly,
                    "RMSE_log1p": rmse_yearly_log1p,
                    "MAE_log1p": mae_yearly_log1p,
                },
                "bass": {
                    "RMSE": rmse_yearly_bass,
                    "MAE": mae_yearly_bass,
                    "RMSE_log1p": rmse_yearly_log1p_bass,
                    "MAE_log1p": mae_yearly_log1p_bass,
                },
            },
            "cumulative_counts": {
                "sssb": {
                    "RMSE": rmse_cum,
                    "MAE": mae_cum,
                    "RMSE_log1p": rmse_cum_log1p,
                    "MAE_log1p": mae_cum_log1p,
                },
                "bass": {
                    "RMSE": rmse_cum_bass,
                    "MAE": mae_cum_bass,
                    "RMSE_log1p": rmse_cum_log1p_bass,
                    "MAE_log1p": mae_cum_log1p_bass,
                },
            },
        },
        "nodewise_spatial_metrics": spatial_node_metrics,
        "poisson_deviance": {
            "sssb": deviance_sssb,
            "bass": deviance_bass,
        },
        "plots": {
            "cumulative_linear": str(fig_path),
            "cumulative_log1p": str(fig_path_log),
            "final_fields_log1p": str(fig_path_fields),
            "observed_vs_predicted_spatial_log1p": str(fig_path_spatial),
            "adoption_rate_linear": str(fig_path_rate),
            "adoption_rate_log1p": str(fig_path_rate_log),
        },
        "topk_hit_concentration": ranking_metrics,
        "calibration_by_risk_bin": calibration_metrics,
        "neighborhood_spatial_metrics": neighborhood_metrics,
        "conditioned_seed_year": int(solver_cfg.seed_year) if getattr(solver_cfg, "condition_on_seed_year", False) else None,
        "instantaneous_metric_years": data.years[eval_start:].astype(int).tolist(),
        "cumulative_metric_years": data.years.astype(int).tolist(),
        "plot_years": data.years.astype(int).tolist(),
        "observed_total_post_seed": float(annual_obs_eval.sum()),
        "model_expected_total_full_window_conditioned": float(annual_model_conditioned.sum()),
        "model_expected_total_post_seed": float(annual_model_eval.sum()),
        "classic_bass_fit": {
            "uniform": bass,
            "population": bass,
            "note": (
                "Uniform and population-informed Bass baselines share the same aggregate Bass p, q, M; they differ only in spatial allocation."
            ),
        },
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[METRICS]")
    print(json.dumps(metrics, indent=2))
    print("[PLOT] wrote:", fig_path)
    print("[PLOT] wrote:", fig_path_log)
    print("[PLOT] wrote:", fig_path_fields)
    print("[PLOT] wrote:", fig_path_spatial)
    print("[DATA] wrote:", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())