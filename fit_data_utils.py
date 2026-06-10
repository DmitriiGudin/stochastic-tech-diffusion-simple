from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import meshio
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from pyproj import Transformer
import matplotlib.tri as mtri


@dataclass(frozen=True)
class SSSBFitData:
    Y: np.ndarray                         # (n_years, n_nodes)
    years: np.ndarray                     # (n_years,)
    population: np.ndarray                # (n_nodes,)
    pv_potential: np.ndarray              # (n_nodes,)
    transmission_distance_km: np.ndarray  # (n_nodes,)
    L: csr_matrix                         # sparse graph Laplacian
    mesh_points_km: np.ndarray            # (n_nodes, 2)
    triangles: np.ndarray                 # (n_triangles, 3)


def load_mesh_points_and_triangles(msh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(msh_path)

    points = np.asarray(mesh.points[:, :2], dtype=float)

    triangles = None
    for block in mesh.cells:
        if block.type == "triangle":
            triangles = np.asarray(block.data, dtype=np.int64)
            break

    if triangles is None:
        raise ValueError(f"No triangle cells found in {msh_path}")

    return points, triangles


def build_normalized_graph_laplacian(
    *,
    n_nodes: int,
    triangles: np.ndarray,
) -> csr_matrix:
    """
    Build normalized graph Laplacian

        (LJ)_i = mean_{j ~ i}(J_j - J_i).

    Matrix form:
        L_ij = 1/deg(i) for neighbors j
        L_ii = -1
    """
    neighbors = [set() for _ in range(n_nodes)]

    for a, b, c in np.asarray(triangles, dtype=np.int64):
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    rows = []
    cols = []
    vals = []

    for i, nb_set in enumerate(neighbors):
        nb = sorted(nb_set)
        deg = len(nb)

        if deg == 0:
            rows.append(i)
            cols.append(i)
            vals.append(0.0)
            continue

        # diagonal
        rows.append(i)
        cols.append(i)
        vals.append(-1.0)

        # neighbors
        w = 1.0 / deg
        for j in nb:
            rows.append(i)
            cols.append(j)
            vals.append(w)

    L = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return L


def _load_lspv_events(lspv_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lspv_csv)

    required = {"xlong", "ylat", "p_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LSPV file missing columns: {sorted(missing)}")

    out = pd.DataFrame({
        "lon": pd.to_numeric(df["xlong"], errors="coerce"),
        "lat": pd.to_numeric(df["ylat"], errors="coerce"),
        "year": pd.to_numeric(df["p_year"], errors="coerce"),
    })

    out = out.dropna(subset=["lon", "lat", "year"]).copy()
    out["year"] = out["year"].astype(int)
    return out


def _events_inside_mesh_mask(
    *,
    mesh_points_km: np.ndarray,
    triangles: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    epsg_project: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return inside-mask and projected event coordinates.
    """
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)
    x_m, y_m = tr.transform(lon, lat)
    xy_km = np.column_stack([np.asarray(x_m) / 1000.0, np.asarray(y_m) / 1000.0])

    triang = mtri.Triangulation(mesh_points_km[:, 0], mesh_points_km[:, 1], triangles)
    finder = triang.get_trifinder()

    tri_id = finder(xy_km[:, 0], xy_km[:, 1])
    inside = tri_id >= 0

    return inside, xy_km


def apply_year_window(
    Y: np.ndarray,
    years: np.ndarray,
    year_window,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict annual count matrix to inclusive year window [start, end].

    year_window may be None or [start, end].
    """
    if year_window is None:
        return Y, years

    if len(year_window) != 2:
        raise ValueError("year_window must be None or [start_year, end_year].")

    start, end = int(year_window[0]), int(year_window[1])
    if start > end:
        raise ValueError("year_window start must be <= end.")

    mask = (years >= start) & (years <= end)
    return Y[mask], years[mask]


def build_annual_node_count_matrix(
    *,
    msh_path: Path,
    lspv_csv: Path,
    epsg_project: int = 5070,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Y[y,i] = number of LSPV adoptions at node i during year y.

    Events are assigned to nearest node after filtering to those inside mesh.
    """
    mesh_points_km, triangles = load_mesh_points_and_triangles(msh_path)
    n_nodes = mesh_points_km.shape[0]

    events = _load_lspv_events(lspv_csv)

    inside, xy_km = _events_inside_mesh_mask(
        mesh_points_km=mesh_points_km,
        triangles=triangles,
        lon=events["lon"].to_numpy(float),
        lat=events["lat"].to_numpy(float),
        epsg_project=epsg_project,
    )

    events_inside = events.loc[inside].copy()
    xy_inside = xy_km[inside]

    if events_inside.empty:
        years = np.array([], dtype=int)
        Y = np.zeros((0, n_nodes), dtype=float)
        return Y, years, mesh_points_km, triangles

    first_year = int(events_inside["year"].min())
    last_year = int(events_inside["year"].max())
    years = np.arange(first_year, last_year + 1, dtype=int)

    tree = cKDTree(mesh_points_km)
    _, node_idx = tree.query(xy_inside, k=1)

    year_to_idx = {int(y): k for k, y in enumerate(years)}

    Y = np.zeros((len(years), n_nodes), dtype=float)

    for year, node in zip(events_inside["year"].to_numpy(int), node_idx.astype(int)):
        Y[year_to_idx[int(year)], int(node)] += 1.0

    return Y, years, mesh_points_km, triangles


def load_node_features(node_features_npz: Path) -> dict:
    data = np.load(node_features_npz, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_sssb_fit_data(
    *,
    msh_path: Path,
    node_features_npz: Path,
    lspv_csv: Path,
    epsg_project: int = 5070,
    population_key: str = "population_smooth_2020",
    year_window=None,
) -> SSSBFitData:
    """
    Build all fixed arrays needed for SSSB likelihood optimization.
    """
    Y, years, mesh_points_km, triangles = build_annual_node_count_matrix(
        msh_path=msh_path,
        lspv_csv=lspv_csv,
        epsg_project=epsg_project,
    )
    Y, years = apply_year_window(Y, years, year_window)

    n_nodes = mesh_points_km.shape[0]
    features = load_node_features(node_features_npz)

    if population_key not in features:
        raise KeyError(f"Missing population key {population_key} in {node_features_npz}")

    population = np.asarray(features[population_key], dtype=float)

    pv = np.asarray(features.get("pv_potential", np.full(n_nodes, np.nan)), dtype=float)
    grid = np.asarray(features.get("transmission_distance_km", np.full(n_nodes, np.nan)), dtype=float)

    if population.shape[0] != n_nodes:
        raise ValueError("population length does not match number of mesh nodes.")
    if pv.shape[0] != n_nodes:
        raise ValueError("pv_potential length does not match number of mesh nodes.")
    if grid.shape[0] != n_nodes:
        raise ValueError("transmission_distance_km length does not match number of mesh nodes.")

    L = build_normalized_graph_laplacian(
        n_nodes=n_nodes,
        triangles=triangles,
    )

    return SSSBFitData(
        Y=Y,
        years=years,
        population=population,
        pv_potential=pv,
        transmission_distance_km=grid,
        L=L,
        mesh_points_km=mesh_points_km,
        triangles=triangles,
    )