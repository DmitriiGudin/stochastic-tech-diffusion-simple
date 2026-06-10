from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy.spatial import cKDTree
from pyproj import Transformer

from shapely.geometry import Point
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Polygon
from shapely.ops import transform as shapely_transform, unary_union
from shapely.strtree import STRtree


# ============================================================
# Mesh loading / coordinate utilities
# ============================================================

def load_mesh_points_km(msh_path: Path) -> np.ndarray:
    """
    Return mesh node coordinates in projected km coordinates.

    Returns
    -------
    points_km : ndarray, shape (n_nodes, 2)
    """
    mesh = meshio.read(msh_path)
    return np.asarray(mesh.points[:, :2], dtype=float)


def load_mesh_triangles(msh_path: Path) -> np.ndarray:
    """
    Return triangle connectivity from a .msh file.

    Returns
    -------
    tri : ndarray, shape (n_triangles, 3)
    """
    mesh = meshio.read(msh_path)

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            return np.asarray(cell_block.data, dtype=np.int64)

    raise ValueError(f"No triangle cells found in {msh_path}")


def mesh_points_lonlat(
    msh_path: Path,
    epsg_project: int = 5070,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert mesh node coordinates from EPSG-projected km coordinates to lon/lat.
    """
    pts_km = load_mesh_points_km(msh_path)
    pts_m = pts_km * 1000.0

    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_m[:, 0], pts_m[:, 1])

    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def project_lonlat_to_km(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg_project: int = 5070,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project lon/lat to EPSG-projected km coordinates.
    """
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)

    x_m, y_m = fwd.transform(
        np.asarray(lon, dtype=float),
        np.asarray(lat, dtype=float),
    )

    return np.asarray(x_m, dtype=float) / 1000.0, np.asarray(y_m, dtype=float) / 1000.0


# ============================================================
# Population CSV loading
# ============================================================

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_col = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]

    raise ValueError(
        f"Could not find any of columns {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_zip_population_csv(pop_csv: Path) -> pd.DataFrame:
    """
    Load ZIP population CSV.

    Expected flexible columns:
      longitude: longitude / lon / lng
      latitude : latitude / lat
      pop2010  : population_2010 / pop2010 / POP2010 / population2010
      pop2020  : population_2020 / pop2020 / POP2020 / population2020

    Returns dataframe with standardized columns:
      longitude, latitude, pop2010, pop2020
    """
    df = pd.read_csv(pop_csv)

    lon_col = _find_column(df, ["longitude", "lon", "lng"])
    lat_col = _find_column(df, ["latitude", "lat"])
    p10_col = _find_column(df, ["population_2010", "pop2010", "POP2010", "population2010"])
    p20_col = _find_column(df, ["population_2020", "pop2020", "POP2020", "population2020"])

    out = pd.DataFrame({
        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
        "pop2010": pd.to_numeric(df[p10_col], errors="coerce"),
        "pop2020": pd.to_numeric(df[p20_col], errors="coerce"),
    })

    out = out.dropna(subset=["longitude", "latitude", "pop2010", "pop2020"]).copy()
    out["pop2010"] = np.clip(out["pop2010"].to_numpy(float), 0.0, None)
    out["pop2020"] = np.clip(out["pop2020"].to_numpy(float), 0.0, None)

    return out


def extrapolate_population(
    pop2010: np.ndarray,
    pop2020: np.ndarray,
    year: float,
) -> np.ndarray:
    """
    Exponential interpolation/extrapolation between 2010 and 2020:

        p(t) = p2010 exp(g (t - 2010)),
        g = log(p2020 / p2010) / 10.

    Rules:
      - if both years are positive: exponential growth/decay
      - if only 2010 is positive: constant at 2010 value
      - if only 2020 is positive: constant at 2020 value
      - otherwise zero
    """
    pop2010 = np.asarray(pop2010, dtype=float)
    pop2020 = np.asarray(pop2020, dtype=float)
    year = float(year)

    out = np.zeros_like(pop2010, dtype=float)

    both = (pop2010 > 0) & (pop2020 > 0)
    if np.any(both):
        g = np.log(pop2020[both] / pop2010[both]) / 10.0
        out[both] = pop2010[both] * np.exp(g * (year - 2010.0))

    only10 = (pop2010 > 0) & (pop2020 <= 0)
    only20 = (pop2010 <= 0) & (pop2020 > 0)

    out[only10] = pop2010[only10]
    out[only20] = pop2020[only20]

    return np.clip(out, 0.0, None)


# ============================================================
# LSPV CSV loading
# ============================================================

def load_lspv_events(lspv_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lspv_csv)

    required = {"xlong", "ylat", "p_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LSPV CSV missing columns: {sorted(missing)}")

    out = pd.DataFrame({
        "longitude": pd.to_numeric(df["xlong"], errors="coerce"),
        "latitude": pd.to_numeric(df["ylat"], errors="coerce"),
        "p_year": pd.to_numeric(df["p_year"], errors="coerce"),
    })

    out = out.dropna(subset=["longitude", "latitude", "p_year"]).copy()
    out["p_year"] = out["p_year"].astype(int)

    return out


def map_lspv_adoptions_nearest_node(
    *,
    msh_path: Path,
    lspv_csv: Path,
    epsg_project: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Map LSPV events inside the mesh to the nearest mesh node.

    Returns
    -------
    node_counts : ndarray, shape (n_nodes,)
    events_inside : dataframe with projected coordinates and nearest node.
    """
    mesh_points = load_mesh_points_km(msh_path)
    events = load_lspv_events(lspv_csv)

    inside, x_km, y_km = zip_points_inside_mesh_mask(
        msh_path=msh_path,
        lon=events["longitude"].to_numpy(float),
        lat=events["latitude"].to_numpy(float),
        epsg_project=epsg_project,
    )

    events_inside = events.loc[inside].copy()
    events_inside["x_km"] = x_km[inside]
    events_inside["y_km"] = y_km[inside]

    node_counts = np.zeros(mesh_points.shape[0], dtype=float)

    if not events_inside.empty:
        tree = cKDTree(mesh_points)
        _, node_idx = tree.query(events_inside[["x_km", "y_km"]].to_numpy(float), k=1)
        events_inside["nearest_node"] = node_idx.astype(int)
        np.add.at(node_counts, node_idx.astype(int), 1.0)
    else:
        events_inside["nearest_node"] = np.array([], dtype=int)

    return node_counts, events_inside


def lspv_year_summary(events_inside: pd.DataFrame) -> dict:
    """
    Global yearly counts for all LSPV events inside the mesh.
    If zero events, returns total_count=0 and years={}.
    """
    total = int(len(events_inside))
    if total == 0:
        return {
            "total_count": 0,
            "first_year": None,
            "last_year": None,
            "years": {},
        }

    first = int(events_inside["p_year"].min())
    last = int(events_inside["p_year"].max())

    counts = (
        events_inside.groupby("p_year")
        .size()
        .reindex(range(first, last + 1), fill_value=0)
    )

    return {
        "total_count": total,
        "first_year": first,
        "last_year": last,
        "years": {str(int(y)): int(v) for y, v in counts.items()},
    }


def lspv_county_year_summary(
    *,
    events_inside: pd.DataFrame,
    county_shp: Path,
    state_codes: list[str],
    county_names: list[str],
) -> dict:
    """
    County-level yearly LSPV counts.

    Uses the global first/last adoption years from events_inside.
    If there are zero global adoptions, returns {}.
    """
    if events_inside.empty:
        return {}

    if not county_names:
        return {}

    first = int(events_inside["p_year"].min())
    last = int(events_inside["p_year"].max())
    years = list(range(first, last + 1))

    counties = gpd.read_file(county_shp).to_crs("EPSG:4326")

    if "NAME" not in counties.columns or "STATEFP" not in counties.columns:
        raise ValueError("County shapefile must contain NAME and STATEFP columns.")

    state_fips = {
        "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
        "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
        "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
        "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
        "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
        "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
        "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
        "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
        "WV": "54", "WI": "55", "WY": "56",
    }

    requested_fips = {state_fips[s] for s in state_codes if s in state_fips}
    requested_counties_clean = {str(c).strip().lower() for c in county_names}

    counties = counties.copy()
    counties["county_clean"] = counties["NAME"].astype(str).str.strip().str.lower()
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)

    counties = counties[
        counties["STATEFP"].isin(requested_fips)
        & counties["county_clean"].isin(requested_counties_clean)
    ].copy()

    if counties.empty:
        return {}

    points = gpd.GeoDataFrame(
        events_inside.copy(),
        geometry=gpd.points_from_xy(events_inside["longitude"], events_inside["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        counties[["NAME", "geometry"]],
        how="left",
        predicate="within",
    )

    out = {}
    for county_name in counties["NAME"].astype(str).tolist():
        sub = joined[joined["NAME"] == county_name]
        counts = (
            sub.groupby("p_year")
            .size()
            .reindex(years, fill_value=0)
        )

        out[county_name] = {
            "total_count": int(len(sub)),
            "years": {str(int(y)): int(v) for y, v in counts.items()},
        }

    return out


def plot_lspv_adoptions_nearest_node(
    *,
    msh_path: Path,
    node_counts: np.ndarray,
    out_png: Path,
    epsg_project: int,
    scale: str = "linear"
) -> None:
    """
    Sparse LSPV plot:
      - draw the mesh in light grey
      - draw colored dots only at nodes with at least one adoption
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pts_km = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)

    pts_m = pts_km * 1000.0
    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_m[:, 0], pts_m[:, 1])
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    node_counts = np.asarray(node_counts, dtype=float)
    nz = node_counts > 0

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    triang = mtri.Triangulation(lon, lat, tri)
    ax.triplot(triang, linewidth=0.25, color="0.75", alpha=0.65)
    
    # Draw outer mesh boundary clearly
    edge_count = {}
    for a, b, c in tri:
        for e in [(a, b), (b, c), (c, a)]:
            e = tuple(sorted(e))
            edge_count[e] = edge_count.get(e, 0) + 1
    
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    
    for a, b in boundary_edges:
        ax.plot(
            [lon[a], lon[b]],
            [lat[a], lat[b]],
            color="black",
            linewidth=1.5,
            alpha=1.0,
            zorder=3,
        )
        
    if scale == "linear":
        plot_counts = node_counts
        colorbar_label = "Adoptions / node"
        vmin = 1.0
        vmax = max(float(np.nanmax(node_counts[nz])), 1.0) if np.any(nz) else 1.0
    elif scale == "log1p":
        plot_counts = np.log1p(node_counts)
        colorbar_label = "ln(1 + adoptions / node)"
        vmin = np.log1p(1.0)
        vmax = max(float(np.nanmax(plot_counts[nz])), np.log1p(1.0)) if np.any(nz) else 1.0
    else:
        raise ValueError("scale must be 'linear' or 'log1p'.")

    if np.any(nz):
        sc = ax.scatter(
            lon[nz],
            lat[nz],
            c=plot_counts[nz],
            s=35,
            linewidths=0.25,
            edgecolors="black",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label=colorbar_label)
    else:
        ax.text(
            0.5,
            0.5,
            "No LSPV adoptions inside mesh",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("LSPV adoptions assigned to nearest mesh node")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")
    
    
# ============================================================
# Transmission line utilities
# ============================================================

def _geometry_to_km(geom):
    """
    Convert projected-meter shapely geometry to projected-km geometry.
    """
    return shapely_transform(lambda x, y, z=None: (np.asarray(x) / 1000.0, np.asarray(y) / 1000.0), geom)


def _extract_lines(geom) -> list:
    """
    Extract LineString components from arbitrary shapely geometry.
    """
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, LineString):
        return [geom]

    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_extract_lines(g))
        return out

    return []


def _mesh_triangle_polygons_km(msh_path: Path) -> list[Polygon]:
    pts = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)

    polys = []
    for a, b, c in tri:
        poly = Polygon([
            tuple(pts[a]),
            tuple(pts[b]),
            tuple(pts[c]),
        ])
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polys.append(poly)
        else:
            poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                polys.append(poly)

    return polys


def _load_transmission_lines_km_near_mesh(
    *,
    transmission_shp: Path,
    msh_path: Path,
    epsg_project: int,
    buffer_km: float,
) -> list:
    """
    Load transmission lines, project to EPSG coordinates, convert meters to km,
    and retain only lines intersecting a buffered mesh domain.
    """
    if not transmission_shp.exists():
        raise FileNotFoundError(transmission_shp)

    gdf = gpd.read_file(transmission_shp)
    if gdf.empty:
        return []

    gdf = gdf.to_crs(f"EPSG:{epsg_project}")

    lines = []
    for geom in gdf.geometry.values:
        geom_km = _geometry_to_km(geom)
        lines.extend(_extract_lines(geom_km))

    if not lines:
        return []

    # Build mesh domain polygon from triangles, then buffer outward.
    tri_polys = _mesh_triangle_polygons_km(msh_path)
    if not tri_polys:
        return []

    domain = unary_union(tri_polys)
    domain_buffered = domain.buffer(float(buffer_km))

    # Fast bbox prefilter before exact intersection.
    minx, miny, maxx, maxy = domain_buffered.bounds
    retained = []
    for line in lines:
        lx0, ly0, lx1, ly1 = line.bounds
        if lx1 < minx or lx0 > maxx or ly1 < miny or ly0 > maxy:
            continue
        if line.intersects(domain_buffered):
            retained.append(line)

    return retained


def _strtree_query(tree: STRtree, geom):
    """
    Compatibility wrapper for Shapely 1.x and 2.x.

    Shapely 1.x STRtree.query returns geometries.
    Shapely 2.x STRtree.query returns integer indices.
    """
    out = tree.query(geom)
    return out


def _tree_items_from_query(query_result, geoms: list):
    if len(query_result) == 0:
        return []

    first = query_result[0]
    if isinstance(first, (int, np.integer)):
        return [geoms[int(i)] for i in query_result]

    return list(query_result)


def _nearest_line_distance_km(point, tree: STRtree, lines: list) -> float:
    """
    Compatibility wrapper for nearest-distance queries.
    """
    nearest = tree.nearest(point)

    # Shapely 2.x returns index; Shapely 1.x returns geometry.
    if isinstance(nearest, (int, np.integer)):
        line = lines[int(nearest)]
    else:
        line = nearest

    return float(point.distance(line))


def map_transmission_distance_to_nodes(
    *,
    msh_path: Path,
    transmission_shp: Path,
    epsg_project: int,
    buffer_km: float = 15.0,
) -> dict:
    """
    Compute nodewise distance to nearest transmission line.

    Rule:
      - if any transmission segment intersects one of the triangles incident
        to a node, that node receives distance 0;
      - otherwise distance is Euclidean distance in projected km coordinates
        to nearest retained transmission-line segment.

    Lines are retained if they intersect the mesh domain buffered by buffer_km.
    """
    pts = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)
    n_nodes = pts.shape[0]

    lines = _load_transmission_lines_km_near_mesh(
        transmission_shp=transmission_shp,
        msh_path=msh_path,
        epsg_project=epsg_project,
        buffer_km=buffer_km,
    )

    distances = np.full(n_nodes, np.nan, dtype=float)

    if not lines:
        return {
            "transmission_distance_km": distances,
            "transmission_lines_retained": np.array([0], dtype=int),
            "transmission_buffer_km": np.array([float(buffer_km)]),
            "transmission_nodes_zero_count": np.array([0], dtype=int),
            "transmission_distance_min_km": np.array([np.nan]),
            "transmission_distance_median_km": np.array([np.nan]),
            "transmission_distance_max_km": np.array([np.nan]),
        }

    tree = STRtree(lines)

    # ------------------------------------------------------------
    # Step 1: zero-distance rule from triangle intersections.
    # ------------------------------------------------------------
    zero_nodes = np.zeros(n_nodes, dtype=bool)

    for a, b, c in tri:
        poly = Polygon([tuple(pts[a]), tuple(pts[b]), tuple(pts[c])])
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            continue

        hits_raw = _strtree_query(tree, poly)
        hits = _tree_items_from_query(hits_raw, lines)

        for line in hits:
            if line.intersects(poly):
                zero_nodes[[a, b, c]] = True
                break

    distances[zero_nodes] = 0.0

    # ------------------------------------------------------------
    # Step 2: nearest distance for all remaining nodes.
    # ------------------------------------------------------------
    remaining = np.flatnonzero(~zero_nodes)

    for i in remaining:
        p = Point(float(pts[i, 0]), float(pts[i, 1]))
        distances[i] = _nearest_line_distance_km(p, tree, lines)

    finite = distances[np.isfinite(distances)]

    return {
        "transmission_distance_km": distances,
        "transmission_lines_retained": np.array([len(lines)], dtype=int),
        "transmission_buffer_km": np.array([float(buffer_km)]),
        "transmission_nodes_zero_count": np.array([int(np.count_nonzero(zero_nodes))], dtype=int),
        "transmission_distance_min_km": np.array([float(np.nanmin(finite)) if finite.size else np.nan]),
        "transmission_distance_median_km": np.array([float(np.nanmedian(finite)) if finite.size else np.nan]),
        "transmission_distance_max_km": np.array([float(np.nanmax(finite)) if finite.size else np.nan]),
    }


def plot_transmission_distance_nodes(
    *,
    msh_path: Path,
    distances_km: np.ndarray,
    out_png: Path,
    epsg_project: int,
) -> None:
    """
    Diagnostic plot of nodewise distance to nearest transmission line.
    Uses log1p scale because distances can have long tails.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pts_km = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)

    pts_m = pts_km * 1000.0
    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_m[:, 0], pts_m[:, 1])
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    vals = np.asarray(distances_km, dtype=float)
    plot_vals = np.log1p(np.clip(vals, 0.0, None))

    vmax = float(np.nanquantile(plot_vals, 0.99)) if plot_vals.size else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    triang = mtri.Triangulation(lon, lat, tri)
    ax.triplot(triang, linewidth=0.2, color="0.8", alpha=0.6)

    finite = np.isfinite(plot_vals)
    s = max(6.0, 5000.0 / max(np.sqrt(len(vals)), 1.0)) * 0.35
    
    finite_vals = plot_vals[finite]
    if finite_vals.size == 0:
        layer_edges = np.array([0.0, vmax])
    else:
        qs = np.linspace(0.0, 1.0, 11)
        layer_edges = np.nanquantile(finite_vals, qs)
        layer_edges = np.unique(layer_edges)
        if layer_edges.size < 2:
            layer_edges = np.array([0.0, vmax])
    
    sc = None
    
    for ell in range(layer_edges.size - 1):
        lo = layer_edges[ell]
        hi = layer_edges[ell + 1]
    
        if ell == layer_edges.size - 2:
            mask = finite & (plot_vals >= lo) & (plot_vals <= hi)
        else:
            mask = finite & (plot_vals >= lo) & (plot_vals < hi)
    
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
    
        idx = idx[np.argsort(plot_vals[idx])]
    
        sc = ax.scatter(
            lon[idx],
            lat[idx],
            c=plot_vals[idx],
            s=s,
            linewidths=0.0,
            vmin=0.0,
            vmax=vmax,
            alpha=0.55 + 0.45 * (ell + 1) / max(layer_edges.size - 1, 1),
            zorder=2 + ell,
        )
    
    if sc is None:
        sc = ax.scatter(
            lon,
            lat,
            c=np.zeros_like(plot_vals),
            s=s,
            linewidths=0.0,
            vmin=0.0,
            vmax=vmax,
            alpha=0.55,
            zorder=2,
        )
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="log1p distance to line [km]")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Distance to nearest transmission line")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")
    
    
# ============================================================
# GIS solar data utilities
# ============================================================

def map_pvout_to_nodes(
    *,
    msh_path: Path,
    pvout_tif: Path,
    epsg_project: int,
) -> dict:
    """
    Sample Photovoltaic Electricity Potential raster at mesh nodes.

    Returns
    -------
    dict containing:
        pv_potential : ndarray, shape (n_nodes,)
    """
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError(
            "PVOUT mapping requires rasterio. Install with e.g. "
            "`conda install -c conda-forge rasterio`."
        ) from e

    if not pvout_tif.exists():
        raise FileNotFoundError(pvout_tif)

    pts_km = load_mesh_points_km(msh_path)
    pts_m = pts_km * 1000.0

    with rasterio.open(pvout_tif) as src:
        raster_crs = src.crs

        if raster_crs is None:
            raise ValueError(f"Raster {pvout_tif} has no CRS.")

        transformer = Transformer.from_crs(
            f"EPSG:{epsg_project}",
            raster_crs,
            always_xy=True,
        )

        xs, ys = transformer.transform(pts_m[:, 0], pts_m[:, 1])
        coords = list(zip(xs, ys))

        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

        nodata = src.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan

    return {
        "pv_potential": vals,
        "pv_potential_valid_count": np.array([int(np.count_nonzero(np.isfinite(vals)))]),
        "pv_potential_min": np.array([float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan]),
        "pv_potential_median": np.array([float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan]),
        "pv_potential_max": np.array([float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan]),
    }


def plot_pv_potential_nodes(
    *,
    msh_path: Path,
    pv_values: np.ndarray,
    out_png: Path,
    epsg_project: int,
) -> None:
    """
    Linear-scale diagnostic plot of PV electricity potential sampled at mesh nodes.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vals = np.asarray(pv_values, dtype=float)
    finite = np.isfinite(vals)

    if np.any(finite):
        vmin = float(np.nanquantile(vals[finite], 0.01))
        vmax = float(np.nanquantile(vals[finite], 0.99))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    sc = _node_circle_plot(
        msh_path=msh_path,
        values=vals,
        ax=ax,
        epsg_project=epsg_project,
        title="PV electricity potential at mesh nodes",
        vmin=vmin,
        vmax=vmax,
        transform="linear",
    )

    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="Specific yield (kWh/kWp) / yr")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")


# ============================================================
# Mesh filtering / node control areas
# ============================================================

def zip_points_inside_mesh_mask(
    *,
    msh_path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    epsg_project: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return mask of ZIP centroids inside the mesh.

    Returns
    -------
    inside : ndarray bool, shape (n_zip,)
    x_km   : projected x coordinates
    y_km   : projected y coordinates
    """
    pts = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)

    x_km, y_km = project_lonlat_to_km(lon, lat, epsg_project=epsg_project)

    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    finder = triang.get_trifinder()

    tri_ids = finder(x_km, y_km)
    inside = tri_ids >= 0

    return inside, x_km, y_km


def nodal_control_areas_km2(msh_path: Path) -> np.ndarray:
    """
    Compute mass-lumped nodal control areas:

        A_i = sum_{T containing i} area(T) / 3.

    Returns
    -------
    A_nodes : ndarray, shape (n_nodes,)
    """
    pts = load_mesh_points_km(msh_path)
    tri = load_mesh_triangles(msh_path)

    x1 = pts[tri[:, 0], 0]
    y1 = pts[tri[:, 0], 1]
    x2 = pts[tri[:, 1], 0]
    y2 = pts[tri[:, 1], 1]
    x3 = pts[tri[:, 2], 0]
    y3 = pts[tri[:, 2], 1]

    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    A_nodes = np.zeros(pts.shape[0], dtype=float)
    share = area / 3.0

    np.add.at(A_nodes, tri[:, 0], share)
    np.add.at(A_nodes, tri[:, 1], share)
    np.add.at(A_nodes, tri[:, 2], share)

    return A_nodes


# ============================================================
# Population-to-node mapping
# ============================================================

def map_population_nearest(
    *,
    mesh_points_km: np.ndarray,
    zip_xy_km: np.ndarray,
    zip_pop: np.ndarray,
) -> np.ndarray:
    """
    Old baseline method:
      assign each ZIP centroid to the nearest mesh node.
    """
    tree = cKDTree(mesh_points_km)
    _, idx = tree.query(zip_xy_km, k=1)

    node_pop = np.zeros(mesh_points_km.shape[0], dtype=float)
    np.add.at(node_pop, idx, zip_pop)

    return node_pop


def map_population_kernel(
    *,
    mesh_points_km: np.ndarray,
    zip_xy_km: np.ndarray,
    zip_pop: np.ndarray,
    length_scale_km: float,
    k_neighbors: int = 32,
    kernel: str = "exponential",
) -> np.ndarray:
    """
    Smooth ZIP populations onto mesh nodes.

    For each ZIP centroid z with population P_z, choose k nearest mesh nodes.
    Assign mass

        P_z * w_i / sum_j w_j

    where

        w_i = exp(-d_i / ell)              for kernel='exponential'
        w_i = exp(-0.5 * (d_i / ell)^2)    for kernel='gaussian'

    This exactly conserves the total included ZIP population up to floating error.
    """
    if length_scale_km <= 0:
        raise ValueError("length_scale_km must be positive.")

    n_nodes = mesh_points_km.shape[0]
    k = int(min(max(1, k_neighbors), n_nodes))

    tree = cKDTree(mesh_points_km)
    dist, idx = tree.query(zip_xy_km, k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    if kernel == "exponential":
        weights = np.exp(-dist / float(length_scale_km))
    elif kernel == "gaussian":
        weights = np.exp(-0.5 * (dist / float(length_scale_km)) ** 2)
    else:
        raise ValueError("kernel must be 'exponential' or 'gaussian'.")

    denom = weights.sum(axis=1)
    good = denom > 0

    weights[good] = weights[good] / denom[good, None]
    weights[~good] = 0.0

    node_pop = np.zeros(n_nodes, dtype=float)

    mass = zip_pop[:, None] * weights
    np.add.at(node_pop, idx.reshape(-1), mass.reshape(-1))

    return node_pop


def build_node_feature_table(
    *,
    msh_path: Path,
    pop_csv: Path,
    epsg_project: int,
    smooth_length_km: float,
    smooth_k_neighbors: int,
    smooth_kernel: str = "exponential",
) -> dict:
    """
    Build per-node mapped features.

    Currently computes:
      - population_nearest_2010
      - population_nearest_2020
      - population_smooth_2010
      - population_smooth_2020
      - population_density_smooth_2010
      - population_density_smooth_2020
      - pv_potential_placeholder
      - transmission_distance_km_placeholder

    Returns a dictionary suitable for np.savez_compressed.
    """
    mesh_points = load_mesh_points_km(msh_path)
    A_nodes = nodal_control_areas_km2(msh_path)

    df_pop = load_zip_population_csv(pop_csv)

    inside, x_km, y_km = zip_points_inside_mesh_mask(
        msh_path=msh_path,
        lon=df_pop["longitude"].to_numpy(float),
        lat=df_pop["latitude"].to_numpy(float),
        epsg_project=epsg_project,
    )

    zip_xy_inside = np.column_stack([x_km[inside], y_km[inside]])
    pop2010_inside = df_pop["pop2010"].to_numpy(float)[inside]
    pop2020_inside = df_pop["pop2020"].to_numpy(float)[inside]

    nearest_2010 = map_population_nearest(
        mesh_points_km=mesh_points,
        zip_xy_km=zip_xy_inside,
        zip_pop=pop2010_inside,
    )
    nearest_2020 = map_population_nearest(
        mesh_points_km=mesh_points,
        zip_xy_km=zip_xy_inside,
        zip_pop=pop2020_inside,
    )

    smooth_2010 = map_population_kernel(
        mesh_points_km=mesh_points,
        zip_xy_km=zip_xy_inside,
        zip_pop=pop2010_inside,
        length_scale_km=smooth_length_km,
        k_neighbors=smooth_k_neighbors,
        kernel=smooth_kernel,
    )
    smooth_2020 = map_population_kernel(
        mesh_points_km=mesh_points,
        zip_xy_km=zip_xy_inside,
        zip_pop=pop2020_inside,
        length_scale_km=smooth_length_km,
        k_neighbors=smooth_k_neighbors,
        kernel=smooth_kernel,
    )

    safe_A = np.where(A_nodes > 0, A_nodes, np.nan)

    pv_placeholder = np.full(mesh_points.shape[0], np.nan, dtype=float)
    transmission_placeholder = np.full(mesh_points.shape[0], np.nan, dtype=float)

    return {
        "mesh_points_km": mesh_points,
        "A_nodes_km2": A_nodes,

        "population_nearest_2010": nearest_2010,
        "population_nearest_2020": nearest_2020,

        "population_smooth_2010": smooth_2010,
        "population_smooth_2020": smooth_2020,

        "population_density_smooth_2010": smooth_2010 / safe_A,
        "population_density_smooth_2020": smooth_2020 / safe_A,

        "pv_potential": pv_placeholder,
        "transmission_distance_km": transmission_placeholder,

        "zip_inside_count": np.array([int(np.count_nonzero(inside))]),
        "zip_total_count": np.array([int(len(df_pop))]),

        "zip_population_inside_2010": np.array([float(pop2010_inside.sum())]),
        "zip_population_inside_2020": np.array([float(pop2020_inside.sum())]),

        "node_population_nearest_sum_2010": np.array([float(nearest_2010.sum())]),
        "node_population_nearest_sum_2020": np.array([float(nearest_2020.sum())]),
        "node_population_smooth_sum_2010": np.array([float(smooth_2010.sum())]),
        "node_population_smooth_sum_2020": np.array([float(smooth_2020.sum())]),

        "smooth_length_km": np.array([float(smooth_length_km)]),
        "smooth_k_neighbors": np.array([int(smooth_k_neighbors)]),
        "smooth_kernel": np.array([smooth_kernel]),
    }


def save_node_features_npz(
    *,
    out_npz: Path,
    features: dict,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **features)
    print(f"[DATA] wrote {out_npz}")


# ============================================================
# Plotting
# ============================================================

def _node_circle_plot(
    *,
    msh_path: Path,
    values: np.ndarray,
    ax,
    epsg_project: int,
    title: str,
    radius_scale: float = 0.35,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    transform: str = "linear",
    n_layers: int = 10,
):
    lon, lat = mesh_points_lonlat(msh_path, epsg_project=epsg_project)

    raw_values = np.asarray(values, dtype=float)

    if transform == "log1p":
        plot_values = np.log1p(np.clip(raw_values, 0.0, None))
    elif transform == "linear":
        plot_values = raw_values
    else:
        raise ValueError("transform must be 'linear' or 'log1p'.")

    finite = np.isfinite(plot_values)

    if vmin is None:
        vmin = float(np.nanmin(plot_values[finite])) if np.any(finite) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(plot_values[finite])) if np.any(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    n = len(raw_values)
    s = max(6.0, 5000.0 / max(np.sqrt(n), 1.0)) * float(radius_scale)

    finite_vals = plot_values[finite]
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
            mask = finite & (plot_values >= lo) & (plot_values <= hi)
        else:
            mask = finite & (plot_values >= lo) & (plot_values < hi)

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        # Critical fix: sort within each layer from smaller to larger.
        idx = idx[np.argsort(plot_values[idx])]

        mappable = ax.scatter(
            lon[idx],
            lat[idx],
            c=plot_values[idx],
            s=s,
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
            c=np.zeros_like(plot_values),
            s=s,
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


def plot_population_comparison(
    *,
    msh_path: Path,
    nearest_values: np.ndarray,
    smooth_values: np.ndarray,
    out_png: Path,
    epsg_project: int,
    year: int,
) -> None:
    """
    Two-panel comparison:
      left  = nearest-node ZIP assignment
      right = smoothed kernel assignment
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vals = np.log1p(np.concatenate([np.asarray(nearest_values, dtype=float), np.asarray(smooth_values, dtype=float)]))
    vmax = float(np.nanquantile(vals, 0.99)) if vals.size else 1.0
    vmax = max(vmax, 1.0)
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    sc0 = _node_circle_plot(
        msh_path=msh_path,
        values=nearest_values,
        ax=axes[0],
        epsg_project=epsg_project,
        title=f"Population / node, nearest ZIP assignment ({year})",
        vmin=vmin,
        vmax=vmax,
        transform="log1p"
    )

    sc1 = _node_circle_plot(
        msh_path=msh_path,
        values=smooth_values,
        ax=axes[1],
        epsg_project=epsg_project,
        title=f"Population / node, kernel smoothing ({year})",
        vmin=vmin,
        vmax=vmax,
        transform="log1p"
    )

    fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02, label="ln(1 + population) / node")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")


def plot_population_smoothed(
    *,
    msh_path: Path,
    smooth_values: np.ndarray,
    out_png: Path,
    epsg_project: int,
    year: int,
) -> None:
    """
    Single-panel smoothed population plot.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.nanquantile(np.log1p(smooth_values), 0.99)) if len(smooth_values) else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    sc = _node_circle_plot(
        msh_path=msh_path,
        values=smooth_values,
        ax=ax,
        epsg_project=epsg_project,
        title=f"Smoothed population / node ({year})",
        vmin=0.0,
        vmax=vmax,
        transform="log1p"
    )

    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="ln(1 + population) / node")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")