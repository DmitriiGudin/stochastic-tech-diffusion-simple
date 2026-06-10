from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import json
import re
import time

import geopandas as gpd
import gmsh
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Transformer


_STATE_ABBR_TO_FIPS = {
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


@dataclass(frozen=True)
class MeshBuildConfig:
    h_km: float = 25.0
    simplify_km: float = 5.0
    epsg_project: int = 5070
    gmsh_algo_2d: int = 6
    msh_version: float = 4.1
    ring_eps: float = 1e-12


def _clean_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def make_region_tag(
    state_codes: Iterable[str],
    county_names: Optional[Iterable[str]] = None,
    *,
    digest_len: int = 10,
) -> str:
    states = [str(s).strip().upper() for s in state_codes if str(s).strip()]
    counties = [str(c).strip() for c in (county_names or []) if str(c).strip()]

    payload = {
        "states": sorted(states),
        "counties": sorted(_clean_name(c) for c in counties),
    }

    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        digest_size=8,
    ).hexdigest()[:digest_len]

    state_part = "_".join(states) if states else "region"
    if counties:
        return f"{state_part}__counties_{digest}"
    return state_part


def pick_largest_polygon(geom) -> Polygon:
    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if (not p.is_empty) and p.area > 0]
        if not polys:
            raise ValueError("No valid polygons in MultiPolygon.")
        return max(polys, key=lambda p: p.area)

    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def load_admin1_states(admin1_shp: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(admin1_shp).to_crs("EPSG:4326")

    if "iso_3166_2" in gdf.columns:
        state_code = gdf["iso_3166_2"].astype(str).str.split("-").str[-1]
    elif "postal" in gdf.columns:
        state_code = gdf["postal"].astype(str)
    elif "postal_code" in gdf.columns:
        state_code = gdf["postal_code"].astype(str)
    else:
        raise ValueError(
            "Could not infer state code. Expected one of: "
            "iso_3166_2, postal, postal_code."
        )

    gdf = gdf.copy()
    gdf["state_code"] = state_code.str.upper().str.strip()
    return gdf[gdf["state_code"].str.fullmatch(r"[A-Z]{2}", na=False)].copy()


def load_us_counties(county_shp: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(county_shp).to_crs("EPSG:4326")

    required = {"NAME", "STATEFP"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"County shapefile missing columns: {sorted(missing)}")

    fips_to_abbr = {v: k for k, v in _STATE_ABBR_TO_FIPS.items()}

    gdf = gdf.copy()
    gdf["county_name_clean"] = gdf["NAME"].astype(str).map(_clean_name)
    gdf["STATEFP"] = gdf["STATEFP"].astype(str).str.zfill(2)
    gdf["state_code"] = gdf["STATEFP"].map(fips_to_abbr)

    return gdf


def build_region_polygon_from_states(
    admin1_shp: Path,
    state_codes: Iterable[str],
) -> Polygon:
    states = {str(s).strip().upper() for s in state_codes if str(s).strip()}
    if not states:
        raise ValueError("state_codes is empty.")

    gdf = load_admin1_states(admin1_shp)
    sel = gdf[gdf["state_code"].isin(states)]

    if sel.empty:
        raise ValueError(f"No states found for state_codes={sorted(states)}")

    geom = unary_union(sel.geometry.values)
    if geom.is_empty:
        raise ValueError("Selected state geometry is empty.")

    return pick_largest_polygon(geom)


def build_region_polygon_from_counties(
    county_shp: Path,
    county_names: Iterable[str],
    state_codes: Optional[Iterable[str]] = None,
) -> Polygon:
    counties = {_clean_name(c) for c in county_names if str(c).strip()}
    if not counties:
        raise ValueError("county_names is empty.")

    gdf = load_us_counties(county_shp)

    if state_codes is not None:
        states = {str(s).strip().upper() for s in state_codes if str(s).strip()}
        if states:
            gdf = gdf[gdf["state_code"].isin(states)].copy()

    sel = gdf[gdf["county_name_clean"].isin(counties)].copy()

    if sel.empty:
        raise ValueError(
            f"No counties found for county_names={sorted(counties)} "
            f"within state_codes={list(state_codes) if state_codes is not None else None}."
        )

    geom = unary_union(sel.geometry.values)
    if geom.is_empty:
        raise ValueError("Selected county geometry is empty.")

    return pick_largest_polygon(geom)


def build_region_polygon(
    *,
    admin1_shp: Path,
    state_codes: Iterable[str],
    county_shp: Optional[Path] = None,
    county_names: Optional[Iterable[str]] = None,
) -> Polygon:
    county_names = list(county_names or [])
    if county_names:
        if county_shp is None:
            raise ValueError("county_shp is required when county_names is nonempty.")
        return build_region_polygon_from_counties(
            county_shp=county_shp,
            county_names=county_names,
            state_codes=state_codes,
        )

    return build_region_polygon_from_states(admin1_shp, state_codes)


def project_polygon_to_km(poly_lonlat: Polygon, epsg_project: int) -> Polygon:
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)

    def project_ring(coords):
        xs, ys = tr.transform(
            [p[0] for p in coords],
            [p[1] for p in coords],
        )
        return [(x / 1000.0, y / 1000.0) for x, y in zip(xs, ys)]

    exterior = project_ring(poly_lonlat.exterior.coords)
    interiors = [project_ring(ring.coords) for ring in poly_lonlat.interiors]

    return Polygon(exterior, interiors)


def simplify_polygon_km(poly_km: Polygon, simplify_km: float) -> Polygon:
    if simplify_km > 0:
        out = poly_km.simplify(float(simplify_km), preserve_topology=True)
    else:
        out = poly_km

    if not out.is_valid:
        out = out.buffer(0)

    if out.is_empty:
        raise ValueError("Simplified polygon is empty.")

    return pick_largest_polygon(out)


def _clean_ring(coords, eps: float):
    coords = list(coords)

    if len(coords) < 3:
        raise ValueError("Ring has fewer than 3 coordinates.")

    if abs(coords[0][0] - coords[-1][0]) < eps and abs(coords[0][1] - coords[-1][1]) < eps:
        coords = coords[:-1]

    cleaned = [coords[0]]
    for x, y in coords[1:]:
        x0, y0 = cleaned[-1]
        if abs(x - x0) >= eps or abs(y - y0) >= eps:
            cleaned.append((x, y))

    if len(cleaned) < 3:
        raise ValueError("Ring collapsed after duplicate cleanup.")

    return cleaned


def _add_polygon_to_gmsh(poly_km: Polygon, h_km: float, ring_eps: float) -> int:
    geo = gmsh.model.geo

    def add_loop(coords):
        coords = _clean_ring(coords, eps=ring_eps)
        pts = [
            geo.addPoint(float(x), float(y), 0.0, float(h_km))
            for x, y in coords
        ]
        lines = [
            geo.addLine(pts[i], pts[i + 1])
            for i in range(len(pts) - 1)
        ]
        lines.append(geo.addLine(pts[-1], pts[0]))
        return geo.addCurveLoop(lines)

    outer = add_loop(poly_km.exterior.coords)
    holes = [add_loop(list(ring.coords)[::-1]) for ring in poly_km.interiors]

    return geo.addPlaneSurface([outer] + holes)


def build_mesh_from_polygon_km(
    poly_km: Polygon,
    out_msh: Path,
    cfg: MeshBuildConfig,
    *,
    verbose: bool = True,
    model_name: str = "region_mesh",
) -> None:
    out_msh.parent.mkdir(parents=True, exist_ok=True)

    h = float(cfg.h_km)
    t0 = time.perf_counter()

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(model_name)

        surf = _add_polygon_to_gmsh(poly_km, h, cfg.ring_eps)
        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf], tag=1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        fid = gmsh.model.mesh.field.add("Constant")
        try:
            gmsh.model.mesh.field.setNumber(fid, "VIn", h)
        except Exception:
            gmsh.model.mesh.field.setNumber(fid, "Lc", h)
        gmsh.model.mesh.field.setAsBackgroundMesh(fid)

        point_entities = gmsh.model.getEntities(0)
        if point_entities:
            gmsh.model.mesh.setSize(point_entities, h)

        gmsh.option.setNumber("Mesh.Algorithm", cfg.gmsh_algo_2d)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        gmsh.model.mesh.generate(2)

        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            pass

        gmsh.option.setNumber("Mesh.MshFileVersion", cfg.msh_version)
        gmsh.write(str(out_msh))

    finally:
        gmsh.finalize()

    print(f"[MESH] wrote {out_msh}")
    print(f"[MESH] elapsed {time.perf_counter() - t0:.2f} s")


def build_mesh_for_region(
    *,
    admin1_shp: Path,
    county_shp: Path,
    state_codes: Iterable[str],
    county_names: Optional[Iterable[str]],
    out_msh: Path,
    cfg: MeshBuildConfig,
    verbose: bool = True,
) -> None:
    poly_lonlat = build_region_polygon(
        admin1_shp=admin1_shp,
        state_codes=state_codes,
        county_shp=county_shp,
        county_names=county_names,
    )
    poly_km = project_polygon_to_km(poly_lonlat, cfg.epsg_project)
    poly_km = simplify_polygon_km(poly_km, cfg.simplify_km)

    build_mesh_from_polygon_km(
        poly_km=poly_km,
        out_msh=out_msh,
        cfg=cfg,
        verbose=verbose,
        model_name=out_msh.stem,
    )


def plot_mesh_lonlat(
    *,
    msh_path: Path,
    out_png: Path,
    epsg_project: int = 5070,
) -> None:
    mesh = meshio.read(msh_path)

    tri_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            tri_cells = cell_block.data
            break

    if tri_cells is None:
        raise ValueError("No triangle cells found in mesh.")

    pts_km = mesh.points[:, :2]
    pts_m = pts_km * 1000.0

    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_m[:, 0], pts_m[:, 1])

    tri = mtri.Triangulation(lon, lat, tri_cells)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.triplot(tri, linewidth=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(msh_path.stem)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[PLOT] wrote {out_png}")


def mesh_summary(msh_path: Path) -> dict:
    mesh = meshio.read(msh_path)

    n_points = mesh.points.shape[0]
    n_triangles = 0
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            n_triangles += cell_block.data.shape[0]

    return {
        "path": str(msh_path),
        "n_points": int(n_points),
        "n_triangles": int(n_triangles),
    }