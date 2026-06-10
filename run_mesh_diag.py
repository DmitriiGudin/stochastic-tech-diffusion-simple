from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List

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


def _parse_list_arg(s: str, *, upper: bool = False) -> List[str]:
    raw = str(s or "").strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        obj = ast.literal_eval(raw)
        if not isinstance(obj, (list, tuple)):
            raise ValueError(f"Expected list/tuple, got {type(obj)}")
        out = [str(x).strip() for x in obj if str(x).strip()]
    elif "," in raw:
        out = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        out = [raw]

    if upper:
        out = [x.upper() for x in out]

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LSPV mesh and map node-level diagnostics.")

    parser.add_argument("--states", required=True, type=str)
    parser.add_argument("--counties", default="", type=str)

    parser.add_argument("--h_km", default=25.0, type=float)
    parser.add_argument("--simplify_km", default=5.0, type=float)
    parser.add_argument("--epsg_project", default=5070, type=int)

    parser.add_argument("--pop_csv", default="data/raw/zip_population_all.csv", type=str)
    parser.add_argument("--plot_pop_year", default=2020, type=int)

    parser.add_argument("--smooth_length_km", default=75.0, type=float)
    parser.add_argument("--smooth_k_neighbors", default=32, type=int)
    parser.add_argument("--smooth_kernel", default="exponential", choices=["exponential", "gaussian"])

    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--lspv_csv", default="data/raw/uspvdb_v4_0_20260414.csv", type=str)
    
    parser.add_argument("--transmission_shp", default="data/raw/Transmission_Lines/Transmission_Lines.shp", type=str)
    parser.add_argument("--transmission_buffer_km", default=15.0, type=float, help="How far outside the mesh to keep transmission lines.")
    
    parser.add_argument("--pvout_tif", default="data/raw/PVOUT.tif", type=str)
    
    parser.add_argument("--adoption_plot_scale", default="linear", choices=["linear", "log1p"])

    args = parser.parse_args()

    states = _parse_list_arg(args.states, upper=True)
    counties = _parse_list_arg(args.counties, upper=False)

    base = Path("data")

    admin1_shp = (
        base
        / "raw"
        / "ne_10m_admin_1_states_provinces_lakes"
        / "ne_10m_admin_1_states_provinces_lakes.shp"
    )
    county_shp = (
        base
        / "raw"
        / "cb_2023_us_county_5m"
        / "cb_2023_us_county_5m.shp"
    )

    pop_csv = Path(args.pop_csv)
    lspv_csv = Path(args.lspv_csv)
    transmission_shp = Path(args.transmission_shp)
    pvout_tif = Path(args.pvout_tif)

    if not admin1_shp.exists():
        raise FileNotFoundError(admin1_shp)
    if not county_shp.exists():
        raise FileNotFoundError(county_shp)
    if not pop_csv.exists():
        raise FileNotFoundError(pop_csv)
    if not lspv_csv.exists():
        raise FileNotFoundError(lspv_csv)
    if not transmission_shp.exists():
        raise FileNotFoundError(transmission_shp)
    if not pvout_tif.exists():
        raise FileNotFoundError(pvout_tif)

    # Computational mesh output
    mesh_dir = base / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostics / derived data output
    out_root = Path("out") / "mesh_diag"
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    region_tag = make_region_tag(states, counties)
    tag = (
        f"{region_tag}"
        f"_h{args.h_km:g}"
        f"_s{args.simplify_km:g}"
        f"_epsg{args.epsg_project}"
    )

    msh_path = mesh_dir / f"{tag}.msh"

    png_mesh = fig_dir / f"{tag}_mesh.png"
    pop_npz = data_dir / f"{tag}_node_features.npz"
    meta_path = data_dir / f"{tag}_metadata.json"

    year = int(args.plot_pop_year)
    png_pop_compare = fig_dir / f"{tag}_population_compare_{year}.png"
    png_pop_smooth = fig_dir / f"{tag}_population_smoothed_{year}.png"
    png_lspv_adoptions = fig_dir / f"{tag}_lspv_adoptions_nearest_node.png"
    png_transmission_distance = fig_dir / f"{tag}_transmission_distance_km.png"
    png_pv_potential = fig_dir / f"{tag}_pv_potential.png"

    cfg = MeshBuildConfig(
        h_km=float(args.h_km),
        simplify_km=float(args.simplify_km),
        epsg_project=int(args.epsg_project),
    )

    print("[RUN] states:", states)
    print("[RUN] counties:", counties if counties else "(none)")
    print("[RUN] h_km:", cfg.h_km)
    print("[RUN] simplify_km:", cfg.simplify_km)
    print("[RUN] epsg_project:", cfg.epsg_project)
    print("[RUN] mesh:", msh_path)
    print("[RUN] population CSV:", pop_csv)

    # ------------------------------------------------------------
    # 1. Mesh
    # ------------------------------------------------------------
    build_mesh_for_region(
        admin1_shp=admin1_shp,
        county_shp=county_shp,
        state_codes=states,
        county_names=counties,
        out_msh=msh_path,
        cfg=cfg,
        verbose=bool(args.verbose),
    )

    plot_mesh_lonlat(
        msh_path=msh_path,
        out_png=png_mesh,
        epsg_project=cfg.epsg_project,
    )

    # ------------------------------------------------------------
    # 2. Node features
    # ------------------------------------------------------------
    features = build_node_feature_table(
        msh_path=msh_path,
        pop_csv=pop_csv,
        epsg_project=cfg.epsg_project,
        smooth_length_km=float(args.smooth_length_km),
        smooth_k_neighbors=int(args.smooth_k_neighbors),
        smooth_kernel=str(args.smooth_kernel),
    )
    
    transmission_features = map_transmission_distance_to_nodes(
        msh_path=msh_path,
        transmission_shp=transmission_shp,
        epsg_project=cfg.epsg_project,
        buffer_km=float(args.transmission_buffer_km),
    )
    features.update(transmission_features)
    
    pv_features = map_pvout_to_nodes(
        msh_path=msh_path,
        pvout_tif=pvout_tif,
        epsg_project=cfg.epsg_project,
    )
    features.update(pv_features)

    save_node_features_npz(
        out_npz=pop_npz,
        features=features,
    )

    if year not in (2010, 2020):
        raise ValueError("For now plot_pop_year must be either 2010 or 2020.")

    nearest_key = f"population_nearest_{year}"
    smooth_key = f"population_smooth_{year}"

    plot_population_comparison(
        msh_path=msh_path,
        nearest_values=features[nearest_key],
        smooth_values=features[smooth_key],
        out_png=png_pop_compare,
        epsg_project=cfg.epsg_project,
        year=year,
    )

    plot_population_smoothed(
        msh_path=msh_path,
        smooth_values=features[smooth_key],
        out_png=png_pop_smooth,
        epsg_project=cfg.epsg_project,
        year=year,
    )
    
    # ------------------------------------------------------------
    # 3. LSPV adoption diagnostics
    # ------------------------------------------------------------
    node_lspv_counts, lspv_inside = map_lspv_adoptions_nearest_node(
        msh_path=msh_path,
        lspv_csv=lspv_csv,
        epsg_project=cfg.epsg_project,
    )
    
    plot_lspv_adoptions_nearest_node(
        msh_path=msh_path,
        node_counts=node_lspv_counts,
        out_png=png_lspv_adoptions,
        epsg_project=cfg.epsg_project,
        scale=args.adoption_plot_scale,
    )
    
    plot_transmission_distance_nodes(
        msh_path=msh_path,
        distances_km=features["transmission_distance_km"],
        out_png=png_transmission_distance,
        epsg_project=cfg.epsg_project,
    )
    
    lspv_global_summary = lspv_year_summary(lspv_inside)
    
    lspv_county_summary = lspv_county_year_summary(
        events_inside=lspv_inside,
        county_shp=county_shp,
        state_codes=states,
        county_names=counties,
    )
    
    plot_pv_potential_nodes(
        msh_path=msh_path,
        pv_values=features["pv_potential"],
        out_png=png_pv_potential,
        epsg_project=cfg.epsg_project,
    )

    # ------------------------------------------------------------
    # 4. Metadata
    # ------------------------------------------------------------
    mesh_info = mesh_summary(msh_path)

    metadata = {
        "states": states,
        "counties": counties,
        "h_km": cfg.h_km,
        "simplify_km": cfg.simplify_km,
        "epsg_project": cfg.epsg_project,
        "mesh": mesh_info,
        "population_csv": str(pop_csv),
        "node_features_npz": str(pop_npz),
        "smooth_length_km": float(args.smooth_length_km),
        "smooth_k_neighbors": int(args.smooth_k_neighbors),
        "smooth_kernel": str(args.smooth_kernel),

        "population_mass_checks": {
            "zip_inside_count": int(features["zip_inside_count"][0]),
            "zip_total_count": int(features["zip_total_count"][0]),

            "zip_population_inside_2010": float(features["zip_population_inside_2010"][0]),
            "zip_population_inside_2020": float(features["zip_population_inside_2020"][0]),

            "node_population_nearest_sum_2010": float(features["node_population_nearest_sum_2010"][0]),
            "node_population_nearest_sum_2020": float(features["node_population_nearest_sum_2020"][0]),

            "node_population_smooth_sum_2010": float(features["node_population_smooth_sum_2010"][0]),
            "node_population_smooth_sum_2020": float(features["node_population_smooth_sum_2020"][0]),
        },
        "lspv_csv": str(lspv_csv),
        "lspv_adoptions": {
            "global": lspv_global_summary,
            "counties": lspv_county_summary,
        },
        "transmission_lines": {
            "shapefile": str(transmission_shp),
            "buffer_km": float(args.transmission_buffer_km),
            "lines_retained": int(features["transmission_lines_retained"][0]),
            "nodes_zero_distance": int(features["transmission_nodes_zero_count"][0]),
            "distance_min_km": float(features["transmission_distance_min_km"][0]),
            "distance_median_km": float(features["transmission_distance_median_km"][0]),
            "distance_max_km": float(features["transmission_distance_max_km"][0]),
        },
        "pvout": {
            "tif": str(pvout_tif),
            "valid_count": int(features["pv_potential_valid_count"][0]),
            "min": float(features["pv_potential_min"][0]),
            "median": float(features["pv_potential_median"][0]),
            "max": float(features["pv_potential_max"][0]),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[RUN] summary:")
    print(json.dumps(metadata, indent=2))
    print(f"[RUN] wrote metadata: {meta_path}")
    
    print("[RUN] transmission-line summary:")
    print(json.dumps({
        "buffer_km": float(args.transmission_buffer_km),
        "lines_retained": int(features["transmission_lines_retained"][0]),
        "nodes_zero_distance": int(features["transmission_nodes_zero_count"][0]),
        "distance_min_km": float(features["transmission_distance_min_km"][0]),
        "distance_median_km": float(features["transmission_distance_median_km"][0]),
        "distance_max_km": float(features["transmission_distance_max_km"][0]),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())