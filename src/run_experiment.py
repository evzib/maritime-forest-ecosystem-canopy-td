from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from canopy_td import CanopyConfig, run_canopy_td, get_default_configs
from preprocessing import load_trajectories, trajectory_centroids_km
from metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.data_dir)
    ids, initial_points, metadata = trajectory_centroids_km(trajectories)
    base_config = CanopyConfig()
    configs = get_default_configs()
    rows = {"Initial Traffic": compute_metrics(initial_points, initial_points, base_config)}
    for name, config in configs.items():
        points = run_canopy_td(initial_points, config)
        rows[name] = compute_metrics(points, initial_points, base_config)
    results = pd.DataFrame(rows).T.round(3)
    results.to_csv(out_dir / "CANOPY_TD_results.csv")
    with open(out_dir / "CANOPY_TD_parameters.json", "w", encoding="utf-8") as f:
        json.dump({"base_config": base_config.__dict__, "configs": {k: v.__dict__ for k, v in configs.items()}, "n_trajectories": len(ids)}, f, indent=2)
    print(results)

if __name__ == "__main__":
    main()
