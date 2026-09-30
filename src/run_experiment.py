Main experiment: compare CANOPY-TD modes with the genetic algorithm baseline.

Loads AIS trajectories, runs CANOPY-TD (Soft and Strong) and the GA baseline,
computes metrics, and saves the results to CSV and JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from canopy_td import CanopyConfig, run_canopy_td, get_default_configs
from preprocessing import load_trajectories, trajectory_centroids_km
from metrics import compute_metrics
from baseline_ga import GAConfig, run_ga_multistart


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--ga_max_offset_km", type=float, default=5.0)
    parser.add_argument("--ga_lambda", type=float, default=5.0)
    parser.add_argument("--ga_runs", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.data_dir)
    ids, initial_points, metadata = trajectory_centroids_km(trajectories)

    base_config = CanopyConfig()
    canopy_configs = get_default_configs()

    rows = {
        "Initial Traffic": compute_metrics(
            initial_points, initial_points, base_config
        )
    }

    # CANOPY-TD modes
    for name, config in canopy_configs.items():
        points = run_canopy_td(initial_points, config)
        rows[name] = compute_metrics(points, initial_points, base_config)

    # External GA baseline
    ga_config = GAConfig(
        lambda_offset=args.ga_lambda,
        max_total_offset_km=args.ga_max_offset_km,
        population_size=80,
        generations=200,
        seed=42,
    )
    _, ga_metrics, ga_info = run_ga_multistart(
        initial_points, ga_config, base_config, n_runs=args.ga_runs
    )
    rows["GA Baseline"] = ga_metrics

    results = pd.DataFrame(rows).T.round(3)
    results.to_csv(out_dir / "CANOPY_TD_results.csv")

    with open(out_dir / "CANOPY_TD_parameters.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_config": base_config.__dict__,
                "canopy_configs": {
                    k: v.__dict__ for k, v in canopy_configs.items()
                },
                "ga_config": ga_config.__dict__,
                "ga_info": ga_info,
                "n_trajectories": len(ids),
            },
            f,
            indent=2,
        )

    print(results)


if __name__ == "__main__":
    main()

