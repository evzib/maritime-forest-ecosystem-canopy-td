Parameter tuning for the genetic algorithm baseline.

Runs the GA with several values of lambda (the weight of the mean route offset
in the fitness function) and prints the resulting metrics. This helps select a
lambda that produces a route offset comparable to CANOPY-TD.
"""

import argparse

from canopy_td import CanopyConfig
from baseline_ga import GAConfig, run_ga_multistart
from preprocessing import load_trajectories, trajectory_centroids_km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_offset_km", type=float, default=5.0)
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    trajectories = load_trajectories(args.data_dir)
    ids, initial_points, _ = trajectory_centroids_km(trajectories)
    base_config = CanopyConfig()

    print(f"Number of trajectories: {len(ids)}")
    print(f"Max offset budget: {args.max_offset_km} km")
    print(f"Number of GA runs per lambda: {args.n_runs}")
    print()

    for lam in [1.0, 2.0, 5.0, 10.0, 20.0]:
        ga_config = GAConfig(
            lambda_offset=lam,
            max_total_offset_km=args.max_offset_km,
            population_size=80,
            generations=200,
            seed=42,
        )
        _, metrics, _ = run_ga_multistart(
            initial_points, ga_config, base_config, n_runs=args.n_runs
        )
        print(
            f"lambda={lam:5.1f}  "
            f"conflicts={metrics['Conflicts']:5d}  "
            f"offset={metrics['Mean route offset, km']:5.2f}  "
            f"congestion={metrics['Congestion score']:9.3f}  "
            f"risk={metrics['Risk score']:10.3f}"
        )


if __name__ == "__main__":
    main()

