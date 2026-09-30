Sanity check for the genetic algorithm baseline.

Runs the GA on a small synthetic dataset and verifies that:
- the maximum offset constraint is respected,
- the GA does not increase the number of conflicts,
- the fitness value is finite.
"""

import numpy as np

from canopy_td import CanopyConfig
from baseline_ga import GAConfig, run_ga
from metrics import compute_metrics


def main():
    rng = np.random.default_rng(0)

    # 10 vessels randomly placed in a 20 x 20 km area
    initial_points = rng.uniform(-10, 10, size=(10, 2))

    base_config = CanopyConfig()
    ga_config = GAConfig(
        population_size=40,
        generations=100,
        lambda_offset=5.0,
        max_total_offset_km=5.0,
        seed=0,
    )

    print("Initial metrics:")
    print(compute_metrics(initial_points, initial_points, base_config))

    points, metrics, info = run_ga(initial_points, ga_config, base_config)

    print("\nGA metrics:")
    print(metrics)
    print("Best fitness:", info["best_fitness"])

    offsets = np.linalg.norm(points - initial_points, axis=1)
    print("\nMax offset:", offsets.max())

    assert offsets.max() <= ga_config.max_total_offset_km + 1e-6, (
        "Offset constraint violated!"
    )

    initial_conflicts = compute_metrics(
        initial_points, initial_points, base_config
    )["Conflicts"]
    assert metrics["Conflicts"] <= initial_conflicts, "GA increased conflicts!"

    assert np.isfinite(info["best_fitness"]), "Fitness is not finite!"

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()

