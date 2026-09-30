External baseline: Genetic Algorithm (GA) for maritime trajectory redistribution.

This module implements a standard genetic algorithm as an independent external
baseline for comparison with CANOPY-TD. Unlike CANOPY-TD, the GA does not use
canopy overlap, local density, or ecosystem pressure. It searches the space of
vessel displacement vectors directly using evolutionary operators.

The GA operates on the same search space as CANOPY-TD (a 2D displacement vector
for each vessel) and is subject to the same maximum cumulative offset constraint,
making the comparison methodologically fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from canopy_td import CanopyConfig
from metrics import compute_metrics


@dataclass
class GAConfig:
    """Configuration parameters for the genetic algorithm."""
    population_size: int = 80
    generations: int = 200
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    mutation_sigma_km: float = 0.3
    elitism: int = 2
    lambda_offset: float = 5.0
    max_total_offset_km: float = 5.0
    seed: int = 42


def _clip_offsets(offsets: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip each vessel's 2D offset so that its norm does not exceed max_norm."""
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    too_far = norms > max_norm
    if np.any(too_far):
        offsets = offsets.copy()
        offsets[too_far[:, 0]] *= (max_norm / norms[too_far[:, 0]])
    return offsets


def _fitness(
    offsets_flat: np.ndarray,
    initial_points_km: np.ndarray,
    config: CanopyConfig,
    lambda_offset: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness = Conflicts + lambda * MeanRouteOffset."""
    points = initial_points_km + offsets_flat.reshape(-1, 2)
    metrics = compute_metrics(points, initial_points_km, config)
    fitness = metrics["Conflicts"] + lambda_offset * metrics["Mean route offset, km"]
    return fitness, metrics


def _random_population(
    n_vessels: int,
    population_size: int,
    max_total_offset_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create an initial random population. Each individual is a flat vector of length 2*N."""
    pop = rng.normal(
        0.0, max_total_offset_km / 3.0, size=(population_size, 2 * n_vessels)
    )
    for i in range(population_size):
        pop[i] = _clip_offsets(
            pop[i].reshape(-1, 2), max_total_offset_km
        ).ravel()
    return pop


def _tournament_selection(
    population: np.ndarray,
    fitness_values: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select one individual using tournament selection."""
    idx = rng.integers(0, len(population), size=tournament_size)
    best = idx[np.argmin(fitness_values[idx])]
    return population[best].copy()


def _blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend crossover (BLX-alpha)."""
    gamma = rng.uniform(-alpha, 1.0 + alpha, size=parent_a.shape)
    child_a = gamma * parent_a + (1.0 - gamma) * parent_b
    child_b = gamma * parent_b + (1.0 - gamma) * parent_a
    return child_a, child_b


def run_ga(
    initial_points_km: np.ndarray,
    ga_config: GAConfig,
    canopy_config: CanopyConfig,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Run the genetic algorithm.

    Returns
    -------
    optimized_points : np.ndarray
        Optimized vessel positions (N x 2).
    best_metrics : dict
        Metrics computed for the best individual.
    info : dict
        Additional information (best fitness, number of generations).
    """
    rng = np.random.default_rng(ga_config.seed)
    n_vessels = initial_points_km.shape[0]

    population = _random_population(
        n_vessels, ga_config.population_size, ga_config.max_total_offset_km, rng
    )
    fitness_values = np.array(
        [
            _fitness(ind, initial_points_km, canopy_config, ga_config.lambda_offset)[0]
            for ind in population
        ]
    )

    best_idx = int(np.argmin(fitness_values))
    best_individual = population[best_idx].copy()
    best_fitness = float(fitness_values[best_idx])

    for _ in range(ga_config.generations):
        new_population = []

        # Elitism
        elite_idx = np.argsort(fitness_values)[: ga_config.elitism]
        for idx in elite_idx:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < ga_config.population_size:
            parent_a = _tournament_selection(
                population, fitness_values, ga_config.tournament_size, rng
            )
            parent_b = _tournament_selection(
                population, fitness_values, ga_config.tournament_size, rng
            )

            if rng.random() < ga_config.crossover_prob:
                child_a, child_b = _blend_crossover(
                    parent_a, parent_b, alpha=0.5, rng=rng
                )
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            for child in (child_a, child_b):
                if rng.random() < ga_config.mutation_prob:
                    child += rng.normal(
                        0.0, ga_config.mutation_sigma_km, size=child.shape
                    )
                child = _clip_offsets(
                    child.reshape(-1, 2), ga_config.max_total_offset_km
                ).ravel()
                new_population.append(child)
                if len(new_population) >= ga_config.population_size:
                    break

        population = np.array(new_population[: ga_config.population_size])
        fitness_values = np.array(
            [
                _fitness(
                    ind, initial_points_km, canopy_config, ga_config.lambda_offset
                )[0]
                for ind in population
            ]
        )

        current_best = int(np.argmin(fitness_values))
        if fitness_values[current_best] < best_fitness:
            best_fitness = float(fitness_values[current_best])
            best_individual = population[current_best].copy()

    optimized_points = initial_points_km + best_individual.reshape(-1, 2)
    _, best_metrics = _fitness(
        best_individual, initial_points_km, canopy_config, ga_config.lambda_offset
    )
    info = {
        "best_fitness": best_fitness,
        "generations": ga_config.generations,
        "lambda_offset": ga_config.lambda_offset,
        "max_total_offset_km": ga_config.max_total_offset_km,
    }
    return optimized_points, best_metrics, info


def run_ga_multistart(
    initial_points_km: np.ndarray,
    ga_config: GAConfig,
    canopy_config: CanopyConfig,
    n_runs: int = 5,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Run GA several times with different random seeds and keep the best result.

    This reduces the influence of random initialization and provides a more
    stable baseline for comparison.
    """
    best_points = None
    best_metrics = None
    best_fitness = float("inf")
    best_info = None

    for run in range(n_runs):
        cfg = GAConfig(**{**ga_config.__dict__, "seed": ga_config.seed + run})
        points, metrics, info = run_ga(initial_points_km, cfg, canopy_config)
        if info["best_fitness"] < best_fitness:
            best_fitness = info["best_fitness"]
            best_points = points
            best_metrics = metrics
            best_info = info

    best_info = dict(best_info or {})
    best_info["n_runs"] = n_runs
    return best_points, best_metrics, best_info

