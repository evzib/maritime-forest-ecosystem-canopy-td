from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class CanopyConfig:
    R0_km: float = 8.0
    alpha: float = 0.25
    d_safe_km: float = 3.704
    epsilon: float = 1e-4
    beta: float = 0.15
    iterations: int = 100
    max_step_km: float = 0.20
    max_total_offset_km: float = 5.0

def pairwise_distances(points_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = points_km[:, None, :] - points_km[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances, diff

def local_density(points_km: np.ndarray, R0_km: float) -> np.ndarray:
    distances, _ = pairwise_distances(points_km)
    return ((distances < R0_km) & (distances > 0)).sum(axis=1).astype(float)

def adaptive_canopy_radius(points_km: np.ndarray, config: CanopyConfig) -> np.ndarray:
    rho = local_density(points_km, config.R0_km)
    return config.R0_km + config.alpha * rho

def canopy_overlap(points_km: np.ndarray, config: CanopyConfig):
    distances, diff = pairwise_distances(points_km)
    radius = adaptive_canopy_radius(points_km, config)
    overlap = np.maximum(0.0, radius[:, None] + radius[None, :] - distances)
    np.fill_diagonal(overlap, 0.0)
    return overlap, distances, diff

def run_canopy_td(initial_points_km: np.ndarray, config: CanopyConfig) -> np.ndarray:
    points = initial_points_km.copy()
    for _ in range(config.iterations):
        overlap, distances, diff = canopy_overlap(points, config)
        rho = local_density(points, config.R0_km)
        unit = np.zeros_like(diff)
        mask = distances > 1e-9
        unit[mask] = diff[mask] / distances[mask, None]
        force = (overlap[:, :, None] * unit).sum(axis=1)
        step = config.beta * force / (rho[:, None] + 1.0)
        step_norm = np.linalg.norm(step, axis=1)
        too_large = step_norm > config.max_step_km
        if np.any(too_large):
            step[too_large] *= (config.max_step_km / step_norm[too_large])[:, None]
        points = points + step
        total = points - initial_points_km
        total_norm = np.linalg.norm(total, axis=1)
        too_far = total_norm > config.max_total_offset_km
        if np.any(too_far):
            points[too_far] = initial_points_km[too_far] + total[too_far] * (config.max_total_offset_km / total_norm[too_far])[:, None]
    return points

def get_default_configs() -> Dict[str, CanopyConfig]:
    return {
        "CANOPY-Soft": CanopyConfig(beta=0.030, iterations=60, max_step_km=0.08, max_total_offset_km=2.0),
        "CANOPY-Strong": CanopyConfig(beta=0.150, iterations=100, max_step_km=0.20, max_total_offset_km=5.0),
        "Stronger Baseline": CanopyConfig(beta=0.220, iterations=140, max_step_km=0.25, max_total_offset_km=7.0),
    }
