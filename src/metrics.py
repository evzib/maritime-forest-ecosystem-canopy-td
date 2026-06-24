from __future__ import annotations
import numpy as np
from canopy_td import CanopyConfig, pairwise_distances, local_density, adaptive_canopy_radius

def compute_metrics(points_km, initial_points_km, config: CanopyConfig):
    distances, _ = pairwise_distances(points_km)
    rho = local_density(points_km, config.R0_km)
    radius = adaptive_canopy_radius(points_km, config)
    n = points_km.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    conflicts = int(((distances < config.d_safe_km) & upper).sum())
    overlap = np.maximum(0.0, radius[:, None] + radius[None, :] - distances)
    canopy_overlap = float(overlap[upper].sum())
    congestion = float((rho ** 2).sum() / n)
    interaction = (distances < (2 * config.R0_km)) & upper
    rho_pair = (rho[:, None] + rho[None, :]) / 2.0
    risk = float((rho_pair[interaction] / (distances[interaction] + config.epsilon)).sum())
    offsets = np.sqrt(((points_km - initial_points_km) ** 2).sum(axis=1))
    return {
        "Conflicts": conflicts,
        "Canopy overlap": canopy_overlap,
        "Congestion score": congestion,
        "Risk score": risk,
        "Mean route offset, km": float(offsets.mean()),
        "Max route offset, km": float(offsets.max()),
    }
