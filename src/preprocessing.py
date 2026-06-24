from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import re
import numpy as np
import pandas as pd

def load_trajectories(data_dir: str | Path) -> Dict[int, pd.DataFrame]:
    data_dir = Path(data_dir)
    trajectories = {}
    for path in sorted(data_dir.glob("traj_*.txt")):
        match = re.match(r"traj_(\d+)\.txt$", path.name)
        if not match:
            continue
        tid = int(match.group(1))
        df = pd.read_csv(path, sep=";")
        df.columns = [str(c).strip().strip('"') for c in df.columns]
        if not {"x", "y"}.issubset(df.columns):
            continue
        for col in ["vx", "vy", "t"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["x", "y", "vx", "vy", "t"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
        if len(df) > 1:
            trajectories[tid] = df
    return trajectories

def lonlat_reference(trajectories: Dict[int, pd.DataFrame]) -> Tuple[float, float, float, float]:
    all_points = pd.concat(trajectories.values(), ignore_index=True)
    mean_lat = float(all_points["y"].mean())
    mean_lon = float(all_points["x"].mean())
    return mean_lon, mean_lat, 111.0 * np.cos(np.deg2rad(mean_lat)), 111.0

def lonlat_to_km(lon, lat, mean_lon, mean_lat, km_per_deg_lon, km_per_deg_lat):
    return np.column_stack([(np.asarray(lon)-mean_lon)*km_per_deg_lon, (np.asarray(lat)-mean_lat)*km_per_deg_lat])

def trajectory_centroids_km(trajectories: Dict[int, pd.DataFrame]) -> Tuple[List[int], np.ndarray, dict]:
    ids = sorted(trajectories)
    mean_lon, mean_lat, km_lon, km_lat = lonlat_reference(trajectories)
    centroids_lonlat = np.array([[trajectories[i]["x"].mean(), trajectories[i]["y"].mean()] for i in ids])
    centroids_km = lonlat_to_km(centroids_lonlat[:,0], centroids_lonlat[:,1], mean_lon, mean_lat, km_lon, km_lat)
    return ids, centroids_km, {"mean_lon": mean_lon, "mean_lat": mean_lat, "km_per_deg_lon": km_lon, "km_per_deg_lat": km_lat}
