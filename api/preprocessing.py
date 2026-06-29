"""
Preprocessing utilities for Ethiopic handwriting recognition
ALIGNED WITH MAX_STROKES=60 TRAINING
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import interp1d

from api.config import (
    RESAMPLE_N, DIRECTION_BINS, MIN_STROKE_POINTS, 
    MAX_STROKES, FEATURE_DIM
)


def _euclidean(p1: dict, p2: dict) -> float:
    return math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])


def _direction_angle(p1: dict, p2: dict) -> float:
    return math.atan2(p2["y"] - p1["y"], p2["x"] - p1["x"])


def _safe_stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    a = np.array(arr, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _resample_stroke(points: List[dict], n: int = RESAMPLE_N) -> np.ndarray:
    """Resample stroke to fixed number of points - MATCHES TRAINING."""
    if len(points) < 2:
        return np.zeros((n, 3), dtype=np.float32)

    dists = [0.0]
    for i in range(1, len(points)):
        dists.append(dists[-1] + _euclidean(points[i-1], points[i]))

    total_len = dists[-1]
    if total_len == 0:
        x = points[0].get("x", 0)
        y = points[0].get("y", 0)
        p = points[0].get("pressure", 0)
        return np.array([[x, y, p]] * n, dtype=np.float32)

    target_dists = np.linspace(0, total_len, n)
    result = np.zeros((n, 3), dtype=np.float32)

    for field_idx, field in enumerate(["x", "y", "pressure"]):
        vals = np.array([p.get(field, 0) for p in points], dtype=np.float32)
        try:
            f = interp1d(
                dists, vals,
                kind="linear",
                bounds_error=False,
                fill_value=(vals[0], vals[-1]),
            )
            result[:, field_idx] = f(target_dists)
        except:
            result[:, field_idx] = vals[0]

    return result


class ConsistentStrokeExtractor:
    """Extracts FIXED-SIZE features for each stroke - MATCHES TRAINING."""

    def __init__(self):
        self.resampled_points = RESAMPLE_N
        self.resampled_features = 3
        self.resampled_dim = self.resampled_points * self.resampled_features
        self.stat_dim = 50
        self.feature_dim = self.resampled_dim + self.stat_dim
        self.stat_names = self._get_stat_names()

    def _get_stat_names(self) -> List[str]:
        stats = []
        stats.extend(['bbox_width', 'bbox_height', 'bbox_aspect_ratio',
                      'stroke_length', 'displacement'])
        stats.extend(['start_x', 'start_y', 'end_x', 'end_y'])
        stats.extend(['straightness', 'centroid_x', 'centroid_y'])
        stats.extend([f'dir_bin_{i}' for i in range(DIRECTION_BINS)])
        stats.extend(['curvature_mean', 'curvature_std', 'curvature_min',
                      'curvature_max', 'total_curvature'])
        stats.extend(['stroke_duration_ms', 'velocity_mean', 'velocity_std',
                      'total_velocity'])
        stats.extend(['accel_mean', 'accel_std', 'accel_min', 'accel_max'])
        stats.extend(['jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max'])
        stats.extend(['pressure_mean', 'pressure_std', 'pressure_min',
                      'pressure_max', 'pressure_gradient', 'pressure_peak_pos'])
        stats.extend(['tilt_x_mean', 'tilt_x_std', 'tilt_x_min', 'tilt_x_max'])
        stats.extend(['tilt_y_mean', 'tilt_y_std', 'tilt_y_min', 'tilt_y_max'])
        stats.extend(['azimuth_mean', 'azimuth_std', 'azimuth_min', 'azimuth_max'])

        while len(stats) < 50:
            stats.append('padding')

        return stats[:50]

    def extract_stroke_features(self, points: List[dict]) -> np.ndarray:
        if len(points) < MIN_STROKE_POINTS:
            return np.zeros(self.feature_dim, dtype=np.float32)

        resampled = _resample_stroke(points, RESAMPLE_N)
        resampled_flat = resampled.flatten()

        stats_dict = self._extract_statistics(points)
        stat_array = np.zeros(self.stat_dim, dtype=np.float32)
        for i, stat_name in enumerate(self.stat_names):
            if i < self.stat_dim:
                stat_array[i] = stats_dict.get(stat_name, 0.0)

        feature_vector = np.concatenate([resampled_flat, stat_array])
        feature_vector = np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return feature_vector.astype(np.float32)

    def _extract_statistics(self, pts: List[dict]) -> Dict[str, float]:
        n = len(pts)
        feats = {}

        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        feats["bbox_width"] = bbox_w
        feats["bbox_height"] = bbox_h
        feats["bbox_aspect_ratio"] = bbox_w / (bbox_h + 1e-6)

        lengths = [_euclidean(pts[i], pts[i+1]) for i in range(n-1)]
        stroke_len = sum(lengths)
        feats["stroke_length"] = stroke_len

        feats["start_x"] = pts[0]["x"]
        feats["start_y"] = pts[0]["y"]
        feats["end_x"] = pts[-1]["x"]
        feats["end_y"] = pts[-1]["y"]

        displacement = _euclidean(pts[0], pts[-1])
        feats["displacement"] = displacement
        feats["straightness"] = displacement / (stroke_len + 1e-6)

        feats["centroid_x"] = float(np.mean(xs))
        feats["centroid_y"] = float(np.mean(ys))

        angles = [_direction_angle(pts[i], pts[i+1]) for i in range(n-1)]
        hist, _ = np.histogram(angles or [0.], bins=DIRECTION_BINS,
                               range=(-math.pi, math.pi))
        hist = hist / (hist.sum() + 1e-6)
        for b in range(DIRECTION_BINS):
            feats[f"dir_bin_{b}"] = float(hist[b])

        curvatures = []
        for i in range(1, len(angles)):
            delta = abs(angles[i] - angles[i-1])
            delta = min(delta, 2 * math.pi - delta)
            curvatures.append(delta)

        curv_stats = _safe_stats(curvatures)
        feats["curvature_mean"] = curv_stats["mean"]
        feats["curvature_std"] = curv_stats["std"]
        feats["curvature_min"] = curv_stats["min"]
        feats["curvature_max"] = curv_stats["max"]
        feats["total_curvature"] = sum(curvatures)

        timestamps = [p.get("timestamp", 0) for p in pts]
        dt_list = [(timestamps[i+1] - timestamps[i]) / 1000.0 for i in range(n-1)]

        feats["stroke_duration_ms"] = float(timestamps[-1] - timestamps[0])

        velocities = [d / (dt + 1e-6) for d, dt in zip(lengths, dt_list)]
        velocities = [min(v, 1e6) for v in velocities]
        vel_stats = _safe_stats(velocities)
        feats["velocity_mean"] = vel_stats["mean"]
        feats["velocity_std"] = vel_stats["std"]
        feats["total_velocity"] = sum(velocities)

        accels = [abs(velocities[i+1] - velocities[i]) / (dt_list[i] + 1e-6)
                  for i in range(len(velocities)-1)]
        accels = [min(a, 1e6) for a in accels]
        acc_stats = _safe_stats(accels)
        feats["accel_mean"] = acc_stats["mean"]
        feats["accel_std"] = acc_stats["std"]
        feats["accel_min"] = acc_stats["min"]
        feats["accel_max"] = acc_stats["max"]

        jerks = [abs(accels[i+1] - accels[i]) / (dt_list[i] + 1e-6)
                 for i in range(len(accels)-1)]
        jerks = [min(j, 1e6) for j in jerks]
        jerk_stats = _safe_stats(jerks)
        feats["jerk_mean"] = jerk_stats["mean"]
        feats["jerk_std"] = jerk_stats["std"]
        feats["jerk_min"] = jerk_stats["min"]
        feats["jerk_max"] = jerk_stats["max"]

        pressures = [p.get("pressure", 0) for p in pts]
        press_stats = _safe_stats(pressures)
        feats["pressure_mean"] = press_stats["mean"]
        feats["pressure_std"] = press_stats["std"]
        feats["pressure_min"] = press_stats["min"]
        feats["pressure_max"] = press_stats["max"]
        feats["pressure_gradient"] = pressures[-1] - pressures[0]

        max_idx = int(np.argmax(pressures)) if pressures else 0
        feats["pressure_peak_pos"] = max_idx / max(n-1, 1)

        tilt_xs = [p.get("tilt_x", 0) for p in pts]
        tilt_ys = [p.get("tilt_y", 0) for p in pts]

        tx_stats = _safe_stats(tilt_xs)
        feats["tilt_x_mean"] = tx_stats["mean"]
        feats["tilt_x_std"] = tx_stats["std"]
        feats["tilt_x_min"] = tx_stats["min"]
        feats["tilt_x_max"] = tx_stats["max"]

        ty_stats = _safe_stats(tilt_ys)
        feats["tilt_y_mean"] = ty_stats["mean"]
        feats["tilt_y_std"] = ty_stats["std"]
        feats["tilt_y_min"] = ty_stats["min"]
        feats["tilt_y_max"] = ty_stats["max"]

        azimuths = [math.atan2(ty, tx) for tx, ty in zip(tilt_xs, tilt_ys)]
        az_stats = _safe_stats(azimuths)
        feats["azimuth_mean"] = az_stats["mean"]
        feats["azimuth_std"] = az_stats["std"]
        feats["azimuth_min"] = az_stats["min"]
        feats["azimuth_max"] = az_stats["max"]

        return feats


def prepare_features_from_strokes(strokes: List[dict], global_mean: np.ndarray, 
                                   global_std: np.ndarray, max_strokes: int = MAX_STROKES) -> np.ndarray:
    """
    Prepare model input features from stroke data.
    ALIGNED WITH TRAINING PREPROCESSING.
    """
    extractor = ConsistentStrokeExtractor()

    stroke_features = []
    n_real = min(len(strokes), max_strokes)

    for stroke in strokes[:n_real]:
        points = stroke.get("points", [])
        feat = extractor.extract_stroke_features(points)
        stroke_features.append(feat)

    # Pad to max_strokes (60)
    while len(stroke_features) < max_strokes:
        stroke_features.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    feature_matrix = np.stack(stroke_features, axis=0)

    # Apply normalization (using training stats)
    stroke_mask = np.zeros(max_strokes, dtype=bool)
    stroke_mask[:n_real] = True
    feature_matrix[stroke_mask] = (feature_matrix[stroke_mask] - global_mean) / global_std
    feature_matrix[stroke_mask] = np.clip(feature_matrix[stroke_mask], -5.0, 5.0)

    return feature_matrix.astype(np.float32)


def process_handwriting_data(json_data: dict, global_mean: np.ndarray, 
                             global_std: np.ndarray, char2idx: dict,
                             idx2char: dict) -> np.ndarray:
    """Process handwriting JSON data and return model-ready features"""

    # Extract strokes
    strokes = json_data.get("strokes", [])
    if not strokes:
        raise ValueError("No strokes found in data")

    return prepare_features_from_strokes(strokes, global_mean, global_std)
