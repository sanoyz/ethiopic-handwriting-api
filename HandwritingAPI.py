"""
═══════════════════════════════════════════════════════════════════════════════
Ethiopic Handwriting Recognition - Complete Unified Server
═══════════════════════════════════════════════════════════════════════════════
This combines:
1. FastAPI REST API (for file upload and programmatic access)
2. Real-time WebSocket interface (for interactive handwriting)
Both use the same MAX_STROKES=60 model and deployment data

VENV COMPATIBLE - Works with Python virtual environment on Windows
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import asyncio
import warnings
import socket
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# WebSocket imports
from aiohttp import web
import websockets

warnings.filterwarnings("ignore")

# ============================================================
# FIX WINDOWS ENCODING
# ============================================================
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================
# CONSTANTS - MUST MATCH TRAINING CONFIGURATION (MAX_STROKES=60)
# ============================================================
RESAMPLE_N = 32
DIRECTION_BINS = 8
MIN_STROKE_POINTS = 2
MAX_STROKES = 60
FEATURE_DIM = 146

# ENHANCED MODEL CAPACITY (matches training)
D_MODEL = 384
N_HEADS = 12
N_LAYERS = 5
D_FF = 768
DROPOUT = 0.15

# ENHANCED MEMORY (matches training)
MEMORY_SIZE = 192
MEMORY_HEADS = 6
PROTOTYPE_DROPOUT = 0.1
USE_CROSS_ATTENTION = True
USE_POSITION_ENCODING = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default paths - USE ABSOLUTE WINDOWS PATHS
DEFAULT_CHECKPOINT = r"C:\YonAPI\models\API_ready_model\best_model.pt"
DEFAULT_DEPLOYMENT = r"C:\YonAPI\deployment_data"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port+max_attempts}")


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
    """Resample stroke to fixed number of points."""
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


# ============================================================
# FEATURE EXTRACTOR - MATCHES TRAINING
# ============================================================

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


# ============================================================
# NEURAL NETWORK MODELS - MATCHES TRAINING (MAX_STROKES=60)
# ============================================================

class MultiHeadPositionAwareMemoryBank(nn.Module):
    def __init__(self, d_model: int, num_prototypes: int, num_heads: int, 
                 max_strokes: int, dropout: float, prototype_dropout: float = 0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_strokes = max_strokes
        
        self.memory = nn.Parameter(
            torch.randn(num_prototypes, d_model) * 0.02
        )
        
        self.position_encoding = nn.Parameter(
            torch.randn(max_strokes, num_prototypes) * 0.02
        )
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.attention_bias = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout)
        self.prototype_dropout = prototype_dropout
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, padding_mask=None, training=False):
        B, S, D = x.shape
        
        memory = self.memory
        
        Q = self.W_q(x)
        K = self.W_k(memory)
        V = self.W_v(memory)
        
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        K = K.view(self.num_prototypes, self.num_heads, self.head_dim)
        K = K.unsqueeze(0).transpose(1, 2)
        K = K.expand(B, -1, -1, -1)
        
        V = V.view(self.num_prototypes, self.num_heads, self.head_dim)
        V = V.unsqueeze(0).transpose(1, 2)
        V = V.expand(B, -1, -1, -1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.position_encoding is not None:
            pos_bias = self.position_encoding[:S, :]
            pos_bias = pos_bias.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, -1, -1)
            scores = scores + pos_bias
        
        scores = scores / self.temperature.abs()
        scores = scores + self.attention_bias
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(padding_mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, S, D)
        
        gate = torch.sigmoid(self.temperature)
        x = self.norm1(x + gate * self.W_o(attended))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights.mean(dim=1)


class InterSentenceCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, memory_size: int, memory_heads: int,
                 max_strokes: int, dropout: float, prototype_dropout: float):
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        
        self.stroke_memory = MultiHeadPositionAwareMemoryBank(
            d_model, memory_size, memory_heads, max_strokes, 
            dropout, prototype_dropout
        )
        
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None, training=False):
        B, S, D = x.shape
        
        x, memory_weights = self.stroke_memory(x, padding_mask, training)
        
        memory = self.stroke_memory.memory.unsqueeze(0).expand(B, -1, -1)
        
        attn_out, attention_weights = self.cross_attn(
            x, memory, memory,
            key_padding_mask=None
        )
        
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x, attention_weights


class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 memory_size: int, memory_heads: int, max_strokes: int,
                 dropout: float, prototype_dropout: float, use_cross_attention: bool):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        if use_cross_attention:
            self.cross_attn = InterSentenceCrossAttention(
                d_model, n_heads, memory_size, memory_heads, 
                max_strokes, dropout, prototype_dropout
            )
        else:
            self.cross_attn = None
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None, training=False):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        if self.cross_attn is not None:
            x, _ = self.cross_attn(x, padding_mask, training)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x


class EthiopicRecognizerWithEnhancedMemory(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int = FEATURE_DIM, 
                 max_strokes: int = MAX_STROKES):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_strokes = max_strokes
        
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT)
        )
        
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_strokes, D_MODEL) * 0.02
        )
        
        self.encoder_layers = nn.ModuleList([
            EnhancedTransformerEncoderLayer(
                d_model=D_MODEL,
                n_heads=N_HEADS,
                d_ff=D_FF,
                memory_size=MEMORY_SIZE,
                memory_heads=MEMORY_HEADS,
                max_strokes=max_strokes,
                dropout=DROPOUT,
                prototype_dropout=PROTOTYPE_DROPOUT,
                use_cross_attention=USE_CROSS_ATTENTION,
            )
            for _ in range(N_LAYERS)
        ])
        
        self.norm = nn.LayerNorm(D_MODEL)
        self.output_layer = nn.Linear(D_MODEL, vocab_size + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, training=False):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, : x.size(1), :]
        
        padding_mask = (x.sum(dim=-1) == 0)
        
        for layer in self.encoder_layers:
            x = layer(x, padding_mask, training)
        
        x = self.norm(x)
        logits = self.output_layer(x)
        return F.log_softmax(logits, dim=-1)


# ============================================================
# POST-PROCESSING CORRECTION MODULE
# ============================================================

class TextCorrector:
    """Post-processing correction for Ethiopic text recognition errors"""
    
    def __init__(self, deployment_data_path: Optional[str] = None):
        self.confusion_map = {}
        self.context_rules = []
        self.common_words = {}
        self.confidence_threshold = 0.6
        self.char_confidence = {}
        
        if deployment_data_path and os.path.exists(deployment_data_path):
            self._load_deployment_data(deployment_data_path)
        else:
            self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default correction rules"""
        self.confusion_map = {
            'ዚ': [('ማ', 0.85), ('መ', 0.10), ('ሞ', 0.05)],
            'ተተ': [('ተ', 0.90), ('ት', 0.05), ('ታ', 0.05)],
            'ዋ': [('ቹ', 0.65), ('ዎ', 0.20), ('ወ', 0.10)],
            'ስ': [('ዛ', 0.60), ('ሰ', 0.25), ('ሳ', 0.10)],
            'ነ': [('ከ', 0.55), ('ኔ', 0.20), ('ን', 0.15)],
            'ተ': [('በ', 0.50), ('ት', 0.30), ('ታ', 0.15)],
            'መ': [('ል', 0.50), ('ም', 0.30), ('ማ', 0.15)],
        }
        
        self.context_rules = [
            (r'ተ([ማሪ])', r'ተማ\1'),
            (r'አ([ዲስ])', r'አዲስ'),
            (r'ተዚ', r'ተማ'),
            (r'\s+', r' '),
            (r'\s([።፡፤፥?!])', r'\1'),
            (r'([።፡፤፥?!])([^\s])', r'\1 \2'),
        ]
        
        self.common_words = {
            'ተማሪ': True,
            'አዲስ': True,
            'አበባ': True,
            'የኢትዮጵያ': True,
            'ዋና': True,
            'ከተማ': True,
            'ናት': True,
            'ፍቅር': True,
            'ሰላም': True,
        }
    
    def _load_deployment_data(self, deployment_data_path: str):
        """Load correction data from training deployment_data folder"""
        print(f"[INFO] Loading deployment data from: {deployment_data_path}")
        
        try:
            top_confusions_path = os.path.join(deployment_data_path, "top_confusions.json")
            if os.path.exists(top_confusions_path):
                with open(top_confusions_path, 'r', encoding='utf-8') as f:
                    top_confusions = json.load(f)
                    for item in top_confusions:
                        target = item.get('target')
                        predicted = item.get('predicted')
                        rate = item.get('rate', 0)
                        if target and predicted and rate > 0.05:
                            if target not in self.confusion_map:
                                self.confusion_map[target] = []
                            self.confusion_map[target].append((predicted, rate))
                print(f"  [OK] Loaded {len(top_confusions)} top confusions")
            
            char_confidence_path = os.path.join(deployment_data_path, "character_confidence.json")
            if os.path.exists(char_confidence_path):
                with open(char_confidence_path, 'r', encoding='utf-8') as f:
                    self.char_confidence = json.load(f)
                    for char, stats in self.char_confidence.items():
                        if stats.get('mean', 1.0) < 0.5:
                            self.confidence_threshold = min(
                                self.confidence_threshold, 
                                stats.get('mean', 0.6)
                            )
                print(f"  [OK] Loaded character confidence data")
            
            correction_data_path = os.path.join(deployment_data_path, "correction_data.json")
            if os.path.exists(correction_data_path):
                with open(correction_data_path, 'r', encoding='utf-8') as f:
                    correction_data = json.load(f)
                    if 'characters' in correction_data:
                        for char in correction_data['characters']:
                            self.common_words[char] = True
                    print(f"  [OK] Loaded correction data")
            
            print(f"[OK] Deployment data loaded successfully!")
            
        except Exception as e:
            print(f"[WARNING] Error loading deployment data: {e}")
            self._initialize_default_rules()
    
    def _get_context(self, text: str, pos: int, window: int = 3) -> str:
        start = max(0, pos - window)
        end = min(len(text), pos + window + 1)
        return text[start:end]
    
    def _context_matches(self, context: str, suggestion: str) -> bool:
        patterns = {
            'ማ': ['ተ', 'ሪ', 'ዎ', 'ሁ'],
            'ሪ': ['ተ', 'ማ', 'ዎ', 'ቹ'],
            'ቹ': ['ማ', 'ሪ', 'ዎ'],
        }
        for pattern_char, followers in patterns.items():
            if suggestion in followers and pattern_char in context:
                return True
        return True
    
    def _apply_character_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        chars = list(text)
        corrections = []
        
        for i, char in enumerate(chars):
            if char in self.confusion_map:
                suggestions = self.confusion_map[char]
                if suggestions:
                    best_suggestion, conf = max(suggestions, key=lambda x: x[1])
                    if conf > self.confidence_threshold:
                        context = self._get_context(text, i)
                        if self._context_matches(context, best_suggestion):
                            chars[i] = best_suggestion
                            corrections.append({
                                'position': i,
                                'original': char,
                                'corrected': best_suggestion,
                                'confidence': conf
                            })
        
        return ''.join(chars), corrections
    
    def _apply_context_rules(self, text: str) -> Tuple[str, List[Dict]]:
        corrections = []
        for pattern, replacement in self.context_rules:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                corrections.append({
                    'pattern': pattern,
                    'replacement': replacement
                })
        return text, corrections
    
    def _apply_dictionary_correction(self, text: str) -> Tuple[str, List[Dict]]:
        words = text.split()
        corrected_words = []
        corrections = []
        
        for i, word in enumerate(words):
            if word in self.common_words:
                corrected_words.append(word)
                continue
            
            best_match = None
            best_score = 0
            
            for dict_word in self.common_words:
                score = SequenceMatcher(None, word, dict_word).ratio()
                if score > 0.7 and score > best_score:
                    best_match = dict_word
                    best_score = score
            
            if best_match and best_score > 0.7:
                corrected_words.append(best_match)
                corrections.append({
                    'position': i,
                    'original': word,
                    'corrected': best_match,
                    'confidence': best_score
                })
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections
    
    def _fix_spacing(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([።፡፤፥?!])', r'\1', text)
        text = re.sub(r'([።፡፤፥?!])([^\s])', r'\1 \2', text)
        return text.strip()
    
    def correct_text(self, text: str, confidence: float = 0.8) -> Dict[str, Any]:
        if not text or len(text) < 1:
            return {
                'original': text,
                'corrected': text,
                'corrections': [],
                'was_corrected': False,
                'confidence': confidence
            }
        
        original = text
        corrected = text
        all_corrections = []
        
        if confidence > 0.9:
            corrected, spacing_corrections = self._apply_context_rules(corrected)
            all_corrections.extend(spacing_corrections)
        else:
            corrected, char_corrections = self._apply_character_corrections(corrected)
            all_corrections.extend(char_corrections)
            
            corrected, context_corrections = self._apply_context_rules(corrected)
            all_corrections.extend(context_corrections)
            
            corrected, dict_corrections = self._apply_dictionary_correction(corrected)
            all_corrections.extend(dict_corrections)
        
        corrected = self._fix_spacing(corrected)
        was_corrected = original != corrected
        
        return {
            'original': original,
            'corrected': corrected,
            'corrections': all_corrections,
            'was_corrected': was_corrected,
            'correction_count': len(all_corrections),
            'confidence': confidence
        }


# ============================================================
# INFERENCE ENGINE
# ============================================================

class HandwritingInferenceEngine:
    def __init__(self, checkpoint_path: str, deployment_data_path: Optional[str] = None):
        print(f"\n{'='*60}")
        print(f"Loading Model for Real-Time Inference (MAX_STROKES=60)")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        except Exception as e:
            print(f"[ERROR] Error loading checkpoint: {e}")
            raise
        
        self.idx2char = checkpoint["idx2char"]
        self.char2idx = checkpoint["char2idx"]
        self.vocab_size = checkpoint["vocab_size"]
        self.global_mean = checkpoint["global_mean"]
        self.global_std = checkpoint["global_std"]
        self.max_strokes = checkpoint.get("max_strokes", MAX_STROKES)
        
        print(f"  [OK] Vocabulary size: {self.vocab_size}")
        print(f"  [OK] Max strokes: {self.max_strokes}")
        print(f"  [OK] Device: {DEVICE}")
        
        self.extractor = ConsistentStrokeExtractor()
        print(f"  [OK] Feature extractor ready")
        
        self.corrector = TextCorrector(deployment_data_path)
        print(f"  [OK] Text corrector initialized")
        
        self.model = EthiopicRecognizerWithEnhancedMemory(
            vocab_size=self.vocab_size,
            feature_dim=FEATURE_DIM,
            max_strokes=self.max_strokes
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [OK] Model loaded: {total_params:,} parameters")
        print(f"{'='*60}\n")
    
    def _prepare_features(self, strokes: List[List[dict]]) -> np.ndarray:
        stroke_features = []
        n_real = min(len(strokes), self.max_strokes)
        
        for stroke in strokes[:n_real]:
            feat = self.extractor.extract_stroke_features(stroke)
            stroke_features.append(feat)
        
        while len(stroke_features) < self.max_strokes:
            stroke_features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        
        feature_matrix = np.stack(stroke_features, axis=0)
        stroke_mask = np.zeros(self.max_strokes, dtype=bool)
        stroke_mask[:n_real] = True
        
        feature_matrix[stroke_mask] = (
            (feature_matrix[stroke_mask] - self.global_mean) / self.global_std
        )
        feature_matrix[stroke_mask] = np.clip(feature_matrix[stroke_mask], -5.0, 5.0)
        feature_matrix = feature_matrix.astype(np.float32)
        
        return feature_matrix
    
    def _greedy_decode(self, log_probs, blank=0) -> List[List[int]]:
        predictions = log_probs.argmax(dim=-1)
        decoded = []
        for pred in predictions:
            prev = blank
            seq = []
            for token in pred:
                token_val = token.item() if torch.is_tensor(token) else token
                if token_val != blank and token_val != prev:
                    seq.append(token_val)
                prev = token_val
            decoded.append(seq)
        return decoded
    
    def _decode_tokens_to_text(self, tokens: List[int]) -> str:
        text_chars = []
        for tok in tokens:
            if tok == 0:
                continue
            char = self.idx2char.get(tok)
            if char is None:
                text_chars.append("?")
            else:
                text_chars.append(char)
        return "".join(text_chars)
    
    def recognize(self, strokes: List[List[dict]]) -> Dict[str, Any]:
        start_time = time.time()
        
        features = self._prepare_features(strokes)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            log_probs = self.model(features_tensor, training=False)
        
        predictions = self._greedy_decode(log_probs)
        original_text = self._decode_tokens_to_text(predictions[0])
        
        probs = torch.exp(log_probs)
        top_probs, _ = probs.max(dim=-1)
        confidence = float(top_probs.mean().cpu().numpy() * 100)
        confidence = min(100.0, max(0.0, confidence))
        
        correction_result = self.corrector.correct_text(original_text, confidence / 100.0)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'text': correction_result['corrected'],
            'original_text': original_text,
            'confidence': confidence,
            'inference_time_ms': round(inference_time, 2),
            'was_corrected': correction_result['was_corrected'],
            'corrections': correction_result['corrections'],
            'correction_count': correction_result['correction_count'],
            'total_strokes': len(strokes),
            'total_points': sum(len(s) for s in strokes)
        }


# ============================================================
# HTML CONTENT
# ============================================================

def get_html_content(ws_port: int, default_checkpoint: str = "", default_deployment: str = "") -> str:
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ethiopic Handwriting Recognition (MAX_STROKES=60)</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 20px 30px;
        }}
        .header h1 {{ font-size: 24px; }}
        .badge {{ background: #10b981; padding: 2px 12px; border-radius: 12px; font-size: 12px; margin-left: 10px; }}
        .connection-status {{
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            display: inline-block;
        }}
        .connection-status.connected {{ background: #10b981; }}
        .connection-status.disconnected {{ background: #ef4444; }}
        .connection-status.connecting {{ background: #f59e0b; }}
        .content {{ display: flex; padding: 20px; gap: 20px; background: #f8fafc; }}
        .canvas-section {{ flex: 2; }}
        .controls-section {{ flex: 1; min-width: 300px; }}
        .card {{ background: white; border-radius: 12px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-bottom: 15px; color: #1e293b; font-size: 16px; }}
        .status-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        .status-item {{ display: flex; justify-content: space-between; align-items: center; }}
        .status-label {{ font-weight: 600; color: #64748b; font-size: 13px; }}
        .status-value {{ font-weight: bold; color: #1e293b; font-size: 14px; }}
        .pressure-bar-container {{ background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden; margin-top: 10px; }}
        .pressure-bar {{ background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: 0%; transition: width 0.05s ease; }}
        .canvas-container {{ background: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        canvas {{ border: 2px solid #e2e8f0; border-radius: 8px; cursor: crosshair; width: 100%; height: auto; background: white; touch-action: none; }}
        .button-group {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }}
        button {{ padding: 10px 20px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; font-size: 14px; }}
        button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        button:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .btn-primary {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
        .btn-success {{ background: #10b981; color: white; }}
        .btn-secondary {{ background: #64748b; color: white; }}
        .btn-info {{ background: #3b82f6; color: white; }}
        .result-box {{ background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; padding: 20px; color: white; margin-top: 20px; }}
        .result-text {{ font-size: 28px; font-family: 'Nyala', 'Arial Unicode MS', monospace; word-break: break-word; margin: 10px 0; color: #34d399; min-height: 50px; }}
        .result-text.corrected {{ color: #fbbf24; }}
        .result-original {{ font-size: 16px; color: #94a3b8; margin-top: 5px; text-decoration: line-through; opacity: 0.7; }}
        .correction-badge {{ background: #f59e0b; color: black; padding: 2px 12px; border-radius: 12px; font-size: 12px; margin-left: 10px; }}
        .stats {{ display: flex; gap: 20px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #334155; }}
        .stat {{ flex: 1; }}
        .stat-label {{ font-size: 11px; opacity: 0.7; }}
        .stat-value {{ font-size: 20px; font-weight: bold; }}
        .status-message {{ background: #1d4ed8; color: white; padding: 12px 20px; border-radius: 10px; font-size: 13px; margin-top: 15px; }}
        input {{ width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; margin-bottom: 10px; font-family: monospace; }}
        .model-info {{ margin-top: 10px; font-size: 12px; padding: 8px; background: #f1f5f9; border-radius: 6px; }}
        .model-info.loaded {{ background: #10b981; color: white; }}
        .model-info.error {{ background: #ef4444; color: white; }}
        .path-info {{ font-size: 11px; color: #94a3b8; margin-top: 5px; padding: 4px 8px; background: #1e293b; border-radius: 4px; word-break: break-all; }}
        @media (max-width: 768px) {{ .content {{ flex-direction: column; }} .controls-section {{ min-width: auto; }} }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>✍️ Ethiopic Handwriting Recognition <span class="badge">MAX_STROKES=60</span></h1>
        <p>Real-time inference with Enhanced Multi-Head Memory + Position Encoding + Corrections</p>
        <div id="connectionStatus" class="connection-status connecting">Connecting...</div>
    </div>
    
    <div class="content">
        <div class="canvas-section">
            <div class="canvas-container">
                <canvas id="handwritingCanvas" width="800" height="500"></canvas>
                <div class="button-group">
                    <button class="btn-secondary" id="clearBtn">Clear Canvas</button>
                    <button class="btn-success" id="saveBtn">Save JSON</button>
                    <button class="btn-primary" id="recognizeBtn">Recognize</button>
                </div>
            </div>
            
            <div class="result-box">
                <div style="font-size: 14px; opacity: 0.8;">
                    Recognition Result
                    <span id="correctionBadge" style="display:none;" class="correction-badge">Corrected</span>
                </div>
                <div class="result-text" id="resultText">—</div>
                <div class="result-original" id="originalText" style="display:none;"></div>
                <div class="stats">
                    <div class="stat"><div class="stat-label">Time</div><div class="stat-value" id="timeValue">-- ms</div></div>
                    <div class="stat"><div class="stat-label">Confidence</div><div class="stat-value" id="confidenceValue">--%</div></div>
                    <div class="stat"><div class="stat-label">Points</div><div class="stat-value" id="pointsValue">0</div></div>
                    <div class="stat"><div class="stat-label">Strokes</div><div class="stat-value" id="strokesValue">0</div></div>
                </div>
                <div id="correctionDetails" style="display:none; font-size:12px; color:#94a3b8; margin-top:5px; padding:5px 10px; background:#1e293b; border-radius:6px;"></div>
            </div>
        </div>
        
        <div class="controls-section">
            <div class="card">
                <h3>Pen Status</h3>
                <div class="status-grid">
                    <div class="status-item"><span class="status-label">Pen Type:</span><span class="status-value" id="penType">Stylus</span></div>
                    <div class="status-item"><span class="status-label">Pressure:</span><span class="status-value" id="pressureValue">0%</span></div>
                    <div class="status-item"><span class="status-label">Tilt X:</span><span class="status-value" id="tiltX">0°</span></div>
                    <div class="status-item"><span class="status-label">Tilt Y:</span><span class="status-value" id="tiltY">0°</span></div>
                </div>
                <div class="pressure-bar-container"><div class="pressure-bar" id="pressureBar"></div></div>
            </div>
            
            <div class="card">
                <h3>Session Stats</h3>
                <div class="status-grid">
                    <div class="status-item"><span class="status-label">Total Strokes:</span><span class="status-value" id="totalStrokes">0</span></div>
                    <div class="status-item"><span class="status-label">Total Points:</span><span class="status-value" id="totalPoints">0</span></div>
                    <div class="status-item"><span class="status-label">Max Strokes:</span><span class="status-value">60</span></div>
                </div>
            </div>
            
            <div class="card">
                <h3>Model (MAX_STROKES=60)</h3>
                <input type="text" id="checkpointPath" placeholder="Checkpoint path" value="{default_checkpoint}">
                <input type="text" id="deploymentDataPath" placeholder="Deployment data path" value="{default_deployment}">
                <div class="button-group">
                    <button class="btn-info" id="loadModelBtn" style="flex:1">Load Model</button>
                    <button class="btn-secondary" id="statusCheckBtn" style="flex:1">Check Status</button>
                </div>
                <div class="model-info" id="modelStatus">Model not loaded</div>
                <div class="path-info" id="pathInfo"></div>
            </div>
            
            <div class="status-message" id="statusMessage">Ready - Write naturally on the canvas.</div>
        </div>
    </div>
</div>

<script>
    let allStrokes = [], currentStroke = [], isDrawing = false;
    let ws = null, modelLoaded = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const MAX_STROKES = 60;
    
    const canvas = document.getElementById('handwritingCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 800; canvas.height = 500;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#1f2937';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 2;
    
    function connectWebSocket() {{
        const wsPort = {ws_port};
        ws = new WebSocket(`ws://localhost:${{wsPort}}`);
        ws.onopen = () => {{
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
            showStatus('Connected to server', 'success');
            reconnectAttempts = 0;
        }};
        ws.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            if (data.type === 'model_loaded') {{
                if (data.success) {{
                    modelLoaded = true;
                    document.getElementById('modelStatus').innerHTML = `Model ready (vocab: ${{data.vocab_size}}, max_strokes: 60)`;
                    document.getElementById('modelStatus').className = 'model-info loaded';
                    document.getElementById('pathInfo').innerHTML = `Deployment data: ${{data.deployment_loaded ? 'Loaded' : 'Not loaded'}}`;
                    showStatus('Model loaded successfully!', 'success');
                    document.getElementById('recognizeBtn').disabled = false;
                }} else {{
                    modelLoaded = false;
                    document.getElementById('modelStatus').innerHTML = `Error: ${{data.error}}`;
                    document.getElementById('modelStatus').className = 'model-info error';
                    document.getElementById('recognizeBtn').disabled = true;
                }}
            }} else if (data.type === 'recognition_result') {{
                if (data.success) {{
                    const resultText = document.getElementById('resultText');
                    resultText.textContent = data.text;
                    
                    if (data.was_corrected) {{
                        document.getElementById('correctionBadge').style.display = 'inline';
                        resultText.className = 'result-text corrected';
                        document.getElementById('originalText').textContent = `Original: "${{data.original_text}}"`;
                        document.getElementById('originalText').style.display = 'block';
                        if (data.corrections && data.corrections.length > 0) {{
                            const details = document.getElementById('correctionDetails');
                            details.style.display = 'block';
                            details.innerHTML = data.corrections.map(c => 
                                `<div>• "${{c.original || c.pattern}}" → "${{c.corrected || c.replacement}}"</div>`
                            ).join('');
                        }}
                    }} else {{
                        document.getElementById('correctionBadge').style.display = 'none';
                        resultText.className = 'result-text';
                        document.getElementById('originalText').style.display = 'none';
                        document.getElementById('correctionDetails').style.display = 'none';
                    }}
                    
                    document.getElementById('timeValue').textContent = `${{data.time}} ms`;
                    document.getElementById('confidenceValue').textContent = `${{data.confidence}}%`;
                    document.getElementById('strokesValue').textContent = data.total_strokes || 0;
                    document.getElementById('pointsValue').textContent = data.total_points || 0;
                    showStatus(`Recognized: "${{data.text}}"`, 'success');
                }} else {{
                    document.getElementById('resultText').textContent = 'Error';
                    showStatus(`Error: ${{data.error}}`, 'error');
                }}
            }} else if (data.type === 'status') {{
                if (data.status === 'loaded') {{
                    modelLoaded = true;
                    document.getElementById('modelStatus').innerHTML = `Ready (vocab: ${{data.vocab_size}})`;
                    document.getElementById('modelStatus').className = 'model-info loaded';
                    document.getElementById('recognizeBtn').disabled = false;
                }} else {{
                    modelLoaded = false;
                    document.getElementById('modelStatus').innerHTML = 'Model not loaded';
                    document.getElementById('modelStatus').className = 'model-info';
                    document.getElementById('recognizeBtn').disabled = true;
                }}
            }}
        }};
        ws.onerror = () => {{
            document.getElementById('connectionStatus').textContent = 'Connection Error';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        }};
        ws.onclose = () => {{
            document.getElementById('connectionStatus').textContent = 'Reconnecting...';
            document.getElementById('connectionStatus').className = 'connection-status connecting';
            if (reconnectAttempts < maxReconnectAttempts) {{
                reconnectAttempts++;
                setTimeout(connectWebSocket, 3000);
            }} else {{
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                showStatus('Could not connect to server', 'error');
            }}
        }};
    }}
    
    function sendCommand(command, data = {{}}) {{
        if (ws && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{ command, ...data }}));
        }} else {{
            showStatus('Not connected to server', 'error');
        }}
    }}
    
    function loadModel() {{
        const checkpointPath = document.getElementById('checkpointPath').value;
        const deploymentPath = document.getElementById('deploymentDataPath').value;
        if (!checkpointPath) {{
            showStatus('Please enter a checkpoint path', 'error');
            return;
        }}
        showStatus('Loading model...', 'info');
        sendCommand('load_model', {{ checkpoint_path: checkpointPath, deployment_data_path: deploymentPath }});
    }}
    
    function checkStatus() {{ sendCommand('status'); }}
    
    function getPressure(e) {{ return e.pressure !== undefined ? Math.min(1, Math.max(0, e.pressure)) : 0.6; }}
    function getTilt(e) {{ return {{ tiltX: e.tiltX || 0, tiltY: e.tiltY || 0 }}; }}
    
    function updatePenStatus(e) {{
        if (!e) {{
            document.getElementById('pressureValue').textContent = '0%';
            document.getElementById('pressureBar').style.width = '0%';
            return;
        }}
        const p = Math.round(getPressure(e) * 100);
        document.getElementById('pressureValue').textContent = `${{p}}%`;
        document.getElementById('pressureBar').style.width = `${{p}}%`;
        const {{ tiltX, tiltY }} = getTilt(e);
        document.getElementById('tiltX').textContent = `${{Math.round(tiltX)}}°`;
        document.getElementById('tiltY').textContent = `${{Math.round(tiltY)}}°`;
    }}
    
    function getCoords(e) {{
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        let cx = (e.clientX - rect.left) * scaleX;
        let cy = (e.clientY - rect.top) * scaleY;
        return {{ x: Math.max(0, Math.min(canvas.width, cx)), y: Math.max(0, Math.min(canvas.height, cy)) }};
    }}
    
    function startDrawing(e) {{
        e.preventDefault();
        isDrawing = true;
        const coords = getCoords(e);
        const pressure = getPressure(e);
        const {{ tiltX, tiltY }} = getTilt(e);
        currentStroke = [{{
            x: coords.x, y: coords.y,
            timestamp: Date.now(),
            pressure,
            tilt_x: tiltX, tilt_y: tiltY,
            twist: 0,
            pen_type: e.pointerType || 'pen',
            device_type: 'pen_tablet'
        }}];
        updatePenStatus(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        const dotSize = Math.max(2, Math.floor(1 + pressure * 6));
        ctx.fillStyle = '#1f2937';
        ctx.beginPath();
        ctx.arc(coords.x, coords.y, dotSize/2, 0, Math.PI * 2);
        ctx.fill();
    }}
    
    function draw(e) {{
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCoords(e);
        const pressure = getPressure(e);
        const {{ tiltX, tiltY }} = getTilt(e);
        currentStroke.push({{
            x: coords.x, y: coords.y,
            timestamp: Date.now(),
            pressure,
            tilt_x: tiltX, tilt_y: tiltY,
            twist: 0,
            pen_type: e.pointerType || 'pen',
            device_type: 'pen_tablet'
        }});
        updatePenStatus(e);
        ctx.lineWidth = 1 + pressure * 6;
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
    }}
    
    function stopDrawing() {{
        if (!isDrawing) return;
        isDrawing = false;
        if (currentStroke.length > 2) {{
            allStrokes.push([...currentStroke]);
            if (allStrokes.length > MAX_STROKES) {{
                showStatus(`Max strokes (${{MAX_STROKES}}) exceeded! Extra strokes will be ignored.`, 'warning');
            }}
            document.getElementById('totalStrokes').textContent = allStrokes.length;
            document.getElementById('totalPoints').textContent = allStrokes.reduce((s, st) => s + st.length, 0);
            document.getElementById('strokesValue').textContent = allStrokes.length;
            document.getElementById('pointsValue').textContent = allStrokes.reduce((s, st) => s + st.length, 0);
        }}
        currentStroke = [];
        updatePenStatus(null);
    }}
    
    function clearCanvas() {{
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        allStrokes = [];
        currentStroke = [];
        document.getElementById('totalStrokes').textContent = '0';
        document.getElementById('totalPoints').textContent = '0';
        document.getElementById('strokesValue').textContent = '0';
        document.getElementById('pointsValue').textContent = '0';
        document.getElementById('resultText').textContent = '—';
        document.getElementById('originalText').style.display = 'none';
        document.getElementById('correctionBadge').style.display = 'none';
        document.getElementById('correctionDetails').style.display = 'none';
        document.getElementById('timeValue').textContent = '-- ms';
        document.getElementById('confidenceValue').textContent = '--%';
        showStatus('Canvas cleared', 'info');
    }}
    
    function saveHandwriting() {{
        if (allStrokes.length === 0) {{ showStatus('Nothing to save', 'warning'); return; }}
        const sampleId = `${{Date.now()}}_${{Math.random().toString(36).substr(2, 6)}}`;
        const data = {{
            sample_id: sampleId,
            timestamp: new Date().toISOString(),
            canvas_size: {{ width: canvas.width, height: canvas.height }},
            strokes: allStrokes.map((s, i) => ({{ stroke_index: i+1, points: s }})),
            total_strokes: allStrokes.length,
            total_points: allStrokes.reduce((s, st) => s + st.length, 0),
            max_strokes_config: MAX_STROKES
        }};
        const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `handwriting_${{sampleId}}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        showStatus('Saved!', 'success');
    }}
    
    function recognize() {{
        if (allStrokes.length === 0) {{ showStatus('Write something first!', 'warning'); return; }}
        if (!modelLoaded) {{ showStatus('Please load a model first!', 'error'); return; }}
        showStatus('Recognizing...', 'info');
        document.getElementById('resultText').textContent = 'Processing...';
        sendCommand('recognize', {{ strokes: allStrokes }});
    }}
    
    function showStatus(msg, type) {{
        const el = document.getElementById('statusMessage');
        el.textContent = msg;
        const colors = {{ info: '#1d4ed8', success: '#10b981', warning: '#f59e0b', error: '#ef4444' }};
        el.style.background = colors[type] || '#1d4ed8';
        if (type !== 'error') {{
            setTimeout(() => {{
                if (el.textContent === msg) {{
                    el.textContent = 'Ready - Write naturally.';
                    el.style.background = '#1d4ed8';
                }}
            }}, 4000);
        }}
    }}
    
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);
    document.getElementById('saveBtn').addEventListener('click', saveHandwriting);
    document.getElementById('recognizeBtn').addEventListener('click', recognize);
    document.getElementById('loadModelBtn').addEventListener('click', loadModel);
    document.getElementById('statusCheckBtn').addEventListener('click', checkStatus);
    
    document.getElementById('checkpointPath').addEventListener('keypress', (e) => {{
        if (e.key === 'Enter') loadModel();
    }});
    document.getElementById('deploymentDataPath').addEventListener('keypress', (e) => {{
        if (e.key === 'Enter') loadModel();
    }});
    
    document.getElementById('recognizeBtn').disabled = true;
    connectWebSocket();
    showStatus('Ready - Write naturally with your pen', 'success');
</script>
</body>
</html>'''


# ============================================================
# FASTAPI APPLICATION
# ============================================================

# Create FastAPI app
app = FastAPI(
    title="Ethiopic Handwriting Recognition API",
    description=f"API for recognizing Ethiopic characters (MAX_STROKES={MAX_STROKES})",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global inference_engine
    
    if os.path.exists(DEFAULT_CHECKPOINT):
        try:
            inference_engine = HandwritingInferenceEngine(
                DEFAULT_CHECKPOINT,
                DEFAULT_DEPLOYMENT if os.path.exists(DEFAULT_DEPLOYMENT) else None
            )
            print(f"[OK] Model loaded automatically on startup")
        except Exception as e:
            print(f"[ERROR] Failed to load model on startup: {e}")


@app.get("/")
async def root():
    return HTMLResponse(get_html_content(8766, DEFAULT_CHECKPOINT, DEFAULT_DEPLOYMENT))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "device": DEVICE,
        "max_strokes": MAX_STROKES
    }


@app.get("/info")
async def model_info():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "vocabulary_size": inference_engine.vocab_size,
        "max_strokes": inference_engine.max_strokes,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "memory_size": MEMORY_SIZE,
        "device": DEVICE,
        "sample_characters": list(inference_engine.idx2char.values())[:10]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        strokes = data.get("strokes", [])
        
        if not strokes:
            raise HTTPException(status_code=400, detail="No strokes found in JSON")
        
        result = inference_engine.recognize(strokes)
        
        return JSONResponse({
            "success": True,
            "predicted_text": result['text'],
            "original_text": result['original_text'],
            "confidence": result['confidence'],
            "was_corrected": result['was_corrected'],
            "corrections": result['corrections'],
            "inference_time_ms": result['inference_time_ms'],
            "timestamp": datetime.now().isoformat()
        })
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_json")
async def predict_json(request: Request):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = await request.json()
        strokes = data.get("strokes", [])
        
        if not strokes:
            raise HTTPException(status_code=400, detail="No strokes found in data")
        
        result = inference_engine.recognize(strokes)
        
        return JSONResponse({
            "success": True,
            "predicted_text": result['text'],
            "original_text": result['original_text'],
            "confidence": result['confidence'],
            "was_corrected": result['was_corrected'],
            "corrections": result['corrections'],
            "inference_time_ms": result['inference_time_ms'],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# WEBSOCKET SERVER
# ============================================================

websocket_engine = None


async def websocket_handler(websocket):
    """Handle WebSocket connections"""
    global websocket_engine
    print("[INFO] WebSocket client connected")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "load_model":
                checkpoint_path = data.get("checkpoint_path", DEFAULT_CHECKPOINT)
                deployment_data_path = data.get("deployment_data_path", DEFAULT_DEPLOYMENT)
                
                try:
                    if not os.path.exists(checkpoint_path):
                        await websocket.send(json.dumps({
                            "type": "model_loaded",
                            "success": False,
                            "error": f"Checkpoint not found: {checkpoint_path}"
                        }))
                        continue
                    
                    websocket_engine = HandwritingInferenceEngine(
                        checkpoint_path,
                        deployment_data_path if os.path.exists(deployment_data_path) else None
                    )
                    
                    await websocket.send(json.dumps({
                        "type": "model_loaded",
                        "success": True,
                        "vocab_size": websocket_engine.vocab_size,
                        "max_strokes": websocket_engine.max_strokes,
                        "deployment_loaded": bool(deployment_data_path and os.path.exists(deployment_data_path))
                    }))
                    print("[OK] Model loaded via WebSocket")
                    
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "model_loaded",
                        "success": False,
                        "error": str(e)
                    }))
            
            elif command == "recognize":
                if websocket_engine is None:
                    await websocket.send(json.dumps({
                        "type": "recognition_result",
                        "success": False,
                        "error": "Model not loaded. Please load a model first."
                    }))
                    continue
                
                strokes = data.get("strokes", [])
                print(f"[INFO] Recognizing {len(strokes)} strokes")
                
                try:
                    result = websocket_engine.recognize(strokes)
                    await websocket.send(json.dumps({
                        "type": "recognition_result",
                        "success": True,
                        "text": result['text'],
                        "original_text": result['original_text'],
                        "was_corrected": result['was_corrected'],
                        "corrections": result['corrections'],
                        "correction_count": result['correction_count'],
                        "confidence": round(result['confidence'], 2),
                        "time": result['inference_time_ms'],
                        "total_strokes": result['total_strokes'],
                        "total_points": result['total_points']
                    }))
                    
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "recognition_result",
                        "success": False,
                        "error": str(e)
                    }))
            
            elif command == "status":
                status = "loaded" if websocket_engine else "not_loaded"
                vocab = websocket_engine.vocab_size if websocket_engine else 0
                await websocket.send(json.dumps({
                    "type": "status",
                    "status": status,
                    "vocab_size": vocab
                }))
                
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")


async def start_websocket_server(port: int):
    """Start WebSocket server"""
    async with websockets.serve(websocket_handler, "localhost", port):
        print(f"[OK] WebSocket server running on ws://localhost:{port}")
        await asyncio.Future()  # Run forever


# ============================================================
# MAIN ENTRY POINT
# ============================================================

async def main():
    """Main entry point"""
    # Find available ports
    ws_port = find_available_port(8766)
    http_port = find_available_port(8081)
    
    print("\n" + "=" * 70)
    print("  Ethiopic Handwriting Recognition - Unified Server")
    print("  Enhanced Multi-Head Memory + Position Encoding (MAX_STROKES=60)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  WebSocket: ws://localhost:{ws_port}")
    print(f"  HTTP API: http://localhost:{http_port}")
    print(f"  Documentation: http://localhost:{http_port}/docs")
    print("=" * 70)
    print("\n  Default Paths:")
    print(f"  Checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"  Deployment Data: {DEFAULT_DEPLOYMENT}")
    print("=" * 70)
    print("\n  Features:")
    print("  1. REST API: /predict (file upload), /predict_json (JSON body)")
    print("  2. WebSocket: Real-time handwriting recognition")
    print("  3. Interactive UI: http://localhost:{http_port}")
    print("  4. Post-processing corrections with deployment data")
    print("=" * 70)
    print("\n  Press Ctrl+C to stop\n")
    
    # Check default paths
    if os.path.exists(DEFAULT_CHECKPOINT):
        print(f"[OK] Model checkpoint found: {DEFAULT_CHECKPOINT}")
    else:
        print(f"[WARNING] Model checkpoint not found: {DEFAULT_CHECKPOINT}")
    
    if os.path.exists(DEFAULT_DEPLOYMENT):
        print(f"[OK] Deployment data found: {DEFAULT_DEPLOYMENT}")
    else:
        print(f"[WARNING] Deployment data not found. Using default corrections.")
    print("")
    
    # Start WebSocket server
    ws_task = asyncio.create_task(start_websocket_server(ws_port))
    
    # Start HTTP server with uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=http_port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())