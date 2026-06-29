"""
Configuration - MUST MATCH TRAINING CONFIGURATION (MAX_STROKES=60)
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - MUST MATCH TRAINING CONFIGURATION (MAX_STROKES=60)
# ══════════════════════════════════════════════════════════════════════════════
RESAMPLE_N = 32
DIRECTION_BINS = 8
MIN_STROKE_POINTS = 2

# UPDATED FOR MAX_STROKES=60 TRAINING
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

# API Configuration
DEFAULT_CHECKPOINT = "models/API_ready_model/best_model.pt"
DEFAULT_DEPLOYMENT_DATA = "deployment_data"
HOST = "0.0.0.0"
PORT = 8000
MAX_UPLOAD_SIZE = 10485760  # 10MB

# Device configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Configuration loaded:")
print(f"   MAX_STROKES: {MAX_STROKES}")
print(f"   D_MODEL: {D_MODEL}")
print(f"   N_LAYERS: {N_LAYERS}")
print(f"   MEMORY_SIZE: {MEMORY_SIZE}")
print(f"   DEVICE: {DEVICE}")
