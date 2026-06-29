"""
Model loading and inference utilities
ALIGNED WITH MAX_STROKES=60 TRAINING
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from api.config import (
    MAX_STROKES, FEATURE_DIM, D_MODEL, N_HEADS, N_LAYERS, 
    D_FF, DROPOUT, MEMORY_SIZE, MEMORY_HEADS, PROTOTYPE_DROPOUT,
    USE_CROSS_ATTENTION, USE_POSITION_ENCODING, DEVICE
)


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK MODELS - EXACTLY AS IN TRAINING (MAX_STROKES=60)
# ══════════════════════════════════════════════════════════════════════════════

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

        # Always use full memory during inference
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


# ══════════════════════════════════════════════════════════════════════════════
# DECODING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def greedy_decode(log_probs, blank=0):
    """Greedy decoding with blank removal"""
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


def decode_tokens_to_text(tokens, idx2char):
    """Convert token sequence to text"""
    text_chars = []
    for tok in tokens:
        if tok == 0:
            continue
        char = idx2char.get(tok)
        if char is None:
            text_chars.append("?")
        else:
            text_chars.append(char)
    return "".join(text_chars)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

class ModelLoader:
    """Handles model loading and inference with MAX_STROKES=60"""

    def __init__(self, model_path: str, device: str = DEVICE):
        self.device = device
        self.model = None
        self.checkpoint = None
        self.idx2char = {}
        self.char2idx = {}
        self.global_mean = None
        self.global_std = None
        self.max_strokes = MAX_STROKES
        self.deployment_data = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"📥 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.checkpoint = checkpoint

        # Get vocab and config
        self.idx2char = checkpoint["idx2char"]
        self.char2idx = checkpoint["char2idx"]
        self.max_strokes = checkpoint.get("max_strokes", MAX_STROKES)
        vocab_size = len(self.idx2char)

        print(f"  ✓ Vocabulary size: {vocab_size}")
        print(f"  ✓ Max strokes: {self.max_strokes}")

        # Create model with correct config
        self.model = EthiopicRecognizerWithEnhancedMemory(
            vocab_size=vocab_size,
            feature_dim=FEATURE_DIM,
            max_strokes=self.max_strokes
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Get normalization stats
        self.global_mean = checkpoint.get("global_mean")
        self.global_std = checkpoint.get("global_std")

        if self.global_mean is None:
            self.global_mean = np.zeros(FEATURE_DIM)
            self.global_std = np.ones(FEATURE_DIM)

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Device: {self.device}")

    def load_deployment_data(self, deployment_path: str):
        """Load deployment data for error correction"""
        if not os.path.exists(deployment_path):
            print(f"⚠️  Deployment data not found: {deployment_path}")
            return

        self.deployment_data = {}
        try:
            # Load top confusions
            top_confusions_path = os.path.join(deployment_path, "top_confusions.json")
            if os.path.exists(top_confusions_path):
                with open(top_confusions_path, 'r', encoding='utf-8') as f:
                    self.deployment_data['top_confusions'] = json.load(f)
                print(f"  ✓ Loaded top confusions: {len(self.deployment_data['top_confusions'])}")

            # Load character confidence
            char_confidence_path = os.path.join(deployment_path, "character_confidence.json")
            if os.path.exists(char_confidence_path):
                with open(char_confidence_path, 'r', encoding='utf-8') as f:
                    self.deployment_data['char_confidence'] = json.load(f)
                print(f"  ✓ Loaded character confidence")

            # Load correction data
            correction_data_path = os.path.join(deployment_path, "correction_data.json")
            if os.path.exists(correction_data_path):
                with open(correction_data_path, 'r', encoding='utf-8') as f:
                    self.deployment_data['correction_data'] = json.load(f)
                print(f"  ✓ Loaded correction data")

            print(f"✅ Deployment data loaded from: {deployment_path}")
        except Exception as e:
            print(f"⚠️  Error loading deployment data: {e}")

    def predict(self, features: np.ndarray) -> Dict[str, any]:
        """Predict text from handwriting features"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            log_probs = self.model(features_tensor, training=False)

        # Decode
        predictions = greedy_decode(log_probs)
        text = decode_tokens_to_text(predictions[0], self.idx2char)

        # Calculate confidence
        probs = torch.exp(log_probs)
        top_probs, _ = probs.max(dim=-1)
        confidence = float(top_probs.mean().cpu().numpy() * 100)
        confidence = min(100.0, max(0.0, confidence))

        return {
            'text': text,
            'confidence': confidence,
            'tokens': predictions[0]
        }

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model": "EthiopicRecognizerWithEnhancedMemory",
            "vocabulary_size": len(self.idx2char),
            "max_strokes": self.max_strokes,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "memory_size": MEMORY_SIZE,
            "memory_heads": MEMORY_HEADS,
            "device": self.device,
            "has_deployment_data": self.deployment_data is not None
        }


def load_model(model_path: str, device: str = DEVICE, deployment_path: str = None) -> ModelLoader:
    """Convenience function to load model with optional deployment data"""
    loader = ModelLoader(model_path, device)
    if deployment_path:
        loader.load_deployment_data(deployment_path)
    return loader
