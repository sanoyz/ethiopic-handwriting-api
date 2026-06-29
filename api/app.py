"""
FastAPI application for Ethiopic handwriting recognition
ALIGNED WITH MAX_STROKES=60 MODEL
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.config import (
    MAX_STROKES, FEATURE_DIM, D_MODEL, N_LAYERS, 
    MEMORY_SIZE, MEMORY_HEADS, DEVICE, DEFAULT_CHECKPOINT,
    DEFAULT_DEPLOYMENT_DATA
)
from api.model_loader import load_model, decode_tokens_to_text
from api.preprocessing import process_handwriting_data
from api.corrector import TextCorrector

# Setup logging with UTF-8 encoding
class UTF8Formatter(logging.Formatter):
    """Custom formatter that handles UTF-8 characters"""
    def format(self, record):
        # Replace emoji with text equivalents if encoding fails
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # Fallback: replace emoji with text
            msg = record.msg
            replacements = {
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⚠️': '[WARNING]',
                '🚀': '[START]',
                '📁': '[FOLDER]',
                '📊': '[STATS]',
                '🎯': '[TARGET]',
                '📚': '[DOCS]',
                '🔧': '[TOOL]',
                '💡': '[TIP]',
                '📋': '[LIST]',
                '📄': '[FILE]',
                '📈': '[CHART]',
                '📉': '[DOWN]',
                '🔍': '[SEARCH]',
                '🎉': '[DONE]',
            }
            for emoji, text in replacements.items():
                msg = msg.replace(emoji, text)
            record.msg = msg
            return super().format(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Update formatter for console handler
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.setFormatter(UTF8Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Ethiopic Handwriting Recognition API",
    description=f"API for recognizing Ethiopic characters from handwriting data (MAX_STROKES={MAX_STROKES})",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instance
model_loader = None
corrector = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_loader, corrector
    
    model_path = os.environ.get("MODEL_PATH", DEFAULT_CHECKPOINT)
    deployment_path = os.environ.get("DEPLOYMENT_PATH", DEFAULT_DEPLOYMENT_DATA)
    
    if not os.path.exists(model_path):
        logger.error(f"[ERROR] Model file not found: {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")
    
    try:
        # Load model with deployment data
        model_loader = load_model(model_path, DEVICE, deployment_path)
        logger.info(f"[OK] Model loaded successfully on {DEVICE}")
        logger.info(f"   Vocabulary size: {len(model_loader.idx2char)}")
        logger.info(f"   Max Strokes: {model_loader.max_strokes}")
        logger.info(f"   D_MODEL: {D_MODEL}")
        logger.info(f"   N_LAYERS: {N_LAYERS}")
        logger.info(f"   MEMORY_SIZE: {MEMORY_SIZE}")
        
        # Initialize corrector
        if model_loader.deployment_data:
            corrector = TextCorrector(model_loader.deployment_data)
            logger.info(f"[OK] Corrector initialized with deployment data")
        else:
            corrector = TextCorrector()
            logger.info(f"[OK] Corrector initialized with default rules")
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with simple UI"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Ethiopic Handwriting Recognition API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .container {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
            .status {{ padding: 10px; background: #e8f5e9; border-radius: 4px; }}
            .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 4px; }}
            .method {{ color: #1565c0; font-weight: bold; }}
            .badge {{ background: #4caf50; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>✍️ Ethiopic Handwriting Recognition API <span class="badge">MAX_STROKES={MAX_STROKES}</span></h1>
        <div class="container">
            <div class="status">✅ API is running</div>
            <p><strong>Model:</strong> Enhanced Multi-Head Memory + Position Encoding</p>
            <p><strong>Max Strokes:</strong> {MAX_STROKES}</p>
            <p><strong>D_MODEL:</strong> {D_MODEL}</p>
            <p><strong>N_LAYERS:</strong> {N_LAYERS}</p>
            <h2>Endpoints:</h2>
            <div class="endpoint">
                <span class="method">POST</span> /predict - Upload handwriting JSON for recognition
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /health - Check API health
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /info - Get model information
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader is not None,
        "device": DEVICE,
        "max_strokes": MAX_STROKES
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_loader.get_model_info()
    return {
        **info,
        "sample_characters": list(model_loader.idx2char.values())[:10],
        "feature_dim": FEATURE_DIM,
        "dropout": DROPOUT,
        "memory_heads": MEMORY_HEADS,
        "use_cross_attention": USE_CROSS_ATTENTION,
        "use_position_encoding": USE_POSITION_ENCODING
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict text from handwriting JSON file
    
    Expected JSON format:
    {
        "strokes": [
            {
                "points": [
                    {"x": 0, "y": 0, "timestamp": 0, "pressure": 0.5, "tilt_x": 0, "tilt_y": 0},
                    ...
                ]
            }
        ]
    }
    """
    
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and parse JSON
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Get filename for logging
        filename = file.filename or "unknown"
        logger.info(f"Processing file: {filename}")
        
        # Process handwriting data
        features = process_handwriting_data(
            data,
            model_loader.global_mean,
            model_loader.global_std,
            model_loader.char2idx,
            model_loader.idx2char
        )
        
        # Run prediction
        result = model_loader.predict(features)
        text = result['text']
        confidence = result['confidence']
        
        # Apply corrections
        if corrector:
            correction_result = corrector.correct_text(text, confidence / 100.0)
            final_text = correction_result['corrected']
            was_corrected = correction_result['was_corrected']
            corrections = correction_result['corrections']
        else:
            final_text = text
            was_corrected = False
            corrections = []
        
        logger.info(f"Prediction: {final_text} (confidence: {confidence:.1f}%)")
        
        return JSONResponse({
            "success": True,
            "predicted_text": final_text,
            "original_text": text,
            "confidence": confidence,
            "was_corrected": was_corrected,
            "corrections": corrections,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "max_strokes": MAX_STROKES,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS
            }
        })
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict_json")
async def predict_json(request: Request):
    """Predict text from JSON body"""
    
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = await request.json()
        
        # Process handwriting data
        features = process_handwriting_data(
            data,
            model_loader.global_mean,
            model_loader.global_std,
            model_loader.char2idx,
            model_loader.idx2char
        )
        
        # Run prediction
        result = model_loader.predict(features)
        text = result['text']
        confidence = result['confidence']
        
        # Apply corrections
        if corrector:
            correction_result = corrector.correct_text(text, confidence / 100.0)
            final_text = correction_result['corrected']
            was_corrected = correction_result['was_corrected']
            corrections = correction_result['corrections']
        else:
            final_text = text
            was_corrected = False
            corrections = []
        
        return JSONResponse({
            "success": True,
            "predicted_text": final_text,
            "original_text": text,
            "confidence": confidence,
            "was_corrected": was_corrected,
            "corrections": corrections,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True
    )