"""
FastAPI Backend for Cancer Detection
Provides /predict endpoint for image classification with Grad-CAM
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import os
import base64
from typing import Optional
from pydantic import BaseModel

from inference import load_predictor, CancerPredictor


# Initialize FastAPI app
app = FastAPI(
    title="Cancer Detection API",
    description="AI-powered histopathologic cancer detection with explainable AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded on startup)
predictor: Optional[CancerPredictor] = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    confidence: float
    heatmap_base64: str
    original_base64: Optional[str] = None
    message: str = "Prediction generated successfully"
    entropy: Optional[float] = None
    pattern_type: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    message: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    try:
        print("Loading cancer detection model...")
        model_path = os.environ.get('MODEL_PATH', 
                                     os.path.join('..', 'ml_pipeline', 'checkpoints', 'best_model.pth'))
        model_type = os.environ.get('MODEL_TYPE', 'densenet')
        
        predictor = load_predictor(model_path, model_type)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("API will start but predictions will fail until model is available")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "model_loaded": predictor is not None,
        "message": "Cancer Detection API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None,
        "message": "API is operational" if predictor is not None else "Model not loaded yet"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict cancer from uploaded histopathologic image
    
    Args:
        file: Uploaded image file (PNG, JPG, TIF)
        
    Returns:
        Prediction with confidence score and Grad-CAM heatmap
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate file type
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )
    
    try:
        # Read image
        contents = await file.read()
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Open image
        image = Image.open(BytesIO(contents))
        
        # Validate image
        if image.mode not in ['RGB', 'L', 'RGBA']:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Image must be RGB, grayscale, or RGBA."
            )
        
        # Make prediction
        result = predictor.predict(image)
        
        # Generate Grad-CAM and Variance
        heatmap_base64, variance = predictor.generate_gradcam(image)
        
        # Determine Pattern Type based on Variance
        # High variance -> Focal/Specific area
        # Low variance -> Diffuse/Spread out
        if result['prediction'] == 'Cancer':
            pattern_type = "Diffuse/Metastatic" if variance < 0.05 else "Focal/Localized"
        else:
            pattern_type = "Benign/Normal"

        # Convert original image to base64 for display (handles TIFFs)
        buffered_original = BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = f"data:image/png;base64,{base64.b64encode(buffered_original.getvalue()).decode()}"
        
        # Build response
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "heatmap_base64": heatmap_base64,
            "original_base64": original_base64,
            "message": f"Analysis complete. Detected: {result['prediction']}",
            "entropy": result.get('entropy'),
            "pattern_type": pattern_type
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(f"Traceback:\n{error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "name": "Cancer Detection API",
        "version": "1.0.0",
        "description": "Educational AI system for histopathologic cancer detection",
        "disclaimer": "FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL DIAGNOSIS.",
        "supported_formats": ["PNG", "JPG", "JPEG", "TIF", "TIFF"],
        "max_file_size": "10MB",
        "model_type": "EfficientNet-B0 with transfer learning",
        "explainability": "Grad-CAM heatmaps for visual interpretation"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Cancer Detection API...")
    print("Educational Use Only - Not for Clinical Diagnosis")
    print("-" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
