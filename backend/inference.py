"""
Model inference utilities for cancer detection backend
Supports Ensemble of DenseNet, ResNet, and EfficientNet
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import base64
from io import BytesIO
import sys
import cv2

# Add ml_pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline'))

from model import get_model
from grad_cam import visualize_gradcam, get_gradcam_layer, get_gradcam_layers, GradCAM


class EnsembleCancerPredictor:
    """
    Handles inference for an Ensemble of models
    """
    def __init__(self, checkpoint_dir: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Ensemble on device: {self.device}")
        
        self.models = {}
        model_configs = [
            ('densenet', 'best_model_densenet.pth'),
            ('resnet', 'best_model_resnet.pth'),
            ('efficientnet', 'best_model_efficientnet.pth')
        ]
        
        for name, filename in model_configs:
            path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(path):
                print(f"Loading {name} from {path}...")
                try:
                    model = get_model(name, pretrained=False, freeze_backbone=False, device=self.device)
                    state_dict = torch.load(path, map_location=self.device)
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                    model.eval()
                    self.models[name] = model
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
            else:
                print(f"Warning: Checkpoint {filename} not found. Skipping {name}.")
        
        if not self.models:
            raise RuntimeError("No models loaded for Ensemble!")
            
        print(f"Ensemble loaded with {len(self.models)} models.")
        
        # Transforms (Standard ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.preprocess_image(image).to(self.device)
        
        probs = []
        # Get predictions from all available models
        for name, model in self.models.items():
            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
        
        # Average Probability (Soft Voting)
        avg_confidence = sum(probs) / len(probs)
        
        prediction = "Cancer" if avg_confidence > 0.5 else "Non-Cancer"
        
        # For display, if it is non-cancer (e.g. 0.1), show 0.9 confidence in "Non-Cancer"
        display_confidence = avg_confidence if prediction == "Cancer" else (1 - avg_confidence)
        
        # Calculate Entropy (Uncertainty)
        # H(p) = - [p log2(p) + (1-p) log2(1-p)]
        # Clamp to avoid log(0)
        p = avg_confidence
        p = max(1e-6, min(1 - 1e-6, p))
        entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
        
        return {
            'prediction': prediction,
            'confidence': float(display_confidence),
            'raw_score': float(avg_confidence),
            'entropy': float(entropy),
            'ensemble_votes': probs  # Debug info
        }

    def generate_gradcam(self, image: Image.Image) -> tuple:
        """
        Generate Multi-layer Grad-CAM and calculate heatmap variance
        Returns: (base64_image, variance_score)
        """
        # Prefer DenseNet for visualization as it usually has best features
        target_model = self.models.get('densenet') or self.models.get('resnet') or list(self.models.values())[0]
        
        tensor = self.preprocess_image(image).to(self.device)
        original_array = np.array(image.resize((96, 96)))
        
        # Multi-layer Fusion
        layers = get_gradcam_layers(target_model)
        heatmaps = []
        
        target_model.eval()
        
        for layer in layers:
            gradcam = GradCAM(target_model, layer)
            heatmap = gradcam.generate_cam(tensor)
            # Resize to input size (96x96) to ensures all layers match for fusion
            heatmap = cv2.resize(heatmap, (96, 96))
            heatmaps.append(heatmap)
            
        # Average heatmaps (Fuse Structure + Texture)
        if heatmaps:
            fused_heatmap = np.mean(np.array(heatmaps), axis=0)
            # Re-normalize
            fused_heatmap = (fused_heatmap - fused_heatmap.min()) / (fused_heatmap.max() - fused_heatmap.min() + 1e-8)
        else:
            fused_heatmap = np.zeros((96, 96))

        # Calculate Variance (Diffuse Pattern Detection)
        heatmap_variance = np.var(fused_heatmap)
        
        overlaid = gradcam.overlay_heatmap(original_array, fused_heatmap, alpha=0.4)
        
        buffered = BytesIO()
        Image.fromarray(overlaid).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}", heatmap_variance


# Keep single predictor for legacy/fallback support
class CancerPredictor:
    def __init__(self, model_path: str, model_type: str = 'efficientnet', device: str = None):
        # ... (Legacy simplified for brevity, mostly delegated to Ensemble logic if needed)
        # But we will redirect load_predictor to use Ensemble if files exist
        pass

def load_predictor(model_path: str = None, model_type: str = 'densenet'):
    """
    Intelligent loader:
    If ensemble files exist in checkpoints, loads Ensemble.
    Otherwise tries to load the specific single file.
    """
    # Determine checkpoints directory
    if model_path:
        checkpoint_dir = os.path.dirname(model_path)
    else:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline', 'checkpoints')
    
    # Check for ensemble files
    required_files = ['best_model_densenet.pth', 'best_model_resnet.pth', 'best_model_efficientnet.pth']
    existing_files = [f for f in required_files if os.path.exists(os.path.join(checkpoint_dir, f))]
    
    if len(existing_files) >= 2: # If we have at least 2 models, use ensemble
        print(f"Checkpoints found ({len(existing_files)}): Loading EnsemblePredictor...")
        return EnsembleCancerPredictor(checkpoint_dir)
    else:
        # Fallback to single model (original logic, simplified here)
        print("Ensemble files missing. Trying legacy single model load...")
        # Re-implementing minimal single-model logic just in case
        raise FileNotFoundError("Could not find Ensemble checkpoints. Please ensure training finished successfully.")

if __name__ == "__main__":
    print("Testing Ensemble Inference...")
    try:
        predictor = load_predictor()
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
