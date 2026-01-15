"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
For explainability in cancer detection model
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activation"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradient"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index (default: predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        # For binary classification with sigmoid output
        if target_class is None:
            # Use the predicted class
            target_class = (output > 0.5).long().item()
        
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only keep positive influences)
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(self, 
                       original_image: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.7,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            original_image: Original RGB image (H, W, 3)
            heatmap: Grad-CAM heatmap (h, w)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Overlaid image as numpy array
        """
        # DEBUG: Confirm usage of new logic in server logs
        print("DEBUG: Applying Smart Overlay (High Visibility)", flush=True)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB using colormap (Jet: Blue=Cold, Red=Hot)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        # Use Original Image as Background (Preserve H&E Visualization)
        background_colored = original_image
        
        # SMART OVERLAY: Only show "Hot" areas, keep "Cold" areas transparent
        
        # 1. Lower Threshold: Show heatmap where value > 0.15 (15% activation)
        # This catches weaker signals that were previously hidden
        
        pixel_alpha = np.zeros_like(heatmap_resized)
        
        # Soft thresholding for smooth edges
        # Map [0.15, 1.0] -> [0.0, 0.9]
        # Max Alpha 0.9 makes it almost solid at peak activation
        threshold = 0.15
        max_alpha = 0.9
        
        mask = heatmap_resized > threshold
        pixel_alpha[mask] = ((heatmap_resized[mask] - threshold) / (1 - threshold)) * max_alpha
        
        # Add singleton dimension for broadcasting (H, W, 1)
        pixel_alpha = pixel_alpha[..., np.newaxis]
        
        # Blend: background * (1 - alpha) + heatmap * alpha
        overlaid = (background_colored * (1 - pixel_alpha) + heatmap_colored * pixel_alpha).astype(np.uint8)
        
        return overlaid


def get_gradcam_layer(model):
    """
    Get the appropriate layer for Grad-CAM based on model architecture
    
    Args:
        model: PyTorch model
        
    Returns:
        Target layer for Grad-CAM
    """
    # Check model type
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # EfficientNet
        if hasattr(backbone, 'features'):
            # Last convolutional layer in EfficientNet
            return backbone.features[-1]
        
        # ResNet
        elif hasattr(backbone, 'layer4'):
            return backbone.layer4[-1]

        # DenseNet
        elif model.__class__.__name__ == 'DenseNetClassifier':
             # Target the last layer of the features (usually norm5)
             return backbone.features[-1]
    
    raise ValueError("Could not determine Grad-CAM layer for this model")


def visualize_gradcam(model,
                     image_tensor: torch.Tensor,
                     original_image: np.ndarray,
                     device: str = 'cpu',
                     save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and visualize Grad-CAM
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        original_image: Original image as numpy array (H, W, 3)
        device: Device to run on
        save_path: Optional path to save visualization
        
    Returns:
        Tuple of (heatmap, overlaid_image)
    """
    # Get target layer
    target_layer = get_gradcam_layer(model)
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Generate heatmap
    heatmap = gradcam.generate_cam(image_tensor)
    
    # Overlay on original image
    overlaid = gradcam.overlay_heatmap(original_image, heatmap, alpha=0.7)
    
    # Optionally save
    if save_path:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title('Overlaid')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {save_path}")
    
    return heatmap, overlaid


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully")
    print("Use visualize_gradcam() function to generate heatmaps")
