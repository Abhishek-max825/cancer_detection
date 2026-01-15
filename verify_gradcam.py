
import sys
import os
import cv2
import numpy as np
import torch

# Add ml_pipeline to path
sys.path.append(os.path.join(os.getcwd(), 'ml_pipeline'))

try:
    from grad_cam import GradCAM
    print("SUCCESS: Imported GradCAM")
except ImportError as e:
    print(f"ERROR: Could not import GradCAM: {e}")
    sys.exit(1)

# Create dummy inputs
# 100x100 Red image
original_image = np.zeros((100, 100, 3), dtype=np.uint8)
original_image[:, :, 1] = 255 # Green channel

# Mock GradCAM (we only need overlay_heatmap)
class MockGradCAM(GradCAM):
    def __init__(self):
        pass

gradcam = MockGradCAM()

# Create dummy heatmap
# Left half = 0.0 (Cold), Right half = 1.0 (Hot)
heatmap = np.zeros((100, 100), dtype=np.float32)
heatmap[:, 50:] = 1.0 

try:
    result = gradcam.overlay_heatmap(original_image, heatmap, alpha=0.7)
    
    # 1. Check Cold Area (x=25) -> Should be Transparent
    # Input was Green (0,255,0). Output should be Green.
    cold_pixel = result[50, 25]
    print(f"Cold Pixel (Input Green): {cold_pixel}")
    
    # Check for Green dominance
    if cold_pixel[1] > 250 and cold_pixel[0] < 10 and cold_pixel[2] < 10:
        print("CHECK 1 PASSED: Cold area is Transparent (Shows Original Green).")
    else:
        print("CHECK 1 FAILED: Cold area is NOT Transparent.")
        
    # 2. Check Hot Area (x=75) -> Should be Heatmap Color
    hot_pixel = result[50, 75]
    print(f"Hot Pixel (Heatmap Red): {hot_pixel}")
    
    if hot_pixel[0] > 50 and hot_pixel[2] < 50:
        print("CHECK 2 PASSED: Hot area shows Heatmap (Red).")
    else:
        print("CHECK 2 FAILED: Hot area is missing Heatmap.")

    # 3. Check Mid-Range Area (x=60, approx 0.4 activation)
    # With threshold 0.15, 0.4 should be visibly colored.
    # Heatmap at x=60 is transitioned... let's force a pixel to 0.4
    
    # Re-run overlay with constant 0.4 heatmap for test
    heatmap_mid = np.full((100, 100), 0.4, dtype=np.float32)
    result_mid = gradcam.overlay_heatmap(original_image, heatmap_mid, alpha=0.9)
    mid_pixel = result_mid[50, 50]
    print(f"Mid Pixel (0.4 Activation): {mid_pixel}")

    # Input Green. Heatmap (0.4) ~ Cyan/Greenish/Yellow.
    # Should NOT be pure Green (0,255,0).
    if mid_pixel[0] > 10 or mid_pixel[2] > 10:
         print("CHECK 3 PASSED: Mid-range activation is Visible (Not just Green).")
    else:
         print("CHECK 3 FAILED: Mid-range activation is invisible (Pure Green).")
    
except Exception as e:
    print(f"ERROR: {e}")
