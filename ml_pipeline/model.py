"""
CNN Model for Cancer Detection
Using Transfer Learning with EfficientNet-B0
"""
import torch
import torch.nn as nn
from torchvision import models


class CancerClassifier(nn.Module):
    """
    Binary classifier for cancer detection using transfer learning
    Based on EfficientNet-B0 pretrained on ImageNet
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze early layers for transfer learning
        """
        super(CancerClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),  # Single output for binary classification
            nn.Sigmoid()  # Sigmoid for binary probability
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 96, 96)
        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True


class ResNetClassifier(nn.Module):
    """
    Alternative model using ResNet18 for cancer detection
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze early layers
        """
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:  # Don't freeze final fully connected layer
                    param.requires_grad = False
        
        # Get number of features
        num_features = self.backbone.fc.in_features
        
        # Replace final layer for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class DenseNetClassifier(nn.Module):
    """
    DenseNet121 model for high-accuracy cancer detection
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(DenseNetClassifier, self).__init__()
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
                
        # Get number of features
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def get_model(model_type: str = 'efficientnet', 
              pretrained: bool = True,
              freeze_backbone: bool = True,
              device: str = 'cpu') -> nn.Module:
    """
    Get model instance
    
    Args:
        model_type: 'efficientnet', 'resnet', or 'densenet'
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone
        device: Device to load model on
        
    Returns:
        Model instance
    """
    if model_type.lower() == 'efficientnet':
        weights = 'DEFAULT' if pretrained else None
        model = CancerClassifier(pretrained=weights, freeze_backbone=freeze_backbone)
    elif model_type.lower() == 'resnet':
        weights = 'DEFAULT' if pretrained else None
        model = ResNetClassifier(pretrained=weights, freeze_backbone=freeze_backbone)
    elif model_type.lower() == 'densenet':
        weights = 'DEFAULT' if pretrained else None
        model = DenseNetClassifier(pretrained=weights, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test EfficientNet
    model = get_model('efficientnet', device=device)
    print(f"\nEfficientNet model created")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 96, 96).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\nModel test successful!")
