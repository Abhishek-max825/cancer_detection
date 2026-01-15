"""
Training script for cancer detection model with Ensemble Logic and Optimization
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import create_dataloaders
from model import get_model # Updated imports
from utils import save_checkpoint

# --- ENSEMBLE DEFINITIONS (Added for Local Training) ---
# NOTE: Ensure these classes are available in model.py or defined here. 
# For simplicity, we will assume model.py is updated or we define usage here.
# Since model.py was not modified in the previous steps for local file, 
# we will rely on get_model but extend it to support the ensemble loop.

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds


def evaluate_ensemble(models_dict, dataloader, device):
    """Evaluate Ensemble"""
    for model in models_dict.values():
        model.eval()
        
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Ensemble Validating"):
            if images is None: continue
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions from all models
            batch_preds = []
            for name, model in models_dict.items():
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs)
            
            # Average probabilities
            avg_probs = torch.mean(torch.stack(batch_preds), dim=0)
            
            preds = (avg_probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    parser = argparse.ArgumentParser(description='Train Cancer Detection Ensemble')
    # Use defaults for local training - Paths relative to CWD
    parser.add_argument('--data_dir', type=str, default='train', help='Training images directory')
    parser.add_argument('--csv_path', type=str, default='train_labels.csv', help='Labels CSV path')
    parser.add_argument('--save_dir', type=str, default='ml_pipeline/checkpoints', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Epochs per model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=60000, help='Max samples (Optimization)')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device Check with Explicit Warnings
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ GPU NOT DETECTED! Training will be VERY SLOW.")
        print("   To use your RTX 3050, run this command:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print(f"Using device: {device}")
    
    # Create dataloaders (Shared for all models)
    print(f"Loading data (Max Samples: {args.max_samples})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=args.csv_path,
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2, # Safer for Windows
        max_samples=args.max_samples
    )
    
    # Define Ensemble Models
    # We need to ensure get_model or manual definitions return these specific architectures
    # For this script we will instantiate them directly if get_model is generic, 
    # but let's assume get_model can handle types
    
    models_to_train = [
        ('DenseNet', 'densenet'),
        ('ResNet', 'resnet'),
        ('EfficientNet', 'efficientnet') 
    ]
    
    trained_models = {}
    
    for name, model_type in models_to_train:
        print(f"\n{'='*30}\nTraining {name}...\n{'='*30}")
        
        model = get_model(model_type, pretrained=True, freeze_backbone=True, device=device)
        
        # Calculate Class Weight (Neg/Pos ratio)
        # Based on your data: 0=35698, 1=24302 -> Ratio ~ 1.46
        pos_weight = torch.tensor([35698 / 24302]).to(device)
        
        # Use Weighted Loss to break the plateau
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        best_acc = 0.0
        best_model_path = os.path.join(args.save_dir, f'best_model_{name.lower()}.pth')
        
        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best {name} model (Acc: {best_acc:.4f})")
                
            # Fine-tuning logic
            if epoch == 5:
                print("Unfreezing backbone for fine-tuning...")
                if hasattr(model, 'unfreeze_backbone'):
                     model.unfreeze_backbone()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * 0.1
                    
        # Load best weights for ensemble
        if os.path.exists(best_model_path):
             model.load_state_dict(torch.load(best_model_path))
        trained_models[name] = model

    print(f"\n{'='*30}\nEvaluating ENSEMBLE...\n{'='*30}")
    ensemble_acc = evaluate_ensemble(trained_models, val_loader, device)
    print(f"\n>>> FINAL ENSEMBLE ACCURACY: {ensemble_acc:.4f} <<<")


if __name__ == "__main__":
    main()
