import json
import os

try:
    with open('cancer_detection_colab.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 5: Define Multiple Models
    cell_5_source = """# 5. Model Architecture (ENSEMBLE: DenseNet + ResNet + EfficientNet)
class DenseNetClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(DenseNetClassifier, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        self.backbone = models.densenet121(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
                
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True

class ResNetClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(ResNetClassifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class EfficientNetClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(EfficientNetClassifier, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
                
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True
"""

    # 2. Update Cell 9: Train ALL Models
    cell_9_source = """# 9. Run Training (ENSEMBLE TRAINING LOOP)
models_to_train = [
    ('DenseNet', DenseNetClassifier(freeze_backbone=True)),
    ('ResNet', ResNetClassifier(freeze_backbone=True)),
    ('EfficientNet', EfficientNetClassifier(freeze_backbone=True))
]

trained_models = {}

for name, model in models_to_train:
    print(f"\\n{'='*20}\\nTraining {name}...\\n{'='*20}")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_acc = 0.0
    best_model_path = f'best_model_{name.lower()}.pth'
    
    # Train Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best {name} model (Acc: {best_acc:.4f})")
            
        # Fine-tuning (simplified: unfreeze halfway)
        if epoch == 5:
            print("Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

    # Reload best weights
    model.load_state_dict(torch.load(best_model_path))
    trained_models[name] = model

print("\\nALL MODELS TRAINED!")
"""

    # 3. Add Cell 10 for Ensemble Evaluation
    cell_10_source = """# 10. Ensemble Evaluation
print("Evaluating Ensemble Performance...")

def evaluate_ensemble(models_dict, dataloader, device):
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

ensemble_acc = evaluate_ensemble(trained_models, val_loader, device)
print(f"\\n>>> ENSEMBLE ACCURACY: {ensemble_acc:.4f} <<<")
"""
    # Create the new cell object
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_10_source.splitlines(True)
    }

    # Update Logic
    found_5 = False
    found_9 = False
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell['source']
            if isinstance(src, list) and len(src) > 0:
                # Cleaner check
                src_str = "".join(src)
                if '# 5. Model Architecture' in src_str or 'class DenseNetClassifier' in src_str:
                    cell['source'] = cell_5_source.splitlines(True)
                    found_5 = True
                elif '# 9. Run Training' in src_str:
                    cell['source'] = cell_9_source.splitlines(True)
                    found_9 = True

    if found_5 and found_9:
        # Check if we already added the ensemble cell (by checking last cell)
        last_cell_src = nb['cells'][-1]['source']
        is_ensemble_cell = False
        if isinstance(last_cell_src, list) and len(last_cell_src) > 0:
            if 'Ensemble Evaluation' in "".join(last_cell_src):
                 is_ensemble_cell = True
        
        if not is_ensemble_cell:
             nb['cells'].append(new_cell)
        
        with open('cancer_detection_colab.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook with Ensemble Logic.")
    else:
        print(f"Failed to find cells. Found 5 (Arch): {found_5}, Found 9 (Train): {found_9}")

except Exception as e:
    print(f"Error: {e}")
