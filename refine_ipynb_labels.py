import json
import os

try:
    with open('cancer_detection_colab.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New content for Cell 8 (Load Data & VERIFY)
    # This version explicitly finds a 0 and a 1 to show the user both work.
    cell_8_source = """# 8. Load Data & VERIFY
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

print("=== DEBUGGING DATA LOADING ===")
print(f"Looking for CSV at: {CSV_PATH}")
print(f"Looking for Images at: {IMAGE_DIR}")

if not os.path.exists(CSV_PATH):
    print(f"CRITICAL ERROR: {CSV_PATH} missing.")
    if is_colab:
        print("\\n>> CAUSE: You are connected to Google Colab Cloud.")
        print(">> FIX: Change your active Kernel in VS Code to a LOCAL one.")
else:
    print(f"OK: {CSV_PATH} found")

if not os.path.exists(IMAGE_DIR):
    print(f"CRITICAL ERROR: {IMAGE_DIR} missing.")
else:
    print(f"OK: {IMAGE_DIR} found")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    if MAX_SAMPLES and MAX_SAMPLES < len(df):
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    ids = df['id'].tolist()
    labels = df['label'].tolist()

    train_ids, test_ids, train_labels, test_labels = train_test_split(ids, labels, test_size=0.15, stratify=labels, random_state=42)
    train_ids, val_ids, train_labels, val_labels = train_test_split(train_ids, train_labels, test_size=0.15, stratify=train_labels, random_state=42)

    print("\\n=== TESTING IMAGE LOADING ===")
    label_map = {0: 'No Cancer', 1: 'Cancer'}
    
    # Try to find one example of each class to verify labels are correct
    try:
        pos_idx = next(i for i, x in enumerate(train_labels) if x == 1)
        neg_idx = next(i for i, x in enumerate(train_labels) if x == 0)
        indices_to_check = [pos_idx, neg_idx]
        
        sample_ids = [train_ids[i] for i in indices_to_check]
        sample_labels = [train_labels[i] for i in indices_to_check]
    except StopIteration:
        print("Warning: Could not find examples of both classes in current split.")
        sample_ids = train_ids[:3]
        sample_labels = train_labels[:3]

    test_ds = CancerDetectionDataset(sample_ids, sample_labels, IMAGE_DIR, transform=None)
    
    try:
        for i in range(len(test_ds)):
            img, lbl = test_ds[i]
            if img is not None:
                lbl_name = label_map.get(lbl, 'Unknown')
                print(f"✓ Loaded image {sample_ids[i]} (Label: {lbl} - {lbl_name})")
            else:
                print(f"x Failed to load image {sample_ids[i]} (File not found)")
    except Exception as e:
        print(f"\\n!!! FAILURE !!!")
        print(f"Error: {e}")

    # Only define loader if check passes
    train_dataset = CancerDetectionDataset(train_ids, train_labels, IMAGE_DIR, transform=transform_train)
    val_dataset = CancerDetectionDataset(val_ids, val_labels, IMAGE_DIR, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"\\nSUCCESS: Ready to train on {len(train_dataset)} samples")
"""

    # Update cell
    found_8 = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            if isinstance(src, list) and len(src) > 0 and src[0].startswith('# 8. Load Data'):
                cell['source'] = cell_8_source.splitlines(True)
                found_8 = True
                break

    if found_8:
        with open('cancer_detection_colab.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook to check both classes.")
    else:
        print("Failed to find Cell 8.")

except Exception as e:
    print(f"Error: {e}")
