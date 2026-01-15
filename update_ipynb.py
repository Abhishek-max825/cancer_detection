import json
import os

try:
    with open('cancer_detection_colab.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New content for Cell 2 (Google Drive handling)
    cell_2_source = """# 2. Data Setup (Google Drive)
import os
import sys
import zipfile
import shutil

# DETECT ENVIRONMENT
is_colab = 'google.colab' in sys.modules or os.path.exists('/content/sample_data')

if is_colab:
    print("⚠️ Running in Google Colab.")
    from google.colab import drive
    
    # Mount Drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # DEFINE YOUR DRIVE PATH HERE
    # Expecting 'train.zip' and 'train_labels.csv' in this folder
    DRIVE_PATH = '/content/drive/MyDrive' 
    
    print(f"Looking for files in: {DRIVE_PATH}")
    
    drive_zip = os.path.join(DRIVE_PATH, 'train.zip')
    drive_csv = os.path.join(DRIVE_PATH, 'train_labels.csv')
    
    # Copy/Extract Zip
    if os.path.exists(drive_zip):
        if not os.path.exists('train'):
            print(f"Found {drive_zip}. Extracting to local Colab runtime...")
            with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Extraction complete!")
        else:
            print("Train folder already exists. Skipping extraction.")
    else:
        print(f"❌ Error: 'train.zip' not found at {drive_zip}")
        print("Please check the path and filename.")

    # Copy CSV
    if os.path.exists(drive_csv):
        print(f"Found {drive_csv}. Copying...")
        shutil.copy(drive_csv, '.')
        print("CSV copied.")
    else:
        print(f"❌ Error: 'train_labels.csv' not found at {drive_csv}")

else:
    print("✅ Running Locally. Checking for files...")
    
    if os.path.exists('train'):
        print("Found 'train' folder!")
    elif os.path.exists('train_images'):
        print("Found 'train_images' folder!")
    else:
        print("Looking for 'train' or 'train_images'...")
        print(f"Files in {os.getcwd()}: {os.listdir()}")
"""

    # New content for Cell 7 (remains mostly similar but cleaner)
    cell_7_source = """# 7. Configuration & Setup
import sys
import os

# DETECT ENVIRONMENT AGAIN
is_colab = 'google.colab' in sys.modules or os.path.exists('/content/sample_data')

if is_colab:
    print("Running in Google Colab.")
    IMAGE_DIR = 'train'
    CSV_PATH = 'train_labels.csv'
else:
    # Standard Local Config
    CSV_PATH = 'train_labels.csv'
    if os.path.exists('train'):
        IMAGE_DIR = 'train'
    elif os.path.exists('train_images'):
        IMAGE_DIR = 'train_images'
    else:
         IMAGE_DIR = 'train' 

print(f"Using Image Directory: {IMAGE_DIR}")
print(f"Using CSV Path: {CSV_PATH}")

BATCH_SIZE = 64
MAX_SAMPLES = None
NUM_EPOCHS = 15

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
"""

    # Update cells
    found_2 = False
    found_7 = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            if isinstance(src, list) and len(src) > 0 and src[0].startswith('# 2. Data Setup'):
                cell['source'] = cell_2_source.splitlines(True)
                found_2 = True
            elif isinstance(src, list) and len(src) > 0 and src[0].startswith('# 7. Configuration'):
                cell['source'] = cell_7_source.splitlines(True)
                found_7 = True

    if found_2 and found_7:
        with open('cancer_detection_colab.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook for Drive.")
    else:
        print(f"Failed to find cells. Found 2: {found_2}, Found 7: {found_7}")

except Exception as e:
    print(f"Error: {e}")
