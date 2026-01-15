# Cancer Detection AI 🎗️

An advanced AI-powered application for histopathologic cancer detection. This project leverages an **ensemble of deep learning models** (DenseNet121, ResNet50, EfficientNet-B0) to classify tissue samples and uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide visual explainability for its predictions.

![Project Banner](https://via.placeholder.com/1200x300?text=Cancer+Detection+AI+Platform) <!-- Replace with actual screenshot if available -->

## 🛠️ Tech Stack & Architecture

This project is built with a modern, full-stack architecture designed for performance and scalability.

### **Backend (Python & AI)**
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (High-performance async API)
- **Deep Learning:** [PyTorch](https://pytorch.org/) (Model training & inference)
- **Image Processing:** [Pillow](https://python-pillow.org/) & [OpenCV](https://opencv.org/)
- **Models:** DenseNet121, ResNet50, EfficientNet-B0 (Transfer Learning)
- **Explainability:** Grad-CAM (Visual Heatmaps)

### **Frontend (Modern Web)**
- **Core:** [React 18](https://react.dev/) (Component-based UI)
- **Build Tool:** [Vite](https://vitejs.dev/) (Lightning-fast dev server)
- **Styling:** [TailwindCSS](https://tailwindcss.com/) (Utility-first styling)
- **Animations:** [Framer Motion](https://www.framer.com/motion/) (Smooth transitions)
- **State Management:** React Hooks & Context

---

## 📂 Project Structure

The project is organized to separate concerns between data, logic, and presentation.

```bash
├── backend/                # 🐍 FastAPI Backend Logic
│   ├── main.py             # API Entry point & Routes
│   └── inference.py        # Model loading, Prediction & Grad-CAM logic
├── frontend/               # ⚛️ React Frontend
│   ├── src/                # UI Components, Pages, and Hooks
│   └── public/             # Static Assets
├── ml_pipeline/            # 🧠 Machine Learning Core
│   ├── train.py            # Main training script (Ensemble Loop)
│   ├── model.py            # Neural Network Architecture Definitions
│   └── data_loader.py      # Custom PyTorch Dataset Class
├── data/                   # 💾 Dataset Storage
│   ├── train/              # Training images (.tif)
│   ├── test/               # Test images (.tif)
│   └── train_labels.csv    # Dataset metadata
├── scripts/                # 🔧 Maintenance & Utilities
│   └── verification/       # Scripts to verify Grad-CAM & Logic
├── notebooks/              # 📓 Jupyter Notebooks for Experiments
└── analysis.txt            # 📊 Training Metrics & Model Insights
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **Node.js 16+** & **npm**
- **CUDA-capable GPU** (Highly recommended for training, optional for inference)

### 1. Backend Setup
Set up the Python environment and install dependencies.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Setup
Install the necessary Node.js packages.

```bash
cd frontend
npm install
```

### 3. Running the Application
We provide a convenience script to launch the full stack.

**Option A: One-Click Launch (Windows)**
Double-click `run_app.bat` or run it from the terminal:
```powershell
.\run_app.bat
```

**Option B: Manual Start**
*Terminal 1 (Backend):*
```bash
cd backend
uvicorn main:app --reload
```
*Terminal 2 (Frontend):*
```bash
cd frontend
npm run dev
```
Open your browser to `http://localhost:5173` (or the port shown in terminal).

---

## 📡 API Documentation

The backend exposes a RESTful API for predictions.

### `POST /predict`
Uploads a tissue image and returns the probability of cancer along with a heatmap.

- **Request:** `multipart/form-data`
    - `file`: Image file (.tif, .png, .jpg)
- **Response:** JSON
    ```json
    {
      "prediction": "Cancer",
      "confidence": 0.985,
      "heatmap_base64": "data:image/png;base64,iVBOR...",
      "original_base64": "data:image/png;base64,Akd2...",
      "pattern_type": "Focal/Localized"
    }
    ```

### `GET /health`
Status check to ensure the API and Model are loaded.

---

## 📊 Training the Model

To retrain the ensemble model with your own data:

1.  **Prepare Data:** Place your `.tif` images in `data/train/` and your `train_labels.csv` in `data/`.
2.  **Run Training:**
    ```bash
    python ml_pipeline/train.py --num_epochs 10 --batch_size 32
    ```
    *Note: The script defaults to looking in the `data/` directory.*
3.  **Review Results:** Check `analysis.txt` for accuracy metrics and training logs.
4.  **Verification:** Run `scripts/verification/verify_gradcam.bat` to test the explainability layer.

---

## ❓ Troubleshooting

**Q: `FileNotFoundError` when training?**
> A: Ensure you have moved your training data to the `data/` folder. The script expects `data/train` and `data/train_labels.csv`.

**Q: App fails to start with "Module not found"?**
> A: Make sure you activated the virtual environment (`.venv\Scripts\activate`) before installing requirements.

**Q: Grad-CAM heatmap looks solid red/green?**
> A: This might happen if the model is untrained or overfitted. Try retraining for more epochs or check `analysis.txt` to see if the model converged using the `verify_gradcam.py` tool.

---

## 📜 License
This project is for **Educational and Research Purposes Only**. It is NOT intended for clinical diagnosis.
