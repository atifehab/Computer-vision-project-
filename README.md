# COTS Detector — Crown-of-Thorns Starfish Classification & Diagnosis

An advanced deep learning project designed to detect and diagnose Crown-of-Thorns Starfish (COTS) in marine environments. This project features a robust **EfficientNet-B0** training pipeline optimized for underwater imagery, a **Grad-CAM interpretability module** for diagnostic visualization, and two deployment options (**Streamlit web app** and **CustomTkinter desktop app**) featuring a premium dark theme.

The Crown-of-Thorns Starfish (*Acanthaster planci*) is a major marine predator responsible for extensive coral reef destruction. Accurate classification under varying lighting, turbidity, and color absorption conditions is critical for robotic ROV deployment and conservation efforts.

---

## 📂 Project Structure

```text
NTI project/
├── app.py                          # Streamlit Deployment App (Web UI)
├── desktop_app.py                  # CustomTkinter Deployment App (Desktop GUI)
├── requirements.txt                # Python environment dependencies
├── README.md                       # Project documentation
├── models/
│   └── cots_model.pth              # Saved model weights (after retraining)
├── cots_model_archive.zip          # Packaged model weights (default/fallback)
└── src/
    ├── EfficientNet_train.py       # Advanced model training script (v2)
    └── gradcam_analysis.py         # Grad-CAM interpretability diagnostic tool
```

---

## 🛠️ EfficientNet_train.py (Training Pipeline)

The original training script suffered from overfitting due to standard training transforms and tight bounding-box crops. The updated training script (`src/EfficientNet_train.py`) implements state-of-the-art techniques for marine datasets:

### 1. Advanced Augmentation Pipeline
*   **Aspect Ratio Distortion Fix**: Replaced generic resizing with `RandomResizedCrop` to ensure scale and ratio invariance.
*   **Underwater Color Shift Simulation**: Randomly absorbs red light channels (which drop off first underwater) and preserves blue/green spectra to simulate varied depths.
*   **Underwater Haze**: Simulates water particulate scattering and backscatter (turbidity).
*   **Spatial Augmentations**: Uses Perspective transforms, Affine shifts, Blur, and Random Erasing to replicate camera angle changes and reef occlusions.

### 2. Loss & Optimization Strategies
*   **Focal Loss**: Employs `FocalLoss` instead of CrossEntropy to prioritize hard-to-classify, low-contrast, or dark purple starfish samples (hard negative mining).
*   **Differential Learning Rates**: Backbone parameters use a lower learning rate (`1e-4`) to preserve pretrained weights, while the custom classifier head uses a higher rate (`1e-3`).
*   **Learning Rate Warmup & Cosine Annealing**: Implements linear learning rate warmup followed by cosine annealing to ensure smooth, stable convergence.
*   **Early Stopping**: Automatically stops training when validation accuracy halts improvement for a consecutive number of epochs.

---

## 🔍 gradcam_analysis.py (Diagnostic Tool)

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to debug and verify the model's decision-making process. By calculating the gradients of the target class with respect to the feature maps of the final convolutional layer of the EfficientNet-B0 backbone:
*   It generates a colored activation heatmap showing where the model is looking.
*   **Why this is crucial**: Prevents the model from cheating on background features (such as bright sand or coral branches) and confirms that it is attending to the actual spiny morphology of the starfish.
*   **Usage**: Runs diagnostic inference on images and outputs a visual side-by-side comparison of the raw image and the Grad-CAM heatmap.

---

## 🚀 Deployment Applications

We provide two deployment frontends, both optimized with a modern, glassmorphic dark theme:

### 1. Streamlit Web App (`app.py`)
Provides an interactive web interface.
*   **Features**: Drag-and-drop file upload, live webcam stream integration, real-time confidence bar animation, and class probability breakdown.
*   **Launch Command**:
    ```bash
    streamlit run app.py --server.port 8501
    ```

### 2. CustomTkinter Desktop App (`desktop_app.py`)
A fast, standalone desktop GUI built using CustomTkinter.
*   **Features**: Native Windows dark-themed window, smooth result card transitions, and threaded background inference to keep the interface highly responsive.
*   **Launch Command**:
    ```bash
    python desktop_app.py
    ```

---

## ⚙️ Installation & Requirements

Ensure you are using Python 3.10+ and run the following commands to install dependencies:

```bash
git clone https://github.com/atifehab/COTS-Detection.git
cd "COTS-Detection"
pip install -r requirements.txt
```

### Key Libraries:
*   `torch` & `torchvision` (PyTorch deep learning library framework)
*   `streamlit` (Web app UI)
*   `customtkinter` (Desktop GUI styling)
*   `scikit-learn` (Accuracy, recall, and evaluation metrics)
*   `opendatasets` (Automatic Kaggle dataset download wrapper)
