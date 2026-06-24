"""
=============================================================================
 Grad-CAM Analysis Tool for COTS Detector
=============================================================================
 Generates Grad-CAM heatmap overlays to visualize what the EfficientNet-B0
 model focuses on when making predictions.

 This is critical for diagnosing WHY the model fails on certain images
 (e.g., dark purple underwater COTS) — it reveals whether the model learned
 actual COTS morphology or just background/color correlations.

 Usage:
   python src/gradcam_analysis.py --image path/to/image.jpg --model cots_model_archive.zip
   python src/gradcam_analysis.py --image path/to/image.jpg --model models/cots_model_v2.pth

 Output:
   Saves gradcam_overlay.jpg in the same directory as the input image.
=============================================================================
"""

import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM (Gradient-weighted Class Activation Mapping) works by:
# 1. Forward pass: record activations at a target convolutional layer
# 2. Backward pass: compute gradients of the target class w.r.t. activations
# 3. Pool gradients globally (GAP) to get channel importance weights
# 4. Weighted sum of activation maps → heatmap
# 5. ReLU to keep only positive contributions
#
# For EfficientNet-B0, the best target layer is the last convolutional layer
# in the features block: model.features[-1]
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM visualization for any CNN model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index to visualize. If None, uses predicted class.

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
            predicted_class: int
            confidence: float
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

        if target_class is None:
            target_class = predicted.item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]    # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU: keep only positive contributions
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap = heatmap.cpu().numpy()

        return heatmap, predicted.item(), confidence.item(), probs[0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image
        heatmap: numpy array (H, W) with values in [0, 1]
        alpha: transparency of the heatmap overlay

    Returns:
        overlay: PIL Image with heatmap overlay
    """
    # Resize heatmap to match original image
    img_w, img_h = original_image.size
    heatmap_resized = np.array(
        Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize((img_w, img_h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap (jet: blue=low, red=high)
    colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    # Blend original image with heatmap
    original_np = np.array(original_image.convert("RGB"))
    overlay = (original_np * (1 - alpha) + colored_heatmap * alpha).astype(np.uint8)

    return Image.fromarray(overlay), Image.fromarray(colored_heatmap)


def create_analysis_figure(original_image, heatmap, predicted_class,
                           confidence, probs, output_path):
    """Create a multi-panel analysis figure."""
    class_names = ["COTS Found", "No COTS"]

    overlay, heatmap_colored = overlay_heatmap(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Grad-CAM Analysis — Prediction: {class_names[predicted_class]} "
        f"({confidence * 100:.1f}%)",
        fontsize=16, fontweight="bold",
    )

    # Panel 1: Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Heatmap only
    axes[1].imshow(heatmap_colored)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red = high attention)", fontsize=13)
    axes[2].axis("off")

    # Add probability text
    prob_text = (
        f"P(COTS) = {probs[0] * 100:.2f}%\n"
        f"P(No COTS) = {probs[1] * 100:.2f}%"
    )
    fig.text(0.5, 0.02, prob_text, ha="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis figure saved → '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path):
    """Load EfficientNet-B0 model from a .pth or .zip file."""
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    # Handle zip archive format (original cots_model directory)
    if model_path.endswith(".zip") or os.path.isdir(model_path):
        if os.path.isdir(model_path):
            zip_path = model_path + "_archive.zip"
            if not os.path.exists(zip_path):
                shutil.make_archive(
                    zip_path.replace(".zip", ""), "zip",
                    os.path.dirname(model_path), os.path.basename(model_path)
                )
            model_path = zip_path
        state_dict = torch.load(model_path, map_location="cpu",
                                weights_only=False)
    else:
        state_dict = torch.load(model_path, map_location="cpu",
                                weights_only=False)

    # Handle state_dict that might have 'classifier.1' with different shape
    # (v2 model uses Sequential(Dropout(0.3), Linear(1280, 2)))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try loading with the v2 classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2),
        )
        model.load_state_dict(state_dict)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM analysis for COTS detection model"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the model file (.pth or .zip)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the analysis figure (default: gradcam_<image>)"
    )
    parser.add_argument(
        "--target-class", type=int, default=None,
        help="Target class for Grad-CAM (0=COTS, 1=No COTS). Default: predicted"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    original_image = Image.open(args.image).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    input_tensor = preprocess(original_image).unsqueeze(0)

    # Initialize Grad-CAM with the last convolutional layer
    # For EfficientNet-B0: model.features[-1] is the final MBConv block
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap
    print("Generating Grad-CAM heatmap...")
    heatmap, pred_class, confidence, probs = grad_cam.generate(
        input_tensor, target_class=args.target_class
    )

    class_names = ["COTS Found", "No COTS"]
    print(f"Prediction: {class_names[pred_class]} ({confidence * 100:.1f}%)")
    print(f"Probabilities: COTS={probs[0]*100:.2f}%, No COTS={probs[1]*100:.2f}%")

    # Generate output path
    if args.output is None:
        base = os.path.splitext(args.image)[0]
        output_path = f"{base}_gradcam.jpg"
    else:
        output_path = args.output

    # Create analysis figure
    create_analysis_figure(
        original_image, heatmap, pred_class, confidence, probs, output_path
    )

    # Diagnostic interpretation
    print("\n" + "=" * 50)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 50)

    # Analyze heatmap concentration
    threshold = 0.5
    hot_ratio = (heatmap > threshold).sum() / heatmap.size
    print(f"Hot region ratio (>{threshold}): {hot_ratio:.2%}")

    if hot_ratio < 0.05:
        print("WARNING: Very focused attention — model may be relying on a ")
        print("small spurious feature rather than overall COTS morphology.")
    elif hot_ratio > 0.5:
        print("WARNING: Very diffuse attention — model may be looking at ")
        print("background/texture rather than the COTS itself.")
    else:
        print("OK: Attention is reasonably concentrated on specific features.")

    if pred_class == 1 and args.target_class is None:
        print("\nThe model predicted No COTS. To see what COTS features it")
        print("looked for (and missed), re-run with: --target-class 0")


if __name__ == "__main__":
    main()
