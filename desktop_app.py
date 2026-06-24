"""
=============================================================================
 COTS Detector — Desktop Application
=============================================================================
 A CustomTkinter desktop app that deploys the fine-tuned EfficientNet-B0
 model for binary classification of Crown-of-Thorns Starfish (COTS).

 Class 0: COTS Found   — The image contains a Crown-of-Thorns Starfish
 Class 1: No COTS      — The image does not contain a COTS

 The app provides:
   - File dialog for image selection (JPG, JPEG, PNG)
   - Image preview with auto-scaling
   - One-click classification
   - Animated confidence bar with color-coded results
   - Probability breakdown for both classes
=============================================================================
"""

import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# CustomTkinter appearance settings for the dark premium theme.
# "System" follows the OS dark/light mode; we force "Dark" for consistency.
# The color theme "blue" provides accent colors for buttons and widgets.
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Color palette used throughout the UI
COLORS = {
    "bg_dark":      "#0a0e1a",      # Main background
    "bg_card":      "#131829",      # Card / panel background
    "bg_card_alt":  "#1a2035",      # Alternate card background
    "border":       "#2a3050",      # Card border color
    "text_primary": "#e2e8f0",      # Primary text (white-ish)
    "text_secondary":"#8892b0",     # Muted text
    "accent":       "#7b2ff7",      # Purple accent
    "accent_hover": "#6922d1",      # Darker accent on hover
    "safe":         "#00d2ff",      # Cyan — No COTS (safe)
    "safe_dark":    "#0099bb",      # Darker cyan
    "danger":       "#ff5252",      # Red — COTS found (alert)
    "danger_dark":  "#cc3333",      # Darker red
    "bar_bg":       "#1e2540",      # Confidence bar background
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
# Loads the fine-tuned EfficientNet-B0 model from the cots_model/ directory.
# The model directory is PyTorch's extracted zip format, so we re-archive it
# into a .zip file that torch.load() can read.
#
# Architecture:
#   EfficientNet-B0 backbone (1280 features) → Linear(1280, 2)
#   Class 0 = COTS, Class 1 = No COTS
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load the fine-tuned EfficientNet-B0 model for COTS classification."""
    import shutil

    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(num_features, 2)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "cots_model_archive.zip")
    new_model_path = os.path.join(base_dir, "models", "cots_model.pth")

    if os.path.exists(new_model_path):
        state_dict = torch.load(new_model_path, map_location="cpu", weights_only=False)
    else:
        if not os.path.exists(zip_path):
            shutil.make_archive(
                zip_path.replace(".zip", ""), "zip",
                base_dir, "cots_model"
            )
        state_dict = torch.load(zip_path, map_location="cpu", weights_only=False)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2),
        )
        model.load_state_dict(state_dict)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# Standard EfficientNet-B0 preprocessing:
#   Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet stats)
# ─────────────────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image, model):
    """
    Run inference on a single PIL image.

    Returns:
        predicted_class (int): 0 = COTS, 1 = No COTS
        confidence (float): 0.0 – 1.0
        probs (list[float]): [prob_cots, prob_no_cots]
    """
    img = image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item(), probs[0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION CLASS
# ─────────────────────────────────────────────────────────────────────────────
class COTSDetectorApp(ctk.CTk):
    """
    Main desktop window for the COTS Detector.

    Layout (top → bottom):
      ┌─────────────────────────────────────┐
      │           Title Bar                  │
      ├──────────────┬──────────────────────┤
      │  Left Panel  │    Right Panel       │
      │  - Upload    │    - Result card     │
      │  - Preview   │    - Confidence bar  │
      │  - Classify  │    - Probabilities   │
      └──────────────┴──────────────────────┘
    """

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("COTS Detector")
        self.geometry("1000x680")
        self.minsize(900, 600)
        self.configure(fg_color=COLORS["bg_dark"])

        # State
        self.current_image = None   # PIL Image loaded by the user
        self.model = None           # Loaded PyTorch model

        # Build UI
        self._build_title_bar()
        self._build_main_content()

        # Load model in background so the UI doesn't freeze
        self._show_status("Loading model...")
        threading.Thread(target=self._load_model_async, daemon=True).start()

    # ── Title Bar ────────────────────────────────────────────────────────
    def _build_title_bar(self):
        """Top banner with app title and subtitle."""
        title_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"],
                                   corner_radius=0, height=90)
        title_frame.pack(fill="x", padx=0, pady=(0, 0))
        title_frame.pack_propagate(False)

        title = ctk.CTkLabel(
            title_frame, text="COTS Detector",
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=COLORS["accent"],
        )
        title.pack(pady=(18, 0))

        subtitle = ctk.CTkLabel(
            title_frame,
            text="Crown-of-Thorns Starfish Classification  |  EfficientNet-B0",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_secondary"],
        )
        subtitle.pack(pady=(2, 0))

    # ── Main Content ─────────────────────────────────────────────────────
    def _build_main_content(self):
        """Two-column layout: left (input) + right (results)."""
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=24, pady=(8, 20))
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self._build_left_panel(container)
        self._build_right_panel(container)

    # ── Left Panel (Input) ───────────────────────────────────────────────
    def _build_left_panel(self, parent):
        """Image upload, preview, and classify button."""
        left = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                            corner_radius=16, border_width=1,
                            border_color=COLORS["border"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Section header
        header = ctk.CTkLabel(
            left, text="Input Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        header.pack(anchor="w", padx=20, pady=(20, 12))

        # Upload button
        self.upload_btn = ctk.CTkButton(
            left, text="Browse Image...",
            font=ctk.CTkFont(size=14),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            corner_radius=10, height=42,
            command=self._on_browse,
        )
        self.upload_btn.pack(fill="x", padx=20, pady=(0, 12))

        # Image preview area
        self.preview_frame = ctk.CTkFrame(
            left, fg_color=COLORS["bg_card_alt"],
            corner_radius=12, border_width=1,
            border_color=COLORS["border"], height=300,
        )
        self.preview_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        self.preview_frame.pack_propagate(False)

        self.preview_label = ctk.CTkLabel(
            self.preview_frame, text="No image selected",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_secondary"],
        )
        self.preview_label.pack(expand=True)

        # Filename display
        self.filename_label = ctk.CTkLabel(
            left, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
        )
        self.filename_label.pack(anchor="w", padx=20, pady=(0, 4))

        # Classify button
        self.classify_btn = ctk.CTkButton(
            left, text="Classify Image",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            corner_radius=10, height=48,
            state="disabled",
            command=self._on_classify,
        )
        self.classify_btn.pack(fill="x", padx=20, pady=(0, 20))

    # ── Right Panel (Results) ────────────────────────────────────────────
    def _build_right_panel(self, parent):
        """Prediction results, confidence bar, probability breakdown."""
        right = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                             corner_radius=16, border_width=1,
                             border_color=COLORS["border"])
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Section header
        header = ctk.CTkLabel(
            right, text="Prediction Results",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        header.pack(anchor="w", padx=20, pady=(20, 12))

        # ── Result card (inner frame) ──
        self.result_card = ctk.CTkFrame(
            right, fg_color=COLORS["bg_card_alt"],
            corner_radius=14, border_width=1,
            border_color=COLORS["border"],
        )
        self.result_card.pack(fill="x", padx=20, pady=(0, 12))

        # Status icon / text
        self.result_icon = ctk.CTkLabel(
            self.result_card, text="[ ? ]",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["text_secondary"],
        )
        self.result_icon.pack(pady=(24, 4))

        # Prediction label
        self.result_label = ctk.CTkLabel(
            self.result_card, text="Waiting for input",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_secondary"],
        )
        self.result_label.pack(pady=(0, 4))

        # Confidence percentage
        self.confidence_label = ctk.CTkLabel(
            self.result_card, text="",
            font=ctk.CTkFont(size=38, weight="bold"),
            text_color=COLORS["text_secondary"],
        )
        self.confidence_label.pack(pady=(4, 2))

        # "Confidence Score" subtitle
        self.conf_subtitle = ctk.CTkLabel(
            self.result_card, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
        )
        self.conf_subtitle.pack(pady=(0, 4))

        # Confidence bar
        self.bar_bg = ctk.CTkFrame(
            self.result_card, fg_color=COLORS["bar_bg"],
            corner_radius=8, height=14,
        )
        self.bar_bg.pack(fill="x", padx=40, pady=(4, 24))
        self.bar_bg.pack_propagate(False)

        self.bar_fill = ctk.CTkFrame(
            self.bar_bg, fg_color=COLORS["accent"],
            corner_radius=8, height=14, width=0,
        )
        self.bar_fill.place(x=0, y=0, relheight=1.0)

        # ── Probability breakdown ──
        prob_header = ctk.CTkLabel(
            right, text="Class Probabilities",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        prob_header.pack(anchor="w", padx=20, pady=(8, 8))

        # COTS probability row
        self.prob_cots_frame = self._make_prob_row(right, "COTS Found", "—")
        # No COTS probability row
        self.prob_nocots_frame = self._make_prob_row(right, "No COTS", "—")

        # ── Status / info label ──
        self.status_label = ctk.CTkLabel(
            right, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
            wraplength=380,
        )
        self.status_label.pack(anchor="w", padx=20, pady=(16, 20))

    # ── Helper: probability row ──────────────────────────────────────────
    def _make_prob_row(self, parent, label_text, value_text):
        """Create a single row showing a class name and its probability."""
        row = ctk.CTkFrame(parent, fg_color=COLORS["bg_card_alt"],
                           corner_radius=10, height=42, border_width=1,
                           border_color=COLORS["border"])
        row.pack(fill="x", padx=20, pady=(0, 6))
        row.pack_propagate(False)

        lbl = ctk.CTkLabel(row, text=label_text,
                           font=ctk.CTkFont(size=13),
                           text_color=COLORS["text_primary"])
        lbl.pack(side="left", padx=16)

        val = ctk.CTkLabel(row, text=value_text,
                           font=ctk.CTkFont(size=14, weight="bold"),
                           text_color=COLORS["text_secondary"])
        val.pack(side="right", padx=16)

        row._value_label = val  # Store ref for later updates
        return row

    # ── Model loading (background thread) ────────────────────────────────
    def _load_model_async(self):
        """Load the model in a background thread to keep UI responsive."""
        try:
            self.model = load_model()
            self.after(0, lambda: self._show_status(
                "Model loaded — EfficientNet-B0 (2 classes). Ready."
            ))
            self.after(0, lambda: self._enable_classify_if_ready())
        except Exception as e:
            self.after(0, lambda: self._show_status(f"Error loading model: {e}"))

    # ── Browse for image ─────────────────────────────────────────────────
    def _on_browse(self):
        """Open a file dialog to select an image."""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img = Image.open(path)
            self.current_image = img
            self._show_preview(img, os.path.basename(path))
            self._enable_classify_if_ready()
            # Reset results when a new image is loaded
            self._reset_results()
        except Exception as e:
            self._show_status(f"Error opening image: {e}")

    # ── Show image preview ───────────────────────────────────────────────
    def _show_preview(self, img: Image.Image, filename: str):
        """Display the selected image in the preview area."""
        self.preview_frame.update_idletasks()
        pw = self.preview_frame.winfo_width() - 20
        ph = self.preview_frame.winfo_height() - 20

        # Scale to fit while keeping aspect ratio
        img_copy = img.copy()
        img_copy.thumbnail((max(pw, 100), max(ph, 100)), Image.LANCZOS)

        self._tk_image = ImageTk.PhotoImage(img_copy)
        self.preview_label.configure(image=self._tk_image, text="")
        self.filename_label.configure(text=filename)

    # ── Classify image ───────────────────────────────────────────────────
    def _on_classify(self):
        """Run inference on the current image."""
        if self.current_image is None or self.model is None:
            return

        self.classify_btn.configure(state="disabled", text="Analyzing...")
        self._show_status("Running inference...")

        # Run prediction in a background thread
        def run():
            pred_class, conf, probs = predict(self.current_image, self.model)
            self.after(0, lambda: self._display_results(pred_class, conf, probs))

        threading.Thread(target=run, daemon=True).start()

    # ── Display results ──────────────────────────────────────────────────
    def _display_results(self, pred_class, confidence, probs):
        """Update the right panel with prediction results."""
        is_cots = pred_class == 0  # Class 0 = COTS Found

        color = COLORS["danger"] if is_cots else COLORS["safe"]
        icon = "[!]" if is_cots else "[OK]"
        label = "COTS Detected!" if is_cots else "No COTS — Safe"

        # Update result card border
        self.result_card.configure(border_color=color)

        # Update icon, label, confidence
        self.result_icon.configure(text=icon, text_color=color)
        self.result_label.configure(text=label, text_color=color)
        self.confidence_label.configure(
            text=f"{confidence * 100:.1f}%", text_color=color
        )
        self.conf_subtitle.configure(text="Confidence Score")

        # Animate the confidence bar
        self._animate_bar(confidence, color)

        # Update probability rows
        cots_color = COLORS["accent"] if pred_class == 0 else COLORS["text_secondary"]
        nocots_color = COLORS["accent"] if pred_class == 1 else COLORS["text_secondary"]

        self.prob_cots_frame._value_label.configure(
            text=f"{probs[0] * 100:.2f}%", text_color=cots_color
        )
        self.prob_nocots_frame._value_label.configure(
            text=f"{probs[1] * 100:.2f}%", text_color=nocots_color
        )

        # Status
        self._show_status(
            "The Crown-of-Thorns Starfish (Acanthaster planci) is one of the "
            "largest predators of coral reefs. Early detection is critical for "
            "reef conservation."
        )

        # Re-enable button
        self.classify_btn.configure(state="normal", text="Classify Image")

    # ── Animate confidence bar ───────────────────────────────────────────
    def _animate_bar(self, target_pct, color, steps=30, delay=15):
        """Smoothly fill the confidence bar from 0 to target_pct."""
        self.bar_fill.configure(fg_color=color)
        bar_width = self.bar_bg.winfo_width()

        current_step = [0]

        def step():
            current_step[0] += 1
            progress = current_step[0] / steps
            # Ease-out curve for smooth deceleration
            eased = 1 - (1 - progress) ** 3
            w = int(bar_width * target_pct * eased)
            self.bar_fill.place(x=0, y=0, relheight=1.0, width=max(w, 1))

            if current_step[0] < steps:
                self.after(delay, step)

        self.bar_fill.place(x=0, y=0, relheight=1.0, width=0)
        self.after(delay, step)

    # ── Reset results ────────────────────────────────────────────────────
    def _reset_results(self):
        """Clear the results panel to the initial state."""
        self.result_card.configure(border_color=COLORS["border"])
        self.result_icon.configure(text="[ ? ]", text_color=COLORS["text_secondary"])
        self.result_label.configure(text="Ready to classify",
                                    text_color=COLORS["text_secondary"])
        self.confidence_label.configure(text="")
        self.conf_subtitle.configure(text="")
        self.bar_fill.place(x=0, y=0, relheight=1.0, width=0)
        self.prob_cots_frame._value_label.configure(
            text="—", text_color=COLORS["text_secondary"])
        self.prob_nocots_frame._value_label.configure(
            text="—", text_color=COLORS["text_secondary"])

    # ── Helpers ──────────────────────────────────────────────────────────
    def _show_status(self, msg):
        self.status_label.configure(text=msg)

    def _enable_classify_if_ready(self):
        if self.current_image is not None and self.model is not None:
            self.classify_btn.configure(state="normal")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = COTSDetectorApp()
    app.mainloop()
