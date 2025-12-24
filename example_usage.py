"""
Example Usage of DIFEM for Violence Classification

This script demonstrates how to:
1. Load DIFEM features extracted from a video
2. Apply basic feature imputation
3. Use a trained Random Forest model
4. Predict whether the video contains violent behavior

Author: Shaimaa Almously
Graduation Project – 2025
"""

import os
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import warnings
from difem.feature_extraction import extract_difem_features  # ✅ إضافة استخراج الميزات

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------
# Step 0: Extract DIFEM features from video JSON
# --------------------------------------------------

FEATURE_JSON_PATH = "data/sample_video_json"
FEATURE_OUTPUT_PATH = "data/features/video_features.npy"

features = extract_difem_features(FEATURE_JSON_PATH)
print("DIFEM feature vector:", features)
print("Feature dimension:", features.shape)

# حفظ الميزات لاستخدامها لاحقًا في التصنيف
np.save(FEATURE_OUTPUT_PATH, features)


# --------------------------------------------------
# Utility: Display video frames (for visualization)
# --------------------------------------------------

def show_frames_only(folder_path, num_frames=10):
    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(('.jpg', '.png'))
    ])[:num_frames]

    if not frame_files:
        print(f"⚠️ No frames found in: {folder_path}")
        return

    images = []
    for f in frame_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2.5, 2.5))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main Inference Pipeline
# --------------------------------------------------

def classify_video(
    feature_path,
    model_path,
    mean_nonviolence_path,
    frames_dir=None
):
    """
    Classifies a video based on DIFEM features.
    """

    # Load DIFEM features
    X = np.load(feature_path)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Load feature means (used for zero-value imputation)
    mean_nonviolence = np.load(mean_nonviolence_path)

    # Replace zero values
    X[X == 0] = mean_nonviolence[X == 0]

    # Load trained model
    model = joblib.load(model_path)

    # Predict
    prediction = model.predict(X)[0]

    label = "Violence" if prediction == 1 else "Non-Violence"
    print(f"✅ Prediction Result: {label}")

    # Optional visualization
    if frames_dir and os.path.exists(frames_dir):
        show_frames_only(frames_dir, num_frames=10)

    return prediction


# --------------------------------------------------
# Script Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    MODEL_FILE = "models/random_forest_difem.pkl"
    MEAN_NONVIOLENCE = "models/feature_means_nonviolence.npy"
    FRAMES_DIR = "data/frames/video_01"

    classify_video(
        feature_path=FEATURE_OUTPUT_PATH,
        model_path=MODEL_FILE,
        mean_nonviolence_path=MEAN_NONVIOLENCE,
        frames_dir=FRAMES_DIR
    )
