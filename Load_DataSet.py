
# DATA LOADING & FEATURE EXTRACTION

import os
import cv2
import json
import numpy as np

with open("config_used.json", "r") as f:
    CONFIG =  json.load(f)

def load_dataset_from_path(path: str):
    """Load images from subfolders as classes, compute color histograms."""
    print(f"\n📂 Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")

    images, labels = [], []
    class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    for idx, cname in enumerate(class_names):
        folder = os.path.join(path, cname)
        files = [f for f in os.listdir(folder) if f.lower().endswith(("jpg", "jpeg", "png", "bmp","tif"))]
        for f in files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, CONFIG["image_size"])
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            images.append(hist)
            labels.append(idx)

    print(f"✅ Loaded {len(images)} samples from {len(class_names)} classes")
    return np.array(images), np.array(labels), class_names