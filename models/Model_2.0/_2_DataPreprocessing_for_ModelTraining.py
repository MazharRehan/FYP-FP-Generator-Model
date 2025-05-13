import numpy as np
import cv2
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
# Target image shape based on avg aspect ratio (0.552)
TARGET_HEIGHT = 1014
TARGET_WIDTH = 560  # int(0.552 * 1014) ~= 560
ROOM_TYPES = [
    "Bedroom", "Bathroom", "Kitchen", "DrawingRoom", "Garage", "Lounge",
    "Backyard", "Stairs", "Store", "OpenSpace", "PrayerRoom", "Staircase",
    "Lobby", "Lawn", "Dining", "ServantQuarter", "Passage", "Laundry",
    "DressingArea", "SideGarden", "Library", "Walls", "Door"
]
PLOT_SIZES = ["5Marla", "10Marla", "20Marla"]
PLOT_AREAS = {"5Marla": 1125, "10Marla": 2250, "20Marla": 4500}  # sq ft

def get_plot_size_from_filename(filename):
    for p in PLOT_SIZES:
        if p.lower() in filename.lower():
            return p
    return None

def preprocess_data(image_dir, metadata_csv, target_size=(TARGET_WIDTH, TARGET_HEIGHT), validation_split=0.2, use_areas=False):
    df = pd.read_csv(metadata_csv)
    df = df.fillna(0)
    images = []
    metadata_features = []
    area_per_pixel_list = []

    # Prepare all image file lookup (basename -> path)
    all_image_paths = glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
    image_lookup = {os.path.basename(path): path for path in all_image_paths}

    for idx, row in df.iterrows():
        filename = os.path.basename(row['FilePath'])
        img_path = image_lookup.get(filename)
        if img_path is None:
            print(f"Image not found for {filename}")
            continue

        # --- Get plot size (from CSV or filename or folder) ---
        plot_size = row.get("PlotSize") or get_plot_size_from_filename(filename)
        if plot_size not in PLOT_SIZES:
            print(f"Unknown plot size for {filename}")
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
            img = (img.astype(np.float32) - 127.5) / 127.5  # [-1, 1] for tanh

            # Area per pixel
            total_area = PLOT_AREAS[plot_size]
            area_per_pixel = total_area / (target_size[0] * target_size[1])
            area_per_pixel_list.append(area_per_pixel)

            # === Conditioning vector ===
            # One-hot plot size
            plot_size_onehot = [1 if plot_size == x else 0 for x in PLOT_SIZES]
            # Room counts
            room_counts = [row.get(f"Count_{rt}", 0) for rt in ROOM_TYPES]
            # Room areas (optional, normalized by total area)
            if use_areas:
                area_features = [
                    float(row.get(f"{rt}1_SqFt", 0)) / total_area if total_area > 0 else 0
                    for rt in ROOM_TYPES
                ]
                features = np.array(plot_size_onehot + room_counts + area_features, dtype=np.float32)
            else:
                features = np.array(plot_size_onehot + room_counts, dtype=np.float32)

            images.append(img)
            metadata_features.append(features)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    X = np.array(images)
    metadata_features = np.array(metadata_features)
    area_per_pixel_arr = np.array(area_per_pixel_list)

    # Stats print
    print(f"Image data: min={X.min()}, max={X.max()}, mean={X.mean()}")
    print(f"Conditioning vector: min={metadata_features.min()}, max={metadata_features.max()}, mean={metadata_features.mean()}")
    print(f"Any NaN in images? {np.isnan(X).any()}, in conditioning? {np.isnan(metadata_features).any()}")

    # Split data
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - validation_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    X_train = X[train_indices]
    X_val = X[val_indices]
    metadata_train = metadata_features[train_indices]
    metadata_val = metadata_features[val_indices]
    area_per_pixel_train = area_per_pixel_arr[train_indices]
    area_per_pixel_val = area_per_pixel_arr[val_indices]
    return X_train, X_val, metadata_train, metadata_val, area_per_pixel_train, area_per_pixel_val