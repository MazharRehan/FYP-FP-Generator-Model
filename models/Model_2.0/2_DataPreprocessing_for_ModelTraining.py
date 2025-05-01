import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical


def preprocess_data(image_dir, metadata_csv, target_size=(256, 256), validation_split=0.2):
    """
    Preprocess floor plan images and metadata for model training.

    Args:
        image_dir: Base directory containing images
        metadata_csv: Path to the metadata CSV file
        target_size: Target size for resized images
        validation_split: Fraction of data for validation

    Returns:
        X_train, y_train, X_val, y_val, metadata_train, metadata_val
    """
    # Load metadata
    df = pd.read_csv(metadata_csv)

    # Extract features for conditioning
    room_count_cols = [col for col in df.columns if col.startswith('Count_')]

    # Create arrays for images and metadata
    images = []
    metadata_features = []

    # Process each image and its metadata
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['FilePath'])
        try:
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)

            # Normalize image to [-1, 1] range (common for GANs)
            img = (img.astype(np.float32) - 127.5) / 127.5

            # Extract room counts as features
            features = row[room_count_cols].values.astype(np.float32)

            # Append to our lists
            images.append(img)
            metadata_features.append(features)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert to numpy arrays
    X = np.array(images)
    metadata_features = np.array(metadata_features)

    # Split data into training and validation
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - validation_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    metadata_train = metadata_features[train_indices]
    metadata_val = metadata_features[val_indices]

    return X_train, X_val, metadata_train, metadata_val