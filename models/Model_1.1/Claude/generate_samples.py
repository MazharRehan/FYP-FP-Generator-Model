# --- START OF FILE generate_samples.py ---

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import argparse
import random

# Import necessary functions and definitions from training script
# Ensure floor_plan_generator.py is accessible
try:
    from floor_plan_generator import (
        COLOR_MAP,
        INDEX_TO_COLOR,
        NUM_OUTPUT_CHANNELS,
        PLOT_TYPES,
        IMG_HEIGHT,
        IMG_WIDTH,
        NUM_CHANNELS # Needed for noise shape
    )
except ImportError:
    print("ERROR: Ensure 'floor_plan_generator.py' is in the same directory or Python path.")
    print("Cannot import necessary definitions.")
    exit()

# --- Generation Function (Adapted from FloorPlanGenerator class) ---

def generate_floor_plan_from_noise(generator_model, plot_type, seed=None):
    """
    Generates a floor plan for a specific plot type using random noise as input.

    Args:
        generator_model: The loaded Keras generator model.
        plot_type (str): Type of plot ('5_marla', '10_marla', or '20_marla').
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.array: Generated floor plan as an RGB image (uint8), or None on error.
    """
    if plot_type not in PLOT_TYPES:
        print(f"Error: Invalid plot_type '{plot_type}'. Must be one of {PLOT_TYPES}")
        return None

    if seed is not None:
        # Set seeds for TensorFlow and NumPy for this specific generation
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed} for {plot_type}")
    else:
         # If no seed, ensure subsequent calls aren't identical if TF seed wasn't reset
         tf.random.set_seed(random.randint(0, 100000)) # Use a random seed


    # Create random noise input
    # Shape: (batch_size, height, width, channels) -> (1, 256, 256, 3)
    noise_input = tf.random.normal([1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])

    # Create condition vector
    condition = np.zeros((1, len(PLOT_TYPES)), dtype=np.float32)
    try:
        plot_idx = PLOT_TYPES.index(plot_type)
        condition[0, plot_idx] = 1
    except ValueError:
        print(f"Error: Plot type '{plot_type}' not found in defined PLOT_TYPES.")
        return None
    condition_tensor = tf.convert_to_tensor(condition)

    # Generate floor plan logits using the loaded model
    try:
        prediction_logits = generator_model([noise_input, condition_tensor], training=False)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None


    # Convert prediction logits to segmentation map (indices)
    # Squeeze the batch dimension [0]
    pred_map_indices = tf.argmax(prediction_logits[0], axis=-1).numpy()

    # Convert segmentation map indices to RGB image using the global mapping
    rgb_image = INDEX_TO_COLOR[pred_map_indices]

    return rgb_image

# --- PNG Export Function (Adapted from FloorPlanGenerator class) ---

def export_floor_plan_png(rgb_image, output_path):
    """
    Exports a floor plan RGB image to a PNG file without axes/whitespace.

    Args:
        rgb_image (np.array): Floor plan as RGB image (H, W, 3).
        output_path (str): Path to save the floor plan (should end in .png).
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Use matplotlib to save, ensuring no axes or extra whitespace
        height, width, _ = rgb_image.shape
        dpi = 96 # Standard DPI, adjust if needed
        fig_height = height / dpi
        fig_width = width / dpi

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1]) # Fill the entire figure
        ax.imshow(rgb_image)
        ax.axis('off')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Close the figure to free memory
        print(f"Saved PNG: {output_path}")

    except Exception as e:
        print(f"Error saving PNG to {output_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample Floor Plans from a trained model using noise.")
    parser.add_argument("--generator_path", type=str, default="generator_model.h5",
                        help="Path to the saved generator HDF5 model file.")
    parser.add_argument("--output_dir", type=str, default="generated_samples_from_noise",
                        help="Directory to save generated sample images.")
    parser.add_argument("--num_per_type", type=int, default=3,
                        help="Number of samples to generate for each plot type.")

    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading generator model from: {args.generator_path}")
    try:
        # Load the generator model (no need to compile for inference)
        generator = models.load_model(args.generator_path, compile=False)
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading generator model: {e}")
        exit()

    # --- Generate Samples ---
    print(f"\nGenerating {args.num_per_type} samples for each plot type: {PLOT_TYPES}")
    os.makedirs(args.output_dir, exist_ok=True)

    for plot_type in PLOT_TYPES:
        print(f"\n--- Generating for: {plot_type} ---")
        plot_type_output_dir = os.path.join(args.output_dir, plot_type)
        os.makedirs(plot_type_output_dir, exist_ok=True)

        for i in range(args.num_per_type):
            sample_seed = i # Use a consistent seed for reproducibility across runs

            # Generate the floor plan image
            generated_rgb = generate_floor_plan_from_noise(generator, plot_type, seed=sample_seed)

            if generated_rgb is not None:
                # Define output filename
                output_filename = f'generated_{plot_type}_seed{sample_seed}.png'
                output_path = os.path.join(plot_type_output_dir, output_filename)

                # Export to PNG
                export_floor_plan_png(generated_rgb, output_path)
            else:
                print(f"Skipping sample {i} for {plot_type} due to generation error.")

    print("\n--- Sample Generation Complete ---")
    print(f"Generated samples saved in: {args.output_dir}")

# --- END OF FILE generate_samples.py ---