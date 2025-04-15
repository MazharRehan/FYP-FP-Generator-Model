# --- START OF FILE evaluate_model.py ---

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import necessary functions and definitions from your training script
# Assuming floor_plan_generator.py is in the same directory or accessible via PYTHONPATH
try:
    from floor_plan_generator import (
        load_dataset,
        create_tf_dataset, # We might need to adapt this slightly or just use parts
        preprocess_image_to_segmentation_map, # Needed if recalculating targets
        COLOR_MAP,
        INDEX_TO_COLOR,
        NUM_OUTPUT_CHANNELS,
        PLOT_TYPES,
        IMG_HEIGHT,
        IMG_WIDTH,
        BATCH_SIZE, # Use same batch size for consistency
        BUFFER_SIZE # Needed if reshuffling test data (usually not)
    )
except ImportError:
    print("ERROR: Ensure 'floor_plan_generator.py' is in the same directory or Python path.")
    print("Cannot import necessary definitions.")
    exit()

# --- Evaluation Function ---

def evaluate_model(generator_path, dataset_path, output_dir, num_visual_samples=5):
    """
    Evaluates the trained generator model on the test dataset.

    Args:
        generator_path (str): Path to the saved generator model (.h5 file).
        dataset_path (str): Path to the base dataset directory.
        output_dir (str): Directory to save evaluation results (visualizations).
        num_visual_samples (int): Number of visual comparison samples to save.
    """
    print(f"Loading trained generator from: {generator_path}")
    try:
        generator = models.load_model(generator_path, compile=False) # Don't need optimizer state
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Error loading generator model: {e}")
        return

    print("Loading and preparing test dataset...")
    try:
        # Load the full dataset first
        images, conditions = load_dataset(dataset_path)
        # Create the TF datasets, we only need the test split here
        # Note: create_tf_dataset performs the train/test split internally
        _, test_dataset = create_tf_dataset(images, conditions)
        print("Test dataset prepared.")
    except Exception as e:
        print(f"Error loading or preparing dataset: {e}")
        return

    # --- Initialize Metrics ---
    # Mean Intersection over Union
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=NUM_OUTPUT_CHANNELS, name='mean_iou')
    # Pixel Accuracy (can calculate manually or use keras metric)
    accuracy_metric = tf.keras.metrics.Accuracy(name='pixel_accuracy')
    # For manual accuracy calculation:
    # total_pixels = 0
    # correct_pixels = 0

    print("Starting evaluation on test set...")
    # Iterate through the test dataset
    for batch_inputs, batch_target_seg_map in tqdm(test_dataset, desc="Evaluating Batches"):
        input_image = batch_inputs['image']
        condition = batch_inputs['condition']

        # Generate predictions (logits)
        predictions_logits = generator([input_image, condition], training=False)
        # Convert logits to predicted class indices
        predictions_seg_map = tf.argmax(predictions_logits, axis=-1)

        # Update metrics
        # Ensure target and prediction are same dtype (usually uint8 or int32/64)
        miou_metric.update_state(batch_target_seg_map, predictions_seg_map)
        accuracy_metric.update_state(tf.reshape(batch_target_seg_map, [-1]),
                                     tf.reshape(predictions_seg_map, [-1]))

        # Manual accuracy calculation (alternative)
        # batch_correct = tf.reduce_sum(tf.cast(tf.equal(batch_target_seg_map, tf.cast(predictions_seg_map, batch_target_seg_map.dtype)), tf.int32))
        # batch_total = tf.size(batch_target_seg_map)
        # correct_pixels += batch_correct.numpy()
        # total_pixels += batch_total.numpy()

    # --- Calculate Final Metrics ---
    final_miou = miou_metric.result().numpy()
    final_accuracy = accuracy_metric.result().numpy()
    # final_manual_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Mean Intersection over Union (mIoU): {final_miou:.4f}")
    print(f"Pixel Accuracy:                     {final_accuracy:.4f}")
    # print(f"Pixel Accuracy (Manual):            {final_manual_accuracy:.4f}")
    print("------------------------")

    # --- Generate Visual Samples ---
    print(f"Generating {num_visual_samples} visual comparison samples...")
    os.makedirs(output_dir, exist_ok=True)

    # Take samples from the beginning of the test set
    sample_count = 0
    for batch_inputs, batch_target_seg_map in test_dataset.take( (num_visual_samples // BATCH_SIZE) + 1 ):
        if sample_count >= num_visual_samples:
            break

        input_image = batch_inputs['image']
        condition = batch_inputs['condition']

        # Generate predictions
        predictions_logits = generator([input_image, condition], training=False)
        predictions_seg_map = tf.argmax(predictions_logits, axis=-1)

        # Save individual images in the batch
        for i in range(len(input_image)):
            if sample_count >= num_visual_samples:
                break

            idx_in_batch = i
            input_img_np = input_image[idx_in_batch].numpy()
            target_map_np = batch_target_seg_map[idx_in_batch].numpy()
            pred_map_np = predictions_seg_map[idx_in_batch].numpy()

            # Determine original plot type for title/filename
            cond_vector = condition[idx_in_batch].numpy()
            plot_type_idx = np.argmax(cond_vector)
            plot_type_name = PLOT_TYPES[plot_type_idx] if plot_type_idx < len(PLOT_TYPES) else "Unknown"

            # Convert maps to RGB
            input_display = ((input_img_np + 1) * 127.5).astype(np.uint8)
            target_rgb = INDEX_TO_COLOR[target_map_np]
            pred_rgb = INDEX_TO_COLOR[pred_map_np]

            # Plotting
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title(f"Input Image ({plot_type_name})")
            plt.imshow(input_display)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth SegMap")
            plt.imshow(target_rgb)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Predicted SegMap")
            plt.imshow(pred_rgb)
            plt.axis('off')

            plt.suptitle(f"Evaluation Sample {sample_count + 1}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

            # Save the figure
            save_path = os.path.join(output_dir, f'eval_sample_{sample_count + 1}_{plot_type_name}.png')
            plt.savefig(save_path)
            plt.close()

            sample_count += 1

    print(f"Saved {sample_count} visual samples to: {output_dir}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Floor Plan Generator model.")
    parser.add_argument("--generator_path", type=str, default="generator_model.h5",
                        help="Path to the saved generator HDF5 model file.")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                        help="Path to the base dataset directory (containing plot type subfolders).")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results (visual samples).")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of visual comparison samples to generate.")

    args = parser.parse_args()

    # Make sure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_model(
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_visual_samples=args.num_samples
    )

    print("\nEvaluation script finished.")

# --- END OF FILE evaluate_model.py ---