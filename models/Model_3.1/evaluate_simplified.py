import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from floor_plan_generator_stage1_simplified_Version2 import Config, build_generator, preprocess_image, prepare_condition_vector
import pandas as pd
from sklearn.model_selection import train_test_split

def load_trained_generator():
    """Load the trained generator model from SavedModel format."""
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'final_generator')  # Path to SavedModel directory
    try:
        # --- CORRECT WAY TO LOAD SAVEDMODEL ---
        print(f"Loading trained generator from SavedModel: {model_path}")
        generator = tf.keras.models.load_model(model_path)
        # --------------------------------------
        print("Loaded trained generator model successfully!")
        return generator
    except Exception as e:
        print(f"Error loading generator from SavedModel: {e}")
        # You might want to see if any .h5 checkpoints exist as a fallback,
        # e.g., generator_epoch_100.h5, if SavedModel loading fails.
        # For now, let's keep it simple.

        # Fallback attempt (optional): Try loading .h5 weights if SavedModel fails
        # This part assumes your Stage 1 training also saved .h5 files.
        # And that `build_generator()` from floor_plan_generator_stage1_simplified_Version2.py
        # correctly defines the architecture for these .h5 weights.
        print(f"Attempting fallback to load weights from .h5 file...")
        generator_h5_fallback = build_generator()  # From floor_plan_generator_stage1_simplified_Version2
        h5_weights_path = os.path.join(Config.CHECKPOINT_DIR, 'generator_epoch_100.h5')  # Assuming epoch 100 is final
        if os.path.exists(h5_weights_path):
            try:
                generator_h5_fallback.load_weights(h5_weights_path)
                print(f"Successfully loaded weights from .h5 file: {h5_weights_path}")
                return generator_h5_fallback
            except Exception as e_h5:
                print(f"Error loading weights from .h5 file '{h5_weights_path}': {e_h5}")
        else:
            print(f".h5 weights file not found at {h5_weights_path}")

        return None  # Return None if all loading attempts fail

def compare_real_vs_generated(metadata_path=Config.METADATA_PATH, data_dir=Config.DATA_DIR, num_samples=5):
    """Compare real floor plans with generated ones."""
    # Load metadata
    df = pd.read_csv(metadata_path)
    df = df[df['PlotSize'] == Config.PLOT_SIZE]
    df = df[df['Version'] == 'V01']  # Use only original floor plans
    
    # Split data to use test set
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_df = test_df.iloc[:num_samples]  # Take only a few samples
    
    # Load generator
    generator = load_trained_generator()
    if generator is None:
        return
    
    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through test samples
    for i, (_, row) in enumerate(test_df.iterrows()):
        # Load real image
        image_path = os.path.join(data_dir, row['FilePath'])
        real_image = preprocess_image(image_path)
        
        # Extract condition vector
        condition = prepare_condition_vector(row)
        condition = np.expand_dims(condition, axis=0)
        
        # Generate fake image
        noise = tf.random.normal([1, Config.LATENT_DIM])
        fake_image = generator([noise, condition], training=False)[0]
        
        # Convert from [-1, 1] to [0, 1] for visualization
        real_image = (real_image + 1.0) / 2.0
        fake_image = (fake_image + 1.0) / 2.0
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"Sample {i+1}: Real vs Generated Floor Plan (10 Marla)", fontsize=16)
        
        # Real image - walls
        plt.subplot(2, 3, 1)
        plt.imshow(real_image[:, :, 0], cmap='gray')
        plt.title("Real - Walls")
        plt.axis('off')
        
        # Real image - rooms
        plt.subplot(2, 3, 2)
        plt.imshow(real_image[:, :, 1], cmap='jet')
        plt.title("Real - Rooms")
        plt.axis('off')
        
        # Real image - combined
        plt.subplot(2, 3, 3)
        combined_real = np.zeros((real_image.shape[0], real_image.shape[1], 3))
        combined_real[:, :, 0] = real_image[:, :, 0]  # Walls in red channel
        combined_real[:, :, 1] = real_image[:, :, 1]  # Room areas in green channel
        plt.imshow(combined_real)
        plt.title("Real - Combined")
        plt.axis('off')
        
        # Generated image - walls
        plt.subplot(2, 3, 4)
        plt.imshow(fake_image[:, :, 0], cmap='gray')
        plt.title("Generated - Walls")
        plt.axis('off')
        
        # Generated image - rooms
        plt.subplot(2, 3, 5)
        plt.imshow(fake_image[:, :, 1], cmap='jet')
        plt.title("Generated - Rooms")
        plt.axis('off')
        
        # Generated image - combined
        plt.subplot(2, 3, 6)
        combined_fake = np.zeros((fake_image.shape[0], fake_image.shape[1], 3))
        combined_fake[:, :, 0] = fake_image[:, :, 0]  # Walls in red channel
        combined_fake[:, :, 1] = fake_image[:, :, 1]  # Room areas in green channel
        plt.imshow(combined_fake)
        plt.title("Generated - Combined")
        plt.axis('off')
        
        # Add metadata information
        bedroom_count = row.get('Count_Bedroom', 0)
        bathroom_count = row.get('Count_Bathroom', 0)
        plt.figtext(0.5, 0.01, 
                   f"Floor Plan: {row['FloorLevel']}-{row['FP_Number']} • Bedrooms: {bedroom_count} • Bathrooms: {bathroom_count}",
                   ha='center', fontsize=12)
        
        # Save comparison
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        plt.savefig(os.path.join(output_dir, f"comparison_{i+1}.png"), dpi=150)
        plt.close()
        
        print(f"Generated comparison for sample {i+1}")
    
    print(f"Evaluation complete! Check {output_dir} directory for results")

def generate_varying_requirements():
    """Generate floor plans with varying requirements."""
    generator = load_trained_generator()
    if generator is None:
        return
    
    # Create output directory
    output_dir = "variation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define variations to test
    variations = [
        {"bedrooms": 1, "bathrooms": 1, "lounge": 1, "drawing_room": 1, "kitchen": 1},
        {"bedrooms": 2, "bathrooms": 2, "lounge": 1, "drawing_room": 1, "kitchen": 1},
        {"bedrooms": 3, "bathrooms": 2, "lounge": 1, "drawing_room": 1, "kitchen": 1},
        {"bedrooms": 2, "bathrooms": 3, "lounge": 1, "drawing_room": 1, "kitchen": 1},
        {"bedrooms": 3, "bathrooms": 3, "lounge": 0, "drawing_room": 1, "kitchen": 1}
    ]
    
    # Generate samples for each variation
    for i, params in enumerate(variations):
        # Create condition vector
        condition = np.zeros(Config.CONDITION_DIM, dtype=np.float32)
        condition[0] = params["bathrooms"] / 4.0
        condition[1] = params["bedrooms"] / 3.0
        condition[2] = min(params["drawing_room"], 1.0)
        condition[3] = min(params["kitchen"], 1.0)
        condition[4] = min(params["lounge"], 1.0)
        
        # Generate multiple samples with same condition
        plt.figure(figsize=(15, 10))
        title = f"B{params['bedrooms']}-BR{params['bathrooms']}"
        if params['lounge'] == 0:
            title += "-NoLounge"
        plt.suptitle(f"Generated Floor Plans with {title}", fontsize=16)
        
        for j in range(6):  # Generate 6 variations
            # Generate random noise
            noise = tf.random.normal([1, Config.LATENT_DIM])
            
            # Generate floor plan
            generated = generator([noise, np.expand_dims(condition, axis=0)], training=False)[0]
            
            # Convert from [-1, 1] to [0, 1]
            generated = (generated + 1.0) / 2.0
            
            # Plot combined visualization
            plt.subplot(2, 3, j+1)
            combined = np.zeros((generated.shape[0], generated.shape[1], 3))
            combined[:, :, 0] = generated[:, :, 0]  # Walls in red channel
            combined[:, :, 1] = generated[:, :, 1]  # Room areas in green channel
            plt.imshow(combined)
            plt.title(f"Variation {j+1}")
            plt.axis('off')
        
        # Add description
        bedroom_str = "Bedroom" if params["bedrooms"] == 1 else "Bedrooms"
        bathroom_str = "Bathroom" if params["bathrooms"] == 1 else "Bathrooms"
        plt.figtext(0.5, 0.01, 
                   f"{params['bedrooms']} {bedroom_str}, {params['bathrooms']} {bathroom_str}, " +
                   f"Drawing Room: {'Yes' if params['drawing_room'] else 'No'}, " +
                   f"Kitchen: {'Yes' if params['kitchen'] else 'No'}, " +
                   f"Lounge: {'Yes' if params['lounge'] else 'No'}",
                   ha='center', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        plt.savefig(os.path.join(output_dir, f"variations_{title}.png"), dpi=150)
        plt.close()
        
        print(f"Generated variations for configuration {i+1}: {title}")
    
    print(f"Variation analysis complete! Check {output_dir} directory for results")

if __name__ == "__main__":
    print("Running evaluation...")
    try:
        compare_real_vs_generated(num_samples=3)  # Start with just a few samples
        generate_varying_requirements()
    except Exception as e:
        print(f"Evaluation failed with error: {e}")