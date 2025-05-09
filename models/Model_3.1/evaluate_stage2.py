import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from floor_plan_generator_stage2 import Config, build_generator, load_metadata, preprocess_image, prepare_condition_vector, visualize_room_types
from sklearn.model_selection import train_test_split

def load_trained_generator():
    """Load the trained generator model."""
    generator = build_generator()
    try:
        generator.load_weights(os.path.join(Config.CHECKPOINT_DIR, 'final_generator'))
        print("Loaded trained generator model successfully!")
        return generator
    except Exception as e:
        print(f"Error loading generator: {e}")
        return None

def calculate_room_accuracy(real_image, generated_image):
    """Calculate room type prediction accuracy."""
    # Get most likely room type for each pixel
    real_room_types = np.argmax(real_image, axis=-1)
    gen_room_types = np.argmax(generated_image, axis=-1)
    
    # Calculate accuracy
    matching_pixels = np.sum(real_room_types == gen_room_types)
    total_pixels = real_image.shape[0] * real_image.shape[1]
    
    return matching_pixels / total_pixels

def evaluate_model(num_samples=5):
    """Evaluate the model's performance on test samples."""
    print("Evaluating Stage 2 model...")
    
    # Create output directory
    output_dir = "evaluation_stage2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata and split for testing
    df = load_metadata()
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_df = test_df.iloc[:num_samples]  # Use only a subset for evaluation
    
    # Load generator
    generator = load_trained_generator()
    if generator is None:
        return
    
    # Track metrics
    accuracies = []
    
    # Evaluate on test samples
    for i, (_, row) in enumerate(test_df.iterrows()):
        # Load real image
        image_path = os.path.join(Config.DATA_DIR, row['FilePath'])
        real_image = preprocess_image(image_path)
        
        # Extract condition vector
        condition = prepare_condition_vector(row)
        condition = np.expand_dims(condition, axis=0)
        
        # Generate floor plan
        noise = tf.random.normal([1, Config.LATENT_DIM])
        generated_image = generator([noise, condition], training=False)[0].numpy()
        
        # Calculate accuracy
        accuracy = calculate_room_accuracy(real_image, generated_image)
        accuracies.append(accuracy)
        
        # Create visualization
        plt.figure(figsize=(18, 8))
        plt.suptitle(f"Sample {i+1}: Real vs. Generated Floor Plan (Accuracy: {accuracy:.2%})", fontsize=16)
        
        # Real floor plan
        plt.subplot(1, 2, 1)
        real_rgb = visualize_room_types(real_image)
        plt.imshow(real_rgb)
        plt.title("Real Floor Plan")
        plt.axis('off')
        
        # Generated floor plan
        plt.subplot(1, 2, 2)
        generated_rgb = visualize_room_types(generated_image)
        plt.imshow(generated_rgb)
        plt.title("Generated Floor Plan")
        plt.axis('off')
        
        # Save comparison
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i+1}.png"), dpi=150)
        plt.close()
        
        # Create detailed room type visualization
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Sample {i+1}: Room Type Comparison", fontsize=16)
        
        # Select key room types to visualize
        key_rooms = ["Wall", "Bathroom", "Bedroom", "Kitchen", "DrawingRoom"]
        
        for j, room in enumerate(key_rooms):
            room_idx = Config.ROOM_TYPES.index(room)
            
            # Real room probability
            plt.subplot(2, 5, j+1)
            plt.imshow(real_image[:, :, room_idx], cmap='jet')
            plt.title(f"Real - {room}")
            plt.axis('off')
            
            # Generated room probability
            plt.subplot(2, 5, j+6)
            plt.imshow(generated_image[:, :, room_idx], cmap='jet')
            plt.title(f"Generated - {room}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"room_types_{i+1}.png"), dpi=150)
        plt.close()
        
        print(f"Processed sample {i+1}: Accuracy = {accuracy:.2%}")
    
    # Calculate and report overall metrics
    avg_accuracy = np.mean(accuracies)
    print(f"\nEvaluation complete!")
    print(f"Average room type accuracy: {avg_accuracy:.2%}")
    print(f"Evaluation results saved in {output_dir}")

if __name__ == "__main__":
    evaluate_model(num_samples=5)