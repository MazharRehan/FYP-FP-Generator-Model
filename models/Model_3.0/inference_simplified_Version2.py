import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from floor_plan_generator_stage1_simplified import Config, build_generator

def generate_floor_plan(bedroom_count, bathroom_count, drawing_room=1, kitchen=1, lounge=1):
    """Generate a floor plan with specified requirements."""
    # Build the generator
    generator = build_generator()
    
    # Load weights if available, otherwise warn that model needs training
    weights_path = os.path.join(Config.CHECKPOINT_DIR, 'final_generator')
    if os.path.exists(weights_path):
        try:
            generator.load_weights(weights_path + '/variables/variables')
            print("Loaded trained model weights!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using untrained model - results will be random!")
    else:
        print("Warning: No trained model found. Using untrained model - results will be random!")
    
    # Create condition vector
    condition = np.zeros(Config.CONDITION_DIM, dtype=np.float32)
    
    # Set room counts
    condition[0] = bathroom_count / 4.0  # Normalized bathroom count
    condition[1] = bedroom_count / 3.0   # Normalized bedroom count
    condition[2] = min(drawing_room, 1.0)  # Drawing room presence
    condition[3] = min(kitchen, 1.0)      # Kitchen presence
    condition[4] = min(lounge, 1.0)       # Lounge presence
    
    # Generate multiple samples with different noise vectors
    num_samples = 4
    plt.figure(figsize=(16, 12))
    
    for i in range(num_samples):
        # Generate random noise
        noise = tf.random.normal([1, Config.LATENT_DIM])
        
        # Generate floor plan
        generated = generator([noise, np.expand_dims(condition, axis=0)], training=False)[0]
        
        # Convert from [-1, 1] to [0, 1]
        generated = (generated + 1.0) / 2.0
        
        # Plot walls
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(generated[:, :, 0], cmap='gray')
        plt.title(f"Sample {i+1} - Walls")
        plt.axis('off')
        
        # Plot rooms
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(generated[:, :, 1], cmap='jet')
        plt.title(f"Sample {i+1} - Rooms")
        plt.axis('off')
        
        # Plot combined visualization
        plt.subplot(num_samples, 3, i*3 + 3)
        combined = np.zeros((generated.shape[0], generated.shape[1], 3))
        combined[:, :, 0] = generated[:, :, 0]  # Walls in red channel
        combined[:, :, 1] = generated[:, :, 1]  # Room areas in green channel
        plt.imshow(combined)
        plt.title(f"Sample {i+1} - Combined")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs("generated", exist_ok=True)
    
    # Save the figure
    plt.savefig(f"generated/floor_plan_B{bedroom_count}_BR{bathroom_count}.png")
    plt.close()
    
    print(f"Floor plan generated with {bedroom_count} bedrooms and {bathroom_count} bathrooms")
    print(f"Results saved to generated/floor_plan_B{bedroom_count}_BR{bathroom_count}.png")

if __name__ == "__main__":
    # Generate example floor plans with different requirements
    generate_floor_plan(bedroom_count=2, bathroom_count=2)
    generate_floor_plan(bedroom_count=3, bathroom_count=3)
    generate_floor_plan(bedroom_count=2, bathroom_count=2, lounge=2)