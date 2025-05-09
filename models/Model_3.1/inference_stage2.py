import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from floor_plan_generator_stage2 import Config, build_generator, visualize_room_types

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

def generate_floor_plan(bedroom_count, bathroom_count, drawing_room=1, kitchen=1, lounge=1, dining=0, garage=0):
    """Generate a floor plan with specified requirements."""
    # Create output directory
    output_dir = "generated_stage2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build and load the generator
    generator = load_trained_generator()
    if generator is None:
        print("Failed to load generator model. Please train the model first.")
        return
    
    # Create condition vector
    condition = np.zeros(Config.CONDITION_DIM, dtype=np.float32)
    
    # Set plot size indicator (10 Marla)
    condition[0] = 1.0
    
    # Set room counts (normalized based on max counts)
    room_indices = {
        "Bathroom": 1,
        "Bedroom": 2,
        "Dining": 3,
        "DrawingRoom": 4,
        "DressingArea": 5,
        "Kitchen": 6,
        "Lounge": 7,
        "Store": 8,
        "Garage": 9,
        "Backyard": 10
    }
    
    max_counts = {
        "Bathroom": 4,
        "Bedroom": 3,
        "Dining": 1,
        "DrawingRoom": 1,
        "Kitchen": 1,
        "Lounge": 1,
        "Garage": 1
    }
    
    # Set room count requirements
    condition[room_indices["Bathroom"]] = bathroom_count / max_counts["Bathroom"]
    condition[room_indices["Bedroom"]] = bedroom_count / max_counts["Bedroom"]
    condition[room_indices["DrawingRoom"]] = drawing_room
    condition[room_indices["Kitchen"]] = kitchen
    condition[room_indices["Lounge"]] = lounge
    condition[room_indices["Dining"]] = dining
    condition[room_indices["Garage"]] = garage
    
    # Generate multiple samples with different noise vectors
    num_samples = 6
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Generated Floor Plans: {bedroom_count} Bedrooms, {bathroom_count} Bathrooms", fontsize=16)
    
    generated_plans = []
    
    for i in range(num_samples):
        # Generate random noise
        noise = tf.random.normal([1, Config.LATENT_DIM])
        
        # Generate floor plan
        generated = generator([noise, np.expand_dims(condition, axis=0)], training=False)[0]
        generated_plans.append(generated)
        
        # Convert room probabilities to RGB visualization
        rgb_floor_plan = visualize_room_types(generated)
        
        # Plot
        plt.subplot(2, 3, i+1)
        plt.imshow(rgb_floor_plan)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    # Save combined figure
    file_name = f"B{bedroom_count}_BR{bathroom_count}"
    if dining: file_name += "_D"
    if garage: file_name += "_G"
    if not lounge: file_name += "_NL"
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), dpi=150)
    plt.close()
    
    # Create a detailed view of the best sample (sample 0 for simplicity)
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Detailed Room Types: {bedroom_count} Bedrooms, {bathroom_count} Bathrooms", fontsize=16)
    
    # Full floor plan
    plt.subplot(2, 5, 3)
    rgb_floor_plan = visualize_room_types(generated_plans[0])
    plt.imshow(rgb_floor_plan)
    plt.title("Complete Floor Plan")
    plt.axis('off')
    
    # Key room type probabilities
    key_rooms = ["Wall", "Bathroom", "Bedroom", "Kitchen", "DrawingRoom", "Lounge", "Dining", "Garage"]
    for i, room in enumerate(key_rooms):
        room_idx = Config.ROOM_TYPES.index(room)
        
        # Skip rooms not requested
        if room == "Lounge" and lounge == 0:
            continue
        if room == "Dining" and dining == 0:
            continue
        if room == "Garage" and garage == 0:
            continue
            
        plt_idx = i+1 if i < 3 else i+2  # Skip the middle position used for the full plan
        if plt_idx > 5: plt_idx += 5      # Move to second row
        
        plt.subplot(2, 5, plt_idx)
        plt.imshow(generated_plans[0][:, :, room_idx], cmap='jet')
        plt.title(f"{room}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, f"{file_name}_details.png"), dpi=150)
    plt.close()
    
    print(f"Generated floor plans with {bedroom_count} bedrooms and {bathroom_count} bathrooms")
    print(f"Results saved to {output_dir}/{file_name}.png")

if __name__ == "__main__":
    # Generate floor plans with different requirements
    print("Generating floor plans with varying room requirements...")
    
    # Common configurations
    generate_floor_plan(bedroom_count=2, bathroom_count=2)
    generate_floor_plan(bedroom_count=3, bathroom_count=2)
    generate_floor_plan(bedroom_count=3, bathroom_count=3)
    
    # With dining room
    generate_floor_plan(bedroom_count=2, bathroom_count=2, dining=1)
    
    # With garage
    generate_floor_plan(bedroom_count=2, bathroom_count=2, garage=1)
    
    # Larger configuration
    generate_floor_plan(bedroom_count=3, bathroom_count=3, dining=1, garage=1)
    
    # Without lounge
    generate_floor_plan(bedroom_count=2, bathroom_count=1, lounge=0)
    
    print("All floor plans generated successfully!")