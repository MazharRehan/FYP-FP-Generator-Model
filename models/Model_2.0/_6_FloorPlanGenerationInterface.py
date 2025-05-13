import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def generate_floor_plan(generator_model_path, metadata_path, output_path,
                        plot_size, num_bedrooms, num_bathrooms, num_kitchens=1,
                        num_lounges=1, num_garages=1, latent_dim=100):
    """
    Generate a floor plan based on user requirements.

    Args:
        generator_model_path: Path to the trained generator model
        metadata_path: Path to the metadata CSV file
        output_path: Path to save the generated floor plan
        plot_size: Plot size (5Marla, 10Marla, or 20Marla)
        num_bedrooms: Number of bedrooms
        num_bathrooms: Number of bathrooms
        num_kitchens: Number of kitchens
        num_lounges: Number of lounges
        num_garages: Number of garages
        latent_dim: Dimension of latent space
    """
    # Load the generator model
    generator = tf.keras.models.load_model(generator_model_path)

    # Load metadata to get features structure
    metadata = pd.read_csv(metadata_path)
    room_count_cols = [col for col in metadata.columns if col.startswith('Count_')]

    # Create condition vector with zeros
    condition = np.zeros((1, len(room_count_cols)))

    # Set values for required rooms
    for i, col in enumerate(room_count_cols):
        room_type = col.replace('Count_', '')
        if room_type == 'Bedroom':
            condition[0, i] = num_bedrooms
        elif room_type == 'Bathroom':
            condition[0, i] = num_bathrooms
        elif room_type == 'Kitchen':
            condition[0, i] = num_kitchens
        elif room_type == 'Lounge':
            condition[0, i] = num_lounges
        elif room_type == 'Garage':
            condition[0, i] = num_garages

    # Generate random noise
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Generate the floor plan
    generated_image = generator.predict([noise, condition])

    # Scale from [-1, 1] to [0, 255]
    generated_image = ((generated_image[0] + 1) * 127.5).astype(np.uint8)

    # Save the generated floor plan
    plt.figure(figsize=(10, 10))
    plt.imshow(generated_image)
    plt.axis('off')
    plt.title(f"{plot_size} - {num_bedrooms} Bedrooms, {num_bathrooms} Bathrooms")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return generated_image

# Example usage
# generate_floor_plan(
#     'models/generator_final.h5',
#     'floor_plan_metadata_extended.csv',
#     'generated_floor_plan.png',
#     '10Marla', 3, 2, 1, 1, 1
# )