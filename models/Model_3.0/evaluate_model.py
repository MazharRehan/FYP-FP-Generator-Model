import os
import numpy as np
import tensorflow as tf
from floor_plan_generator_stage1 import Config, load_metadata, preprocess_image, prepare_condition_vector
from floor_plan_generator_stage1 import build_generator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_model():
    """Load the trained generator model."""
    # Build the generator
    generator = build_generator()
    
    # Load the weights
    generator.load_weights(os.path.join(Config.CHECKPOINT_DIR, 'final_generator'))
    
    return generator

def evaluate_model():
    """Evaluate the model on a test set."""
    # Load metadata
    df = load_metadata()
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Load the generator
    generator = load_model()
    
    # Create output directory
    output_dir = "evaluation/stage1/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on test samples
    for i, (_, row) in enumerate(test_df.iloc[:20].iterrows()):
        # Load real image
        image_path = os.path.join(Config.DATA_DIR, row['FilePath'])
        real_image = preprocess_image(image_path)
        
        # Extract condition vector
        condition = prepare_condition_vector(row)
        condition = np.expand_dims(condition, axis=0)
        
        # Generate fake image
        noise = tf.random.normal([1, Config.LATENT_DIM])
        fake_image = generator([noise, condition], training=False)[0]
        
        # Convert from [-1, 1] to [0, 1]
        real_image = (real_image + 1.0) / 2.0
        fake_image = (fake_image + 1.0) / 2.0
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Real image - walls
        plt.subplot(2, 4, 1)
        plt.imshow(real_image[:, :, 0], cmap='gray')
        plt.title("Real - Walls")
        plt.axis('off')
        
        # Real image - rooms
        plt.subplot(2, 4, 2)
        plt.imshow(real_image[:, :, 1], cmap='jet')
        plt.title("Real - Rooms")
        plt.axis('off')
        
        # Real image - combined
        plt.subplot(2, 4, 3)
        combined_real = np.zeros((real_image.shape[0], real_image.shape[1], 3))
        combined_real[:, :, 0] = real_image[:, :, 0]
        combined_real[:, :, 1] = real_image[:, :, 1]
        plt.imshow(combined_real)
        plt.title("Real - Combined")
        plt.axis('off')
        
        # Empty subplot
        plt.subplot(2, 4, 4)
        plt.axis('off')
        
        # Fake image - walls
        plt.subplot(2, 4, 5)
        plt.imshow(fake_image[:, :, 0], cmap='gray')
        plt.title("Generated - Walls")
        plt.axis('off')
        
        # Fake image - rooms
        plt.subplot(2, 4, 6)
        plt.imshow(fake_image[:, :, 1], cmap='jet')
        plt.title("Generated - Rooms")
        plt.axis('off')
        
        # Fake image - combined
        plt.subplot(2, 4, 7)
        combined_fake = np.zeros((fake_image.shape[0], fake_image.shape[1], 3))
        combined_fake[:, :, 0] = fake_image[:, :, 0]
        combined_fake[:, :, 1] = fake_image[:, :, 1]
        plt.imshow(combined_fake)
        plt.title("Generated - Combined")
        plt.axis('off')
        
        # Empty subplot
        plt.subplot(2, 4, 8)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{i:03d}.png'))
        plt.close()
    
    print(f"Evaluation complete. Results saved in {output_dir}")

if __name__ == "__main__":
    evaluate_model()