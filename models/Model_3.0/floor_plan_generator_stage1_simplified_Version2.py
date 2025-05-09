import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Data parameters
    DATA_DIR = "dataset/"
    METADATA_PATH = "floor_plan_metadata_v5_includingArea.csv"
    IMAGE_WIDTH = 849
    IMAGE_HEIGHT = 1570
    PLOT_SIZE = "10Marla"
    
    # Model parameters - SIMPLIFIED!
    LATENT_DIM = 100
    CONDITION_DIM = 16
    
    # Smaller model dimensions for reduced memory usage
    TARGET_SIZE = 128  # Using a small square target size
    
    # Training parameters
    BATCH_SIZE = 2  # Very small batch size to conserve memory
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    BETA_2 = 0.999
    
    # Output directories
    CHECKPOINT_DIR = "checkpoints/stage1/"
    LOG_DIR = "logs/stage1/"
    SAMPLE_DIR = "samples/stage1/"

# Create directories if they don't exist
for dir_path in [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.SAMPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data Loading and Preprocessing
def load_metadata():
    """Load and filter metadata for 10 Marla plans."""
    df = pd.read_csv(Config.METADATA_PATH)
    df = df[df['PlotSize'] == Config.PLOT_SIZE]
    # Use only original floor plans (V01) for training to avoid duplicates
    df = df[df['Version'] == 'V01']
    return df

def preprocess_image(image_path):
    """Load and preprocess image for boundary detection - SIMPLIFIED."""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        
        # Extract walls (black pixels)
        walls = np.all(img == [0, 0, 0], axis=-1).astype(np.float32)
        
        # Extract room boundaries (any colored area)
        rooms = np.any(img > 0, axis=-1).astype(np.float32)
        
        # Combine: 1 channel for walls, 1 for room areas
        boundary_map = np.stack([walls, rooms], axis=-1)
        
        # Resize to small target size
        boundary_map = cv2.resize(boundary_map, 
                                (Config.TARGET_SIZE, Config.TARGET_SIZE),
                                interpolation=cv2.INTER_NEAREST)
        
        # Normalize to [-1, 1]
        boundary_map = boundary_map * 2.0 - 1.0
        
        return boundary_map
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return empty array of correct shape in case of error
        return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, 2), dtype=np.float32)

def prepare_condition_vector(row):
    """Extract conditional inputs from metadata - SIMPLIFIED."""
    # Get important room counts
    room_counts = []
    
    # Extract bathroom count (normalized)
    room_counts.append(row.get('Count_Bathroom', 0) / 4.0)
    
    # Extract bedroom count (normalized)
    room_counts.append(row.get('Count_Bedroom', 0) / 3.0)
    
    # Extract other important rooms (presence as 0 or 1)
    room_types = ['DrawingRoom', 'Kitchen', 'Dining', 'Lounge', 'Garage']
    for room in room_types:
        count_col = f'Count_{room}'
        if count_col in row and not pd.isna(row[count_col]):
            room_counts.append(min(float(row[count_col]), 1.0))
        else:
            room_counts.append(0.0)
    
    # Total area (normalized)
    room_counts.append(row.get('TotalAreaSqFt', 2275.0) / 2275.0)
    
    # Convert to numpy array and pad/truncate to desired length
    condition_vector = np.array(room_counts, dtype=np.float32)
    
    if len(condition_vector) < Config.CONDITION_DIM:
        padding = np.zeros(Config.CONDITION_DIM - len(condition_vector), dtype=np.float32)
        condition_vector = np.concatenate([condition_vector, padding])
    else:
        condition_vector = condition_vector[:Config.CONDITION_DIM]
    
    return condition_vector

def data_generator(dataframe, batch_size):
    """Generator function for batches of training data."""
    num_samples = len(dataframe)
    while True:
        # Shuffle DataFrame at the start of each epoch
        df_shuffled = dataframe.sample(frac=1.0, random_state=np.random.randint(1000))
        
        for i in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[i:i+batch_size]
            
            # Prepare batch data
            batch_images = []
            batch_conditions = []
            
            for _, row in batch_df.iterrows():
                # Construct image path
                image_path = os.path.join(Config.DATA_DIR, row['FilePath'])
                
                # Load and preprocess image
                img = preprocess_image(image_path)
                batch_images.append(img)
                
                # Extract condition vector
                condition = prepare_condition_vector(row)
                batch_conditions.append(condition)
            
            # Convert to arrays
            batch_images = np.array(batch_images)
            batch_conditions = np.array(batch_conditions)
            
            yield batch_images, batch_conditions

# Model Architecture - SIMPLIFIED!
def build_generator():
    """Very simple generator for floor plan boundaries."""
    # Latent vector input
    noise_input = layers.Input(shape=(Config.LATENT_DIM,), name='noise_input')
    
    # Conditional input (room counts, etc.)
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')
    
    # Combine noise and condition
    combined_input = layers.Concatenate()([noise_input, condition_input])
    
    # Start with a dense layer to create initial feature map
    x = layers.Dense(8 * 8 * 256)(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)
    
    # Upsampling blocks
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x) # 16x16
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x) # 32x32
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x) # 64x64
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(16, 4, strides=2, padding='same')(x) # 128x128
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer - 2 channels for walls and room areas
    x = layers.Conv2D(2, 4, padding='same', activation='tanh')(x)
    
    return models.Model([noise_input, condition_input], x, name='generator')

def build_discriminator():
    """Simple discriminator for floor plan boundaries."""
    # Image input
    image_input = layers.Input(shape=(Config.TARGET_SIZE, Config.TARGET_SIZE, 2), 
                              name='image_input')
    
    # Conditional input
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')
    
    # Process image
    x = layers.Conv2D(16, 4, strides=2, padding='same')(image_input) # 64x64
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(32, 4, strides=2, padding='same')(x) # 32x32
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x) # 16x16
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x) # 8x8
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Flatten and process condition
    x = layers.Flatten()(x)
    
    # Process condition
    condition_features = layers.Dense(32)(condition_input)
    condition_features = layers.LeakyReLU(alpha=0.2)(condition_features)
    
    # Combine
    combined = layers.Concatenate()([x, condition_features])
    
    # Output
    x = layers.Dense(1)(combined)
    
    return models.Model([image_input, condition_input], x, name='discriminator')

# Simple GAN without WGAN-GP for easier implementation
class FloorPlanGAN(models.Model):
    def __init__(self, generator, discriminator):
        super(FloorPlanGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, gen_optimizer, disc_optimizer):
        super(FloorPlanGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        # Use standard binary cross-entropy loss for simplicity
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    def train_step(self, data):
        real_images, conditions = data
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise
        noise = tf.random.normal([batch_size, Config.LATENT_DIM])
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            fake_images = self.generator([noise, conditions], training=True)
            
            # Get discriminator outputs
            real_output = self.discriminator([real_images, conditions], training=True)
            fake_output = self.discriminator([fake_images, conditions], training=True)
            
            # Calculate losses
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Get gradients and update discriminator
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator([noise, conditions], training=True)
            
            # Get discriminator output for fake images
            fake_output = self.discriminator([fake_images, conditions], training=True)
            
            # Calculate generator loss - want discriminator to think fakes are real
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        
        # Get gradients and update generator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {
            "d_loss": disc_loss,
            "g_loss": gen_loss
        }

# Sample Generation Function
def generate_samples(generator, condition_samples, epoch):
    """Generate and save sample floor plans."""
    noise = tf.random.normal([len(condition_samples), Config.LATENT_DIM])
    generated_images = generator([noise, condition_samples], training=False)
    
    # Convert from [-1, 1] to [0, 1] for visualization
    generated_images = (generated_images + 1) / 2
    
    plt.figure(figsize=(12, 10))
    
    for i in range(min(8, len(generated_images))):
        # Plot walls (channel 0)
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.title('Walls')
        plt.axis('off')
        
        # Plot room areas (channel 1)
        plt.subplot(4, 4, i+9)
        plt.imshow(generated_images[i, :, :, 1], cmap='jet')
        plt.title('Room Areas')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAMPLE_DIR, f'samples_epoch_{epoch:03d}.png'))
    plt.close()

# Training Script
def train():
    # Enable eager execution for better error messages
    print("Configuring TensorFlow...")
    tf.config.run_functions_eagerly(True)
    
    print("Loading metadata...")
    df = load_metadata()
    print(f"Found {len(df)} floor plans for {Config.PLOT_SIZE}")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    
    # Create data generators
    train_gen = data_generator(train_df, Config.BATCH_SIZE)
    val_gen = data_generator(val_df, Config.BATCH_SIZE)
    
    # Calculate steps per epoch
    train_steps = max(1, len(train_df) // Config.BATCH_SIZE)
    val_steps = max(1, len(val_df) // Config.BATCH_SIZE)
    
    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Print model summaries
    generator.summary()
    discriminator.summary()
    
    # Create GAN model
    gan = FloorPlanGAN(generator, discriminator)
    
    # Compile with optimizers
    gen_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                   beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    disc_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                    beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    
    gan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
    
    # Create sample condition vectors for visualization
    sample_conditions = []
    for _, row in val_df.iloc[:4].iterrows():
        sample_conditions.append(prepare_condition_vector(row))
    sample_conditions = np.array(sample_conditions)
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Training
        for step in range(train_steps):
            try:
                real_images, conditions = next(train_gen)
                losses = gan.train_step([real_images, conditions])
                
                if step % 5 == 0:
                    print(f"Step {step}/{train_steps} - "
                          f"d_loss: {losses['d_loss']:.4f}, "
                          f"g_loss: {losses['g_loss']:.4f}")
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Generate samples at the end of each epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            try:
                generate_samples(generator, sample_conditions, epoch + 1)
            except Exception as e:
                print(f"Error generating samples: {e}")
        
        # Save model weights periodically
        if (epoch + 1) % 10 == 0:
            try:
                generator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'generator_epoch_{epoch+1:03d}.h5'))
                discriminator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1:03d}.h5'))
                print(f"Saved model checkpoint at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
    
    # Save final model
    try:
        generator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_generator'))
        discriminator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_discriminator'))
        print("Training complete! Final models saved.")
    except Exception as e:
        print(f"Error saving final models: {e}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")