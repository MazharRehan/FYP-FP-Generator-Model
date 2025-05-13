# --- START OF FILE floor_plan_generator_stage1_Version3_AllData.py ---
"""
!pip install numpy>=2.1.0  tensorflow==2.18.0 matplotlib==3.10.0 scikit-learn==1.3.2 pandas==2.2.3 pillow==11.2.1 opencv-python==4.11.0.86
"""
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
import sys # Added for error exit

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Data parameters
    DATA_DIR = "/kaggle/input/floorplan/dataset/"
    METADATA_PATH = "/kaggle/input/floorplan/floor_plan_metadata_consolidated_v6.csv" # Use the latest CSV name

    # Model parameters - SIMPLIFIED!
    LATENT_DIM = 100
    # Adjust CONDITION_DIM based on the new condition vector structure
    # New structure: 3 (plot_size_one_hot) + 1 (bath_norm) + 1 (bed_norm) + 5 (presence) = 10
    CONDITION_DIM = 10 

    # Smaller model dimensions for reduced memory usage
    TARGET_SIZE = 128  # Using a small square target size (keep this for simplified model)
    IMAGE_CHANNELS = 2 # 1 for walls, 1 for room areas

    # Training parameters
    BATCH_SIZE = 4  # Slightly increase if memory allows, still small
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    BETA_2 = 0.999

    # Output directories
    CHECKPOINT_DIR = "checkpoints/stage1_all_data/"
    LOG_DIR = "logs/stage1_all_data/"
    SAMPLE_DIR = "samples/stage1_all_data/"

# Create directories if they don't exist
for dir_path in [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.SAMPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data Loading and Preprocessing
def load_metadata():
    """Load metadata for ALL floor plans."""
    try:
        df = pd.read_csv(Config.METADATA_PATH)
        # No filtering needed anymore
        # df = df[df['PlotSize'] == Config.PLOT_SIZE] # REMOVED
        # df = df[df['Version'] == 'V01'] # REMOVED
        print(f"Loaded metadata for {len(df)} floor plans.")
        # Basic check for required columns
        required_cols = ['PlotType', 'FileName', 'Count_Bathroom', 'Count_Bedroom']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Metadata CSV missing one or more required columns: {required_cols}")
            sys.exit(1) # Exit if essential columns are missing
        return df
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {Config.METADATA_PATH}")
        sys.exit(1) # Exit if metadata is missing
    except Exception as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1) # Exit on other loading errors


def preprocess_image(image_path):
    """Load and preprocess image for boundary detection - SIMPLIFIED."""
    try:
        # Load image using Pillow (robust to formats) then convert
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)

        # Check if image has 3 channels (RGB)
        if img.ndim != 3 or img.shape[2] != 3:
             print(f"Warning: Image {os.path.basename(image_path)} is not RGB, shape: {img.shape}. Skipping.")
             return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMAGE_CHANNELS), dtype=np.float32) - 1.0 # Return normalized zero array

        # Extract walls (black pixels [0, 0, 0])
        # Use a small tolerance in case of slight variations in black
        walls = np.all(img <= 5, axis=-1).astype(np.float32)

        # Extract room boundaries (any non-black and non-white pixel)
        # Assumes background is white [255, 255, 255]
        non_black = np.any(img > 5, axis=-1)
        non_white = np.any(img < 250, axis=-1)
        rooms = (non_black & non_white).astype(np.float32)

        # Combine: 1 channel for walls, 1 for room areas
        boundary_map = np.stack([walls, rooms], axis=-1)

        # Resize to small target size
        boundary_map = cv2.resize(boundary_map,
                                (Config.TARGET_SIZE, Config.TARGET_SIZE),
                                interpolation=cv2.INTER_NEAREST)

        # Ensure the resized map has the correct number of channels after resize
        if boundary_map.ndim == 2: # If resize somehow collapsed channels
             boundary_map = np.expand_dims(boundary_map, axis=-1)
             # If only one channel resulted, duplicate or handle appropriately
             if Config.IMAGE_CHANNELS == 2:
                  print(f"Warning: Resize resulted in single channel for {os.path.basename(image_path)}. Duplicating.")
                  boundary_map = np.concatenate([boundary_map, boundary_map], axis=-1)
             # Add padding if needed - this case should be rare with INTER_NEAREST on multi-channel input
             while boundary_map.shape[-1] < Config.IMAGE_CHANNELS:
                 boundary_map = np.concatenate([boundary_map, np.zeros_like(boundary_map[..., :1])], axis=-1)


        # Normalize to [-1, 1]
        boundary_map = boundary_map * 2.0 - 1.0

        return boundary_map
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMAGE_CHANNELS), dtype=np.float32) - 1.0
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return normalized zero array of correct shape in case of error
        return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMAGE_CHANNELS), dtype=np.float32) - 1.0

def prepare_condition_vector(row):
    """Extract conditional inputs from metadata - ADAPTED FOR ALL PLOT SIZES."""
    conditions = []

    # 1. One-hot encode plot size
    plot_size = row.get('PlotType', 'Unknown') # Use PlotType from CSV
    if plot_size == '5Marla':
        conditions.extend([1.0, 0.0, 0.0])
    elif plot_size == '10Marla':
        conditions.extend([0.0, 1.0, 0.0])
    elif plot_size == '20Marla':
        conditions.extend([0.0, 0.0, 1.0])
    else:
        conditions.extend([0.0, 0.0, 0.0]) # Vector for unknown/other

    # 2. Normalized room counts (adjust normalization factors based on potential max across all plot sizes)
    # Using slightly higher maxes observed or expected in larger plots
    conditions.append(row.get('Count_Bathroom', 0) / 6.0) # Max bathrooms ~6?
    conditions.append(row.get('Count_Bedroom', 0) / 7.0) # Max bedrooms ~7?

    # 3. Presence of other important rooms (0 or 1)
    room_types_presence = ['DrawingRoom', 'Kitchen', 'Dining', 'Lounge', 'Garage']
    for room in room_types_presence:
        count_col = f'Count_{room}'
        # Check if column exists and value is not NaN and greater than 0
        if count_col in row and pd.notna(row[count_col]) and row[count_col] > 0:
            conditions.append(1.0)
        else:
            conditions.append(0.0)

    # Convert to numpy array
    condition_vector = np.array(conditions, dtype=np.float32)

    # Ensure vector is exactly CONDITION_DIM long (pad/truncate)
    current_len = len(condition_vector)
    if current_len < Config.CONDITION_DIM:
        padding = np.zeros(Config.CONDITION_DIM - current_len, dtype=np.float32)
        condition_vector = np.concatenate([condition_vector, padding])
    elif current_len > Config.CONDITION_DIM:
        condition_vector = condition_vector[:Config.CONDITION_DIM]

    # Check final length
    if len(condition_vector) != Config.CONDITION_DIM:
         print(f"Error: Final condition vector length ({len(condition_vector)}) != Config.CONDITION_DIM ({Config.CONDITION_DIM}) for row: {row.get('FileName')}")
         # Return a zero vector of correct dimension on error
         return np.zeros(Config.CONDITION_DIM, dtype=np.float32)


    return condition_vector


def data_generator(dataframe, batch_size):
    """Generator function for batches of training data."""
    num_samples = len(dataframe)
    while True:
        # Shuffle DataFrame at the start of each epoch
        df_shuffled = dataframe.sample(frac=1.0).reset_index(drop=True) # Shuffle and reset index

        for i in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[i:min(i + batch_size, num_samples)] # Handle last batch potentially smaller

            if batch_df.empty: continue # Skip if batch is empty

            batch_images = []
            batch_conditions = []

            for _, row in batch_df.iterrows():
                # Construct image path using FileName column from CSV
                # Assumes FileName column contains the name like '5Marla_GF_FP_001_V01.png'
                # and DATA_DIR is the flat directory containing these files.
                image_filename = row.get('FileName')
                if pd.isna(image_filename):
                     print(f"Warning: Missing FileName in metadata row. Skipping.")
                     continue

                image_path = os.path.join(Config.DATA_DIR, image_filename)

                # Load and preprocess image
                img = preprocess_image(image_path)
                # Only append if preprocessing was successful (check shape/value range if needed)
                if img.shape == (Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMAGE_CHANNELS):
                     batch_images.append(img)

                     # Extract condition vector
                     condition = prepare_condition_vector(row)
                     batch_conditions.append(condition)
                else:
                     print(f"Warning: Preprocessed image has wrong shape {img.shape} for {image_filename}. Skipping sample.")


            # Only yield if batch is not empty after potential skips
            if batch_images and batch_conditions:
                 # Convert to arrays
                 batch_images = np.array(batch_images)
                 batch_conditions = np.array(batch_conditions)

                 # Check shapes before yielding
                 if batch_images.shape[0] == batch_conditions.shape[0] and batch_images.shape[0] > 0:
                      yield batch_images, batch_conditions
                 else:
                      print(f"Warning: Batch shape mismatch or empty batch. Images: {batch_images.shape}, Conditions: {batch_conditions.shape}. Skipping batch.")

# Model Architecture - SIMPLIFIED! (Updated input shapes)
def build_generator():
    """Very simple generator for floor plan boundaries."""
    noise_input = layers.Input(shape=(Config.LATENT_DIM,), name='noise_input')
    # Use updated CONDITION_DIM
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')

    combined_input = layers.Concatenate()([noise_input, condition_input])

    # Start with a dense layer
    # Adjust initial units based on TARGET_SIZE
    initial_size = Config.TARGET_SIZE // 16 # Assuming 4 upsampling blocks (2^4=16)
    if initial_size < 4: initial_size = 4 # Ensure minimum size
    x = layers.Dense(initial_size * initial_size * 256)(combined_input)
    x = layers.BatchNormalization()(x) # Added Batch Norm
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((initial_size, initial_size, 256))(x)

    # Upsampling blocks
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x) # size*2
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x) # size*4
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x) # size*8
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(16, 4, strides=2, padding='same')(x) # size*16 = TARGET_SIZE
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Output layer - IMAGE_CHANNELS for walls and room areas
    x = layers.Conv2D(Config.IMAGE_CHANNELS, 4, padding='same', activation='tanh')(x)

    return models.Model([noise_input, condition_input], x, name='generator')

def build_discriminator():
    """Simple discriminator for floor plan boundaries."""
    # Use IMAGE_CHANNELS
    image_input = layers.Input(shape=(Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMAGE_CHANNELS),
                              name='image_input')
    # Use updated CONDITION_DIM
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')

    # Process image
    x = layers.Conv2D(16, 4, strides=2, padding='same')(image_input) # 64x64
    # No Batch Norm typically in first layer of D
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(32, 4, strides=2, padding='same')(x) # 32x32
    x = layers.BatchNormalization()(x) # Added Batch Norm
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, 4, strides=2, padding='same')(x) # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x) # 8x8 or smaller
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Flatten image features
    x = layers.Flatten()(x)

    # Process condition input separately and prepare for concatenation
    # Map condition vector to a higher dimension before concat
    condition_features = layers.Dense(128)(condition_input) # Project conditions
    condition_features = layers.LeakyReLU(alpha=0.2)(condition_features)


    # Combine flattened image features and processed condition features
    combined = layers.Concatenate()([x, condition_features])

    # Dense layers for classification
    x = layers.Dense(256)(combined)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x) # Added dropout

    # Output layer (logits)
    x = layers.Dense(1)(x)

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
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function # Add tf.function decorator for potential performance improvement
    def train_step(self, data):
        real_images, conditions = data
        batch_size = tf.shape(real_images)[0]

        # Generate random noise
        noise = tf.random.normal([batch_size, Config.LATENT_DIM])

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            fake_images = self.generator([noise, conditions], training=True)

            # Add noise to real images only (label smoothing can also be used)
            real_images_noisy = real_images #+ tf.random.normal(tf.shape(real_images), mean=0.0, stddev=0.02)

            real_output = self.discriminator([real_images_noisy, conditions], training=True)
            fake_output = self.discriminator([fake_images, conditions], training=True)

            real_loss = self.loss_fn(tf.ones_like(real_output) * 0.9, real_output) # Label smoothing
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # Clip gradients if necessary (optional)
        # disc_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in disc_gradients]
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Train generator
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator([noise, conditions], training=True)
            fake_output = self.discriminator([fake_images, conditions], training=True)
            # Generator wants discriminator to output 1 (real)
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Clip gradients if necessary (optional)
        # gen_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gen_gradients]
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {
            "d_loss": disc_loss,
            "g_loss": gen_loss
        }

# Sample Generation Function
def generate_samples(generator, condition_samples, epoch):
    """Generate and save sample floor plans."""
    if condition_samples is None or len(condition_samples) == 0:
         print("Warning: No sample conditions provided for generating samples.")
         return

    noise = tf.random.normal([len(condition_samples), Config.LATENT_DIM])
    generated_images = generator([noise, condition_samples], training=False)

    # Convert from [-1, 1] to [0, 1] for visualization
    generated_images = (generated_images.numpy() + 1) / 2

    num_samples_to_show = min(8, len(generated_images)) # Show max 8 samples (4 pairs)
    if num_samples_to_show == 0: return # Skip if somehow no images generated

    plt.figure(figsize=(12, 6)) # Adjust figure size

    for i in range(num_samples_to_show):
        # Plot walls (channel 0)
        plt.subplot(2, num_samples_to_show, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray_r') # Use inverted gray cmap
        plt.title(f'Walls {i+1}')
        plt.axis('off')

        # Plot room areas (channel 1)
        plt.subplot(2, num_samples_to_show, i + num_samples_to_show + 1)
        plt.imshow(generated_images[i, :, :, 1], cmap='viridis') # Use a distinct cmap
        plt.title(f'Rooms {i+1}')
        plt.axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(Config.SAMPLE_DIR, f'samples_epoch_{epoch:04d}.png'))
    plt.close()
    print(f"Generated samples for epoch {epoch}")


# Training Script
def train():
    # Enable eager execution? Generally not recommended for performance with tf.function
    # tf.config.run_functions_eagerly(True) # Keep commented unless debugging step-by-step
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Set up a MirroredStrategy for multi-GPU support
    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: ', strategy.num_replicas_in_sync)

    # Create and compile your model within the strategy scope
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    print("Loading metadata...")
    df = load_metadata()

    # Split data into train and validation sets
    if len(df) < 2:
         print("Error: Not enough data samples to split into train/validation.")
         sys.exit(1)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")

    # Create data generators
    train_gen = data_generator(train_df, Config.BATCH_SIZE)
    val_gen = data_generator(val_df, Config.BATCH_SIZE) # Validation generator (optional for GANs)

    # Calculate steps per epoch
    train_steps = max(1, len(train_df) // Config.BATCH_SIZE)
    # val_steps = max(1, len(val_df) // Config.BATCH_SIZE) # Not typically used directly in GAN training loop

    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()

    print("--- Generator Summary ---")
    generator.summary()
    print("\n--- Discriminator Summary ---")
    discriminator.summary()

    # Create GAN model
    gan = FloorPlanGAN(generator, discriminator)

    # Compile with optimizers
    gen_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                   beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    disc_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                    beta_1=Config.BETA_1, beta_2=Config.BETA_2)

    gan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)

    # Create sample condition vectors for visualization from validation set
    sample_conditions = []
    num_samples_to_generate = 4 # Generate fewer samples
    for _, row in val_df.iloc[:num_samples_to_generate].iterrows():
        sample_conditions.append(prepare_condition_vector(row))

    if not sample_conditions:
         print("Warning: Could not create sample conditions from validation data.")
         sample_conditions = None # Handle this case in generate_samples
    else:
         sample_conditions = np.array(sample_conditions)
         print(f"Created {len(sample_conditions)} sample conditions for visualization.")

    print("\nStarting training...")

    # --- Training Loop ---
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        progbar = tf.keras.utils.Progbar(train_steps)
        d_loss_epoch = []
        g_loss_epoch = []

        for step in range(train_steps):
            try:
                real_images, conditions = next(train_gen)

                # Check if batch is valid before training step
                if real_images.shape[0] == 0 or conditions.shape[0] == 0 or real_images.shape[0] != conditions.shape[0]:
                     print(f"Warning: Invalid batch at step {step}. Skipping.")
                     continue

                losses = gan.train_step([real_images, conditions])
                d_loss_epoch.append(losses['d_loss'])
                g_loss_epoch.append(losses['g_loss'])

                progbar.update(step + 1, values=[("d_loss", losses['d_loss']), ("g_loss", losses['g_loss'])])

            except StopIteration:
                 print("Data generator exhausted for epoch. Resetting...")
                 # Reset generator if needed (shouldn't happen with infinite generator)
                 train_gen = data_generator(train_df, Config.BATCH_SIZE)
                 break # Move to next epoch if generator stops unexpectedly
            except Exception as e:
                print(f"\nError during training step {step}: {e}")
                # Optionally: break epoch or continue
                continue

        # Log average losses for the epoch
        avg_d_loss = np.mean(d_loss_epoch) if d_loss_epoch else 0
        avg_g_loss = np.mean(g_loss_epoch) if g_loss_epoch else 0
        print(f"Epoch {epoch+1} Avg Losses - d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}")


        # Generate samples periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            try:
                if sample_conditions is not None:
                     generate_samples(generator, sample_conditions, epoch + 1)
            except Exception as e:
                print(f"Error generating samples: {e}")

        # Save model weights periodically
        if (epoch + 1) % 10 == 0:
            try:
                gen_save_path = os.path.join(Config.CHECKPOINT_DIR, f'generator_epoch_{epoch+1:04d}.weights.h5')
                disc_save_path = os.path.join(Config.CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1:04d}.weights.h5')
                generator.save_weights(gen_save_path)
                discriminator.save_weights(disc_save_path)
                print(f"Saved model weights checkpoint at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving weights checkpoint: {e}")

    # Save final model weights
    try:
        generator.save_weights(os.path.join(Config.CHECKPOINT_DIR, 'generator_final.weights.h5'))
        discriminator.save_weights(os.path.join(Config.CHECKPOINT_DIR, 'discriminator_final.weights.h5'))
        # Save full models (optional, can be larger)
        # generator.save(os.path.join(Config.CHECKPOINT_DIR, 'generator_final_model'))
        # discriminator.save(os.path.join(Config.CHECKPOINT_DIR, 'discriminator_final_model'))
        print("\nTraining complete! Final model weights saved.")
    except Exception as e:
        print(f"Error saving final model weights: {e}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n--- Training failed with error ---")
        print(e)
        import traceback
        traceback.print_exc()


# --- END OF FILE floor_plan_generator_stage1_Version3_AllData.py ---