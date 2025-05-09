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

    # Model parameters
    LATENT_DIM = 128
    CONDITION_DIM = 32
    INITIAL_DIM = 64  # Initial filter size

    # Initial smaller dimensions for stage 1 - using powers of 2 for clean division
    GEN_INPUT_WIDTH = 128  # Must be divisible by 16 (2^4)
    GEN_INPUT_HEIGHT = 128  # Must be divisible by 16 (2^4)

    # Training parameters
    BATCH_SIZE = 4  # Reduced batch size for better stability
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5  # Adam optimizer parameter
    BETA_2 = 0.999  # Adam optimizer parameter

    # WGAN-GP parameters
    GP_WEIGHT = 10.0
    CRITIC_ITERATIONS = 5

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
    """Load and preprocess image for boundary detection."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # Extract walls (black pixels)
    walls = np.all(img == [0, 0, 0], axis=-1).astype(np.float32)

    # Extract room boundaries (any colored area)
    rooms = np.any(img > 0, axis=-1).astype(np.float32)

    # Combine: 1 channel for walls, 1 for room areas
    boundary_map = np.stack([walls, rooms], axis=-1)

    # Resize to target dimensions (use exact dimensions that match our model architecture)
    target_height = Config.GEN_INPUT_HEIGHT * 8  # Final output size
    target_width = Config.GEN_INPUT_WIDTH * 8

    boundary_map = cv2.resize(boundary_map,
                              (target_width, target_height),
                              interpolation=cv2.INTER_NEAREST)

    # Normalize to [-1, 1]
    boundary_map = boundary_map * 2.0 - 1.0

    return boundary_map

def prepare_condition_vector(row):
    """Extract conditional inputs from metadata."""
    # Get room counts from the row
    room_counts = []
    
    # Extract bathroom count
    room_counts.append(row['Count_Bathroom'] / 4.0)  # Normalize by max count
    
    # Extract bedroom count
    room_counts.append(row['Count_Bedroom'] / 3.0)  # Normalize by max count
    
    # Extract other important rooms (presence as 0 or 1)
    room_types = ['DrawingRoom', 'Kitchen', 'Dining', 'Lounge', 'Garage']
    for room in room_types:
        count_col = f'Count_{room}'
        if count_col in row:
            room_counts.append(min(row[count_col], 1.0))  # Binary presence
        else:
            room_counts.append(0.0)
    
    # Total area (normalized)
    room_counts.append(row['TotalAreaSqFt'] / 2275.0)  # Normalize by max area
    
    # Convert to numpy array
    condition_vector = np.array(room_counts, dtype=np.float32)
    
    # Pad to condition dimension if necessary
    if len(condition_vector) < Config.CONDITION_DIM:
        padding = np.zeros(Config.CONDITION_DIM - len(condition_vector), dtype=np.float32)
        condition_vector = np.concatenate([condition_vector, padding])
    
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

# Model Architecture
def build_generator():
    """U-Net-based generator with conditional inputs."""
    # Latent vector input
    noise_input = layers.Input(shape=(Config.LATENT_DIM,), name='noise_input')

    # Conditional input (room counts, etc.)
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')

    # Combine noise and condition
    combined_input = layers.Concatenate()([noise_input, condition_input])

    # Calculate sizes to ensure symmetrical downsampling and upsampling
    # Make sure these divide evenly by 2^num_downsamples
    target_height = Config.GEN_INPUT_HEIGHT
    target_width = Config.GEN_INPUT_WIDTH

    # Make sure dimensions are powers of 2 for clean division
    initial_height = target_height
    initial_width = target_width

    # First dense layer to create initial feature map
    x = layers.Dense(initial_height * initial_width * Config.INITIAL_DIM)(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((initial_height, initial_width, Config.INITIAL_DIM))(x)

    # Save the shapes for the skip connections
    skip_shapes = []
    skip_connections = []

    # Encoder path
    filter_sizes = [Config.INITIAL_DIM * mult for mult in [1, 2, 4, 8]]

    # Encoder blocks - save input shape before each convolution
    for filters in filter_sizes:
        # Save for skip connection
        skip_connections.append(x)
        skip_shapes.append((x.shape[1], x.shape[2]))  # Height, Width

        # Conv block
        x = layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    # Bottleneck
    # Inject condition at bottleneck
    condition_dense = layers.Dense(filter_sizes[-1])(condition_input)
    condition_dense = layers.LeakyReLU(0.2)(condition_dense)
    condition_dense = layers.Dense(int(x.shape[1]) * int(x.shape[2]) * filter_sizes[-1])(condition_dense)
    condition_reshaped = layers.Reshape((int(x.shape[1]), int(x.shape[2]), filter_sizes[-1]))(condition_dense)
    x = layers.Concatenate()([x, condition_reshaped])

    # Extra convolution at bottleneck
    x = layers.Conv2D(filter_sizes[-1] * 2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Decoder blocks with skip connections
    # We'll use the stored shapes to ensure proper dimensions
    for i, filters in enumerate(reversed(filter_sizes)):
        # Print shape information for debugging
        print(f"Upsampling to shape: {skip_shapes[-(i + 1)]}")

        # Upsample to match skip connection dimensions
        if i == 0:
            x = layers.Conv2DTranspose(
                filters, 4, strides=2,
                padding='same',
                output_padding=(1, 1)  # Only include if necessary
            )(x)
        else:
            x = layers.Conv2DTranspose(
                filters, 4, strides=2,
                padding='same'  # No output_padding when it's (0, 0)
            )(x)

        # Ensure exact shape match before concatenation
        if x.shape[1:3] != skip_connections[-(i + 1)].shape[1:3]:
            target_h, target_w = skip_shapes[-(i + 1)]
            x = layers.Resizing(target_h, target_w)(x)

        # Add skip connection
        x = layers.Concatenate()([x, skip_connections[-(i + 1)]])

    # Final upsampling to target size
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer - 2 channels (walls, room areas)
    x = layers.Conv2D(2, 4, padding='same', activation='tanh')(x)

    return models.Model([noise_input, condition_input], x, name='generator')

def build_discriminator():
    """PatchGAN discriminator with conditional input."""
    # Image input - match generator output size
    image_height = Config.GEN_INPUT_HEIGHT * 8
    image_width = Config.GEN_INPUT_WIDTH * 8

    # Image input
    image_input = layers.Input(shape=(image_height, image_width, 2),
                               name='image_input')

    # Conditional input
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')

    # Process conditional input
    condition_dense = layers.Dense(image_height * image_width)(condition_input)
    condition_reshaped = layers.Reshape((image_height, image_width, 1))(condition_dense)

    # Concatenate image with condition
    x = layers.Concatenate()([image_input, condition_reshaped])

    # Downsampling layers
    filter_sizes = [64, 128, 256, 512]

    for filters in filter_sizes:
        x = layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        # Use layer normalization instead of batch normalization for WGAN-GP
        x = layers.LayerNormalization()(x)

    # Output layer (no sigmoid for WGAN)
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return models.Model([image_input, condition_input], x, name='discriminator')

# WGAN-GP Functions
def wasserstein_loss(y_true, y_pred):
    """Wasserstein loss function."""
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_images, fake_images, conditions):
    """Gradient penalty for WGAN-GP."""
    batch_size = tf.shape(real_images)[0]
    
    # Generate random interpolation factors
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    
    # Create interpolated images
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Get discriminator output for interpolated images
        pred = discriminator([interpolated, conditions])
    
    # Calculate gradients with respect to inputs
    gradients = tape.gradient(pred, interpolated)
    # Compute the Euclidean norm of the gradients
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    # Calculate the gradient penalty
    gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))
    
    return gradient_penalty

class FloorPlanGAN(models.Model):
    def __init__(self, generator, discriminator):
        super(FloorPlanGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, gen_optimizer, disc_optimizer):
        super(FloorPlanGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        
    def train_step(self, data):
        real_images, conditions = data
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator multiple times
        for i in range(Config.CRITIC_ITERATIONS):
            # Generate random noise
            noise = tf.random.normal([batch_size, Config.LATENT_DIM])
            
            with tf.GradientTape() as tape:
                # Generate fake images
                fake_images = self.generator([noise, conditions], training=True)
                
                # Get discriminator outputs for real and fake images
                real_output = self.discriminator([real_images, conditions], training=True)
                fake_output = self.discriminator([fake_images, conditions], training=True)
                
                # Calculate Wasserstein loss
                d_loss_real = -tf.reduce_mean(real_output)
                d_loss_fake = tf.reduce_mean(fake_output)
                
                # Calculate gradient penalty
                gp = gradient_penalty(self.discriminator, real_images, fake_images, conditions)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake + Config.GP_WEIGHT * gp
            
            # Get gradients and update discriminator
            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        noise = tf.random.normal([batch_size, Config.LATENT_DIM])
        
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = self.generator([noise, conditions], training=True)
            
            # Get discriminator output for fake images
            fake_output = self.discriminator([fake_images, conditions], training=True)
            
            # Calculate generator loss
            g_loss = -tf.reduce_mean(fake_output)
        
        # Get gradients and update generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "gradient_penalty": gp,
            "g_loss": g_loss
        }

# Sample Generation Function
def generate_samples(generator, condition_samples, epoch):
    """Generate and save sample floor plans."""
    noise = tf.random.normal([len(condition_samples), Config.LATENT_DIM])
    generated_images = generator([noise, condition_samples], training=False)
    
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
    # Enable eager execution for better debugging
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
    train_steps = len(train_df) // Config.BATCH_SIZE
    val_steps = len(val_df) // Config.BATCH_SIZE
    
    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Print model summaries
    generator.summary()
    discriminator.summary()
    
    # Create WGAN model
    wgan = FloorPlanGAN(generator, discriminator)
    
    # Compile with optimizers
    gen_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                   beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    disc_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                    beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    
    wgan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(Config.CHECKPOINT_DIR, 'model_epoch_{epoch:03d}'),
        save_weights_only=True,
        save_freq=train_steps * 5  # Save every 5 epochs
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=Config.LOG_DIR,
        update_freq='epoch'
    )
    
    # Create sample condition vectors for visualization
    sample_conditions = []
    for _, row in val_df.iloc[:8].iterrows():
        sample_conditions.append(prepare_condition_vector(row))
    sample_conditions = np.array(sample_conditions)
    
    # Define custom callback for sample generation
    class SampleGenerationCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0 or epoch == 0:
                generate_samples(generator, sample_conditions, epoch + 1)
    
    sample_callback = SampleGenerationCallback()
    
    print("Starting training...")
    
    # Custom training loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Training
        for step in range(train_steps):
            real_images, conditions = next(train_gen)
            losses = wgan.train_step([real_images, conditions])
            
            if step % 10 == 0:
                print(f"Step {step}/{train_steps} - "
                      f"d_loss: {losses['d_loss']:.4f}, "
                      f"g_loss: {losses['g_loss']:.4f}")
        
        # Generate samples at the end of each epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generate_samples(generator, sample_conditions, epoch + 1)
        
        # Save model weights
        if (epoch + 1) % 5 == 0:
            generator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'generator_epoch_{epoch+1:03d}'))
            discriminator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1:03d}'))
    
    # Save final model
    generator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_generator'))
    discriminator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_discriminator'))
    
    print("Training complete!")

if __name__ == "__main__":
    train()