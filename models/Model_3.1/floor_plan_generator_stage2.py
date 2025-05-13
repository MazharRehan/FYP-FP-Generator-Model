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
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class Config:
    # Data parameters
    DATA_DIR = "dataset/"
    METADATA_PATH = "floor_plan_metadata_consolidated_v6.csv"
    IMAGE_WIDTH = 849
    IMAGE_HEIGHT = 1570
    PLOT_SIZE = "10Marla"
    
    # Model parameters - Enhanced for Stage 2
    LATENT_DIM = 128
    CONDITION_DIM = 32
    NUM_ROOM_TYPES = 23  # Expanded to handle all room types
    
    # Training dimensions
    TARGET_SIZE = 256  # Increased resolution for Stage 2
    
    # Training parameters
    BATCH_SIZE = 2
    EPOCHS = 150
    LEARNING_RATE = 0.0001  # Reduced learning rate for stability
    BETA_1 = 0.5
    BETA_2 = 0.999
    
    # Lambda values for loss components
    LAMBDA_ADV = 1.0      # Adversarial loss weight
    LAMBDA_PIXEL = 10.0   # Pixel reconstruction loss weight
    LAMBDA_ROOM = 5.0     # Room type classification loss weight
    LAMBDA_STRUCT = 2.0   # Structural coherence loss weight
    LAMBDA_COUNT = 1.0    # Room count consistency loss weight
    
    # Output directories
    CHECKPOINT_DIR = "checkpoints/stage2/"
    LOG_DIR = "logs/stage2/"
    SAMPLE_DIR = "samples/stage2/"
    
    # Room type mapping for visualization
    ROOM_TYPES = [
        "Background", "Wall", "Bathroom", "Bedroom", "Dining", 
        "DrawingRoom", "DressingArea", "Entrance", "Kitchen", "Lounge",
        "Staircase", "Store", "Garage", "Balcony", "Lawn",
        "Porch", "Terrace", "Veranda", "Office", "Prayer",
        "Study", "Amber", "Backyard"
    ]
    
    # Room type colors for visualization (RGB format)
    ROOM_COLORS = {
        "Background": [255, 255, 255],
        "Wall": [0, 0, 0],
        "Bathroom": [0, 0, 255],
        "Bedroom": [255, 0, 0],
        "Dining": [255, 192, 0],
        "DrawingRoom": [0, 176, 80],
        "DressingArea": [255, 102, 0],
        "Entrance": [255, 255, 0],
        "Kitchen": [255, 128, 0],
        "Lounge": [255, 255, 0],
        "Staircase": [0, 176, 240],
        "Store": [112, 48, 160],
        "Garage": [192, 0, 0],
        "Balcony": [146, 208, 80],
        "Lawn": [0, 255, 255],
        "Porch": [255, 153, 204],
        "Terrace": [96, 96, 96],
        "Veranda": [255, 102, 204],
        "Office": [204, 204, 255],
        "Prayer": [204, 204, 204],
        "Study": [102, 0, 102],
        "Amber": [153, 51, 0],
        "Backyard": [0, 204, 153]
    }

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

def extract_room_masks(image_array):
    """Extract separate masks for each room type from an RGB image."""
    height, width, _ = image_array.shape
    
    # Initialize empty array for room type masks
    room_masks = np.zeros((height, width, Config.NUM_ROOM_TYPES), dtype=np.float32)
    
    # Set background mask (white areas)
    background_mask = np.all(image_array == [255, 255, 255], axis=-1)
    room_masks[:, :, 0] = background_mask
    
    # Set wall mask (black pixels)
    wall_mask = np.all(image_array == [0, 0, 0], axis=-1)
    room_masks[:, :, 1] = wall_mask
    
    # Process each room type
    for i, room_type in enumerate(Config.ROOM_TYPES[2:], start=2):  # Skip Background and Wall
        color = Config.ROOM_COLORS[room_type]
        mask = np.all(image_array == color, axis=-1)
        room_masks[:, :, i] = mask
    
    return room_masks

def preprocess_image(image_path):
    """Load and preprocess image for room type segmentation."""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        
        # Extract room type masks
        room_masks = extract_room_masks(img)
        
        # Resize to target size
        resized_masks = np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.NUM_ROOM_TYPES), dtype=np.float32)
        
        for i in range(Config.NUM_ROOM_TYPES):
            mask = room_masks[:, :, i]
            resized_mask = cv2.resize(mask, 
                                     (Config.TARGET_SIZE, Config.TARGET_SIZE),
                                     interpolation=cv2.INTER_NEAREST)
            resized_masks[:, :, i] = resized_mask
        
        # Ensure one-hot encoding (each pixel belongs to exactly one room type)
        # If no room type is detected, assign to background
        sum_masks = np.sum(resized_masks, axis=-1, keepdims=True)
        no_assignment = (sum_masks == 0)
        resized_masks[:, :, 0] = np.logical_or(resized_masks[:, :, 0], no_assignment[:, :, 0])
        
        # Normalize to range [0, 1]
        # Ensure each pixel sums to 1 across all room types
        sum_masks = np.sum(resized_masks, axis=-1, keepdims=True)
        normalized_masks = resized_masks / (sum_masks + 1e-8)
        
        return normalized_masks
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return empty array of correct shape in case of error
        return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.NUM_ROOM_TYPES), dtype=np.float32)

def prepare_condition_vector(row):
    """Extract conditional inputs from metadata with enhanced features."""
    # Initialize condition vector
    condition = []
    
    # Plot size (already filtered to 10 Marla, but could be useful later)
    condition.append(1.0)  # 10 Marla indicator
    
    # Room counts - extract these directly from the row (normalized)
    room_types = ['Bathroom', 'Bedroom', 'Dining', 'DrawingRoom', 'DressingArea', 
                 'Kitchen', 'Lounge', 'Store', 'Garage', 'Backyard']
    
    # Add normalized room counts
    max_counts = {'Bathroom': 4, 'Bedroom': 3, 'DrawingRoom': 1, 'Kitchen': 1, 'Dining': 1,
                 'DressingArea': 1, 'Lounge': 1, 'Store': 1, 'Garage': 1, 'Backyard': 1}
    
    for room in room_types:
        count_col = f'Count_{room}'
        if count_col in row and not pd.isna(row[count_col]):
            count = float(row[count_col])
            max_count = max_counts.get(room, 1)
            condition.append(min(count / max_count, 1.0))
        else:
            condition.append(0.0)
    
    # Add total area (normalized)
    condition.append(row.get('TotalAreaSqFt', 2275.0) / 2275.0)
    
    # Add room area proportions (how much space each room takes)
    for room in room_types[:6]:  # Just use the main rooms
        area_col = f'Area_{room}'
        if area_col in row and not pd.isna(row[area_col]) and row.get('TotalAreaSqFt', 0) > 0:
            area_proportion = float(row.get(area_col, 0)) / float(row.get('TotalAreaSqFt', 2275.0))
            condition.append(min(area_proportion, 1.0))
        else:
            condition.append(0.0)
    
    # Convert to numpy array and ensure fixed length
    condition_vector = np.array(condition, dtype=np.float32)
    
    # Pad or truncate to desired length
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

# Enhanced Model Architecture for Stage 2
def build_generator():
    # Latent vector input
    noise_input = layers.Input(shape=(Config.LATENT_DIM,), name='noise_input')
    # Conditional input (room counts, areas, etc.)
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')

    # Process condition with a small network
    c_processed = layers.Dense(128)(condition_input)
    c_processed = layers.LeakyReLU(0.2)(c_processed)
    c_processed = layers.Dense(128)(c_processed)
    c_processed = layers.LeakyReLU(0.2)(c_processed)  # Shape: (None, 128)

    # Combine noise and processed condition
    z = layers.Concatenate()([noise_input, c_processed])  # Shape: (None, LATENT_DIM + 128)

    # Initial dense layer, project to a starting spatial resolution
    # Start at 8x8
    x = layers.Dense(8 * 8 * 512)(z)  # 8*8*512 = 32768
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 512))(x)  # Shape: (None, 8, 8, 512)

    # Upsampling block 1: 8x8 -> 16x16
    x = layers.UpSampling2D()(x)  # (None, 16, 16, 512)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 16, 16, 256)

    # Upsampling block 2: 16x16 -> 32x32
    x = layers.UpSampling2D()(x)  # (None, 32, 32, 256)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 32, 32, 128)

    # Upsampling block 3: 32x32 -> 64x64
    x = layers.UpSampling2D()(x)  # (None, 64, 64, 128)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 64, 64, 64)

    # --- Inject Condition Here or at 128x128 ---
    # Let's try injecting condition early, e.g., after the 8x8 or 16x16 stage,
    # or make the 'bottleneck_condition' match the spatial dimension of 'x' before concatenation.
    # The original error was at a later stage. Let's stick to fixing that specific concatenation.

    # Upsampling block 4: 64x64 -> 128x128
    x = layers.UpSampling2D()(x)  # (None, 128, 128, 64)
    x = layers.Conv2D(64, 3, padding='same')(x)  # Keep 64 channels or adjust
    x = layers.BatchNormalization()(x)
    x_128 = layers.LeakyReLU(0.2)(x)  # x_128 is (None, 128, 128, 64)

    # Prepare condition to be spatially injected at 128x128
    # 'c_processed' is (None, 128)
    condition_spatial = layers.Dense(128 * 128 * 1)(
        c_processed)  # Dense layer to match spatial dimensions if needed, or reshape directly
    condition_spatial = layers.LeakyReLU(0.2)(condition_spatial)
    # The original bottleneck_condition:
    # bottleneck_condition = layers.Dense(16 * 16)(c) # c was (None, 128)
    # bottleneck_condition = layers.Reshape((16, 16, 1))(bottleneck_condition)
    # bottleneck_condition = layers.UpSampling2D(size=(8, 8))(bottleneck_condition) # -> (128, 128, 1)

    # Let's use the original logic for bottleneck_condition but ensure 'x' is also 128x128
    # 'c_processed' (derived from 'c') is (None, 128)
    bottleneck_cond_features = layers.Dense(16 * 16 * 4)(c_processed)  # Project to 16x16x4 = 1024 features
    bottleneck_cond_features = layers.LeakyReLU(0.2)(bottleneck_cond_features)
    bottleneck_cond_reshaped = layers.Reshape((16, 16, 4))(bottleneck_cond_features)  # (None, 16, 16, 4)
    condition_to_concat = layers.UpSampling2D(size=(8, 8))(bottleneck_cond_reshaped)  # (None, 128, 128, 4)

    # Concatenate x_128 with condition_to_concat
    x = layers.Concatenate()([x_128, condition_to_concat])  # x_128(128,128,64), condition(128,128,4) -> (128,128,68)

    # Upsampling block 5: 128x128 -> 256x256
    x = layers.UpSampling2D()(x)  # (None, 256, 256, 68)
    x = layers.Conv2D(32, 3, padding='same')(x)  # Reduce channels
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 256, 256, 32)

    # Removed the problematic skip connection for now.
    # If you want to add a skip connection, ensure dimensions match.
    # For example, if you had an encoder stage at 256x256xN, you could concatenate it here.

    # Final refinement layers
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 256, 256, 64)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)  # (None, 256, 256, 32)

    # Output layer
    output_logits = layers.Conv2D(Config.NUM_ROOM_TYPES, 1, padding='same')(x)  # (None, 256, 256, NUM_ROOM_TYPES)
    output = layers.Activation('softmax', name='room_type_output')(output_logits)

    return models.Model([noise_input, condition_input], output, name='generator')

def build_discriminator():
    """Enhanced PatchGAN discriminator with architectural awareness."""
    # Image input - Room type segmentation (one-hot encoded)
    image_input = layers.Input(shape=(Config.TARGET_SIZE, Config.TARGET_SIZE, Config.NUM_ROOM_TYPES), 
                              name='image_input')
    
    # Conditional input
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name='condition_input')
    
    # Process image input
    x = layers.Conv2D(32, 4, strides=2, padding='same')(image_input)  # 128x128
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)  # 64x64
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)  # 32x32
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)  # 16x16
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)  # 8x8
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    
    # Process condition
    c = layers.Dense(512)(condition_input)
    c = layers.LeakyReLU(0.2)(c)
    c = layers.Dense(8 * 8 * 1)(c)
    c = layers.Reshape((8, 8, 1))(c)
    
    # Combine image features with condition
    x = layers.Concatenate()([x, c])
    
    # Final convolution for PatchGAN output
    x = layers.Conv2D(1, 4, padding='same')(x)
    
    return models.Model([image_input, condition_input], x, name='discriminator')

# Custom Loss Functions for Architectural Awareness
def room_count_consistency_loss(y_true, y_pred, target_counts):
    """Ensure the number of rooms matches target counts."""
    loss = 0.0

    # These are the room types for which 'target_counts' provides data
    # Their indices in Config.ROOM_TYPES (after skipping Background and Wall) will be 0-9
    # if they are the first 10 after Background and Wall.
    # Let's be explicit based on Config.ROOM_TYPES

    # Room types for which we have counts in target_counts
    # This list MUST match the order and content of room_types in prepare_condition_vector
    # that contribute to conditions[:, 1:11]
    counted_room_type_names = ['Bathroom', 'Bedroom', 'Dining', 'DrawingRoom', 'DressingArea',
                               'Kitchen', 'Lounge', 'Store', 'Garage', 'Backyard']

    if target_counts.shape[1] != len(counted_room_type_names):
        raise ValueError(f"target_counts has {target_counts.shape[1]} columns, "
                         f"but expected {len(counted_room_type_names)} for counted_room_type_names.")

    for i, room_name_in_target_counts in enumerate(counted_room_type_names):
        try:
            # Find the actual index of this room_name in Config.ROOM_TYPES
            # This is the channel index in y_pred
            pred_channel_idx = Config.ROOM_TYPES.index(room_name_in_target_counts)
        except ValueError:
            # This room name (from counted_room_type_names) is not in Config.ROOM_TYPES
            # This would be a configuration error.
            print(
                f"Warning: Room '{room_name_in_target_counts}' for count loss not found in Config.ROOM_TYPES. Skipping.")
            continue

        # Get target count for this room type from the i-th column of target_counts
        target = target_counts[:, i]

        # Calculate predicted count (sum of probabilities for the specific pred_channel_idx)
        # The division factor here might need review for its logic (as discussed before)
        pred_count = tf.reduce_sum(y_pred[:, :, :, pred_channel_idx], axis=[1, 2]) / (
                    Config.TARGET_SIZE * Config.TARGET_SIZE / 100.0)  # Added .0 for float division

        # Calculate mean squared error
        room_loss = tf.reduce_mean(tf.square(pred_count - target))
        loss += room_loss

    return loss / float(len(counted_room_type_names) + 1e-8)  # Average loss over counted types

def structural_coherence_loss(y_pred):
    """Encourage structural coherence in the floor plan."""
    # Walls should be continuous
    walls = y_pred[:, :, :, 1]  # Wall channel
    
    # Calculate gradients in x and y directions
    gx = walls[:, :, 1:] - walls[:, :, :-1]
    gy = walls[:, 1:, :] - walls[:, :-1, :]
    
    # Encourage smoothness within walls (L1 norm of gradients)
    smoothness_loss = tf.reduce_mean(tf.abs(gx)) + tf.reduce_mean(tf.abs(gy))
    
    # Ensure walls separate different room types
    # This encourages walls between different room types
    room_types = y_pred[:, :, :, 2:]  # All room types except background and walls
    
    # Create shifted versions to compute differences between adjacent pixels
    room_types_x = room_types[:, :, 1:]
    room_types_y = room_types[:, 1:, :]
    
    # Calculate room differences in x and y directions
    room_diff_x = tf.reduce_sum(tf.abs(room_types[:, :, :-1] - room_types_x), axis=-1)
    room_diff_y = tf.reduce_sum(tf.abs(room_types[:, :-1, :] - room_types_y), axis=-1)
    
    # Where rooms differ, walls should be present
    wall_x = walls[:, :, :-1]
    wall_y = walls[:, :-1, :]
    
    # Calculate wall presence where needed
    wall_needed_loss = (
        tf.reduce_mean(room_diff_x * (1 - wall_x)) +
        tf.reduce_mean(room_diff_y * (1 - wall_y))
    )
    
    return smoothness_loss + 0.5 * wall_needed_loss

class FloorPlanGANStage2(models.Model):
    def __init__(self, generator, discriminator):
        super(FloorPlanGANStage2, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, gen_optimizer, disc_optimizer):
        super(FloorPlanGANStage2, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        # Sparse categorical crossentropy for room type classification
        self.room_loss = tf.keras.losses.CategoricalCrossentropy()
        # Adversarial loss
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    def train_step(self, data):
        real_images, conditions = data
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise
        noise = tf.random.normal([batch_size, Config.LATENT_DIM])
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            fake_images = self.generator([noise, conditions], training=True)
            
            # Add noise to inputs for regularization
            real_images_noisy = real_images + tf.random.normal(tf.shape(real_images), mean=0.0, stddev=0.05)
            real_images_noisy = tf.clip_by_value(real_images_noisy, 0, 1)
            
            fake_images_noisy = fake_images + tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=0.05)
            fake_images_noisy = tf.clip_by_value(fake_images_noisy, 0, 1)
            
            # Get discriminator outputs
            real_output = self.discriminator([real_images_noisy, conditions], training=True)
            fake_output = self.discriminator([fake_images_noisy, conditions], training=True)
            
            # Calculate losses - use soft labels for GAN stability
            valid = tf.ones_like(real_output) * 0.9  # Soft label for real
            fake = tf.zeros_like(fake_output) + 0.1  # Soft label for fake
            
            real_loss = self.adversarial_loss(valid, real_output)
            fake_loss = self.adversarial_loss(fake, fake_output)
            disc_loss = (real_loss + fake_loss) / 2
        
        # Get gradients and update discriminator
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # Prepare room count targets for consistency loss
        # Extract from conditions - assuming first 10 elements after the plot size
        # represent normalized room counts for the main rooms
        room_counts = conditions[:, 1:11]  # Adjust based on your condition vector format
        
        # Train generator with multiple objectives
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator([noise, conditions], training=True)
            
            # Discriminator output for fake images
            fake_output = self.discriminator([fake_images, conditions], training=True)
            
            # Adversarial loss - fool the discriminator
            gen_adversarial_loss = self.adversarial_loss(
                tf.ones_like(fake_output), fake_output
            )
            
            # Room type classification loss - match the target room types
            # Using sparse categorical cross-entropy
            gen_room_type_loss = self.room_loss(real_images, fake_images)
            
            # Structural coherence loss - ensure walls are continuous
            gen_structural_loss = structural_coherence_loss(fake_images)
            
            # Room count consistency loss - ensure correct number of rooms
            gen_count_loss = room_count_consistency_loss(real_images, fake_images, room_counts)
            
            # Combined generator loss with weights
            gen_loss = (
                Config.LAMBDA_ADV * gen_adversarial_loss +
                Config.LAMBDA_PIXEL * gen_room_type_loss +
                Config.LAMBDA_STRUCT * gen_structural_loss +
                Config.LAMBDA_COUNT * gen_count_loss
            )
        
        # Get gradients and update generator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {
            "d_loss": disc_loss,
            "g_loss": gen_loss,
            "g_adv_loss": gen_adversarial_loss,
            "g_room_loss": gen_room_type_loss,
            "g_struct_loss": gen_structural_loss,
            "g_count_loss": gen_count_loss
        }

def visualize_room_types(room_type_map):
    """Convert room type probabilities to RGB image."""
    # Initialize RGB image
    height, width, _ = room_type_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get the most likely room type for each pixel
    room_type_indices = np.argmax(room_type_map, axis=-1)
    
    # Assign color based on room type
    for i, room_type in enumerate(Config.ROOM_TYPES):
        mask = (room_type_indices == i)
        if np.any(mask):
            color = Config.ROOM_COLORS[room_type]
            rgb_image[mask] = color
    
    return rgb_image

def generate_samples(generator, condition_samples, epoch):
    """Generate and save sample floor plans with room types."""
    noise = tf.random.normal([len(condition_samples), Config.LATENT_DIM])
    generated_images = generator([noise, condition_samples], training=False)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(min(4, len(generated_images))):
        # Get room type probabilities
        room_probs = generated_images[i]
        
        # Convert to RGB visualization
        rgb_floor_plan = visualize_room_types(room_probs)
        
        # Plot RGB floor plan
        plt.subplot(2, 4, i+1)
        plt.imshow(rgb_floor_plan)
        plt.title(f'Sample {i+1}')
        plt.axis('off')
        
        # Plot the most probable room type for specific channels
        # Walls
        plt.subplot(2, 4, i+5)
        wall_probs = room_probs[:, :, 1]  # Wall channel
        plt.imshow(wall_probs, cmap='gray')
        plt.title(f'Sample {i+1} - Walls')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAMPLE_DIR, f'samples_epoch_{epoch:03d}.png'))
    plt.close()
    
    # Additional visualization showing individual room types
    plt.figure(figsize=(20, 15))
    sample_idx = 0  # Use the first sample
    room_probs = generated_images[sample_idx]
    
    # Plot for each major room type
    key_rooms = ["Wall", "Bathroom", "Bedroom", "Kitchen", "DrawingRoom", "Dining", "Lounge", "Garage"]
    for i, room in enumerate(key_rooms):
        room_idx = Config.ROOM_TYPES.index(room)
        plt.subplot(2, 4, i+1)
        plt.imshow(room_probs[:, :, room_idx], cmap='jet')
        plt.title(f'{room}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAMPLE_DIR, f'room_types_epoch_{epoch:03d}.png'))
    plt.close()

def train():
    """Train the Stage 2 floor plan generator."""
    # Enable eager execution for better error messages
    print("Configuring TensorFlow...")
    # tf.config.run_functions_eagerly(True) # Comment this out for performance
    
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

    # ... (previous code in train()) ...

    # Check if we can load Stage 1 weights
    stage1_model_dir_path = os.path.join("checkpoints/stage1/", "final_generator")  # Path to the SavedModel DIRECTORY

    # More robust check for a valid SavedModel directory
    is_stage1_saved_model_valid = os.path.isdir(stage1_model_dir_path) and \
                                  os.path.exists(os.path.join(stage1_model_dir_path, "saved_model.pb"))

    # Build Stage 2 models (these are the models we will actually train)
    generator = build_generator()  # Stage 2 generator from floor_plan_generator_stage2.py
    discriminator = build_discriminator()  # Stage 2 discriminator

    print("Stage 2 Generator Summary:")
    generator.summary()
    print("\nStage 2 Discriminator Summary:")
    discriminator.summary()

    # Try to transfer Stage 1 weights if available
    if is_stage1_saved_model_valid:
        print(f"\nFound Stage 1 SavedModel at: {stage1_model_dir_path}. Attempting to load and transfer learning...")
        try:
            # ---- CORRECT WAY TO LOAD STAGE 1 SAVEDMODEL ----
            print(f"Loading Stage 1 generator model from: {stage1_model_dir_path}")
            # This loads the complete Stage 1 generator model (architecture + weights)
            generator_stage1_loaded = tf.keras.models.load_model(stage1_model_dir_path)
            print("Stage 1 generator model loaded successfully.")
            # generator_stage1_loaded.summary() # Optional: print summary of loaded Stage 1 model for comparison
            # -------------------------------------------------

            # Now, transfer compatible weights from generator_stage1_loaded to the new Stage 2 'generator'
            print("Transferring weights from loaded Stage 1 generator to Stage 2 generator...")
            num_transferred_layers = 0
            num_skipped_shape_mismatch = 0
            num_skipped_no_s1_layer = 0

            for layer_s2 in generator.layers:  # Iterate over layers of the Stage 2 generator
                try:
                    # Try to get the corresponding layer by name from the loaded Stage 1 generator
                    layer_s1 = generator_stage1_loaded.get_layer(name=layer_s2.name)

                    s1_weights = layer_s1.get_weights()
                    s2_weights = layer_s2.get_weights()

                    if s1_weights and s2_weights:  # Check if both layers have weights
                        # Check if all weight tensors within the layer have matching shapes
                        if all(w1.shape == w2.shape for w1, w2 in zip(s1_weights, s2_weights)):
                            layer_s2.set_weights(s1_weights)
                            num_transferred_layers += 1
                        else:
                            print(f"  - Skipping layer '{layer_s2.name}': Incompatible weight shapes.")
                            # For detailed debugging of shape mismatches:
                            # for i, (w1, w2) in enumerate(zip(s1_weights, s2_weights)):
                            #     if w1.shape != w2.shape:
                            #         print(f"    - Weight {i}: Stage 1 shape {w1.shape}, Stage 2 shape {w2.shape}")
                            num_skipped_shape_mismatch += 1
                        # else:
                        # print(f"  - Layer '{layer_s2.name}' or its Stage 1 counterpart has no weights.")
                        pass

                except ValueError:
                    # This means a layer with layer_s2.name was not found in generator_stage1_loaded
                    # print(f"  - Layer '{layer_s2.name}' not found in Stage 1 model. Skipping.")
                    num_skipped_no_s1_layer += 1
                except Exception as e_transfer:
                    print(f"  - Error transferring weights for layer '{layer_s2.name}': {e_transfer}")

            print(f"Weight transfer attempt complete.")
            print(f"  Successfully transferred weights for {num_transferred_layers} layers.")
            if num_skipped_shape_mismatch > 0:
                print(f"  Skipped {num_skipped_shape_mismatch} layers due to shape mismatches.")
            if num_skipped_no_s1_layer > 0:
                print(f"  Skipped {num_skipped_no_s1_layer} layers not found in Stage 1 model by name.")

        except Exception as e_load:
            print(f"ERROR: Could not load Stage 1 model or transfer weights: {e_load}")
            print("Proceeding with randomly initialized weights for Stage 2 generator.")
    else:
        print(f"\nStage 1 SavedModel not found or invalid at '{stage1_model_dir_path}'.")
        print("Training Stage 2 generator from scratch.")

    # Create GAN model (using the Stage 2 generator, which may now have some transferred weights)
    gan = FloorPlanGANStage2(generator, discriminator)
    
    # Compile with optimizers
    gen_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE,
                                   beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    disc_optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE * 0.5,  # Slower discriminator
                                    beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    
    gan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
    
    # Create sample condition vectors for visualization
    sample_conditions = []
    for _, row in val_df.iloc[:4].iterrows():
        sample_conditions.append(prepare_condition_vector(row))
    sample_conditions = np.array(sample_conditions)
    
    print("Starting training...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Training
        total_d_loss = 0
        total_g_loss = 0
        total_g_adv_loss = 0
        total_g_room_loss = 0
        total_g_struct_loss = 0
        total_g_count_loss = 0
        
        for step in range(train_steps):
            try:
                real_images, conditions = next(train_gen)
                losses = gan.train_step([real_images, conditions])
                
                # Accumulate losses for reporting
                total_d_loss += losses['d_loss']
                total_g_loss += losses['g_loss']
                total_g_adv_loss += losses['g_adv_loss']
                total_g_room_loss += losses['g_room_loss']
                total_g_struct_loss += losses['g_struct_loss']
                total_g_count_loss += losses['g_count_loss']
                
                if step % 5 == 0:
                    print(f"Step {step}/{train_steps} - "
                          f"d_loss: {losses['d_loss']:.4f}, "
                          f"g_loss: {losses['g_loss']:.4f}, "
                          f"room_loss: {losses['g_room_loss']:.4f}")
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Report average losses for the epoch
        avg_d_loss = total_d_loss / train_steps
        avg_g_loss = total_g_loss / train_steps
        avg_g_adv_loss = total_g_adv_loss / train_steps
        avg_g_room_loss = total_g_room_loss / train_steps
        avg_g_struct_loss = total_g_struct_loss / train_steps
        avg_g_count_loss = total_g_count_loss / train_steps
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
              f"d_loss: {avg_d_loss:.4f}, "
              f"g_loss: {avg_g_loss:.4f}, "
              f"g_adv_loss: {avg_g_adv_loss:.4f}, "
              f"g_room_loss: {avg_g_room_loss:.4f}, "
              f"g_struct_loss: {avg_g_struct_loss:.4f}, "
              f"g_count_loss: {avg_g_count_loss:.4f}")
        
        # Generate samples periodically
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
        
        # Report total training time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"Total training time: {hours}h {minutes}m {seconds}s")
    except Exception as e:
        print(f"Error saving final models: {e}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")