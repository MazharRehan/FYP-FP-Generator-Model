# --- START OF FILE floor_plan_generator.py ---

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
# import json # No longer needed if not saving metadata here
from datetime import datetime

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
BUFFER_SIZE = 400
BATCH_SIZE = 16
IMG_HEIGHT = 256  # Standard size for model input/output
IMG_WIDTH = 256   # Standard size for model input/output
NUM_CHANNELS = 3  # Input is RGB image (used as conditional input)
EPOCHS = 80       # <<< UPDATED: Default epochs set to 80 >>>
PLOT_TYPES = ['5_marla', '10_marla', '20_marla']

# Define the color mapping - Ensure this matches generate_metadata.py if consistency is needed across scripts
# Using the map provided in floor_plan_generator.py initially
COLOR_MAP = {
    'Bedroom': [255, 0, 0],          # Red
    'Bathroom': [0, 0, 255],         # Blue
    'Kitchen': [255, 165, 0],        # Orange
    'Drawing Room': [0, 128, 0],     # Green
    'Garage': [165, 42, 42],         # Brown
    'Lounge': [255, 255, 0],         # Yellow (NOTE: Name differs slightly from metadata script)
    'Backyard': [50, 205, 50],       # Lime Green
    'Stairs': [0, 128, 128],         # Teal
    'Storage': [128, 0, 128],        # Purple (NOTE: Name differs slightly from metadata script)
    'Open Space': [0, 255, 255],     # Cyan
    'Prayer Room': [127, 127, 127],  # Crimson / Grayish
    'Staircase': [153, 51, 255],     # Violet
    'Lobby': [255, 0, 255],          # Magenta
    'Lawn': [64, 224, 208],          # Turquoise
    'Dining': [255, 192, 203],       # Pink (NOTE: Check RGB vs metadata script)
    'Servant': [75, 0, 130],         # Indigo (NOTE: Name differs slightly from metadata script)
    'Passage': [128, 128, 0],        # Olive Green
    'Laundry': [230, 230, 250],      # Lavender
    'Dressing': [255, 127, 80],      # Coral (NOTE: Name differs slightly from metadata script)
    'Side Garden': [255, 215, 0],    # Gold
    'Library': [255, 191, 0],        # Amber
    'Walls': [0, 0, 0],              # Black
    'Door': [128, 0, 0],             # Mahogany
    'Background': [255, 255, 255]    # White
}
NUM_OUTPUT_CHANNELS = len(COLOR_MAP) # Number of classes = number of entries in COLOR_MAP

# --- Mappings for Preprocessing and Visualization ---
# Create reverse mapping from RGB tuple to class index
RGB_TO_INDEX = {tuple(color): i for i, color in enumerate(COLOR_MAP.values())}
# Create mapping from class index to RGB color
INDEX_TO_COLOR = np.array(list(COLOR_MAP.values()), dtype=np.uint8)

# --- Data Loading and Preprocessing ---

def load_dataset(dataset_path, plot_types=None):
    """
    Load dataset from the given path for specified plot types.
    Assumes input images are PNG.

    Args:
        dataset_path: Path to the base dataset directory (e.g., './dataset')
        plot_types: List of plot types to include (e.g., ['5_marla', '10_marla'])

    Returns:
        images: List of floor plan images (normalized to [-1, 1])
        conditions: List of condition vectors (plot type one-hot encoded)
    """
    if plot_types is None:
        plot_types = PLOT_TYPES

    images = []
    conditions = []
    plot_type_map = {ptype: i for i, ptype in enumerate(PLOT_TYPES)} # Map name to index

    print(f"Scanning dataset path: {dataset_path}")
    if not os.path.isdir(dataset_path):
         raise FileNotFoundError(f"Dataset base directory not found: {dataset_path}")

    for plot_type_name in plot_types:
        plot_dir = os.path.join(dataset_path, plot_type_name)

        if not os.path.exists(plot_dir):
            print(f"Warning: Directory '{plot_dir}' not found. Skipping.")
            continue

        print(f"Processing directory: {plot_dir}")
        files = [f for f in os.listdir(plot_dir) if f.lower().endswith('.png')]
        print(f"Found {len(files)} PNG files in {plot_type_name}.")

        if plot_type_name not in plot_type_map:
            print(f"Warning: Plot type '{plot_type_name}' from directory name is not in the defined PLOT_TYPES list. Skipping files in this directory.")
            continue

        plot_index = plot_type_map[plot_type_name]

        for file in files:
            file_path = os.path.join(plot_dir, file)
            try:
                # Load and preprocess image
                img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
                img_array = img_to_array(img)

                # Normalize RGB image to [-1, 1] (input to generator)
                img_array_normalized = (img_array / 127.5) - 1

                # Create one-hot encoding for plot type condition
                condition = np.zeros(len(PLOT_TYPES), dtype=np.float32)
                condition[plot_index] = 1

                images.append(img_array_normalized)
                conditions.append(condition)
            except Exception as e:
                print(f"Error loading or processing image {file_path}: {e}")

    if not images:
         raise ValueError("No valid images were loaded. Check dataset paths and image files.")

    return np.array(images, dtype=np.float32), np.array(conditions, dtype=np.float32)

def preprocess_image_to_segmentation_map(image_normalized):
    """
    Convert a normalized RGB image to a segmentation map (target for generator).
    Each pixel will be assigned a class index based on its *closest* RGB value in COLOR_MAP.

    Args:
        image_normalized: RGB image array with values in range [-1, 1]

    Returns:
        segmentation_map: Array (IMG_HEIGHT, IMG_WIDTH) of class indices (uint8)
    """
    # Convert from [-1, 1] to [0, 255] uint8
    img_uint8 = ((image_normalized + 1) * 127.5).astype(np.uint8)

    # Initialize segmentation map (using a default index, e.g., background)
    background_index = RGB_TO_INDEX.get(tuple(COLOR_MAP['Background']), NUM_OUTPUT_CHANNELS -1) # Default to last index if 'Background' missing
    segmentation_map = np.full((img_uint8.shape[0], img_uint8.shape[1]), background_index, dtype=np.uint8)

    # --- Optimization: Vectorized approach for color matching ---
    # Reshape image to (H*W, 3) and colors to (N_colors, 3)
    pixels = img_uint8.reshape(-1, 3)
    color_array = np.array(list(COLOR_MAP.values()), dtype=np.uint8)

    # Calculate squared distances (broadcasted)
    # Using float32 for intermediate distance calculation to avoid overflow
    distances = np.sum((pixels[:, np.newaxis, :].astype(np.float32) - color_array[np.newaxis, :, :].astype(np.float32))**2, axis=2)

    # Find the index of the minimum distance for each pixel
    closest_indices = np.argmin(distances, axis=1)

    # Reshape back to image dimensions
    segmentation_map = closest_indices.reshape(img_uint8.shape[0], img_uint8.shape[1]).astype(np.uint8)

    # --- Old Pixel-by-pixel approach (kept for reference, much slower) ---
    # for y in range(img_uint8.shape[0]):
    #     for x in range(img_uint8.shape[1]):
    #         rgb = tuple(img_uint8[y, x])
    #         # Find index of the closest color in the color map
    #         min_dist = float('inf')
    #         best_index = background_index
    #         for color_tuple, index in RGB_TO_INDEX.items():
    #             dist = sum((int(color_tuple[i]) - int(rgb[i])) ** 2 for i in range(3))
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 best_index = index
    #             # Optional: Break early if exact match found
    #             if dist == 0:
    #                 break
    #         segmentation_map[y, x] = best_index
    # -------------------------------------------------------------------

    return segmentation_map


def create_tf_dataset(images, conditions):
    """
    Preprocesses images to segmentation maps and creates TensorFlow datasets.

    Args:
        images: Array of normalized input floor plan images [-1, 1]
        conditions: Array of condition vectors

    Returns:
        train_dataset: tf.data.Dataset for training: ({'image': norm_image, 'condition': cond}, seg_map)
        test_dataset: tf.data.Dataset for testing: ({'image': norm_image, 'condition': cond}, seg_map)
    """
    print("Preprocessing images to segmentation maps...")
    segmentation_maps = []
    # Using list comprehension for potentially better readability with tqdm if needed later
    segmentation_maps = [preprocess_image_to_segmentation_map(img) for img in images]
    segmentation_maps = np.array(segmentation_maps, dtype=np.uint8) # Target maps are uint8 indices

    print("Splitting dataset...")
    # Input to the model is the normalized image and condition
    # Target is the segmentation map
    X_train, X_test, y_train, y_test, cond_train, cond_test = train_test_split(
        images, segmentation_maps, conditions, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    print("Creating TensorFlow datasets...")
    # Train Dataset: Yields a tuple: (dictionary_of_inputs, target_output)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'image': X_train, 'condition': cond_train}, # Inputs dict
        y_train                                       # Target segmentation map
    )).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Test Dataset: Yields the same structure
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'image': X_test, 'condition': cond_test},   # Inputs dict
        y_test                                       # Target segmentation map
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

# --- Loss Functions ---

def generator_loss(disc_generated_output, gen_output_logits, target_seg_map):
    """
    Loss function for the generator (Pix2Pix style adapted for segmentation).

    Args:
        disc_generated_output: Discriminator output logits for generated images.
        gen_output_logits: Generator output logits (NUM_OUTPUT_CHANNELS deep).
        target_seg_map: Ground truth segmentation map (indices).

    Returns:
        total_loss: Combined adversarial and pixel-wise segmentation loss.
    """
    # Adversarial Loss: Generator wants discriminator to think generated images are real
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output)

    # Segmentation Loss (Pixel-wise): Compare generator output logits to target indices
    # Use SparseCategoricalCrossentropy because target is indices, output is logits
    pixel_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        target_seg_map, gen_output_logits)

    # Total Generator Loss (Lambda can be tuned, 100 is common in Pix2Pix for L1)
    lambda_pixel = 100
    total_loss = gan_loss + lambda_pixel * pixel_loss

    return total_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Loss function for the discriminator.

    Args:
        disc_real_output: Discriminator output logits for real images + conditions.
        disc_generated_output: Discriminator output logits for generated images + conditions.

    Returns:
        total_loss: Discriminator loss.
    """
    # Real Loss: Discriminator should identify real images as real (label 1)
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output)

    # Generated Loss: Discriminator should identify generated images as fake (label 0)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)

    # Total Discriminator Loss
    total_loss = real_loss + generated_loss

    return total_loss

# --- Model Architecture (U-Net Generator, PatchGAN Discriminator) ---

def downsample_block(x, filters, kernel_size=4, strides=2, apply_batchnorm=True):
    """Downsampling block used in both Generator and Discriminator."""
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer=initializer, use_bias=not apply_batchnorm)(x) # Use bias if no BN
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def upsample_block(x, filters, kernel_size=4, strides=2, apply_dropout=False):
    """Upsampling block used in Generator."""
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.ReLU()(x) # Often ReLU in generator decoder
    return x

def build_generator():
    """Builds the U-Net based Generator model."""
    # --- Inputs ---
    # Input is the original normalized RGB image, used for conditioning structure
    image_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], name='image_input')
    # Condition is the one-hot plot type vector
    condition_input = layers.Input(shape=[len(PLOT_TYPES)], name='condition_input')

    # --- Condition Preprocessing ---
    # Expand condition to match spatial dimensions HxWx1
    condition_expanded = layers.Dense(IMG_HEIGHT * IMG_WIDTH * 1)(condition_input)
    condition_reshaped = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(condition_expanded)

    # Combine image and condition - Channels become NUM_CHANNELS + 1
    inputs_combined = layers.Concatenate()([image_input, condition_reshaped]) # Shape: (H, W, C+1)

    # --- Encoder (Downsampling Path) ---
    # Based on Pix2Pix architecture (adjust filter numbers if needed)
    down_stack = [
        downsample_block(inputs_combined, 64, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample_block(inputs_combined, 128),                        # (bs, 64, 64, 128)
        downsample_block(inputs_combined, 256),                        # (bs, 32, 32, 256)
        downsample_block(inputs_combined, 512),                        # (bs, 16, 16, 512)
        downsample_block(inputs_combined, 512),                        # (bs, 8, 8, 512)
        downsample_block(inputs_combined, 512),                        # (bs, 4, 4, 512)
        downsample_block(inputs_combined, 512),                        # (bs, 2, 2, 512)
        downsample_block(inputs_combined, 512),                        # (bs, 1, 1, 512) Bottleneck
    ]

    # --- Decoder (Upsampling Path) ---
    up_stack = [
        upsample_block(down_stack[-1], 512, apply_dropout=True), # (bs, 2, 2, 1024) with skip connection
        upsample_block(down_stack[-1], 512, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample_block(down_stack[-1], 512, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample_block(down_stack[-1], 512),                     # (bs, 16, 16, 1024)
        upsample_block(down_stack[-1], 256),                     # (bs, 32, 32, 512)
        upsample_block(down_stack[-1], 128),                     # (bs, 64, 64, 256)
        upsample_block(down_stack[-1], 64),                      # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    # Final layer to get back to original image size and output channels (logits)
    last = layers.Conv2DTranspose(NUM_OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation=None) # <<< Output logits

    x = inputs_combined # Start with the combined input
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1]) # Reverse skips except the last one (bottleneck)

    # Upsampling with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x) # Final layer

    return models.Model(inputs=[image_input, condition_input], outputs=x, name='generator')


def build_discriminator():
    """Builds the PatchGAN Discriminator model."""
    initializer = tf.random_normal_initializer(0., 0.02)

    # --- Inputs ---
    # Input is the target (real) or generated (fake) segmentation map (one-hot encoded)
    # Shape: (H, W, NUM_OUTPUT_CHANNELS)
    segmap_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, NUM_OUTPUT_CHANNELS], name='segmap_input')
    # Condition is the one-hot plot type vector
    condition_input = layers.Input(shape=[len(PLOT_TYPES)], name='condition_input')

    # --- Condition Preprocessing ---
    # Expand condition to match spatial dimensions HxWx1
    condition_expanded = layers.Dense(IMG_HEIGHT * IMG_WIDTH * 1)(condition_input)
    condition_reshaped = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(condition_expanded)

    # Combine segmentation map and condition
    # Channels = NUM_OUTPUT_CHANNELS + 1
    inputs_combined = layers.Concatenate()([segmap_input, condition_reshaped]) # (bs, H, W, N_classes+1)

    # --- PatchGAN Architecture ---
    # Sequence of Conv2D layers reducing spatial dimensions
    # Layer 1: (bs, 128, 128, 64)
    down1 = downsample_block(inputs_combined, 64, apply_batchnorm=False)(inputs_combined)
    # Layer 2: (bs, 64, 64, 128)
    down2 = downsample_block(down1, 128)(down1)
    # Layer 3: (bs, 32, 32, 256)
    down3 = downsample_block(down2, 256)(down2)

    # Intermediate layer (optional, common in PatchGAN)
    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU(0.2)(batchnorm1)

    # Final layer reducing to a patch output
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = layers.Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1) Output logits

    return tf.keras.Model(inputs=[segmap_input, condition_input], outputs=last, name='discriminator')


# --- Main Class ---

class FloorPlanGenerator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        print("--- Generator Summary ---")
        self.generator.summary()
        print("\n--- Discriminator Summary ---")
        self.discriminator.summary()

        # Initialize optimizers
        self.generator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)

        # Setup checkpoint
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        # Initialize logs for TensorBoard
        self.log_dir = 'logs/'
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "fit", current_time)
        )

        # Load latest checkpoint if exists
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        if tf.train.latest_checkpoint(self.checkpoint_dir):
            print(f"Restored checkpoint from {tf.train.latest_checkpoint(self.checkpoint_dir)}")
        else:
            print("Initializing from scratch.")

    def load_data(self):
        """Loads images and conditions, creates TF datasets."""
        images, conditions = load_dataset(self.dataset_path)
        self.train_dataset, self.test_dataset = create_tf_dataset(images, conditions)
        # Get one batch from test dataset for generating samples during training
        self.sample_test_batch = next(iter(self.test_dataset))


    @tf.function
    def train_step(self, batch_inputs, target_seg_map):
        """
        Performs a single training step for both Generator and Discriminator.

        Args:
            batch_inputs: Dictionary {'image': normalized_rgb_image, 'condition': one_hot_condition}
            target_seg_map: Ground truth segmentation map (indices)

        Returns:
            gen_loss: Generator loss for this step.
            disc_loss: Discriminator loss for this step.
        """
        input_image = batch_inputs['image']
        condition = batch_inputs['condition']

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1. Generate fake segmentation map (logits)
            gen_output_logits = self.generator([input_image, condition], training=True)

            # 2. Prepare inputs for Discriminator
            # Real pair: target segmentation map (one-hot) + condition
            target_seg_map_one_hot = tf.one_hot(target_seg_map, depth=NUM_OUTPUT_CHANNELS, dtype=tf.float32)
            # Fake pair: generated segmentation map (one-hot from logits) + condition
            # Use argmax to get indices, then one-hot. Stop gradient flow to generator here for disc input.
            gen_output_seg_map = tf.argmax(gen_output_logits, axis=-1)
            gen_output_seg_map_one_hot = tf.one_hot(gen_output_seg_map, depth=NUM_OUTPUT_CHANNELS, dtype=tf.float32)

            # 3. Get Discriminator outputs
            disc_real_output = self.discriminator([target_seg_map_one_hot, condition], training=True)
            # Reuse the generator output for the fake input to the discriminator
            disc_generated_output = self.discriminator([gen_output_seg_map_one_hot, condition], training=True)

            # 4. Calculate losses
            gen_total_loss = generator_loss(disc_generated_output, gen_output_logits, target_seg_map)
            disc_total_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # 5. Calculate and apply gradients
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_total_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_total_loss, disc_total_loss

    def train(self, epochs=EPOCHS):
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train for.
        """
        # Load data if not already loaded
        if not hasattr(self, 'train_dataset'):
             self.load_data()

        print(f"Starting training for {epochs} epochs...")
        steps_per_epoch = tf.data.experimental.cardinality(self.train_dataset).numpy()
        if steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
             print("Warning: Could not determine steps per epoch.")
             steps_per_epoch = 1 # Avoid division by zero if unknown

        for epoch in range(epochs):
            start_time = datetime.now()
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0

            # Iterate over batches
            for step, (batch_inputs, batch_target) in enumerate(self.train_dataset):
                gen_loss, disc_loss = self.train_step(batch_inputs, batch_target)
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss

                if step % 100 == 0: # Print progress periodically
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch} - "
                          f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

            # Calculate average losses for the epoch
            avg_gen_loss = epoch_gen_loss / steps_per_epoch
            avg_disc_loss = epoch_disc_loss / steps_per_epoch

            # Log average losses to TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('epoch_gen_loss', avg_gen_loss, step=epoch)
                tf.summary.scalar('epoch_disc_loss', avg_disc_loss, step=epoch)

            # Save checkpoint periodically (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Checkpoint saved for epoch {epoch+1}")

            # Generate and save sample images periodically (e.g., every 5 epochs)
            if (epoch + 1) % 5 == 0:
                 print("Generating sample images...")
                 self.generate_and_save_images(epoch + 1, self.sample_test_batch)

            time_taken = datetime.now() - start_time
            print(f"Epoch {epoch+1}/{epochs} completed in {time_taken}. "
                  f"Avg Gen Loss: {avg_gen_loss:.4f}, Avg Disc Loss: {avg_disc_loss:.4f}\n")

        # --- Post-Training ---
        # Save the final models
        print("Saving final models...")
        self.generator.save('generator_model.h5')
        self.discriminator.save('discriminator_model.h5')
        print("Models saved.")

        # Generate final samples
        print("Generating final sample images...")
        self.generate_and_save_images(epochs, self.sample_test_batch)

        print("Training completed!")


    def generate_and_save_images(self, epoch, sample_batch):
        """
        Generates images from the generator using a sample test batch and saves them.

        Args:
            epoch (int): Current epoch number (for naming files).
            sample_batch (tuple): A tuple containing ({'image': images, 'condition': conditions}, target_maps)
                                  from the test dataset.
        """
        sample_inputs = sample_batch[0]
        sample_target_maps = sample_batch[1]
        sample_images = sample_inputs['image']
        sample_conditions = sample_inputs['condition']

        try:
            predictions_logits = self.generator([sample_images, sample_conditions], training=False)
            predictions_seg_map = tf.argmax(predictions_logits, axis=-1) # Get class indices

            output_dir = os.path.join('generated_samples', f'epoch_{epoch}')
            os.makedirs(output_dir, exist_ok=True)

            num_samples_to_show = min(BATCH_SIZE, 4) # Show first few samples

            plt.figure(figsize=(15, 5 * num_samples_to_show))

            for i in range(num_samples_to_show):
                # --- Input Image (Normalized RGB) ---
                plt.subplot(num_samples_to_show, 3, i*3 + 1)
                plt.title("Input Image")
                # De-normalize for display
                input_display = (sample_images[i].numpy() + 1) * 127.5
                plt.imshow(input_display.astype(np.uint8))
                plt.axis('off')

                # --- Ground Truth Segmentation Map (Converted to RGB) ---
                plt.subplot(num_samples_to_show, 3, i*3 + 2)
                plt.title("Ground Truth")
                target_rgb = INDEX_TO_COLOR[sample_target_maps[i].numpy()]
                plt.imshow(target_rgb)
                plt.axis('off')

                # --- Predicted Segmentation Map (Converted to RGB) ---
                plt.subplot(num_samples_to_show, 3, i*3 + 3)
                plt.title("Prediction")
                pred_rgb = INDEX_TO_COLOR[predictions_seg_map[i].numpy()]
                plt.imshow(pred_rgb)
                plt.axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'comparison_epoch_{epoch}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved sample comparison image to {save_path}")

        except Exception as e:
            print(f"Error generating/saving images for epoch {epoch}: {e}")


    def generate_floor_plan_from_noise(self, plot_type, seed=None):
        """
        Generates a floor plan for a specific plot type using random noise as input.

        Args:
            plot_type (str): Type of plot ('5_marla', '10_marla', or '20_marla').
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            np.array: Generated floor plan as an RGB image (uint8).
        """
        if plot_type not in PLOT_TYPES:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Must be one of {PLOT_TYPES}")

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Create random noise input (matching the shape the generator expects for the 'image' input)
        # Note: The generator was trained using real images as input. Using pure noise might
        # produce less coherent results compared to translating an existing simple layout or sketch.
        # However, this function fulfills the request for noise-based generation.
        noise_input = tf.random.normal([1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])

        # Create condition vector
        condition = np.zeros((1, len(PLOT_TYPES)), dtype=np.float32)
        plot_idx = PLOT_TYPES.index(plot_type)
        condition[0, plot_idx] = 1
        condition_tensor = tf.convert_to_tensor(condition)

        # Generate floor plan logits
        prediction_logits = self.generator([noise_input, condition_tensor], training=False)

        # Convert prediction logits to segmentation map (indices)
        pred_map_indices = tf.argmax(prediction_logits[0], axis=-1).numpy() # Get batch 0

        # Convert segmentation map indices to RGB image
        rgb_image = INDEX_TO_COLOR[pred_map_indices]

        return rgb_image

    def export_floor_plan_png(self, rgb_image, output_path):
        """
        Exports a floor plan RGB image to a PNG file.

        Args:
            rgb_image (np.array): Floor plan as RGB image (H, W, 3).
            output_path (str): Path to save the floor plan (should end in .png).
        """
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir: # Avoid error if saving in current directory
                os.makedirs(output_dir, exist_ok=True)

            # Use matplotlib to save, ensuring no axes or extra whitespace
            height, width, _ = rgb_image.shape
            dpi = 96 # Standard DPI, adjust if needed
            fig_height = height / dpi
            fig_width = width / dpi

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1]) # Fill the entire figure
            ax.imshow(rgb_image)
            ax.axis('off')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig) # Close the figure to free memory
            print(f"Saved PNG floor plan to: {output_path}")

        except Exception as e:
            print(f"Error saving PNG to {output_path}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FOLDER = 'dataset'
    OUTPUT_GENERATION_FOLDER = 'output_generated'
    RUN_TRAINING = True # Set to False to skip training if models/checkpoints exist
    NUM_EXAMPLES_TO_GENERATE = 3 # Number of examples per plot type

    # --- Initialization ---
    print("Initializing Floor Plan Generator...")
    generator_instance = FloorPlanGenerator(DATASET_FOLDER)

    # --- Training ---
    if RUN_TRAINING:
        print("\n--- Starting Model Training ---")
        generator_instance.train(epochs=EPOCHS) # Use default epochs (80)
        print("\n--- Training Finished ---")
    else:
        print("\n--- Skipping Training (RUN_TRAINING=False) ---")
        # Ensure checkpoint is loaded if skipping training
        if not tf.train.latest_checkpoint(generator_instance.checkpoint_dir):
             print("Warning: Skipping training, but no checkpoint found. Generation might fail or use random weights.")


    # --- Generation ---
    print(f"\n--- Generating {NUM_EXAMPLES_TO_GENERATE} Examples per Plot Type ---")
    for plot_type in PLOT_TYPES:
        print(f"Generating for: {plot_type}")
        for i in range(NUM_EXAMPLES_TO_GENERATE):
            try:
                # Generate floor plan using random noise
                floor_plan_rgb = generator_instance.generate_floor_plan_from_noise(plot_type, seed=i) # Use index as seed

                # Define output path
                output_filename = f'floor_plan_{plot_type}_{i+1}.png'
                output_path = os.path.join(OUTPUT_GENERATION_FOLDER, plot_type, output_filename)

                # Export to PNG
                generator_instance.export_floor_plan_png(floor_plan_rgb, output_path)

            except Exception as e:
                print(f"Error generating or exporting {plot_type} example {i+1}: {e}")

    print("\n--- Floor Plan Generation Examples Complete! ---")


# --- Example Workflow Comment (Updated) ---
"""
Example Workflow:

1. Prepare Dataset:
   - Ensure your dataset is organized in './dataset/' with subfolders:
     'dataset/5_marla/*.png'
     'dataset/10_marla/*.png'
     'dataset/20_marla/*.png'

2. Install Dependencies:
   pip install tensorflow numpy matplotlib scikit-learn Pillow

3. (Optional) Generate Metadata for Analysis (Separate Script):
   # If you need detailed metadata analysis, run the metadata script:
   # python generate_metadata.py --dataset dataset --output data/raw

4. Run Training and Generation:
   # This script handles both training and generation based on the RUN_TRAINING flag.
   # It will train (or load checkpoint), then generate examples.
   python floor_plan_generator.py

   - Training checkpoints are saved in './training_checkpoints/'
   - TensorBoard logs are saved in './logs/'
   - Sample comparisons during training are saved in './generated_samples/'
   - Final generated examples (from noise) are saved in './output_generated/'

5. Monitor Training (Optional):
   # Open a separate terminal in the project directory and run:
   # tensorboard --logdir logs

"""
# --- END OF FILE floor_plan_generator.py ---