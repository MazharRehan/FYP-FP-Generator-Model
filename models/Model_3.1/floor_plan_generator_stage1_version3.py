# floor_plan_generator_stage1_version3.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import cv2
import time # For epoch timing

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Data parameters
    DATA_DIR = "dataset/"
    METADATA_PATH = "floor_plan_metadata_v5_includingArea.csv"
    IMAGE_WIDTH = 849  # Original, for reference
    IMAGE_HEIGHT = 1570 # Original, for reference
    PLOT_SIZE = "10Marla" # Focusing on one plot size for now

    # Model parameters
    LATENT_DIM = 100
    CONDITION_DIM = 16 # Keep as per simplified, can be expanded
    TARGET_SIZE = 128  # Output image size (square)
    IMG_CHANNELS = 2   # 0: Walls, 1: Interior Space

    # Training parameters
    BATCH_SIZE = 4 # Try a slightly larger batch size if memory allows
    EPOCHS = 200   # GANs often need more epochs
    LEARNING_RATE_G = 0.0001
    LEARNING_RATE_D = 0.0001 # Can be same or different
    BETA_1 = 0.5
    BETA_2 = 0.999

    # LSGAN labels
    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0 # For LSGAN, can also be -1.0 or other values. Using 0 for simplicity.

    # Output directories
    CHECKPOINT_DIR = "checkpoints/stage1_v3/"
    LOG_DIR = "logs/stage1_v3/" # Not implementing TensorBoard explicitly here for brevity
    SAMPLE_DIR = "samples/stage1_v3/"

# Create directories
for dir_path in [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.SAMPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Data Loading and Preprocessing ---
def load_metadata():
    df = pd.read_csv(Config.METADATA_PATH)
    df = df[df['PlotSize'] == Config.PLOT_SIZE]
    df = df[df['Version'] == 'V01']
    return df


def preprocess_image(image_path):
    """Load and preprocess image for Stage 1 (Walls and Interior Space)."""
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)  # img is uint8

        # 1. Walls (black pixels) - Keep as boolean initially
        # Assuming walls are RGB(0,0,0)
        walls_mask_bool = np.all(img == [0, 0, 0], axis=-1)  # Boolean mask

        # 2. Background (white pixels) - Keep as boolean
        # Assuming background is RGB(255,255,255)
        background_mask_bool = np.all(img == [255, 255, 255], axis=-1)  # Boolean mask

        # 3. Interior Space (non-wall, non-background) - Perform logical operations on booleans
        # Interior space is anything that is NOT a wall AND NOT background
        interior_space_mask_bool = np.logical_not(walls_mask_bool) & np.logical_not(background_mask_bool)
        # Or, equivalently and perhaps more clearly:
        # interior_space_mask_bool = ~(walls_mask_bool | background_mask_bool)

        # Convert boolean masks to float32 for stacking
        walls_float = walls_mask_bool.astype(np.float32)
        interior_space_float = interior_space_mask_bool.astype(np.float32)

        # Stack channels: [walls, interior_space]
        target_map = np.stack([walls_float, interior_space_float], axis=-1)

        # Resize
        resized_map = cv2.resize(target_map,
                                 (Config.TARGET_SIZE, Config.TARGET_SIZE),
                                 interpolation=cv2.INTER_NEAREST)  # Crucial for masks

        # Normalize to [-1, 1] for tanh output
        resized_map = (resized_map * 2.0) - 1.0

        return resized_map
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Ensure the returned zero array matches the expected channel order if needed for debugging later
        return np.zeros((Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMG_CHANNELS), dtype=np.float32)


def prepare_condition_vector(row):
    """Extract conditional inputs - same as simplified for now."""
    room_counts = []
    room_counts.append(row.get('Count_Bathroom', 0) / 4.0) # Normalize
    room_counts.append(row.get('Count_Bedroom', 0) / 3.0)  # Normalize
    room_types = ['DrawingRoom', 'Kitchen', 'Dining', 'Lounge', 'Garage']
    for room in room_types:
        room_counts.append(min(float(row.get(f'Count_{room}', 0)), 1.0)) # Presence
    room_counts.append(row.get('TotalAreaSqFt', 2275.0) / 2275.0) # Normalize area

    condition_vector = np.array(room_counts, dtype=np.float32)
    # Pad or truncate to Config.CONDITION_DIM
    current_len = len(condition_vector)
    if current_len < Config.CONDITION_DIM:
        padding = np.zeros(Config.CONDITION_DIM - current_len, dtype=np.float32)
        condition_vector = np.concatenate([condition_vector, padding])
    else:
        condition_vector = condition_vector[:Config.CONDITION_DIM]
    return condition_vector

def data_generator(dataframe, batch_size):
    num_samples = len(dataframe)
    while True:
        df_shuffled = dataframe.sample(frac=1.0) # Shuffle each epoch
        for i in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[i:i+batch_size]
            batch_images = []
            batch_conditions = []
            for _, row in batch_df.iterrows():
                image_path = os.path.join(Config.DATA_DIR, row['FilePath'])
                img = preprocess_image(image_path)
                batch_images.append(img)
                condition = prepare_condition_vector(row)
                batch_conditions.append(condition)
            yield np.array(batch_images), np.array(batch_conditions)

# --- U-Net Generator ---
def downsample_block(x, filters, kernel_size=4, strides=2, apply_batchnorm=True, name_prefix=""):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=f"{name_prefix}_conv")(x)
    if apply_batchnorm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.LeakyReLU(name=f"{name_prefix}_leakyrelu")(x)
    return x

def upsample_block(x, skip_input, filters, kernel_size=4, strides=2, apply_dropout=False, name_prefix=""):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=f"{name_prefix}_convtranspose")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    if apply_dropout:
        x = layers.Dropout(0.5, name=f"{name_prefix}_dropout")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip_input])
    return x

def build_generator_unet():
    noise_input = layers.Input(shape=(Config.LATENT_DIM,), name="noise_input")
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name="condition_input")

    # Process condition and merge with noise
    cond_dense = layers.Dense(Config.TARGET_SIZE * Config.TARGET_SIZE * 1, name="cond_dense_initial")(condition_input) # Expand to image size with 1 channel
    cond_reshaped = layers.Reshape((Config.TARGET_SIZE, Config.TARGET_SIZE, 1), name="cond_reshaped_initial")(cond_dense)

    # Initial projection for noise to match image dimensions roughly for a deeper network if needed
    # For a U-Net, we typically start from the image directly or a projected noise.
    # Here, we'll project noise to a small feature map and concatenate with upsampled condition later.

    # Generator U-Net Structure
    # Output size: Config.TARGET_SIZE (e.g., 128)
    # Encoder
    # Input is TARGET_SIZE x TARGET_SIZE x (IMG_CHANNELS for image-to-image, or projected noise for noise-to-image)
    # For this conditional GAN, we start by projecting noise + condition to an initial feature map

    # Project noise to a small spatial dimension to start the "image" path
    # For a 128x128 output, starting at 4x4 or 8x8 is common.
    # Let's start by projecting combined noise + condition to an initial small feature map
    initial_filters = 512
    s = 4 # starting size
    
    merged_input = layers.Concatenate(name="merge_noise_cond")([noise_input, condition_input])
    x = layers.Dense(s * s * initial_filters, use_bias=False, name="initial_dense_projection")(merged_input)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.LeakyReLU(name="initial_leakyrelu")(x)
    x = layers.Reshape((s, s, initial_filters), name="initial_reshape")(x) # e.g., (4, 4, 512)

    # Encoder path for U-Net (if starting from a base image/projection)
    # For pure generation, we can see it as an expansive path
    # Example U-Net structure (adjust filter counts and layers based on TARGET_SIZE)
    # For 128x128 output: 4->8->16->32->64->128
    
    # Let's use a more traditional U-Net structure assuming input is an image.
    # For generation, the "input" to the encoder is often the projected noise.
    # However, we are generating based on noise and condition.
    # Let's build an expansive path primarily.
    
    # We'll simplify and make it more like a DCGAN generator expanded
    # but with skip connections conceptually. A true image-to-image U-Net has a clear input image.
    # Let's try an architecture that progressively builds up features.

    # TARGET_SIZE = 128
    # Block 1: 4x4x512 -> 8x8x512
    up1 = layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False, name="up1_convtranspose")(x)
    up1 = layers.BatchNormalization(name="up1_bn")(up1)
    up1 = layers.ReLU(name="up1_relu")(up1) # (8, 8, 512)
    
    # Block 2: 8x8x512 -> 16x16x256
    up2 = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False, name="up2_convtranspose")(up1)
    up2 = layers.BatchNormalization(name="up2_bn")(up2)
    up2 = layers.ReLU(name="up2_relu")(up2) # (16, 16, 256)

    # Block 3: 16x16x256 -> 32x32x128
    up3 = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False, name="up3_convtranspose")(up2)
    up3 = layers.BatchNormalization(name="up3_bn")(up3)
    up3 = layers.ReLU(name="up3_relu")(up3) # (32, 32, 128)

    # Block 4: 32x32x128 -> 64x64x64
    up4 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, name="up4_convtranspose")(up3)
    up4 = layers.BatchNormalization(name="up4_bn")(up4)
    up4 = layers.ReLU(name="up4_relu")(up4) # (64, 64, 64)

    # Block 5: 64x64x64 -> 128x128x(IMG_CHANNELS)
    # Inject condition here if not done earlier, or ensure it's part of the flow
    # Let's reshape and upsample condition_input to match up4's spatial dimensions if needed
    # For simplicity in this version, we relied on the initial merged_input
    
    output_image = layers.Conv2DTranspose(Config.IMG_CHANNELS, 4, strides=2, padding='same',
                                           activation='tanh', name="final_convtranspose_output")(up4) # (128, 128, 2)

    return models.Model(inputs=[noise_input, condition_input], outputs=output_image, name="generator_unet_style")


# --- PatchGAN Discriminator ---
def build_discriminator_patchgan():
    image_input = layers.Input(shape=(Config.TARGET_SIZE, Config.TARGET_SIZE, Config.IMG_CHANNELS), name="image_input")
    condition_input = layers.Input(shape=(Config.CONDITION_DIM,), name="condition_input")

    # Process and reshape condition to concatenate with image features
    # Expand condition to match spatial dimensions of one of the conv layers
    # Example: for a 32x32 patch output, condition could be shaped to 32x32xN
    # For PatchGAN, condition is often concatenated after some conv layers on the image
    # Let's make condition spatial to merge with image features
    # Target patch size could be TARGET_SIZE / (2^4) = 128/16 = 8x8 for 4 downsampling layers
    cond_channels = 16 # Number of channels for processed condition
    cond_spatial_size = Config.TARGET_SIZE // (2**4) # e.g. 128 / 16 = 8
    
    # Process condition to be spatial
    c = layers.Dense(cond_spatial_size * cond_spatial_size * cond_channels, name="disc_cond_dense")(condition_input)
    c = layers.Reshape((cond_spatial_size, cond_spatial_size, cond_channels), name="disc_cond_reshape")(c) # (8,8,16)

    # Discriminator path
    # Output Patch size related to number of strided convolutions.
    # For TARGET_SIZE=128, 4 downsampling layers of stride 2: 128->64->32->16->8. Patch is 8x8.
    
    # Layer 1: 128x128x2 -> 64x64x64
    d1 = downsample_block(image_input, 64, apply_batchnorm=False, name_prefix="d1") # No BN on first layer is common
    # Layer 2: 64x64x64 -> 32x32x128
    d2 = downsample_block(d1, 128, name_prefix="d2")
    # Layer 3: 32x32x128 -> 16x16x256
    d3 = downsample_block(d2, 256, name_prefix="d3")
    # Layer 4: 16x16x256 -> 8x8x512
    d4 = downsample_block(d3, 512, name_prefix="d4") # (8,8,512)

    # Concatenate condition features with image features
    merged_features = layers.Concatenate(name="disc_merge_img_cond")([d4, c]) # (8,8, 512+cond_channels)

    # Final layer for PatchGAN output (no sigmoid for LSGAN/WGAN)
    # For LSGAN, the output is the raw value.
    patch_output = layers.Conv2D(1, 4, strides=1, padding='same', name="patchgan_output")(merged_features) # (8,8,1)

    return models.Model(inputs=[image_input, condition_input], outputs=patch_output, name="discriminator_patchgan")

# --- LSGAN Training Step ---
class LSGAN(models.Model):
    # ... (init and compile methods as before with clipping) ...

    def train_step(self, data):
        real_images, conditions = data
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, Config.LATENT_DIM])

        tf.print("--- Discriminator Training ---")
        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            fake_images = self.generator([noise, conditions], training=True)
            tf.print("fake_images (min, max, has_nan):", tf.reduce_min(fake_images), tf.reduce_max(fake_images),
                     tf.reduce_any(tf.math.is_nan(fake_images)))

            real_output = self.discriminator([real_images, conditions], training=True)
            tf.print("real_output (min, max, has_nan):", tf.reduce_min(real_output), tf.reduce_max(real_output),
                     tf.reduce_any(tf.math.is_nan(real_output)))

            fake_output = self.discriminator([fake_images, conditions], training=True)  # Pass the *same* fake_images
            tf.print("fake_output (min, max, has_nan):", tf.reduce_min(fake_output), tf.reduce_max(fake_output),
                     tf.reduce_any(tf.math.is_nan(fake_output)))

            disc_real_loss = self.mse_loss(tf.ones_like(real_output) * Config.REAL_LABEL, real_output)
            tf.print("disc_real_loss (has_nan):", disc_real_loss, tf.reduce_any(tf.math.is_nan(disc_real_loss)))

            disc_fake_loss = self.mse_loss(tf.ones_like(fake_output) * Config.FAKE_LABEL, fake_output)
            tf.print("disc_fake_loss (has_nan):", disc_fake_loss, tf.reduce_any(tf.math.is_nan(disc_fake_loss)))

            disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
            tf.print("disc_loss (has_nan):", disc_loss, tf.reduce_any(tf.math.is_nan(disc_loss)))

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # tf.print("disc_gradients (first few, check for NaNs):", [tf.reduce_any(tf.math.is_nan(g)) if g is not None else False for g in disc_gradients[:3]])

        # Apply gradient clipping for discriminator (as implemented before)
        if self.disc_clip_value is not None:
            clipped_disc_gradients = []
            for i, grad in enumerate(disc_gradients):
                if grad is not None:
                    # tf.print(f"Disc grad {i} before clip (min, max, has_nan):", tf.reduce_min(grad), tf.reduce_max(grad), tf.reduce_any(tf.math.is_nan(grad)))
                    clipped_grad = tf.clip_by_value(grad, -self.disc_clip_value, self.disc_clip_value)
                    # tf.print(f"Disc grad {i} after clip (min, max, has_nan):", tf.reduce_min(clipped_grad), tf.reduce_max(clipped_grad), tf.reduce_any(tf.math.is_nan(clipped_grad)))
                    clipped_disc_gradients.append(clipped_grad)
                else:
                    clipped_disc_gradients.append(None)
            disc_gradients = clipped_disc_gradients

        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        tf.print("\n--- Generator Training ---")

        # Train Generator
        with tf.GradientTape() as gen_tape:
            fake_images_for_gen = self.generator([noise, conditions], training=True)  # Generator is in training mode
            tf.print("fake_images_for_gen (min, max, has_nan):", tf.reduce_min(fake_images_for_gen),
                     tf.reduce_max(fake_images_for_gen), tf.reduce_any(tf.math.is_nan(fake_images_for_gen)))

            # Discriminator is used for inference by the generator, its weights are fixed during this step
            fake_output_for_gen = self.discriminator([fake_images_for_gen, conditions],
                                                     training=False)  # CHANGED training=False
            tf.print("fake_output_for_gen (min, max, has_nan):", tf.reduce_min(fake_output_for_gen),
                     tf.reduce_max(fake_output_for_gen), tf.reduce_any(tf.math.is_nan(fake_output_for_gen)))

            gen_loss = self.mse_loss(tf.ones_like(fake_output_for_gen) * Config.REAL_LABEL, fake_output_for_gen)
            tf.print("gen_loss (has_nan):", gen_loss, tf.reduce_any(tf.math.is_nan(gen_loss)))

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        # tf.print("gen_gradients (first few, check for NaNs):", [tf.reduce_any(tf.math.is_nan(g)) if g is not None else False for g in gen_gradients[:3]])

        # Apply gradient clipping for generator (as implemented before)
        if self.gen_clip_value is not None:
            clipped_gen_gradients = []
            for i, grad in enumerate(gen_gradients):
                if grad is not None:
                    # tf.print(f"Gen grad {i} before clip (min, max, has_nan):", tf.reduce_min(grad), tf.reduce_max(grad), tf.reduce_any(tf.math.is_nan(grad)))
                    clipped_grad = tf.clip_by_value(grad, -self.gen_clip_value, self.gen_clip_value)
                    # tf.print(f"Gen grad {i} after clip (min, max, has_nan):", tf.reduce_min(clipped_grad), tf.reduce_max(clipped_grad), tf.reduce_any(tf.math.is_nan(clipped_grad)))
                    clipped_gen_gradients.append(clipped_grad)
                else:
                    clipped_gen_gradients.append(None)
            gen_gradients = clipped_gen_gradients

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        tf.print("--- End of Train Step ---")

        return {"d_loss": disc_loss, "g_loss": gen_loss}

# --- Sample Generation ---
def generate_samples(generator, condition_samples, epoch, prefix=""):
    noise = tf.random.normal([len(condition_samples), Config.LATENT_DIM])
    generated_images = generator([noise, condition_samples], training=False)
    generated_images = (generated_images + 1.0) / 2.0 # Denormalize from [-1,1] to [0,1]

    fig, axs = plt.subplots(len(condition_samples), Config.IMG_CHANNELS,
                            figsize=(Config.IMG_CHANNELS * 3, len(condition_samples) * 3))
    if len(condition_samples) == 1:
        axs = np.expand_dims(axs, axis=0) # Ensure axs is 2D for single sample

    for i in range(len(condition_samples)):
        # Walls
        axs[i, 0].imshow(generated_images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axs[i, 0].set_title(f"Sample {i+1} - Walls")
        axs[i, 0].axis('off')
        # Interior Space
        axs[i, 1].imshow(generated_images[i, :, :, 1], cmap='gray', vmin=0, vmax=1) # Use gray for binary mask
        axs[i, 1].set_title(f"Sample {i+1} - Interior")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAMPLE_DIR, f"{prefix}samples_epoch_{epoch:04d}.png"))
    plt.close(fig)

# --- Training Script ---
def train():
    tf.config.run_functions_eagerly(True) # Uncomment for debugging, slows down training
    print("Loading metadata...")
    df = load_metadata()
    print(f"Found {len(df)} floor plans for {Config.PLOT_SIZE} (V01 only).")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    train_dataset = data_generator(train_df, Config.BATCH_SIZE)
    # val_dataset = data_generator(val_df, Config.BATCH_SIZE) # For validation if needed

    print("Building models (U-Net Generator, PatchGAN Discriminator)...")
    generator = build_generator_unet()
    discriminator = build_discriminator_patchgan()

    print("Generator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()

    gan = LSGAN(generator, discriminator)

    gen_optimizer = optimizers.Adam(Config.LEARNING_RATE_G, beta_1=Config.BETA_1, beta_2=Config.BETA_2)
    disc_optimizer = optimizers.Adam(Config.LEARNING_RATE_D, beta_1=Config.BETA_1, beta_2=Config.BETA_2)

    # MODIFIED compile call
    gan.compile(gen_optimizer=gen_optimizer,
                disc_optimizer=disc_optimizer,
                gen_clip_value=1.0,  # Clip generator gradients to [-1.0, 1.0]
                disc_clip_value=1.0)  # Clip discriminator gradients to [-1.0, 1.0]

    # Prepare sample conditions for visualization during training
    sample_rows = val_df.sample(min(4, len(val_df))) # Take up to 4 samples from validation set
    sample_conditions_np = np.array([prepare_condition_vector(row) for _, row in sample_rows.iterrows()])

    print(f"Starting LSGAN training for {Config.EPOCHS} epochs...")
    steps_per_epoch = len(train_df) // Config.BATCH_SIZE

    for epoch in range(Config.EPOCHS):
        epoch_start_time = time.time()
        total_g_loss = 0
        total_d_loss = 0
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        for step in range(steps_per_epoch):
            real_imgs, conds = next(train_dataset)
            if real_imgs.shape[0] != Config.BATCH_SIZE: # Skip partial batches at the end
                continue
            losses = gan.train_step([real_imgs, conds])
            total_d_loss += losses['d_loss']
            total_g_loss += losses['g_loss']

            if (step + 1) % max(1, steps_per_epoch // 5) == 0 : # Print 5 times per epoch
                print(f"  Step {step+1}/{steps_per_epoch} - D Loss: {losses['d_loss']:.4f}, G Loss: {losses['g_loss']:.4f}")
        
        avg_d_loss = total_d_loss / steps_per_epoch
        avg_g_loss = total_g_loss / steps_per_epoch
        epoch_time_taken = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} summary: Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, Time: {epoch_time_taken:.2f}s")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if len(sample_conditions_np) > 0:
                generate_samples(generator, sample_conditions_np, epoch + 1, prefix=f"{Config.PLOT_SIZE}_")
                print(f"Generated samples for epoch {epoch+1}.")

        if (epoch + 1) % 20 == 0: # Save checkpoints every 20 epochs
            generator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'generator_epoch_{epoch+1:04d}.h5'))
            discriminator.save_weights(os.path.join(Config.CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1:04d}.h5'))
            print(f"Saved model weights at epoch {epoch+1}")

    print("\nTraining complete!")
    try:
        generator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_generator')) # Save as SavedModel
        discriminator.save(os.path.join(Config.CHECKPOINT_DIR, 'final_discriminator'))
        print("Final generator and discriminator models saved in SavedModel format.")
    except Exception as e:
        print(f"Error saving final models: {e}")

if __name__ == "__main__":
    train()