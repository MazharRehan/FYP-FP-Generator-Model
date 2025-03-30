import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import json
from datetime import datetime

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define parameters
BUFFER_SIZE = 400
BATCH_SIZE = 16
IMG_HEIGHT = 256  # Resized height
IMG_WIDTH = 256   # Resized width
NUM_CHANNELS = 3  # RGB
EPOCHS = 200
PLOT_TYPES = ['5_marla', '10_marla', '20_marla']
# Number of output channels = number of room types + walls
NUM_OUTPUT_CHANNELS = 24  # Based on your color coding scheme

# Define the color mapping for visualization
COLOR_MAP = {
    'Bedroom': [255, 0, 0],          # Red
    'Bathroom': [0, 0, 255],         # Blue
    'Kitchen': [255, 165, 0],        # Orange
    'Drawing Room': [0, 128, 0],     # Green
    'Garage': [165, 42, 42],         # Brown
    'Lounge': [255, 255, 0],         # Yellow
    'Backyard': [50, 205, 50],       # Lime Green
    'Stairs': [0, 128, 128],         # Teal
    'Storage': [128, 0, 128],        # Purple
    'Open Space': [0, 255, 255],     # Cyan
    'Prayer Room': [127, 127, 127],  # Crimson
    'Staircase': [153, 51, 255],     # Violet
    'Lobby': [255, 0, 255],          # Magenta
    'Lawn': [64, 224, 208],          # Turquoise
    'Dining': [255, 192, 203],       # Pink
    'Servant': [75, 0, 130],         # Indigo
    'Passage': [128, 128, 0],        # Olive Green
    'Laundry': [230, 230, 250],      # Lavender
    'Dressing': [255, 127, 80],      # Coral
    'Side Garden': [255, 215, 0],    # Gold
    'Library': [255, 191, 0],        # Amber
    'Walls': [0, 0, 0],              # Black
    'Door': [128, 0, 0],             # Mahogany
    'Background': [255, 255, 255]    # White
}

# Create reverse mapping from RGB to room type index
RGB_TO_INDEX = {}
INDEX_TO_COLOR = []
for i, (room_type, color) in enumerate(COLOR_MAP.items()):
    RGB_TO_INDEX[tuple(color)] = i
    INDEX_TO_COLOR.append(color)

INDEX_TO_COLOR = np.array(INDEX_TO_COLOR)

def load_dataset(dataset_path, plot_types=None):
    """
    Load dataset from the given path for specified plot types.
    
    Args:
        dataset_path: Path to the dataset directory
        plot_types: List of plot types to include (e.g., ['5_marla', '10_marla'])
        
    Returns:
        images: List of floor plan images
        conditions: List of condition vectors (plot type one-hot encoded)
    """
    if plot_types is None:
        plot_types = PLOT_TYPES
        
    images = []
    conditions = []
    
    for i, plot_type in enumerate(plot_types):
        plot_dir = os.path.join(dataset_path, plot_type)
        
        # Skip if directory doesn't exist
        if not os.path.exists(plot_dir):
            print(f"Warning: Directory {plot_dir} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        
        for file in files:
            file_path = os.path.join(plot_dir, file)
            
            # Load and preprocess image
            img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img)
            
            # Normalize to [-1, 1]
            img_array = (img_array / 127.5) - 1
            
            # Create one-hot encoding for plot type condition
            condition = np.zeros(len(PLOT_TYPES))
            condition[i] = 1
            
            images.append(img_array)
            conditions.append(condition)
    
    return np.array(images), np.array(conditions)

def preprocess_image_to_segmentation_map(image):
    """
    Convert an RGB image to a segmentation map based on color mapping.
    Each pixel will be assigned a class index based on its RGB value.
    
    Args:
        image: RGB image array with values in range [-1, 1]
        
    Returns:
        segmentation_map: Array of class indices
    """
    # Convert from [-1, 1] to [0, 255]
    img = ((image + 1) * 127.5).astype(np.uint8)
    
    # Initialize segmentation map with background class
    segmentation_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Assign class indices based on RGB values
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rgb = tuple(img[y, x])
            # Find closest color in the color map
            closest_color = min(RGB_TO_INDEX.keys(),
                                key = lambda c: sum((int(c[i]) - int(rgb[i])) ** 2 for i in range(3)))

            # key=lambda c: sum((c[i] - rgb[i])**2 for i in range(3)))
            segmentation_map[y, x] = RGB_TO_INDEX[closest_color]
    
    return segmentation_map

def preprocess_dataset(images, conditions):
    """
    Preprocess the dataset for training.
    
    Args:
        images: Array of floor plan images
        conditions: Array of condition vectors
        
    Returns:
        segmentation_maps: Array of segmentation maps
        conditions: Array of condition vectors
    """
    segmentation_maps = []
    
    for img in images:
        segmentation_map = preprocess_image_to_segmentation_map(img)
        segmentation_maps.append(segmentation_map)
    
    return np.array(segmentation_maps), conditions

def generator_loss(disc_generated_output, gen_output, target):
    """
    Loss function for the generator.
    
    Args:
        disc_generated_output: Discriminator output on generated images
        gen_output: Generator output
        target: Ground truth segmentation maps
        
    Returns:
        total_loss: Combined adversarial and pixel-wise loss
    """
    # Binary cross entropy loss for the GAN
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output)
    
    # Pixel-wise loss (sparse categorical crossentropy for segmentation)
    pixel_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        target, gen_output)
    
    # Total loss
    total_loss = gan_loss + 100 * pixel_loss  # Higher weight on pixel loss
    
    return total_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Loss function for the discriminator.
    
    Args:
        disc_real_output: Discriminator output on real images
        disc_generated_output: Discriminator output on generated images
        
    Returns:
        total_loss: Discriminator loss
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_loss = real_loss + generated_loss
    
    return total_loss

def upsample_block(x, filters, kernel_size=4, strides=2, apply_dropout=False):
    """
    Upsampling block for the generator.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of the kernel
        strides: Stride length
        apply_dropout: Whether to apply dropout
        
    Returns:
        x: Output tensor
    """
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, 
                              padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    
    x = layers.LeakyReLU(0.2)(x)
    
    return x

def downsample_block(x, filters, kernel_size=4, strides=2, apply_batchnorm=True):
    """
    Downsampling block for the generator and discriminator.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of the kernel
        strides: Stride length
        apply_batchnorm: Whether to apply batch normalization
        
    Returns:
        x: Output tensor
    """
    x = layers.Conv2D(filters, kernel_size, strides=strides, 
                     padding='same', use_bias=False)(x)
    
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    
    x = layers.LeakyReLU(0.2)(x)
    
    return x

def build_generator():
    """
    Build the generator model (U-Net architecture).
    
    Returns:
        model: Generator model
    """
    # Input: Random noise + condition vector
    noise_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
    condition_input = layers.Input(shape=[len(PLOT_TYPES)])
    
    # Expand condition to match spatial dimensions
    condition_expanded = layers.Dense(IMG_HEIGHT * IMG_WIDTH)(condition_input)
    condition_expanded = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(condition_expanded)
    
    # Concatenate noise and condition
    x = layers.Concatenate()([noise_input, condition_expanded])
    
    # Encoder
    # For simplicity, we apply an initial convolution to combine the inputs
    x = layers.Conv2D(64, 4, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Downsampling blocks
    encoder_outputs = []
    down_filters = [64, 128, 256, 512, 512, 512]
    
    for filters in down_filters:
        x = downsample_block(x, filters)
        encoder_outputs.append(x)
    
    # Bottleneck
    bottleneck = downsample_block(encoder_outputs[-1], 512)
    
    # Decoder with skip connections
    up_filters = [512, 512, 512, 256, 128, 64]
    apply_dropout = [True, True, True, False, False, False]
    
    x = bottleneck
    
    for i, (filters, dropout) in enumerate(zip(up_filters, apply_dropout)):
        x = upsample_block(x, filters, apply_dropout=dropout)
        # Skip connection (except for the last layer)
        if i < len(encoder_outputs):
            x = layers.Concatenate()([x, encoder_outputs[-(i+1)]])
    
    # Output layer (num channels = number of room types)
    # output = layers.Conv2DTranspose(NUM_OUTPUT_CHANNELS, 4, strides=1, padding='same', activation=None)(x)
    output = layers.Conv2DTranspose(NUM_OUTPUT_CHANNELS, 4, strides=2,
                                    padding='same', activation=None)(x)

    return models.Model([noise_input, condition_input], output, name='generator')

def build_discriminator():
    """
    Build the discriminator model (PatchGAN).
    
    Returns:
        model: Discriminator model
    """
    # Input: Generated or real image + condition vector
    # image_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
    image_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, NUM_OUTPUT_CHANNELS])
    condition_input = layers.Input(shape=[len(PLOT_TYPES)])
    
    # Expand condition to match spatial dimensions
    condition_expanded = layers.Dense(IMG_HEIGHT * IMG_WIDTH)(condition_input)
    condition_expanded = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(condition_expanded)
    
    # Concatenate image and condition
    x = layers.Concatenate()([image_input, condition_expanded])
    
    # Downsampling blocks
    filters = [64, 128, 256, 512]
    
    for i, f in enumerate(filters):
        # For the first layer, don't apply batch normalization
        x = downsample_block(x, f, apply_batchnorm=(i != 0))
    
    # Output layer (PatchGAN)
    output = layers.Conv2D(1, 4, strides=1, padding='same')(x)
    
    return models.Model([image_input, condition_input], output, name='discriminator')

class FloorPlanGenerator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        
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
        
        # Initialize logs
        self.log_dir = 'logs/'
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            train_dataset: TensorFlow dataset for training
            test_dataset: TensorFlow dataset for testing
        """
        print("Loading dataset...")
        images, conditions = load_dataset(self.dataset_path)
        
        print(f"Found {len(images)} images.")
        if len(images) == 0:
            raise ValueError("No images found in the dataset.")
        
        print("Preprocessing dataset...")
        segmentation_maps, conditions = preprocess_dataset(images, conditions)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test, cond_train, cond_test = train_test_split(
            images, segmentation_maps, conditions, test_size=0.2, random_state=42
        )
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {'image': X_train, 'condition': cond_train}, y_train
        )).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            {'image': X_test, 'condition': cond_test}, y_test
        )).batch(BATCH_SIZE)
        
        return train_dataset, test_dataset
    
    @tf.function
    def train_step(self, input_image, target, condition):
        """
        Training step for the GAN.
        
        Args:
            input_image: Input image
            target: Target segmentation map
            condition: Condition vector
            
        Returns:
            gen_loss: Generator loss
            disc_loss: Discriminator loss
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake image
            gen_output = self.generator([input_image, condition], training=True)

            target_one_hot = tf.one_hot(target, depth=NUM_OUTPUT_CHANNELS)

            # Discriminator outputs
            # disc_real_output = self.discriminator([input_image, condition], training=True)
            # disc_real_output = self.discriminator([target, condition], training=True)
            disc_real_output = self.discriminator([target_one_hot, condition], training=True)
            # disc_generated_output = self.discriminator([gen_output, condition], training=True)
            gen_output_one_hot = tf.one_hot(tf.argmax(gen_output, axis=-1), depth=NUM_OUTPUT_CHANNELS)
            disc_generated_output = self.discriminator([gen_output_one_hot, condition], training=True)

            # Calculate losses
            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
        # Calculate gradients
        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def train(self, epochs=EPOCHS):
        """
        Train the GAN.
        
        Args:
            epochs: Number of epochs
        """
        train_dataset, test_dataset = self.load_and_preprocess_data()
        
        print("Starting training...")
        for epoch in range(epochs):
            start = datetime.now()
            
            # Training
            for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataset):
                input_image = batch_inputs['image']
                condition = batch_inputs['condition']
                
                gen_loss, disc_loss = self.train_step(input_image, batch_targets, condition)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx} - "
                          f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
                    
                    # Log to TensorBoard
                    with self.summary_writer.as_default():
                        tf.summary.scalar('gen_loss', gen_loss, step=epoch * len(train_dataset) + batch_idx)
                        tf.summary.scalar('disc_loss', disc_loss, step=epoch * len(train_dataset) + batch_idx)
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                
                # Generate and save sample images
                self.generate_and_save_images(epoch + 1, test_dataset)
            
            print(f"Time taken for epoch {epoch+1}: {datetime.now() - start}")
        
        # Save the final model
        self.generator.save('generator_model.h5')
        self.discriminator.save('discriminator_model.h5')
        
        # Generate final samples
        self.generate_and_save_images(epochs, test_dataset)
        
        print("Training completed!")
    
    def generate_and_save_images(self, epoch, test_dataset):
        """
        Generate and save sample floor plans.
        
        Args:
            epoch: Current epoch
            test_dataset: Test dataset
        """
        # Take a sample from the test dataset
        for batch_inputs, _ in test_dataset.take(1):
            input_image = batch_inputs['image']
            condition = batch_inputs['condition']
            
            # Generate images for each condition type
            for i, plot_type in enumerate(PLOT_TYPES):
                # Create condition vector for this plot type
                # cond = np.zeros((1, len(PLOT_TYPES)))
                # cond[0, i] = 1
                cond = tf.convert_to_tensor(np.eye(len(PLOT_TYPES))[i:i + 1], dtype=tf.float32)
                
                # Generate floor plan
                prediction = self.generator([input_image[:1], cond], training=False)
                
                # Convert prediction to segmentation map
                pred_map = tf.argmax(prediction[0], axis=-1)
                
                # Convert segmentation map to RGB image
                rgb_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                
                for y in range(IMG_HEIGHT):
                    for x in range(IMG_WIDTH):
                        class_idx = pred_map[y, x].numpy()
                        rgb_image[y, x] = INDEX_TO_COLOR[class_idx]
                
                # Save the generated image
                output_dir = f'generated_samples/epoch_{epoch}'
                os.makedirs(output_dir, exist_ok=True)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(rgb_image)
                plt.axis('off')
                plt.title(f'Generated {plot_type} Floor Plan')
                plt.savefig(f'{output_dir}/{plot_type}_sample.png')
                plt.close()
    
    def generate_floor_plan(self, plot_type, seed=None):
        """
        Generate a floor plan for a specific plot type.
        
        Args:
            plot_type: Type of plot ('5_marla', '10_marla', or '20_marla')
            seed: Random seed for reproducibility
            
        Returns:
            rgb_image: Generated floor plan as RGB image
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
        # Create random noise
        noise = np.random.normal(0, 1, (1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        
        # Create condition vector
        condition = np.zeros((1, len(PLOT_TYPES)))
        plot_idx = PLOT_TYPES.index(plot_type)
        condition[0, plot_idx] = 1
        
        # Generate floor plan
        prediction = self.generator([noise, condition], training=False)
        
        # Convert prediction to segmentation map
        pred_map = tf.argmax(prediction[0], axis=-1)
        
        # Convert segmentation map to RGB image
        rgb_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        for y in range(IMG_HEIGHT):
            for x in range(IMG_WIDTH):
                class_idx = pred_map[y, x].numpy()
                rgb_image[y, x] = INDEX_TO_COLOR[class_idx]
        
        return rgb_image
    
    def export_floor_plan(self, rgb_image, output_path, format='png'):
        """
        Export a floor plan to file.
        
        Args:
            rgb_image: Floor plan as RGB image
            output_path: Path to save the floor plan
            format: Output format ('png', 'svg', or 'dxf')
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'png':
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        elif format == 'svg':
            # Basic SVG export
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
            plt.close()
            
        elif format == 'dxf':
            # DXF export requires more complex conversion
            # This is a placeholder - you'll need a library like ezdxf for proper DXF export
            print("DXF export feature requires additional implementation with a library like ezdxf.")
            print("Saving as PNG instead for now.")
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(output_path.replace('.dxf', '.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else:
            raise ValueError(f"Unsupported format: {format}")

# Example usage
if __name__ == "__main__":
    # Initialize the floor plan generator
    generator = FloorPlanGenerator('dataset')
    
    # Train the model
    generator.train(epochs=EPOCHS)
    
    # Generate and export floor plans
    for plot_type in PLOT_TYPES:
        for i in range(5):  # Generate 5 examples of each type
            # Generate floor plan
            floor_plan = generator.generate_floor_plan(plot_type, seed=i)
            
            # Export to different formats
            generator.export_floor_plan(
                floor_plan, 
                f'output/{plot_type}/floor_plan_{i}.png', 
                format='png'
            )
            
            generator.export_floor_plan(
                floor_plan, 
                f'output/{plot_type}/floor_plan_{i}.svg', 
                format='svg'
            )
            
            generator.export_floor_plan(
                floor_plan, 
                f'output/{plot_type}/floor_plan_{i}.dxf', 
                format='dxf'
            )
    
    print("All floor plans generated and exported successfully!")