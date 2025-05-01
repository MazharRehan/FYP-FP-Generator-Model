import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_generator(latent_dim, condition_dim, output_shape=(256, 256, 3)):
    """
    Build generator model for CGAN.

    Args:
        latent_dim: Dimension of random noise input
        condition_dim: Dimension of condition vector (room counts)
        output_shape: Shape of output image

    Returns:
        Keras Model
    """
    # Random noise input
    noise_input = layers.Input(shape=(latent_dim,))

    # Condition input (room counts)
    condition_input = layers.Input(shape=(condition_dim,))

    # Concatenate noise and condition
    combined_input = layers.Concatenate()([noise_input, condition_input])

    # First dense layer with enough units for 8x8 feature maps with 256 channels
    x = layers.Dense(8 * 8 * 256)(combined_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)

    # Transposed convolution blocks (upsampling)
    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 32x32 -> 64x64
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 64x64 -> 128x128
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 128x128 -> 256x256
    x = layers.Conv2DTranspose(16, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Final output layer with tanh activation (generates images in [-1, 1])
    output = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)

    # Create and return model
    model = models.Model([noise_input, condition_input], output)
    return model


def build_discriminator(input_shape=(256, 256, 3), condition_dim=None):
    """
    Build discriminator model for CGAN.

    Args:
        input_shape: Shape of input image
        condition_dim: Dimension of condition vector

    Returns:
        Keras Model
    """
    # Image input
    image_input = layers.Input(shape=input_shape)

    # Condition input
    condition_input = layers.Input(shape=(condition_dim,))

    # Process condition through dense layers
    condition_x = layers.Dense(input_shape[0] * input_shape[1])(condition_input)
    condition_x = layers.Reshape((input_shape[0], input_shape[1], 1))(condition_x)

    # Concatenate image with condition along channel dimension
    x = layers.Concatenate()([image_input, condition_x])

    # Convolutional layers
    x = layers.Conv2D(32, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Flatten and output single value
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([image_input, condition_input], output)
    return model


# Build the CGAN
def build_cgan(latent_dim, condition_dim, generator, discriminator):
    """
    Build the complete CGAN model.

    Args:
        latent_dim: Dimension of random noise input
        condition_dim: Dimension of condition vector
        generator: Generator model
        discriminator: Discriminator model

    Returns:
        Keras Model
    """
    # For the combined model, we only train the generator
    discriminator.trainable = False

    # Random noise and condition inputs
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(condition_dim,))

    # Generate an image using the generator
    generated_image = generator([noise_input, condition_input])

    # Classify the generated image with the discriminator
    validity = discriminator([generated_image, condition_input])

    # Build and compile the combined model
    model = models.Model([noise_input, condition_input], validity)
    return model