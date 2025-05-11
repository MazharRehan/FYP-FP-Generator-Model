import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(latent_dim, condition_dim, output_shape=(560, 1014, 3)):
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(condition_dim,))
    x = layers.Concatenate()([noise_input, condition_input])
    x = layers.Dense(8 * 8 * 256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)  # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)   # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)   # 64x64
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # Upsample to target shape
    x = layers.Conv2DTranspose(32, 4, strides=(target_shape_div(output_shape, 64)), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(16, 4, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    output = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    model = models.Model([noise_input, condition_input], output)
    return model

def target_shape_div(output_shape, source_dim):
    # Returns (stride_h, stride_w) to reach output_shape from source_dim
    stride_h = output_shape[1] // source_dim
    stride_w = output_shape[0] // source_dim
    return (stride_h, stride_w)

def build_discriminator(input_shape=(560, 1014, 3), condition_dim=None):
    image_input = layers.Input(shape=input_shape)
    condition_input = layers.Input(shape=(condition_dim,))
    cond = layers.Dense(8 * 8 * 1)(condition_input)
    cond = layers.LeakyReLU(0.2)(cond)
    cond = layers.Reshape((8, 8, 1))(cond)
    cond = layers.Lambda(lambda x: tf.image.resize(x, (input_shape[0], input_shape[1]), method='nearest'))(cond)
    x = layers.Concatenate()([image_input, cond])
    x = layers.Conv2D(32, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model([image_input, condition_input], output)
    return model

def build_cgan(latent_dim, condition_dim, generator, discriminator):
    discriminator.trainable = False
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(condition_dim,))
    generated_image = generator([noise_input, condition_input])
    validity = discriminator([generated_image, condition_input])
    model = models.Model([noise_input, condition_input], validity)
    return model