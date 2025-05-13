import matplotlib.pyplot as plt
from _2_DataPreprocessing_for_ModelTraining import preprocess_data
from _3_CGANModelArchitecture import build_generator, build_discriminator, build_cgan
from _4_ModelTrainingScript import train_cgan, save_generated_images
from tensorflow.keras.optimizers import Adam
import numpy as np

if __name__ == "__main__":
    latent_dim = 100
    batch_size = 8
    epochs = 30
    save_interval = 10
    image_size = (560, 1014)  # width, height based on avg aspect ratio

    X_train, X_val, metadata_train, metadata_val, area_per_pixel_train, area_per_pixel_val = preprocess_data(
        "./dataset",
        "floor_plan_metadata_extended.csv",
        target_size=image_size,
        use_areas=False  # Set to True to add room areas to conditioning vector
    )

    condition_dim = metadata_train.shape[1]
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    print(f"Condition dimension: {condition_dim}")

    generator = build_generator(latent_dim, condition_dim, output_shape=(*image_size, 3))
    discriminator = build_discriminator(input_shape=(*image_size, 3), condition_dim=condition_dim)

    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-4, 0.5, clipvalue=1.0),
        metrics=['accuracy']
    )
    cgan = build_cgan(latent_dim, condition_dim, generator, discriminator)
    cgan.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-4, 0.5, clipvalue=1.0)
    )

    generator.summary()
    discriminator.summary()

    history = train_cgan(generator, discriminator, cgan,
                         X_train, metadata_train,
                         latent_dim, batch_size, epochs, save_interval)

    generator.save('models/generator_final.h5')
    discriminator.save('models/discriminator_final.h5')

    plt.figure(figsize=(10, 5))
    plt.plot(history['d_losses'], label='Discriminator Loss')
    plt.plot(history['g_losses'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')