if __name__ == "__main__":
    # Parameters
    latent_dim = 100
    batch_size = 32
    epochs = 500
    save_interval = 10
    image_size = (256, 256)

    # Load and preprocess data
    X_train, X_val, metadata_train, metadata_val = preprocess_data(
        "./dataset",
        "floor_plan_metadata_extended.csv",
        target_size=image_size
    )

    condition_dim = metadata_train.shape[1]

    print(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    print(f"Condition dimension: {condition_dim}")

    # Build models
    generator = build_generator(latent_dim, condition_dim, output_shape=(*image_size, 3))
    discriminator = build_discriminator(input_shape=(*image_size, 3), condition_dim=condition_dim)

    # Compile discriminator
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(0.0002, 0.5),
        metrics=['accuracy']
    )

    # Build combined CGAN model
    cgan = build_cgan(latent_dim, condition_dim, generator, discriminator)

    # Compile combined model
    cgan.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(0.0002, 0.5)
    )

    # Print model summaries
    generator.summary()
    discriminator.summary()

    # Train the model
    history = train_cgan(generator, discriminator, cgan,
                         X_train, metadata_train,
                         latent_dim, batch_size, epochs, save_interval)

    # Save final models
    generator.save('models/generator_final.h5')
    discriminator.save('models/discriminator_final.h5')

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['d_losses'], label='Discriminator Loss')
    plt.plot(history['g_losses'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')