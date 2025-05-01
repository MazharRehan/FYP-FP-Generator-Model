def train_cgan(generator, discriminator, cgan, X_train, metadata_train,
               latent_dim, batch_size=32, epochs=100, save_interval=10):
    """
    Train the CGAN model.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        cgan: Combined CGAN model
        X_train: Training images
        metadata_train: Training metadata features
        latent_dim: Dimension of random noise input
        batch_size: Batch size for training
        epochs: Number of training epochs
        save_interval: Interval to save generated images

    Returns:
        Training history
    """
    # Set up directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)

    # Number of training examples
    half_batch = batch_size // 2

    # Lists to store loss values
    d_losses = []
    g_losses = []

    # Training loop
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]
        real_conditions = metadata_train[idx]

        # Generate a half batch of fake images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        idx = np.random.randint(0, metadata_train.shape[0], half_batch)
        gen_conditions = metadata_train[idx]
        gen_images = generator.predict([noise, gen_conditions])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_conditions],
                                                   np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([gen_images, gen_conditions],
                                                   np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Generate random noise and conditions
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        idx = np.random.randint(0, metadata_train.shape[0], batch_size)
        sampled_conditions = metadata_train[idx]

        # Train the generator (via the discriminator's error)
        g_loss = cgan.train_on_batch([noise, sampled_conditions],
                                     np.ones((batch_size, 1)))

        # Store the losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")

        # Save generated images at intervals
        if (epoch + 1) % save_interval == 0:
            save_generated_images(epoch + 1, generator, metadata_train, latent_dim)

        # Save models at intervals
        if (epoch + 1) % save_interval == 0:
            generator.save(f'models/generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'models/discriminator_epoch_{epoch + 1}.h5')

    return {'d_losses': d_losses, 'g_losses': g_losses}


def save_generated_images(epoch, generator, metadata, latent_dim, n_samples=16):
    """
    Generate and save images during training.

    Args:
        epoch: Current epoch number
        generator: Generator model
        metadata: Metadata for conditioning
        latent_dim: Dimension of random noise input
        n_samples: Number of images to generate
    """
    # Generate random noise and select random conditions
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    idx = np.random.randint(0, metadata.shape[0], n_samples)
    sampled_conditions = metadata[idx]

    # Generate images
    gen_images = generator.predict([noise, sampled_conditions])

    # Scale images from [-1, 1] to [0, 1]
    gen_images = (gen_images + 1) / 2.0

    # Plot and save
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'generated_images/epoch_{epoch}.png')
    plt.close()