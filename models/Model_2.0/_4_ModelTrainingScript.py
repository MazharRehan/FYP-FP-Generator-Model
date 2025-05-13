import os
import numpy as np
import matplotlib.pyplot as plt

def train_cgan(generator, discriminator, cgan, X_train, metadata_train,
               latent_dim, batch_size=8, epochs=30, save_interval=10):
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)
    half_batch = batch_size // 2
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]
        real_conditions = metadata_train[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        idx = np.random.randint(0, metadata_train.shape[0], half_batch)
        gen_conditions = metadata_train[idx]
        gen_images = generator.predict([noise, gen_conditions])
        d_loss_real = discriminator.train_on_batch([real_images, real_conditions], np.ones((half_batch, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch([gen_images, gen_conditions], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        idx = np.random.randint(0, metadata_train.shape[0], batch_size)
        sampled_conditions = metadata_train[idx]
        g_loss = cgan.train_on_batch([noise, sampled_conditions], np.ones((batch_size, 1)) * 0.9)
        d_losses.append(float(np.mean(d_loss)))
        g_losses.append(float(np.mean(g_loss)))
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")
        if np.isnan(d_loss).any() or np.isnan(g_loss).any():
            print("NaN loss detected! Stopping training.")
            break
        if (epoch + 1) % save_interval == 0:
            save_generated_images(epoch + 1, generator, metadata_train, latent_dim, X_train.shape[1:3])
            generator.save(f'models/generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'models/discriminator_epoch_{epoch + 1}.h5')
    return {'d_losses': d_losses, 'g_losses': g_losses}

def save_generated_images(epoch, generator, metadata, latent_dim, target_shape, n_samples=16):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    idx = np.random.randint(0, metadata.shape[0], n_samples)
    sampled_conditions = metadata[idx]
    gen_images = generator.predict([noise, sampled_conditions])
    gen_images = (gen_images + 1) / 2.0
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_images[i].reshape((*target_shape, 3)))  # automatic reshaping
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'generated_images/epoch_{epoch}.png')
    plt.close()