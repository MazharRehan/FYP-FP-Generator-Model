def evaluate_floor_plans(generator, X_test, metadata_test, latent_dim, num_samples=50):
    """
    Evaluate generated floor plans against test data.

    Args:
        generator: Trained generator model
        X_test: Test images
        metadata_test: Test metadata
        latent_dim: Dimension of latent space
        num_samples: Number of samples to evaluate
    """
    # Generate floor plans using test conditions
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    idx = np.random.randint(0, metadata_test.shape[0], num_samples)
    sampled_conditions = metadata_test[idx]
    sampled_real_images = X_test[idx]

    # Generate images
    generated_images = generator.predict([noise, sampled_conditions])

    # Scale images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0
    sampled_real_images = (sampled_real_images + 1) / 2.0

    # Calculate structural similarity (SSIM) between real and generated images
    from skimage.metrics import structural_similarity as ssim

    ssim_values = []
    for i in range(num_samples):
        # Convert to grayscale for SSIM
        real_gray = np.mean(sampled_real_images[i], axis=2)
        gen_gray = np.mean(generated_images[i], axis=2)

        # Calculate SSIM
        ssim_val = ssim(real_gray, gen_gray)
        ssim_values.append(ssim_val)

    print(f"Average SSIM: {np.mean(ssim_values)}")

    # Visualize some examples
    plt.figure(figsize=(20, 10))
    for i in range(min(5, num_samples)):
        # Real image
        plt.subplot(2, 5, i + 1)
        plt.imshow(sampled_real_images[i])
        plt.title(f"Real {i + 1}")
        plt.axis('off')

        # Generated image
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(generated_images[i])
        plt.title(f"Generated {i + 1}")
        plt.axis('off')

    plt.savefig('evaluation_comparison.png')
    plt.close()

    return np.mean(ssim_values)