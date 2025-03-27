import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from floor_plan_generator import FloorPlanGenerator, PLOT_TYPES, COLOR_MAP
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_generator(generator, test_dataset, num_samples=5):
    """
    Evaluate the generator model on test data.
    
    Args:
        generator: Trained generator model
        test_dataset: Test dataset
        num_samples: Number of samples to evaluate
    """
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Get samples from test dataset
    samples = []
    for batch_inputs, batch_targets in test_dataset.take(num_samples):
        samples.append((batch_inputs, batch_targets))
    
    # Evaluate each sample
    for i, (batch_inputs, batch_targets) in enumerate(samples):
        input_image = batch_inputs['image']
        condition = batch_inputs['condition']
        
        # Generate prediction
        prediction = generator.generator([input_image, condition], training=False)
        pred_maps = tf.argmax(prediction, axis=-1)
        
        # Plot results
        fig, axes = plt.subplots(3, len(input_image), figsize=(20, 12))
        
        for j in range(len(input_image)):
            # Original image
            axes[0, j].imshow((input_image[j] + 1) * 0.5)  # Convert from [-1, 1] to [0, 1]
            axes[0, j].set_title(f'Input Image {j+1}')
            axes[0, j].axis('off')
            
            # Ground truth
            axes[1, j].imshow(batch_targets[j], cmap='tab20')
            axes[1, j].set_title(f'Ground Truth {j+1}')
            axes[1, j].axis('off')
            
            # Prediction
            axes[2, j].imshow(pred_maps[j], cmap='tab20')
            axes[2, j].set_title(f'Prediction {j+1}')
            axes[2, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'evaluation_results/sample_{i+1}_comparison.png')
        plt.close()
        
        # Calculate confusion matrix for quantitative evaluation
        for j in range(len(input_image)):
            y_true = batch_targets[j].numpy().flatten()
            y_pred = pred_maps[j].numpy().flatten()
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
            plt.title(f'Normalized Confusion Matrix - Sample {i+1}, Image {j+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'evaluation_results/sample_{i+1}_image_{j+1}_confusion_matrix.png')
            plt.close()

def visualize_generated_samples(generator, num_samples=5):
    """
    Generate and visualize floor plans for each plot type.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate for each plot type
    """
    # Create output directory
    os.makedirs('generated_samples', exist_ok=True)
    
    # Generate samples for each plot type
    for plot_type in PLOT_TYPES:
        plt.figure(figsize=(20, 4 * num_samples))
        
        for i in range(num_samples):
            # Generate floor plan
            floor_plan = generator.generate_floor_plan(plot_type, seed=i)
            
            # Plot
            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(floor_plan)
            plt.title(f'{plot_type} Floor Plan - Sample {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'generated_samples/{plot_type}_samples.png')
        plt.close()

def create_legend():
    """
    Create a legend for the color-coded room types.
    """
    # Create a figure for the legend
    plt.figure(figsize=(12, 10))
    
    # Create patches for legend
    patches = []
    labels = []
    
    for room_type, color in COLOR_MAP.items():
        # Convert RGB color to matplotlib format (0-1 range)
        color_normalized = [c / 255 for c in color]
        
        # Create a patch and add it to the list
        patch = plt.Rectangle((0, 0), 1, 1, fc=color_normalized)
        patches.append(patch)
        labels.append(room_type)
    
    # Create the legend
    plt.legend(patches, labels, loc='center', fontsize=12)
    plt.axis('off')
    plt.title('Floor Plan Color Legend', fontsize=16)
    
    # Save the legend
    plt.savefig('color_legend.png')
    plt.close()

if __name__ == "__main__":
    # Load the trained model
    generator = FloorPlanGenerator('dataset')
    
    # Check if the model exists and load it
    if os.path.exists('generator_model.h5'):
        generator.generator = tf.keras.models.load_model('generator_model.h5')
        print("Loaded trained generator model.")
    else:
        print("Trained model not found. Please train the model first.")
        exit()
    
    # Load test dataset
    _, test_dataset = generator.load_and_preprocess_data()
    
    # Evaluate the model
    evaluate_generator(generator, test_dataset)
    
    # Generate and visualize samples
    visualize_generated_samples(generator)
    
    # Create color legend
    create_legend()
    
    print("Evaluation completed. Results saved to 'evaluation_results' and 'generated_samples' directories.")