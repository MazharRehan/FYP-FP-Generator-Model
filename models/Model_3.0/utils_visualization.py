import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_boundary_map(boundary_map, save_path=None, title=None):
    """
    Visualize a boundary map with walls and room areas.
    
    Args:
        boundary_map: Numpy array of shape (H, W, 2)
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Normalize if needed
    if boundary_map.max() > 1.0 or boundary_map.min() < 0.0:
        boundary_map = (boundary_map + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot walls
    ax[0].imshow(boundary_map[:, :, 0], cmap='gray')
    ax[0].set_title('Walls')
    ax[0].axis('off')
    
    # Plot rooms
    ax[1].imshow(boundary_map[:, :, 1], cmap='jet')
    ax[1].set_title('Room Areas')
    ax[1].axis('off')
    
    # Plot combined visualization
    combined = np.zeros((boundary_map.shape[0], boundary_map.shape[1], 3))
    combined[:, :, 0] = boundary_map[:, :, 0]  # Walls in red channel
    combined[:, :, 1] = boundary_map[:, :, 1]  # Room areas in green channel
    ax[2].imshow(combined)
    ax[2].set_title('Combined Visualization')
    ax[2].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_batch(real_batch, generated_batch, save_dir, epoch, batch_idx=0):
    """
    Visualize a batch of real and generated floor plans.
    
    Args:
        real_batch: Batch of real floor plans
        generated_batch: Batch of generated floor plans
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        batch_idx: Batch index
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = min(8, real_batch.shape[0])
    
    plt.figure(figsize=(16, 8))
    
    for i in range(batch_size):
        # Real floor plan - walls
        plt.subplot(4, batch_size, i+1)
        plt.imshow(real_batch[i, :, :, 0], cmap='gray')
        plt.title(f'Real Walls {i+1}')
        plt.axis('off')
        
        # Real floor plan - rooms
        plt.subplot(4, batch_size, i+1+batch_size)
        plt.imshow(real_batch[i, :, :, 1], cmap='jet')
        plt.title(f'Real Rooms {i+1}')
        plt.axis('off')
        
        # Generated floor plan - walls
        plt.subplot(4, batch_size, i+1+2*batch_size)
        plt.imshow(generated_batch[i, :, :, 0], cmap='gray')
        plt.title(f'Gen Walls {i+1}')
        plt.axis('off')
        
        # Generated floor plan - rooms
        plt.subplot(4, batch_size, i+1+3*batch_size)
        plt.imshow(generated_batch[i, :, :, 1], cmap='jet')
        plt.title(f'Gen Rooms {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_epoch_{epoch:03d}_batch_{batch_idx:03d}.png'))
    plt.close()