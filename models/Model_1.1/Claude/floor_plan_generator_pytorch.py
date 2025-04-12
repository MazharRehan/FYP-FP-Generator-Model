import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from datetime import datetime

from floor_plan_generator import FloorPlanGenerator

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
    'Dining': [225, 192, 203],       # Pink
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

class FloorPlanDataset(Dataset):
    def __init__(self, dataset_path, plot_types=None, transform=None):
        self.dataset_path = dataset_path
        self.plot_types = plot_types if plot_types else PLOT_TYPES
        self.transform = transform
        self.images = []
        self.conditions = []
        
        # Load dataset
        for i, plot_type in enumerate(self.plot_types):
            plot_dir = os.path.join(self.dataset_path, plot_type)
            
            # Skip if directory doesn't exist
            if not os.path.exists(plot_dir):
                print(f"Warning: Directory {plot_dir} not found. Skipping.")
                continue
                
            files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
            
            for file in files:
                file_path = os.path.join(plot_dir, file)
                
                # Load image
                img = Image.open(file_path).convert('RGB')
                
                # Create one-hot encoding for plot type condition
                condition = np.zeros(len(self.plot_types))
                condition[i] = 1
                
                self.images.append(img)
                self.conditions.append(condition)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        condition = self.conditions[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, condition

def preprocess_image_to_segmentation_map(image):
    """
    Convert an RGB image to a segmentation map based on color mapping.
    Each pixel will be assigned a class index based on its RGB value.
    
    Args:
        image: RGB image array with values in range [0, 1]
        
    Returns:
        segmentation_map: Array of class indices
    """
    # Convert from [0, 1] to [0, 255]
    img = (image * 255).astype(np.uint8)
    
    # Initialize segmentation map with background class
    segmentation_map = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
    
    # Assign class indices based on RGB values
    for y in range(img.shape[1]):
        for x in range(img.shape[2]):
            rgb = tuple(img[:, y, x])
            # Find closest color in the color map
            closest_color = min(RGB_TO_INDEX.keys(), 
                              key=lambda c: sum((c[i] - rgb[i])**2 for i in range(3)))
            segmentation_map[y, x] = RGB_TO_INDEX[closest_color]
    
    return segmentation_map

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Encoder
            self._down_block(3 + len(PLOT_TYPES), 64, 4, 2, 1),  # Input: 3 + condition channels
            self._down_block(64, 128, 4, 2, 1),
            self._down_block(128, 256, 4, 2, 1),
            self._down_block(256, 512, 4, 2, 1),
            self._down_block(512, 512, 4, 2, 1),
            self._down_block(512, 512, 4, 2, 1),
            self._down_block(512, 512, 4, 2, 1),
            self._down_block(512, 512, 4, 2, 1),
            # Decoder
            self._up_block(512, 512, 4, 2, 1, dropout=True),
            self._up_block(1024, 512, 4, 2, 1, dropout=True),
            self._up_block(1024, 512, 4, 2, 1, dropout=True),
            self._up_block(1024, 256, 4, 2, 1),
            self._up_block(512, 128, 4, 2, 1),
            self._up_block(256, 64, 4, 2, 1),
            nn.ConvTranspose2d(128, NUM_OUTPUT_CHANNELS, 4, 2, 1)
        )
    
    def _down_block(self, in_channels, out_channels, kernel_size, stride, padding, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _up_block(self, in_channels, out_channels, kernel_size, stride, padding, dropout=False):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
        layers.append(nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x, condition):
        # Expand condition to match spatial dimensions
        condition = condition.view(condition.size(0), condition.size(1), 1, 1)
        condition = condition.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, condition], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            self._down_block(3 + len(PLOT_TYPES), 64, 4, 2, 1, apply_batchnorm=False),
            self._down_block(64, 128, 4, 2, 1),
            self._down_block(128, 256, 4, 2, 1),
            self._down_block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    
    def _down_block(self, in_channels, out_channels, kernel_size, stride, padding, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x, condition):
        # Expand condition to match spatial dimensions
        condition = condition.view(condition.size(0), condition.size(1), 1, 1)
        condition = condition.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, condition], dim=1)
        return self.main(x)

class FloorPlanGenerator:
    def __init__(self, dataset_path, learning_rate=0.0002, beta1=0.5):
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        
        # Setup checkpoint
        self.checkpoint_dir = './training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize logs
        self.log_dir = 'logs/'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            train_dataset: DataLoader for training
            test_dataset: DataLoader for testing
        """
        print("Loading dataset...")
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # Normalize to [-1, 1]
        ])
        dataset = FloorPlanDataset(self.dataset_path, transform=transform)
        print(f"Found {len(dataset)} images.")
        
        # Split into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
    
    def generator_loss(self, disc_generated_output, gen_output, target):
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
        bce_loss = nn.BCEWithLogitsLoss()
        gan_loss = bce_loss(disc_generated_output, torch.ones_like(disc_generated_output))
        
        # Pixel-wise loss (cross entropy for segmentation)
        ce_loss = nn.CrossEntropyLoss()
        pixel_loss = ce_loss(gen_output, target)
        
        # Total loss
        total_loss = gan_loss + 100 * pixel_loss  # Higher weight on pixel loss
        return total_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Loss function for the discriminator.
        
        Args:
            disc_real_output: Discriminator output on real images
            disc_generated_output: Discriminator output on generated images
            
        Returns:
            total_loss: Discriminator loss
        """
        bce_loss = nn.BCEWithLogitsLoss()
        real_loss = bce_loss(disc_real_output, torch.ones_like(disc_real_output))
        generated_loss = bce_loss(disc_generated_output, torch.zeros_like(disc_generated_output))
        total_loss = real_loss + generated_loss
        return total_loss
    
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
        # Move data to device
        input_image = input_image.to(self.device)
        target = target.to(self.device)
        condition = condition.to(self.device)
        
        # Train generator
        self.generator_optimizer.zero_grad()
        gen_output = self.generator(input_image, condition)
        disc_generated_output = self.discriminator(gen_output, condition)
        gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
        gen_loss.backward()
        self.generator_optimizer.step()
        
        # Train discriminator
        self.discriminator_optimizer.zero_grad()
        disc_real_output = self.discriminator(input_image, condition)
        disc_generated_output = self.discriminator(gen_output.detach(), condition)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        disc_loss.backward()
        self.discriminator_optimizer.step()
        
        return gen_loss.item(), disc_loss.item()
    
    def train(self, epochs=EPOCHS):
        """
        Train the GAN.
        
        Args:
            epochs: Number of epochs
        """
        train_loader, test_loader = self.load_and_preprocess_data()
        
        print("Starting training...")
        for epoch in range(epochs):
            start = datetime.now()
            
            # Training
            for batch_idx, (input_image, condition) in enumerate(train_loader):
                segmentation_maps = torch.stack([torch.tensor(preprocess_image_to_segmentation_map(img.numpy())) for img in input_image])
                gen_loss, disc_loss = self.train_step(input_image, segmentation_maps, condition)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx} - "
                          f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
                    
                    # Log to file
                    with open(os.path.join(self.log_dir, "training_log.txt"), "a") as log_file:
                        log_file.write(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx} - "
                                       f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}\n")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict()
                }, os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch+1}.pth"))
                
                # Generate and save sample images
                self.generate_and_save_images(epoch + 1, test_loader)
            
            print(f"Time taken for epoch {epoch+1}: {datetime.now() - start}")
        
        # Save the final model
        torch.save(self.generator.state_dict(), 'generator_model.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator_model.pth')
        
        # Generate final samples
        self.generate_and_save_images(epochs, test_loader)
        
        print("Training completed!")
    
    def generate_and_save_images(self, epoch, test_loader):
        """
        Generate and save sample floor plans.
        
        Args:
            epoch: Current epoch
            test_loader: Test dataset
        """
        # Take a sample from the test dataset
        for input_image, condition in test_loader:
            input_image = input_image.to(self.device)
            condition = condition.to(self.device)
            
            # Generate images for each condition type
            for i, plot_type in enumerate(PLOT_TYPES):
                # Create condition vector for this plot type
                cond = torch.zeros((1, len(PLOT_TYPES)), device=self.device)
                cond[0, i] = 1
                
                # Generate floor plan
                prediction = self.generator(input_image[:1], cond)
                
                # Convert prediction to segmentation map
                pred_map = torch.argmax(prediction[0], dim=0)
                
                # Convert segmentation map to RGB image
                rgb_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                
                for y in range(IMG_HEIGHT):
                    for x in range(IMG_WIDTH):
                        class_idx = pred_map[y, x].item()
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
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create random noise
        noise = torch.randn((1, 3, IMG_HEIGHT, IMG_WIDTH), device=self.device)

        # Create condition vector
        condition = torch.zeros((1, len(PLOT_TYPES)), device=self.device)
        plot_idx = PLOT_TYPES.index(plot_type)
        condition[0, plot_idx] = 1

        # Generate floor plan
        prediction = self.generator(noise, condition)

        # Convert prediction to segmentation map
        pred_map = torch.argmax(prediction[0], dim=0)

        # Convert segmentation map to RGB image
        rgb_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        for y in range(IMG_HEIGHT):
            for x in range(IMG_WIDTH):
                class_idx = pred_map[y, x].item()
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
        import argparse

        parser = argparse.ArgumentParser(description='Train and generate floor plans with GAN')
        parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'],
                            help='Mode of operation: "train" or "generate"')
        parser.add_argument('--dataset_path', type=str, default='dataset',
                            help='Path to the dataset directory')
        parser.add_argument('--output_dir', type=str, default='output',
                            help='Directory to save outputs')
        parser.add_argument('--epochs', type=int, default=EPOCHS,
                            help='Number of epochs for training')
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                            help='Batch size for training')
        parser.add_argument('--plot_type', type=str, choices=PLOT_TYPES,
                            help='Type of plot to generate (required for "generate" mode)')
        parser.add_argument('--num_samples', type=int, default=1,
                            help='Number of samples to generate (required for "generate" mode)')
        parser.add_argument('--learning_rate', type=float, default=0.0002,
                            help='Learning rate for training')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Beta1 parameter for Adam optimizer')
        parser.add_argument('--checkpoint', type=str,
                            help='Path to load a saved model checkpoint')

        args = parser.parse_args()

        if args.mode == 'train':
            # Initialize the floor plan generator
            generator = FloorPlanGenerator(dataset_path=args.dataset_path,
                                           learning_rate=args.learning_rate,
                                           beta1=args.beta1)

            # Train the model
            generator.train(epochs=args.epochs)

        elif args.mode == 'generate':
            if not args.plot_type:
                raise ValueError('Plot type is required for "generate" mode')

            # Initialize the floor plan generator
            generator = FloorPlanGenerator(dataset_path=args.dataset_path)

            # Check if the model exists and load it
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint)
                generator.generator.load_state_dict(checkpoint['generator_state_dict'])
                generator.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                generator.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
                generator.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
                print("Loaded model checkpoint.")
            else:
                generator.generator.load_state_dict(torch.load('generator_model.pth'))
                generator.discriminator.load_state_dict(torch.load('discriminator_model.pth'))
                print("Loaded trained generator model.")

            # Generate and export floor plans
            for i in range(args.num_samples):
                # Generate floor plan
                floor_plan = generator.generate_floor_plan(args.plot_type, seed=i)

                # Export to different formats
                output_dir = f'{args.output_dir}/{args.plot_type}'
                os.makedirs(output_dir, exist_ok=True)
                generator.export_floor_plan(
                    floor_plan,
                    f'{output_dir}/floor_plan_{i}.png',
                    format='png'
                )

                generator.export_floor_plan(
                    floor_plan,
                    f'{output_dir}/floor_plan_{i}.svg',
                    format='svg'
                )

                generator.export_floor_plan(
                    floor_plan,
                    f'{output_dir}/floor_plan_{i}.dxf',
                    format='dxf'
                )

            print("All floor plans generated and exported successfully!")