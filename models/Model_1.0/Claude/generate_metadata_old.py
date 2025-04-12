import os
import json
import numpy as np
from PIL import Image
import argparse
import cv2
from datetime import datetime

# Define color mapping
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

# Dimensions mapping
DIMENSION_MAP = {
    '5_marla': {
        'feet': [25, 45],
        'inches': [6.338, 11.338],
        'pixels': [608, 1088],
        'dpi': 96,
        'aspect_ratio': 0.559,
        'bit_depth': 24
    },
    '10_marla': {
        'feet': [35, 65],
        'inches': [8.833, 16.344],
        'pixels': [849, 1570],
        'dpi': 96,
        'aspect_ratio': 0.541,
        'bit_depth': 24
    },
    '20_marla': {
        'feet': [50, 90],
        'inches': [12.583, 22.594],
        'pixels': [1209, 2170],
        'dpi': 96,
        'aspect_ratio': 0.557,
        'bit_depth': 24
    }
}

def analyze_floor_plan(image_path):
    """
    Analyze a floor plan image to extract room information.
    
    Args:
        image_path: Path to the floor plan image
        
    Returns:
        metadata: Dictionary containing metadata about the floor plan
    """
    # Extract file name components
    file_name = os.path.basename(image_path)
    name_parts = os.path.splitext(file_name)[0].split('_')
    
    plot_type = name_parts[0]
    floor_level = name_parts[1]
    plan_type = name_parts[2]
    fp_number = name_parts[3]
    version = name_parts[4]
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB for easier color matching
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize room counts
    room_counts = {room_type: 0 for room_type in COLOR_MAP.keys()}
    room_areas = {room_type: 0 for room_type in COLOR_MAP.keys()}
    
    total_area = 0
    
    # Analyze colors to count rooms and calculate areas
    for y in range(image_rgb.shape[0]):
        for x in range(image_rgb.shape[1]):
            pixel = image_rgb[y, x]
            
            # Find the closest color in our color map
            min_distance = float('inf')
            closest_room = None
            
            for room_type, color in COLOR_MAP.items():
                distance = sum((pixel[i] - color[i])**2 for i in range(3))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_room = room_type
            
            # If distance is too large, ignore this pixel
            if min_distance < 3000:  # Threshold for color matching
                room_areas[closest_room] += 1
                total_area += 1
    
    # Count distinct connected components for each room type
    for room_type, color in COLOR_MAP.items():
        # Skip walls and background
        if room_type in ['Walls', 'Background', 'Door']:
            continue
        
        # Create mask for this room type
        lower_bound = np.array([max(0, c-20) for c in color])
        upper_bound = np.array([min(255, c+20) for c in color])
        
        mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        # First component is the background, so we subtract 1
        room_counts[room_type] = max(0, num_labels - 1)
    
    # Get dimension info
    plot_type_key = plot_type.replace('Marla', '_marla')
    dimensions = DIMENSION_MAP.get(plot_type_key, {})
    
    # Create metadata
    metadata = {
        'file_name': file_name,
        'plot_type': plot_type,
        'floor_level': floor_level,
        'plan_type': plan_type,
        'fp_number': fp_number,
        'version': version,
        'dimensions': dimensions,
        'room_counts': {k: v for k, v in room_counts.items() if v > 0},
        'room_areas_pixels': {k: v for k, v in room_areas.items() if v > 0},
        'total_area_pixels': total_area,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return metadata

def process_dataset(dataset_path, output_path):
    """
    Process all floor plan images in the dataset and generate metadata.
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save the metadata
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Find all plot type directories
    plot_types = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for plot_type in plot_types:
        plot_dir = os.path.join(dataset_path, plot_type)
        
        # Create output directory for this plot type
        plot_output_dir = os.path.join(output_path, plot_type, 'metadata', 'json')
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Find all PNG files
        png_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        
        print(f"Processing {len(png_files)} files in {plot_type}...")
        
        for png_file in png_files:
            image_path = os.path.join(plot_dir, png_file)
            
            try:
                # Analyze the floor plan
                metadata = analyze_floor_plan(image_path)
                
                # Save metadata
                json_path = os.path.join(plot_output_dir, os.path.splitext(png_file)[0] + '.json')
                
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Processed: {png_file}")
                
            except Exception as e:
                print(f"Error processing {png_file}: {str(e)}")
    
    print("Metadata generation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate metadata for floor plan dataset')
    parser.add_argument('--dataset', default='dataset', help='Path to the dataset directory')
    parser.add_argument('--output', default='data/raw', help='Path to save the metadata')
    
    args = parser.parse_args()
    # path is = './dataset'
    # output is = './data/raw'
    
    process_dataset(args.dataset, args.output)