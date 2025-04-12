# --- START OF FILE generate_metadata.py ---

import os
import json
import numpy as np
import cv2 # Using OpenCV for image processing
import argparse
import re
import logging
from datetime import datetime
from collections import defaultdict

# --- Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base directory containing subfolders like '5_marla', '10_marla' (can be overridden by args)
DEFAULT_BASE_DATA_DIR = "./dataset"
# Base output directory (can be overridden by args)
DEFAULT_OUTPUT_DIR = "./data/raw"

# Define color mapping (Name -> RGB) - Consistent with original spec
COLOR_MAP = {
    'Bedroom': [255, 0, 0],          # Red
    'Bathroom': [0, 0, 255],         # Blue
    'Kitchen': [255, 165, 0],        # Orange
    'Drawing Room': [0, 128, 0],     # Green
    'Garage': [165, 42, 42],         # Brown
    'Lounge/Sitting Area': [255, 255, 0], # Yellow (Corrected Name)
    'Backyard': [50, 205, 50],       # Lime Green
    'Stairs': [0, 128, 128],         # Teal
    'Storage Room/Store': [128, 0, 128], # Purple (Corrected Name)
    'Open Space': [0, 255, 255],     # Cyan
    'Prayer Room': [127, 127, 127],  # Grayish/Check RGB - Assuming original was right
    'Staircase': [153, 51, 255],     # Violet
    'Lobby': [255, 0, 255],          # Magenta
    'Lawn': [64, 224, 208],          # Turquoise
    'Dining': [225, 192, 203],       # Pink
    'Servant Quarters': [75, 0, 130], # Indigo (Corrected Name)
    'Passage': [128, 128, 0],        # Olive Green
    'Laundry': [230, 230, 250],      # Lavender
    'Dressing Area': [255, 127, 80], # Coral (Corrected Name)
    'Side Garden': [255, 215, 0],    # Gold
    'Library': [255, 191, 0],        # Amber
    'Amber': [255, 191, 0],          # Explicitly add Amber if needed, shares color w Library
    # --- Structural/Non-counted ---
    'Walls': [0, 0, 0],              # Black
    'Door': [128, 0, 0],             # Mahogany
    'Background': [255, 255, 255]    # White (Often ignored but good to define)
}

# List of room types that should be counted using connected components
# Excludes Walls, Door, Background
ROOM_TYPES_TO_COUNT = [
    'Bedroom', 'Bathroom', 'Kitchen', 'Drawing Room', 'Garage',
    'Lounge/Sitting Area', 'Backyard', 'Stairs', 'Storage Room/Store',
    'Open Space', 'Prayer Room', 'Staircase', 'Lobby', 'Lawn', 'Dining',
    'Servant Quarters', 'Passage', 'Laundry', 'Dressing Area',
    'Side Garden', 'Library', 'Amber' # Note: Library/Amber share color, count may be combined
]

# Dimensions mapping (Canonical PlotSize -> Dict) - Consistent with original spec
DIMENSION_MAP = {
    '5Marla': {
        'feet': [25, 45], 'inches': [6.338, 11.338], 'pixels': [608, 1088],
        'dpi': 96, 'aspect_ratio': 0.559, 'bit_depth': 24
    },
    '10Marla': {
        'feet': [35, 65], 'inches': [8.833, 16.344], 'pixels': [849, 1570],
        'dpi': 96, 'aspect_ratio': 0.541, 'bit_depth': 24
    },
    '20Marla': {
        'feet': [50, 90], 'inches': [12.583, 22.594], 'pixels': [1209, 2170],
        'dpi': 96, 'aspect_ratio': 0.557, 'bit_depth': 24
    }
}

# Regex for Filename Parsing (from step 1.2.py)
# Example: 5Marla_GF_FP_001_V01.png
FILENAME_PATTERN = re.compile(
    r"^(?P<PlotSize>\d+Marla)_"
    r"(?P<FloorLevel>GF)_"
    r"(?P<PlanType>FP)_"
    r"(?P<FP_Number>\d+)_"
    r"(?P<Version>V\d+)"
    r"\.png$",
    re.IGNORECASE
)

# Minimum pixel area for a region to be counted (from auto_count_rooms.py)
# Helps filter noise/small artifacts. Adjust if needed.
MIN_AREA_THRESHOLD = 80 # Pixels

# --- Helper Function to create RGB -> Name mapping ---
def get_rgb_to_name_map(color_map):
    rgb_map = {}
    for name, rgb in color_map.items():
        rgb_tuple = tuple(rgb)
        if rgb_tuple in rgb_map:
             # Handle duplicate colors (like Library/Amber) - Append name
             existing_name = rgb_map[rgb_tuple]
             if isinstance(existing_name, list):
                 existing_name.append(name)
             else:
                 rgb_map[rgb_tuple] = [existing_name, name]
        else:
            rgb_map[rgb_tuple] = name
    return rgb_map

RGB_TO_NAME_MAP = get_rgb_to_name_map(COLOR_MAP)

# --- Core Analysis Function ---

def analyze_floor_plan_updated(image_path):
    """
    Analyzes a floor plan image using filename parsing and connected components
    for accurate room counting and area calculation.

    Args:
        image_path (str): Path to the floor plan image.

    Returns:
        dict: Dictionary containing metadata about the floor plan, or None on error.
    """
    file_name = os.path.basename(image_path)
    logging.debug(f"Analyzing: {file_name}")

    # 1. Parse Filename
    match = FILENAME_PATTERN.match(file_name)
    if not match:
        logging.warning(f"Filename pattern mismatch, skipping: {file_name}")
        return None

    try:
        extracted_data = match.groupdict()
        plot_type = extracted_data['PlotSize']
        # Normalize plot_type capitalization for DIMENSION_MAP lookup (e.g., 5marla -> 5Marla)
        plot_type_canonical = plot_type[:-5].capitalize() + "Marla" # Simple capitalization


    except Exception as e:
        logging.error(f"Error parsing filename components for {file_name}: {e}")
        return None

    # 2. Load Image
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = image_rgb.shape
        logging.debug(f"Image loaded: {width}x{height}")

    except Exception as e:
        logging.error(f"Error loading image {file_name}: {e}")
        return None

    # 3. Room Counting and Area Calculation using Connected Components
    room_counts = defaultdict(int)
    room_areas_pixels = defaultdict(int)
    total_valid_area_pixels = 0 # Sum of areas for counted room types

    # Efficiently create masks for all defined colors
    masks = {}
    for rgb_tuple, name_or_list in RGB_TO_NAME_MAP.items():
        # Create mask for this exact RGB color
        mask = cv2.inRange(image_rgb, np.array(rgb_tuple), np.array(rgb_tuple))
        masks[rgb_tuple] = mask

    logging.debug(f"Starting connected components analysis for {len(ROOM_TYPES_TO_COUNT)} room types.")
    processed_room_names = set() # Keep track if a room type was processed

    for room_name in ROOM_TYPES_TO_COUNT:
        target_rgb_list = COLOR_MAP.get(room_name)
        if not target_rgb_list:
            logging.warning(f"Color definition not found for room type: {room_name}. Skipping count.")
            continue

        target_rgb_tuple = tuple(target_rgb_list)

        # Avoid processing same color mask multiple times if names share color
        if room_name in processed_room_names:
            continue

        # Get the binary mask for this color
        binary_mask = masks.get(target_rgb_tuple)
        if binary_mask is None or np.count_nonzero(binary_mask) == 0:
            # logging.debug(f"No pixels found for {room_name} ({target_rgb_tuple})")
            continue # Skip if no pixels match this color

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        count_for_room = 0
        area_for_room = 0

        # Iterate through components (label 0 is background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA_THRESHOLD:
                count_for_room += 1
                area_for_room += area

        # Handle shared colors (assign count/area to all names sharing the color)
        name_at_rgb = RGB_TO_NAME_MAP.get(target_rgb_tuple)
        room_names_for_color = []
        if isinstance(name_at_rgb, list):
            room_names_for_color.extend(name_at_rgb)
        elif isinstance(name_at_rgb, str):
            room_names_for_color.append(name_at_rgb)

        for rn in room_names_for_color:
            if rn in ROOM_TYPES_TO_COUNT: # Only update if it's a type we care about counting
                room_counts[rn] = count_for_room
                room_areas_pixels[rn] = area_for_room
                processed_room_names.add(rn) # Mark as processed
                logging.debug(f"  Counted {rn}: {count_for_room} regions, {area_for_room}px area.")


    # Calculate total area from the counted regions
    total_valid_area_pixels = sum(room_areas_pixels.values())

    # 4. Get Dimension Info
    dimensions = DIMENSION_MAP.get(plot_type_canonical)
    if not dimensions:
        logging.warning(f"Dimension info not found for plot type '{plot_type_canonical}' from file {file_name}")
        dimensions = {} # Assign empty dict if not found

    # Check if image dimensions match expected dimensions
    expected_pixels = dimensions.get('pixels')
    if expected_pixels and (width, height) != tuple(expected_pixels):
        logging.warning(f"Image dimensions ({width}x{height}) mismatch expected ({expected_pixels[0]}x{expected_pixels[1]}) for {file_name}")


    # 5. Create Metadata Dictionary (Structure like original JSON)
    #    <<<<< APPLY FIX HERE >>>>>
    metadata = {
        'file_name': file_name,
        'plot_type': extracted_data['PlotSize'], # Use value directly from filename
        'floor_level': extracted_data['FloorLevel'],
        'plan_type': extracted_data['PlanType'],
        'fp_number': extracted_data['FP_Number'],
        'version': extracted_data['Version'],
        'dimensions': dimensions,
        # Filter out rooms with 0 count AND CAST TO PYTHON INT
        'room_counts': {k: int(v) for k, v in room_counts.items() if v > 0},
        # Filter out rooms with 0 area AND CAST TO PYTHON INT
        'room_areas_pixels': {k: int(v) for k, v in room_areas_pixels.items() if v > 0},
        # CAST TOTAL AREA TO PYTHON INT
        'total_area_pixels': int(total_valid_area_pixels),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    logging.debug(f"Metadata generated for {file_name}")
    return metadata

# --- Main Processing Function ---

def process_dataset(dataset_path, output_path):
    """
    Processes all compatible floor plan images in the dataset directory
    structure and saves metadata as JSON files.

    Args:
        dataset_path (str): Path to the base dataset directory (e.g., './dataset').
        output_path (str): Path to the base output directory (e.g., './data/raw').
    """
    logging.info(f"Starting dataset processing from: {dataset_path}")
    logging.info(f"Outputting JSON metadata to: {output_path}")

    if not os.path.isdir(dataset_path):
        logging.error(f"Dataset directory not found: {dataset_path}")
        return

    processed_count = 0
    error_count = 0

    # Find subdirectories (like '5_marla', '10_marla')
    plot_type_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    if not plot_type_dirs:
        logging.warning(f"No subdirectories found in {dataset_path}. Expected directories like '5_marla', etc.")
        return

    for plot_dir_name in plot_type_dirs:
        current_subdir = os.path.join(dataset_path, plot_dir_name)
        logging.info(f"--- Processing directory: {current_subdir} ---")

        # Create corresponding output directory structure
        # e.g., ./data/raw/5_marla/metadata/json/
        plot_output_dir = os.path.join(output_path, plot_dir_name, 'metadata', 'json')
        try:
            os.makedirs(plot_output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create output directory {plot_output_dir}: {e}. Skipping this directory.")
            continue

        # Find all PNG files in the current subdirectory
        try:
            png_files = [f for f in os.listdir(current_subdir) if f.lower().endswith('.png')]
            logging.info(f"Found {len(png_files)} PNG files.")
        except OSError as e:
             logging.error(f"Could not read directory {current_subdir}: {e}. Skipping.")
             continue

        subdir_processed = 0
        subdir_error = 0

        for png_file in png_files:
            image_path = os.path.join(current_subdir, png_file)
            try:
                # Analyze the floor plan using the updated logic
                metadata = analyze_floor_plan_updated(image_path)

                if metadata:
                    # Save metadata to JSON
                    json_filename = os.path.splitext(png_file)[0] + '.json'
                    json_path = os.path.join(plot_output_dir, json_filename)

                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    logging.info(f"Successfully processed and saved metadata for: {png_file}")
                    subdir_processed += 1
                else:
                    # Analysis function returned None (e.g., pattern mismatch, file error)
                    logging.warning(f"Skipped generating metadata for: {png_file}")
                    subdir_error += 1

            except Exception as e:
                # Catch any unexpected errors during analysis or saving
                logging.error(f"Unexpected error processing {png_file}: {e}", exc_info=False) # Set True for full traceback
                subdir_error += 1

        logging.info(f"Finished directory {plot_dir_name}. Processed: {subdir_processed}, Skipped/Errors: {subdir_error}")
        processed_count += subdir_processed
        error_count += subdir_error

    logging.info("--- Dataset Processing Complete ---")
    logging.info(f"Total files processed successfully: {processed_count}")
    logging.info(f"Total files skipped or failed: {error_count}")

# --- Command-Line Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate JSON metadata for floor plan dataset using connected components analysis.')
    parser.add_argument('--dataset', default=DEFAULT_BASE_DATA_DIR,
                        help=f'Path to the base dataset directory containing plot size subfolders (default: {DEFAULT_BASE_DATA_DIR})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to the base output directory for saving metadata (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--min_area', type=int, default=MIN_AREA_THRESHOLD,
                        help=f'Minimum pixel area for a region to be counted (default: {MIN_AREA_THRESHOLD})')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Update MIN_AREA_THRESHOLD if provided via argument
    MIN_AREA_THRESHOLD = args.min_area

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Run the processing
    process_dataset(args.dataset, args.output)

# --- END OF FILE generate_metadata.py ---