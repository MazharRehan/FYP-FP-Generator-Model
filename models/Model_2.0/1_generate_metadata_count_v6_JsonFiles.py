# --- START OF FILE generate_metadata.py ---

import os
import json
import numpy as np
import cv2  # Using OpenCV for image processing
import argparse
import re
import logging
from datetime import datetime
from collections import defaultdict

# --- Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base directory containing all PNG files directly (can be overridden by args)
DEFAULT_BASE_DATA_DIR = "../Model_3.1/dataset"
# Base output directory (can be overridden by args)
DEFAULT_OUTPUT_DIR = "./data/"

# --- NEW: Standardized Image Dimensions (assumed for all input images) ---
STANDARDIZED_WIDTH_PX = 512
STANDARDIZED_HEIGHT_PX = 927
STANDARDIZED_TOTAL_PIXELS = STANDARDIZED_WIDTH_PX * STANDARDIZED_HEIGHT_PX
STANDARDIZED_ASPECT_RATIO = STANDARDIZED_WIDTH_PX / STANDARDIZED_HEIGHT_PX

# Define color mapping (Name -> RGB)
COLOR_MAP = {
    'Bedroom': [255, 0, 0], 'Bathroom': [0, 0, 255], 'Kitchen': [255, 165, 0],
    'DrawingRoom': [0, 128, 0], 'Garage': [165, 42, 42], 'Lounge': [255, 255, 0],
    'Backyard': [50, 205, 50], 'Stairs': [0, 128, 128], 'Store': [128, 0, 128],
    'OpenSpace': [0, 255, 255], 'PrayerRoom': [127, 127, 127], 'Staircase': [153, 51, 255],
    'Lobby': [255, 0, 255], 'Lawn': [64, 224, 208], 'Dining': [225, 192, 203],
    'ServantQuarter': [75, 0, 130], 'Passage': [128, 128, 0], 'Laundry': [230, 230, 250],
    'DressingArea': [255, 127, 80], 'SideGarden': [255, 215, 0], 'Library': [255, 191, 0],
    'Wall': [0, 0, 0], 'Door': [128, 0, 0], 'Background': [255, 255, 255]
}

ROOM_TYPES_TO_COUNT = [
    'Bedroom', 'Bathroom', 'Kitchen', 'DrawingRoom', 'Garage', 'Lounge', 'Backyard',
    'Stairs', 'Store', 'OpenSpace', 'PrayerRoom', 'Staircase', 'Lobby', 'Lawn',
    'Dining', 'ServantQuarter', 'Passage', 'Laundry', 'DressingArea', 'SideGarden', 'Library'
]

# --- UPDATED: Physical Plot Information Map ---
# Stores physical dimensions and calculates physical area.
# Original pixel/DPI info can be kept for historical reference if needed.
PHYSICAL_PLOT_INFO_MAP = {
    '5Marla': {
        'physical_dims_ft': [25, 45],  # Width, Height in feet
        'physical_area_sqft': 25 * 45,  # 1125 sq ft
        # Original nominal values, for reference
        'original_nominal_pixels': [608, 1088], 'original_nominal_aspect_ratio': 0.559,
    },
    '10Marla': {
        'physical_dims_ft': [35, 65],  # Width, Height in feet
        'physical_area_sqft': 35 * 65,  # 2275 sq ft
        'original_nominal_pixels': [849, 1570], 'original_nominal_aspect_ratio': 0.541,
    },
    '20Marla': {
        'physical_dims_ft': [50, 90],  # Width, Height in feet
        'physical_area_sqft': 50 * 90,  # 4500 sq ft
        'original_nominal_pixels': [1209, 2170], 'original_nominal_aspect_ratio': 0.557,
    }
}

# Regex for Filename Parsing
FILENAME_PATTERN = re.compile(
    r"^(?P<PlotSize>\d+Marla)_"
    r"(?P<FloorLevel>GF)_"
    r"(?P<PlanType>FP)_"
    r"(?P<FP_Number>\d+)_"
    r"(?P<Version>V\d+)"
    r"\.png$",
    re.IGNORECASE
)

MIN_AREA_THRESHOLD = 80


def get_rgb_to_name_map(color_map):
    rgb_map = {}
    for name, rgb in color_map.items():
        rgb_tuple = tuple(rgb)
        if rgb_tuple in rgb_map:
            existing_name = rgb_map[rgb_tuple]
            if isinstance(existing_name, list):
                existing_name.append(name)
            else:
                rgb_map[rgb_tuple] = [existing_name, name]
        else:
            rgb_map[rgb_tuple] = name
    return rgb_map


RGB_TO_NAME_MAP = get_rgb_to_name_map(COLOR_MAP)


def analyze_floor_plan_updated(image_path):
    file_name = os.path.basename(image_path)
    logging.debug(f"Analyzing: {file_name}")

    match = FILENAME_PATTERN.match(file_name)
    if not match:
        logging.warning(f"Filename pattern mismatch, skipping: {file_name}")
        return None

    extracted_data = match.groupdict()
    plot_size_from_filename = extracted_data['PlotSize']

    # Normalize plot_size_from_filename capitalization if necessary for map lookup
    # (e.g., if filenames are '5marla' but map keys are '5Marla')
    # Assuming direct match for now, e.g. "5Marla" in filename matches "5Marla" key
    plot_info = PHYSICAL_PLOT_INFO_MAP.get(plot_size_from_filename)
    if not plot_info:
        # Attempt normalization (e.g. 5marla -> 5Marla)
        normalized_plot_key = plot_size_from_filename[:-5].capitalize() + "Marla"
        plot_info = PHYSICAL_PLOT_INFO_MAP.get(normalized_plot_key)
        if not plot_info:
            logging.warning(
                f"Physical plot info not found for '{plot_size_from_filename}' (or '{normalized_plot_key}') in PHYSICAL_PLOT_INFO_MAP for file {file_name}. Cannot calculate physical areas.")
            # Decide if to proceed or return None. For now, proceed without physical area calcs.
            # return None
            physical_area_sqft_total_plot = 0
            area_per_pixel_sqft = 0  # Set to 0 if plot info not found
        else:
            physical_area_sqft_total_plot = plot_info['physical_area_sqft']
    else:
        physical_area_sqft_total_plot = plot_info['physical_area_sqft']

    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: raise ValueError(f"Could not load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image_rgb.shape
        logging.debug(f"Image loaded: {img_width}x{img_height}")

        # Verify if image dimensions match the expected standardized dimensions
        current_image_total_pixels = img_width * img_height
        if (img_width, img_height) != (STANDARDIZED_WIDTH_PX, STANDARDIZED_HEIGHT_PX):
            logging.warning(
                f"Image dimensions ({img_width}x{img_height}) for {file_name} "
                f"mismatch expected standardized dimensions ({STANDARDIZED_WIDTH_PX}x{STANDARDIZED_HEIGHT_PX}). "
                "Area per pixel calculations will use ACTUAL image dimensions."
            )
        # If physical_area_sqft_total_plot is available and current_image_total_pixels is > 0
        if physical_area_sqft_total_plot > 0 and current_image_total_pixels > 0:
            area_per_pixel_sqft = physical_area_sqft_total_plot / current_image_total_pixels
        else:
            area_per_pixel_sqft = 0  # Cannot calculate if total physical area or pixel count is zero/unknown


    except Exception as e:
        logging.error(f"Error loading image {file_name}: {e}")
        return None

    room_counts = defaultdict(int)
    room_areas_pixels = defaultdict(int)
    masks = {tuple(rgb): cv2.inRange(image_rgb, np.array(rgb), np.array(rgb)) for name, rgb in COLOR_MAP.items()}

    processed_room_names = set()
    for room_name in ROOM_TYPES_TO_COUNT:
        target_rgb_list = COLOR_MAP.get(room_name)
        if not target_rgb_list: continue
        target_rgb_tuple = tuple(target_rgb_list)
        if room_name in processed_room_names: continue

        binary_mask = masks.get(target_rgb_tuple)
        if binary_mask is None or np.count_nonzero(binary_mask) == 0: continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        count_for_room, area_for_room = 0, 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA_THRESHOLD:
                count_for_room += 1
                area_for_room += area

        name_at_rgb = RGB_TO_NAME_MAP.get(target_rgb_tuple)
        room_names_for_color = []
        if isinstance(name_at_rgb, list):
            room_names_for_color.extend(name_at_rgb)
        elif isinstance(name_at_rgb, str):
            room_names_for_color.append(name_at_rgb)

        for rn in room_names_for_color:
            if rn in ROOM_TYPES_TO_COUNT:
                room_counts[rn] = count_for_room
                room_areas_pixels[rn] = area_for_room
                processed_room_names.add(rn)
                logging.debug(f"  Counted {rn}: {count_for_room} regions, {area_for_room}px area.")

    total_counted_area_pixels = sum(room_areas_pixels.values())

    # Calculate physical areas if area_per_pixel_sqft is valid
    room_areas_sqft = {}
    total_counted_area_sqft = 0
    if area_per_pixel_sqft > 0:
        room_areas_sqft = {k: round(v * area_per_pixel_sqft, 2) for k, v in room_areas_pixels.items() if v > 0}
        total_counted_area_sqft = round(total_counted_area_pixels * area_per_pixel_sqft, 2)

    metadata = {
        'file_name': file_name,
        'parsed_plot_type': extracted_data['PlotSize'],
        'floor_level': extracted_data['FloorLevel'],
        'plan_type': extracted_data['PlanType'],
        'fp_number': extracted_data['FP_Number'],
        'version': extracted_data['Version'],
        'image_actual_width_px': int(img_width),
        'image_actual_height_px': int(img_height),
        'image_assumed_standardized_width_px': STANDARDIZED_WIDTH_PX,  # For reference
        'image_assumed_standardized_height_px': STANDARDIZED_HEIGHT_PX,  # For reference
        'plot_physical_area_sqft': physical_area_sqft_total_plot,
        'area_per_pixel_sqft': round(area_per_pixel_sqft, 8) if area_per_pixel_sqft > 0 else 0,
        'room_counts': {k: int(v) for k, v in room_counts.items() if v > 0},
        'room_areas_pixels': {k: int(v) for k, v in room_areas_pixels.items() if v > 0},
        'total_counted_area_pixels': int(total_counted_area_pixels),
        'room_areas_sqft': room_areas_sqft,  # Will be empty if area_per_pixel_sqft is 0
        'total_counted_area_sqft': total_counted_area_sqft,  # Will be 0 if area_per_pixel_sqft is 0
        'physical_plot_info_ref': plot_info if plot_info else {},  # Reference to original map entry
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    logging.debug(f"Metadata generated for {file_name}")
    return metadata


def process_dataset(dataset_path, output_path_base):
    logging.info(f"Starting dataset processing from: {dataset_path}")
    logging.info(f"Outputting JSON metadata to subfolders within: {output_path_base}")
    if not os.path.isdir(dataset_path):
        logging.error(f"Dataset directory not found: {dataset_path}")
        return

    processed_count, error_count = 0, 0
    try:
        all_files = os.listdir(dataset_path)
        png_files = [f for f in all_files if f.lower().endswith('.png') and FILENAME_PATTERN.match(f)]
        logging.info(f"Found {len(png_files)} potential PNG files matching pattern in {dataset_path}.")
    except OSError as e:
        logging.error(f"Could not read dataset directory {dataset_path}: {e}")
        return

    if not png_files:
        logging.warning(f"No PNG files matching the expected filename pattern found in {dataset_path}.")
        return

    for png_file in png_files:
        image_path = os.path.join(dataset_path, png_file)
        filename_match = FILENAME_PATTERN.match(png_file)  # Already checked, but good for direct access
        parsed_plot_size = filename_match.groupdict()['PlotSize']
        plot_specific_output_dir = os.path.join(output_path_base, parsed_plot_size, 'metadata', 'json')
        try:
            os.makedirs(plot_specific_output_dir, exist_ok=True)
        except OSError as e:
            logging.error(
                f"Could not create output directory {plot_specific_output_dir}: {e}. Skipping file {png_file}.")
            error_count += 1;
            continue

        try:
            metadata = analyze_floor_plan_updated(image_path)
            if metadata:
                json_filename = os.path.splitext(png_file)[0] + '.json'
                json_path = os.path.join(plot_specific_output_dir, json_filename)
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logging.info(f"Successfully processed and saved metadata for: {png_file} to {json_path}")
                processed_count += 1
            else:
                logging.warning(f"Skipped generating metadata for: {png_file} (analysis returned None)")
                error_count += 1
        except Exception as e:
            logging.error(f"Unexpected error processing {png_file}: {e}", exc_info=False)
            error_count += 1

    logging.info(f"--- Dataset Processing Complete ---")
    logging.info(f"Total files processed successfully: {processed_count}")
    logging.info(f"Total files skipped or failed: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate JSON metadata for floor plan dataset.')
    parser.add_argument('--dataset', default=DEFAULT_BASE_DATA_DIR,
                        help=f'Path to the flat dataset directory (default: {DEFAULT_BASE_DATA_DIR})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to the base output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--min_area', type=int, default=MIN_AREA_THRESHOLD,
                        help=f'Minimum pixel area for a region (default: {MIN_AREA_THRESHOLD})')
    parser.add_argument('--std_width', type=int, default=STANDARDIZED_WIDTH_PX,
                        help=f'Assumed standardized width of input images in pixels (default: {STANDARDIZED_WIDTH_PX})')
    parser.add_argument('--std_height', type=int, default=STANDARDIZED_HEIGHT_PX,
                        help=f'Assumed standardized height of input images in pixels (default: {STANDARDIZED_HEIGHT_PX})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Update global constants from command-line arguments
    MIN_AREA_THRESHOLD = args.min_area
    STANDARDIZED_WIDTH_PX = args.std_width
    STANDARDIZED_HEIGHT_PX = args.std_height
    STANDARDIZED_TOTAL_PIXELS = STANDARDIZED_WIDTH_PX * STANDARDIZED_HEIGHT_PX
    STANDARDIZED_ASPECT_RATIO = STANDARDIZED_WIDTH_PX / STANDARDIZED_HEIGHT_PX if STANDARDIZED_HEIGHT_PX > 0 else 0

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Using standardized image dimensions: {STANDARDIZED_WIDTH_PX}x{STANDARDIZED_HEIGHT_PX} px")
    process_dataset(args.dataset, args.output)

# --- END OF FILE generate_metadata.py ---