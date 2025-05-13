# --- START OF UPDATED generate_metadata_v3.py (incorporating v2 features and our discussion) ---

import os
import json
import numpy as np
import cv2  # Using OpenCV for image processing
import argparse
import re
import logging
import sys
import pandas as pd
from datetime import datetime
from collections import defaultdict

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_BASE_DATA_DIR = "../Model_3.1/dataset"  # Flat structure
DEFAULT_OUTPUT_DIR = "./data/new"
DEFAULT_CSV_OUTPUT_FILENAME = "floor_plan_metadata_consolidated.csv"

# --- Standardized Image Dimensions (assumed for all input images) ---
# These will be updated by command-line arguments if provided
STANDARDIZED_WIDTH_PX = 512
STANDARDIZED_HEIGHT_PX = 927

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

PHYSICAL_PLOT_INFO_MAP = {
    '5Marla': {'physical_dims_ft': [25, 45], 'physical_area_sqft': 25 * 45},
    '10Marla': {'physical_dims_ft': [35, 65], 'physical_area_sqft': 35 * 65},
    '20Marla': {'physical_dims_ft': [50, 90], 'physical_area_sqft': 50 * 90}
}

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


def analyze_floor_plan(image_path, standardized_width_px, standardized_height_px):
    file_name = os.path.basename(image_path)
    logging.debug(f"Analyzing: {file_name}")

    match = FILENAME_PATTERN.match(file_name)
    if not match:
        logging.warning(f"Filename pattern mismatch, skipping: {file_name}")
        return None
    extracted_data = match.groupdict()
    plot_size_from_filename = extracted_data['PlotSize']

    plot_info = PHYSICAL_PLOT_INFO_MAP.get(plot_size_from_filename)
    if not plot_info:
        normalized_plot_key = plot_size_from_filename[:-5].capitalize() + "Marla"
        plot_info = PHYSICAL_PLOT_INFO_MAP.get(normalized_plot_key)
        if not plot_info:
            logging.warning(
                f"Physical plot info not found for '{plot_size_from_filename}' (or '{normalized_plot_key}') for {file_name}.")
            plot_physical_area_sqft_total = 0
        else:
            plot_physical_area_sqft_total = plot_info['physical_area_sqft']
    else:
        plot_physical_area_sqft_total = plot_info['physical_area_sqft']

    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: raise ValueError(f"Could not load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_actual_height, img_actual_width, _ = image_rgb.shape
        logging.debug(f"Image loaded: {img_actual_width}x{img_actual_height}")

        actual_image_total_pixels = img_actual_width * img_actual_height
        if (img_actual_width, img_actual_height) != (standardized_width_px, standardized_height_px):
            logging.warning(
                f"Image dimensions ({img_actual_width}x{img_actual_height}) for {file_name} "
                f"mismatch expected standardized dimensions ({standardized_width_px}x{standardized_height_px}). "
                "Calculations will use ACTUAL image dimensions."
            )

        area_per_pixel_sqft = 0
        if plot_physical_area_sqft_total > 0 and actual_image_total_pixels > 0:
            area_per_pixel_sqft = plot_physical_area_sqft_total / actual_image_total_pixels

    except Exception as e:
        logging.error(f"Error loading image {file_name}: {e}")
        return None

    room_counts = defaultdict(int)
    room_total_areas_pixels = defaultdict(int)  # Total pixel area for each room type
    room_instance_areas_pixels = defaultdict(list)  # List of pixel areas for each instance of a room type

    # Pre-create all masks once
    color_masks = {tuple(rgb): cv2.inRange(image_rgb, np.array(rgb), np.array(rgb)) for name, rgb in COLOR_MAP.items()}

    processed_room_types_for_mask = set()
    for room_name_config in ROOM_TYPES_TO_COUNT:  # e.g. "Bedroom"
        target_rgb_list = COLOR_MAP.get(room_name_config)
        if not target_rgb_list: continue
        target_rgb_tuple = tuple(target_rgb_list)

        # Avoid re-processing the same mask if multiple room names share a color
        if target_rgb_tuple in processed_room_types_for_mask:
            continue

        binary_mask = color_masks.get(target_rgb_tuple)
        if binary_mask is None or np.count_nonzero(binary_mask) == 0: continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        current_mask_instance_count = 0
        current_mask_total_pixel_area = 0
        current_mask_instance_pixel_areas = []

        for i in range(1, num_labels):  # Label 0 is the background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA_THRESHOLD:
                current_mask_instance_count += 1
                current_mask_total_pixel_area += area
                current_mask_instance_pixel_areas.append(area)

        # Assign counts and areas to all room names that map to this target_rgb_tuple
        room_names_for_this_color = RGB_TO_NAME_MAP.get(target_rgb_tuple)
        if not isinstance(room_names_for_this_color, list):
            room_names_for_this_color = [room_names_for_this_color]

        for rn in room_names_for_this_color:
            if rn in ROOM_TYPES_TO_COUNT:  # Ensure it's a room type we're designated to count
                room_counts[rn] = current_mask_instance_count
                room_total_areas_pixels[rn] = current_mask_total_pixel_area
                room_instance_areas_pixels[rn] = sorted(current_mask_instance_pixel_areas.copy(),
                                                        reverse=True)  # Store sorted
                logging.debug(
                    f"  Processed {rn} (Color {target_rgb_tuple}): {current_mask_instance_count} instances, {current_mask_total_pixel_area}px total area.")

        processed_room_types_for_mask.add(target_rgb_tuple)

    overall_total_counted_pixels = sum(room_total_areas_pixels.values())

    room_total_areas_sqft = {}
    room_instance_areas_sqft = defaultdict(list)
    overall_total_counted_sqft = 0

    if area_per_pixel_sqft > 0:
        for room_type, total_px_area in room_total_areas_pixels.items():
            room_total_areas_sqft[room_type] = round(total_px_area * area_per_pixel_sqft, 2)

        for room_type, instance_px_list in room_instance_areas_pixels.items():
            instance_sqft_list = [round(px * area_per_pixel_sqft, 2) for px in instance_px_list]
            room_instance_areas_sqft[room_type] = instance_sqft_list  # Already sorted

        overall_total_counted_sqft = round(overall_total_counted_pixels * area_per_pixel_sqft, 2)

    metadata = {
        'file_name': file_name,
        'parsed_plot_type': extracted_data['PlotSize'],
        'floor_level': extracted_data['FloorLevel'],
        'plan_type': extracted_data['PlanType'],
        'fp_number': extracted_data['FP_Number'],
        'version': extracted_data['Version'],
        'image_actual_width_px': int(img_actual_width),
        'image_actual_height_px': int(img_actual_height),
        'image_configured_standardized_width_px': standardized_width_px,
        'image_configured_standardized_height_px': standardized_height_px,
        'plot_physical_area_sqft': plot_physical_area_sqft_total,
        'calculated_area_per_pixel_sqft': round(area_per_pixel_sqft, 8) if area_per_pixel_sqft > 0 else 0,
        'room_counts': {k: int(v) for k, v in room_counts.items() if v > 0},
        'room_total_areas_pixels': {k: int(v) for k, v in room_total_areas_pixels.items() if v > 0},
        'overall_total_counted_pixels': int(overall_total_counted_pixels),
        'room_instance_areas_pixels': {k: [int(px) for px in v] for k, v in room_instance_areas_pixels.items() if v},
        # Store pixel areas for instances
        'room_total_areas_sqft': room_total_areas_sqft,
        'overall_total_counted_sqft': overall_total_counted_sqft,
        'room_instance_areas_sqft': dict(room_instance_areas_sqft),  # Already contains lists of sqft areas
        'physical_plot_info_ref': plot_info if plot_info else {},
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    logging.debug(f"Metadata generated for {file_name}")
    return metadata


def save_consolidated_metadata_to_csv(all_metadata_list, csv_output_filename):
    if not all_metadata_list:
        logging.warning("No metadata to save to CSV.")
        return

    flattened_data_for_csv = []
    for meta_item in all_metadata_list:
        flat_dict = {
            'FileName': meta_item.get('file_name'),
            'PlotType': meta_item.get('parsed_plot_type'),
            'FloorLevel': meta_item.get('floor_level'),
            'PlanType': meta_item.get('plan_type'),
            'FP_Number': meta_item.get('fp_number'),
            'Version': meta_item.get('version'),
            'ImageActualWidthPx': meta_item.get('image_actual_width_px'),
            'ImageActualHeightPx': meta_item.get('image_actual_height_px'),
            'PlotPhysicalAreaSqFt': meta_item.get('plot_physical_area_sqft'),
            'AreaPerPixelSqFt': meta_item.get('calculated_area_per_pixel_sqft'),
            'OverallTotalCountedPixels': meta_item.get('overall_total_counted_pixels'),
            'OverallTotalCountedSqFt': meta_item.get('overall_total_counted_sqft'),
        }

        # Add room counts
        for room, count in meta_item.get('room_counts', {}).items():
            flat_dict[f'Count_{room}'] = count

        # Add total room areas (pixels and sqft)
        for room, area_px in meta_item.get('room_total_areas_pixels', {}).items():
            flat_dict[f'TotalPx_{room}'] = area_px
        for room, area_sqft in meta_item.get('room_total_areas_sqft', {}).items():
            flat_dict[f'TotalSqFt_{room}'] = area_sqft

        # Add individual room instance areas (pixels and sqft)
        # Max instances to record per room type (to avoid too many columns) - adjust if needed
        max_instances_per_room_type_in_csv = 5
        for room, instances_sqft_list in meta_item.get('room_instance_areas_sqft', {}).items():
            instances_px_list = meta_item.get('room_instance_areas_pixels', {}).get(room, [])
            for i in range(min(len(instances_sqft_list), max_instances_per_room_type_in_csv)):
                flat_dict[f'{room}{i + 1}_SqFt'] = instances_sqft_list[i]
                if i < len(instances_px_list):  # ensure px list is also available
                    flat_dict[f'{room}{i + 1}_Px'] = instances_px_list[i]

        flattened_data_for_csv.append(flat_dict)

    df = pd.DataFrame(flattened_data_for_csv)

    # Define a logical order for columns
    base_cols = ['FileName', 'PlotType', 'FloorLevel', 'PlanType', 'FP_Number', 'Version',
                 'ImageActualWidthPx', 'ImageActualHeightPx', 'PlotPhysicalAreaSqFt', 'AreaPerPixelSqFt',
                 'OverallTotalCountedPixels', 'OverallTotalCountedSqFt']

    # Dynamically get other column categories
    count_cols = sorted([col for col in df.columns if col.startswith('Count_')])
    total_px_cols = sorted([col for col in df.columns if col.startswith('TotalPx_')])
    total_sqft_cols = sorted([col for col in df.columns if col.startswith('TotalSqFt_')])

    # Instance columns (more complex to sort perfectly but this is a good start)
    instance_cols = sorted([col for col in df.columns if any(
        rt in col and ("_SqFt" in col or "_Px" in col) and col[len(rt):len(rt) + 1].isdigit() for rt in
        ROOM_TYPES_TO_COUNT)])

    ordered_columns = base_cols + count_cols + total_px_cols + total_sqft_cols + instance_cols

    # Ensure all expected columns are in the DataFrame, add if missing (with NaN)
    for col in ordered_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[ordered_columns]  # Reorder

    try:
        df.to_csv(csv_output_filename, index=False, encoding='utf-8')
        logging.info(f"Consolidated metadata successfully saved to {csv_output_filename}")
    except Exception as e:
        logging.error(f"Error saving consolidated metadata to CSV: {e}")


def process_images_in_dataset(dataset_path, output_path_base, csv_filename, std_width, std_height):
    logging.info(f"Starting dataset processing from: {dataset_path}")
    logging.info(f"Outputting JSON metadata to subfolders within: {output_path_base}")
    logging.info(f"Consolidated CSV will be saved to: {csv_filename}")
    logging.info(f"Using configured standardized image dimensions: {std_width}x{std_height} px")

    if not os.path.isdir(dataset_path):
        logging.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    all_generated_metadata = []
    processed_file_count = 0
    error_file_count = 0

    try:
        all_files_in_dataset = os.listdir(dataset_path)
        png_image_files = [f for f in all_files_in_dataset if f.lower().endswith('.png') and FILENAME_PATTERN.match(f)]
        logging.info(f"Found {len(png_image_files)} PNG files matching pattern in {dataset_path}.")
    except OSError as e:
        logging.error(f"Could not read dataset directory {dataset_path}: {e}")
        return

    if not png_image_files:
        logging.warning(f"No PNG files matching the expected filename pattern found in {dataset_path}.")
        return

    for png_file_name in png_image_files:
        full_image_path = os.path.join(dataset_path, png_file_name)

        filename_match = FILENAME_PATTERN.match(png_file_name)  # Already validated
        parsed_plot_size_for_output = filename_match.groupdict()['PlotSize']

        plot_specific_json_output_dir = os.path.join(output_path_base, parsed_plot_size_for_output, 'metadata', 'json')
        try:
            os.makedirs(plot_specific_json_output_dir, exist_ok=True)
        except OSError as e:
            logging.error(
                f"Could not create output directory {plot_specific_json_output_dir}: {e}. Skipping file {png_file_name}.")
            error_file_count += 1
            continue

        try:
            metadata_dict = analyze_floor_plan(full_image_path, std_width, std_height)
            if metadata_dict:
                json_output_filename = os.path.splitext(png_file_name)[0] + '.json'
                full_json_output_path = os.path.join(plot_specific_json_output_dir, json_output_filename)
                with open(full_json_output_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)

                all_generated_metadata.append(metadata_dict)
                logging.info(f"Successfully processed and saved JSON for: {png_file_name} to {full_json_output_path}")
                processed_file_count += 1
            else:
                logging.warning(f"Skipped generating metadata for: {png_file_name} (analysis returned None)")
                error_file_count += 1
        except Exception as e:
            logging.error(f"Unexpected error processing {png_file_name}: {e}", exc_info=False)
            error_file_count += 1

    logging.info("--- Image Processing Complete ---")
    logging.info(f"Total files processed successfully for JSON: {processed_file_count}")
    logging.info(f"Total files skipped or failed: {error_file_count}")

    if all_generated_metadata:
        save_consolidated_metadata_to_csv(all_generated_metadata, csv_filename)
    else:
        logging.info("No metadata was generated, so CSV file was not created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate JSON and consolidated CSV metadata for floor plan dataset.')
    parser.add_argument('--dataset', default=DEFAULT_BASE_DATA_DIR,
                        help=f'Path to the flat dataset directory containing PNG floor plans (default: {DEFAULT_BASE_DATA_DIR})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to the base output directory for saving JSON metadata (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--csv_output', default=DEFAULT_CSV_OUTPUT_FILENAME,
                        help=f'Filename for the consolidated CSV output (default: {DEFAULT_CSV_OUTPUT_FILENAME})')
    parser.add_argument('--min_area', type=int, default=MIN_AREA_THRESHOLD,
                        help=f'Minimum pixel area for a region to be counted (default: {MIN_AREA_THRESHOLD})')
    parser.add_argument('--std_width', type=int, default=STANDARDIZED_WIDTH_PX,
                        help=f'Configured standardized width of input images in pixels (default: {STANDARDIZED_WIDTH_PX})')
    parser.add_argument('--std_height', type=int, default=STANDARDIZED_HEIGHT_PX,
                        help=f'Configured standardized height of input images in pixels (default: {STANDARDIZED_HEIGHT_PX})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    MIN_AREA_THRESHOLD = args.min_area  # Update global from arg

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    process_images_in_dataset(
        args.dataset,
        args.output,
        args.csv_output,
        args.std_width,
        args.std_height
    )

# --- END OF UPDATED generate_metadata_v3.py ---