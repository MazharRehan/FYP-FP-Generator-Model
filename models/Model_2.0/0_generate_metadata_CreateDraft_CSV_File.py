import os
import re
import pandas as pd
import logging
import glob

# --- Configuration ---

# Path to the BASE directory containing the floor plan images
BASE_DATA_DIR = "../Model_3.1/dataset"

# Output CSV file name for the metadata
METADATA_FILENAME = "floor_plan_metadata_v6_counts.csv" # Version 6

# Updated resolution (Width x Height)
STANDARDIZED_WIDTH = 512
STANDARDIZED_HEIGHT = 927

# CORRECTED Regex pattern - Expects PlotSize prefix
# Example: 5Marla_GF_FP_001_V01.png
FILENAME_PATTERN = re.compile(
    r"^(?P<PlotSize>\d+Marla)_"  # Expecting PlotSize prefix
    r"(?P<FloorLevel>GF)_"
    r"(?P<PlanType>FP)_"
    r"(?P<FP_Number>\d+)_"
    r"(?P<Version>V\d+)"
    r"\.png$",
    re.IGNORECASE
)

# List of room types (same as before)
ROOM_TYPES = [
    'Bedroom', 'Bathroom', 'Kitchen', 'DrawingRoom', 'Garage',
    'Lounge', 'Backyard', 'Stairs', 'Store',
    'Open Space', 'Prayer Room', 'Staircase', 'Lobby', 'Lawn', 'Dining',
    'ServantQuarters', 'Passage', 'Laundry', 'DressingArea',
    'SideGarden', 'Library',
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---

def process_dataset_structure(base_dir):
    """
    Scans the dataset directory, parses filenames, and extracts metadata.
    """
    all_metadata = []
    total_processed = 0
    total_skipped = 0

    logging.info(f"Scanning dataset directory: {base_dir}")

    if not os.path.isdir(base_dir):
        logging.error(f"Error: Dataset directory not found at {base_dir}")
        return None

    file_paths = glob.glob(os.path.join(base_dir, '*.png'))
    logging.info(f"Found {len(file_paths)} PNG files in the dataset directory.")

    processed_in_subdir = 0
    skipped_in_subdir = 0

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        match = FILENAME_PATTERN.match(filename)

        if match:
            try:
                extracted_data = match.groupdict()  # Includes 'PlotSize' from filename

                # Create Metadata Record
                metadata = extracted_data  # Start with data from filename regex
                metadata['Filename'] = filename
                metadata['RelativePath'] = os.path.relpath(file_path, base_dir).replace('\\', '/')

                # Use the standardized dimensions
                metadata['TargetWidth'] = STANDARDIZED_WIDTH
                metadata['TargetHeight'] = STANDARDIZED_HEIGHT

                # Initialize room counts to 0 (Placeholder)
                for room_type in ROOM_TYPES:
                    metadata[f'Count_{room_type}'] = 0

                all_metadata.append(metadata)
                processed_in_subdir += 1

            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                skipped_in_subdir += 1
        else:
            logging.warning(f"Filename did not match expected pattern: {filename}")
            skipped_in_subdir += 1

    logging.info(f"Finished processing. Processed: {processed_in_subdir}, Skipped/Failed: {skipped_in_subdir}")
    total_processed += processed_in_subdir
    total_skipped += skipped_in_subdir

    logging.info(f"--- Total across all files ---")
    logging.info(f"Successfully processed: {total_processed}, Skipped/Failed: {total_skipped}")
    return all_metadata

# --- save_metadata_to_csv function remains the same ---
def save_metadata_to_csv(metadata_list, output_filename):
    """Saves the list of metadata dictionaries to a CSV file."""
    if not metadata_list:
        logging.warning("No metadata was generated. CSV file will not be created.")
        return

    try:
        df = pd.DataFrame(metadata_list)
        base_cols = ['RelativePath', 'Filename', 'PlotSize', 'FloorLevel', 'PlanType', 'FP_Number', 'Version', 'TargetWidth', 'TargetHeight']
        count_cols = sorted([col for col in df.columns if col.startswith('Count_')])
        all_cols_ordered = base_cols + count_cols
        for col in all_cols_ordered:
             if col not in df.columns:
                 df[col] = None
        df = df[all_cols_ordered]
        df.to_csv(output_filename, index=False, encoding='utf-8')
        logging.info(f"Metadata successfully saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving metadata to CSV: {e}")

# --- Execution ---
if __name__ == "__main__":
    metadata = process_dataset_structure(BASE_DATA_DIR)
    if metadata:
        save_metadata_to_csv(metadata, METADATA_FILENAME)
