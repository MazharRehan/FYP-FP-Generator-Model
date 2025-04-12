import os
import re
import pandas as pd
import logging
import glob

# --- Configuration ---

# <<< CHANGE THIS >>> Path to the BASE directory containing subfolders like '5_marla', '10_marla'
BASE_DATA_DIR = "./dataset"

# Output CSV file name for the metadata
METADATA_FILENAME = "floor_plan_metadata_v3.csv" # Version 3

# Mapping from directory names to canonical PlotSize and dimensions
PLOT_INFO_FROM_DIR = {
    "5_marla": ("5Marla", (608, 1088)),
    "10_marla": ("10Marla", (849, 1570)),
    "20_marla": ("20Marla", (1209, 2170)),
}

# CORRECTED Regex pattern - Expects PlotSize prefix again
# Example: 5Marla_GF_FP_001_V01.png
FILENAME_PATTERN = re.compile(
    r"^(?P<PlotSize>\d+Marla)_"  # Expecting PlotSize again
    r"(?P<FloorLevel>GF)_"
    r"(?P<PlanType>FP)_"
    r"(?P<FP_Number>\d+)_"
    r"(?P<Version>V\d+)"
    r"\.png$",
    re.IGNORECASE
)

# List of room types (same as before)
ROOM_TYPES = [
    'Bedroom', 'Bathroom', 'Kitchen', 'Drawing Room', 'Garage',
    'Lounge/Sitting Area', 'Backyard', 'Stairs', 'Storage Room/Store',
    'Open Space', 'Prayer Room', 'Staircase', 'Lobby', 'Lawn', 'Dining',
    'Servant Quarters', 'Passage', 'Laundry', 'Dressing Area',
    'Side Garden', 'Library', 'Amber',
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---

def process_dataset_structure(base_dir):
    """
    Scans subdirectories, parses filenames (expecting PlotSize prefix),
    validates against directory, extracts metadata.
    """
    all_metadata = []
    total_processed = 0
    total_skipped = 0

    logging.info(f"Scanning base directory: {base_dir}")

    if not os.path.isdir(base_dir):
        logging.error(f"Error: Base directory not found at {base_dir}")
        return None

    for dir_name, (canonical_plot_size, (width, height)) in PLOT_INFO_FROM_DIR.items():
        current_subdir = os.path.join(base_dir, dir_name)
        logging.info(f"--- Processing subdirectory: {current_subdir} for expected PlotSize: {canonical_plot_size} ---")

        if not os.path.isdir(current_subdir):
            logging.warning(f"Subdirectory not found: {current_subdir}. Skipping.")
            continue

        file_paths = glob.glob(os.path.join(current_subdir, '*.png'))
        logging.info(f"Found {len(file_paths)} PNG files in {dir_name}.")

        processed_in_subdir = 0
        skipped_in_subdir = 0

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            match = FILENAME_PATTERN.match(filename)

            if match:
                try:
                    extracted_data = match.groupdict() # Includes 'PlotSize' from filename
                    filename_plot_size = extracted_data['PlotSize']

                    # --- Validation Step ---
                    # Normalize both for comparison (e.g., handle '5marla' vs '5Marla')
                    if filename_plot_size.lower() != canonical_plot_size.lower():
                        logging.warning(
                            f"Mismatch! File '{filename}' in directory '{dir_name}' "
                            f"has PlotSize '{filename_plot_size}' in name, "
                            f"but expected '{canonical_plot_size}' based on directory. "
                            f"Using directory '{canonical_plot_size}' for metadata."
                        )
                        # Decide how to handle mismatch: skip file or trust directory?
                        # Current logic: Trust directory, use canonical_plot_size & dims.

                    # --- Create Metadata Record ---
                    metadata = extracted_data # Start with data from filename regex
                    metadata['Filename'] = filename
                    metadata['RelativePath'] = os.path.relpath(file_path, base_dir).replace('\\', '/')

                    # **Override/Set PlotSize and Dimensions based on DIRECTORY**
                    metadata['PlotSize'] = canonical_plot_size # Use the one from PLOT_INFO_FROM_DIR
                    metadata['TargetWidth'] = width
                    metadata['TargetHeight'] = height

                    # Initialize room counts to 0 (Placeholder)
                    for room_type in ROOM_TYPES:
                        metadata[f'Count_{room_type}'] = 0

                    all_metadata.append(metadata)
                    processed_in_subdir += 1

                except Exception as e:
                    logging.error(f"Error processing matched file {filename} in {dir_name}: {e}")
                    skipped_in_subdir += 1
            else:
                # This warning should now only appear for filenames TRULY not matching the full pattern
                logging.warning(f"Filename did not match expected pattern in {dir_name}: {filename}")
                skipped_in_subdir += 1

        logging.info(f"Finished {dir_name}. Processed: {processed_in_subdir}, Skipped/Failed: {skipped_in_subdir}")
        total_processed += processed_in_subdir
        total_skipped += skipped_in_subdir

    logging.info(f"--- Total across all subdirectories ---")
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