#!/bin/bash

# Base directory for preprocessed data
BASE_DIR="preprocessed"

# List of plot sizes
PLOT_SIZES=("5Marla" "10Marla" "20Marla")

# Number of Floor Plans to create (adjust as needed)
NUM_FPS=10  # Example: Creating 3 floor plans for each plot size

# Floor Level, Plan Type, and Version
FLOOR_LEVEL="GF"    # Ground Floor
PLAN_TYPE="FP"      # Floor Plan
VERSION="V01"       # Versioning

# Room types for masks
ROOM_TYPES=("Bedroom" "Bathroom" "Kitchen" "Lounge")

# Loop through each plot size
for PLOT in "${PLOT_SIZES[@]}"; do
    # Create main plot directory inside preprocessed
    mkdir -p "$BASE_DIR/$PLOT"

    # Loop to create multiple floor plans
    for ((i=1; i<=NUM_FPS; i++)); do
        # Format FP number as 3-digit (e.g., 001, 002, 003)
        FP_NUM=$(printf "%03d" $i)

        # Construct the folder name
        FP_FOLDER="${PLOT}_${FLOOR_LEVEL}_${PLAN_TYPE}_${FP_NUM}_${VERSION}"

        # Create directories
        FP_PATH="$BASE_DIR/$PLOT/$FP_FOLDER"
        MASKS_PATH="$FP_PATH/masks"

        mkdir -p "$FP_PATH"
        mkdir -p "$MASKS_PATH"

        # Create empty files for demonstration (uncomment if needed)
        # touch "$FP_PATH/${FP_FOLDER}.png"    # Preprocessed image
        touch "$FP_PATH/${FP_FOLDER}.json"   # Metadata file

        # Create mask files
        #for ROOM in "${ROOM_TYPES[@]}"; do
            # touch "$MASKS_PATH/${FP_FOLDER}_${ROOM}_mask.png"
        #done

        echo "Created structure for: $FP_FOLDER"
    done
done

echo "Directory structure created successfully."
