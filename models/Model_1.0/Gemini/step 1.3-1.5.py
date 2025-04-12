import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2 # Using OpenCV for resizing
import torch
from torch.utils.data import Dataset, DataLoader, Sampler # Import Sampler
from torchvision import transforms
import logging
from tqdm.notebook import tqdm # Use standard tqdm if not in notebook
import random # Needed for shuffling groups

# --- Configuration --- (Same as before)
METADATA_FILENAME = "./floor_plan_metadata_v2.csv" # <<< MAKE SURE this is the correct CSV name
BASE_DATA_DIR = "./dataset"
BATCH_SIZE = 4
NUM_WORKERS = 0

# --- Color Mapping --- (Same as before)
COLOR_TO_ID_MAP = {
    (0, 0, 0): 0, (128, 0, 0): 22, (255, 0, 0): 1, (0, 0, 255): 2, (255, 165, 0): 3,
    (0, 128, 0): 4, (165, 42, 42): 5, (255, 255, 0): 6, (50, 205, 50): 7, (0, 128, 128): 8,
    (128, 0, 128): 9, (0, 255, 255): 10, (127, 127, 127): 11, (153, 51, 255): 12,
    (255, 0, 255): 13, (64, 224, 208): 14, (225, 192, 203): 15, (75, 0, 130): 16,
    (128, 128, 0): 17, (230, 230, 250): 18, (255, 127, 80): 19, (255, 215, 0): 20,
    (255, 191, 0): 21,
}
ID_TO_COLOR_MAP = {v: k for k, v in COLOR_TO_ID_MAP.items()}
NUM_CLASSES = max(COLOR_TO_ID_MAP.values()) + 1

# --- Room Types & Plot Sizes --- (Same as before)
ROOM_TYPES = [
    'Bedroom', 'Bathroom', 'Kitchen', 'Drawing Room', 'Garage', 'Lounge/Sitting Area',
    'Backyard', 'Stairs', 'Storage Room/Store', 'Open Space', 'Prayer Room', 'Staircase',
    'Lobby', 'Lawn', 'Dining', 'Servant Quarters', 'Passage', 'Laundry', 'Dressing Area',
    'Side Garden', 'Library', 'Amber',
]
ROOM_COUNT_COLS = [f'Count_{rt}' for rt in ROOM_TYPES]
PLOT_SIZE_CATEGORIES = ['5Marla', '10Marla', '20Marla']

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Optional: Address Pandas Future Warning (Can be ignored for now)
# pd.set_option('future.no_silent_downcasting', True)

# --- Preprocessing Functions (preprocess_image, prepare_condition_vector) ---
# (These functions remain exactly the same as in the previous corrected version)
def preprocess_image(image_path, target_width, target_height, color_map):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.int64)
        for rgb_tuple, class_id in color_map.items():
             if class_id == 0 and rgb_tuple == (0,0,0): continue
             matches = np.all(img_np == np.array(rgb_tuple).reshape(1, 1, 3), axis=2)
             mask[matches] = class_id
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        return resized_mask
    except FileNotFoundError: logging.error(f"Image file not found: {image_path}"); return None
    except Exception as e: logging.error(f"Error processing image {image_path}: {e}"); return None

def prepare_condition_vector(metadata_row, plot_size_categories, room_count_cols):
    try:
        plot_size = metadata_row['PlotSize']
        plot_size_index = plot_size_categories.index(plot_size)
        plot_size_one_hot = np.zeros(len(plot_size_categories), dtype=np.float32)
        plot_size_one_hot[plot_size_index] = 1.0
        # Note: The FutureWarning originates here, but doesn't break things yet.
        room_counts = metadata_row[room_count_cols].fillna(0).values.astype(np.float32)
        condition_vector = np.concatenate((plot_size_one_hot, room_counts))
        return condition_vector
    except KeyError as e: logging.error(f"Missing column: {e}. Row: {metadata_row.get('Filename', 'N/A')}"); return None
    except ValueError as e: logging.error(f"Plot size error: {metadata_row.get('PlotSize', 'N/A')}. Error: {e}"); return None
    except Exception as e: logging.error(f"Condition vector error for {metadata_row.get('Filename', 'N/A')}: {e}"); return None

# --- Dataset Class (FloorPlanDataset) ---
class FloorPlanDataset(Dataset):
    def __init__(self, metadata_df, base_dir, color_to_id_map,
                 plot_size_categories, room_count_cols, transform=None):
        self.metadata_df = metadata_df
        self.base_dir = base_dir
        self.color_to_id_map = color_to_id_map
        self.plot_size_categories = plot_size_categories
        self.room_count_cols = room_count_cols
        self.transform = transform
        logging.info(f"Dataset initialized with {len(self.metadata_df)} samples. NUM_CLASSES={NUM_CLASSES}")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        metadata_row = self.metadata_df.iloc[idx]
        image_rel_path = metadata_row['RelativePath']
        image_full_path = os.path.join(self.base_dir, image_rel_path)
        target_width = metadata_row['TargetWidth']
        target_height = metadata_row['TargetHeight']

        mask_np = preprocess_image(image_full_path, target_width, target_height, self.color_to_id_map)
        if mask_np is None:
            dummy_mask = torch.zeros((int(target_height), int(target_width)), dtype=torch.long) # Use int() for safety
            dummy_cond_len = len(self.plot_size_categories) + len(self.room_count_cols)
            dummy_cond = torch.zeros((dummy_cond_len,), dtype=torch.float)
            return dummy_mask, dummy_cond

        condition_np = prepare_condition_vector(metadata_row, self.plot_size_categories, self.room_count_cols)
        if condition_np is None:
            mask_tensor = torch.from_numpy(mask_np).long()
            dummy_cond_len = len(self.plot_size_categories) + len(self.room_count_cols)
            dummy_cond = torch.zeros((dummy_cond_len,), dtype=torch.float)
            return mask_tensor, dummy_cond

        mask_tensor = torch.from_numpy(mask_np).long()
        condition_tensor = torch.from_numpy(condition_np).float()

        sample = {'mask': mask_tensor, 'condition': condition_tensor}
        if self.transform: sample = self.transform(sample)
        return sample['mask'], sample['condition']


# --- NEW: Custom Sampler ---

class GroupedBatchSampler(Sampler[list[int]]):
    """
    Custom sampler that yields batches of indices, ensuring all indices
    in a batch belong to the same group (e.g., same PlotSize).
    """
    def __init__(self, metadata_df: pd.DataFrame, group_col: str, batch_size: int, shuffle: bool = True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing the metadata.
            group_col (str): The column name to group by (e.g., 'PlotSize').
            batch_size (int): The desired batch size.
            shuffle (bool): Whether to shuffle indices within groups and the order of groups.
        """
        super().__init__(metadata_df) # Provide data_source
        self.metadata_df = metadata_df
        self.group_col = group_col
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by the specified column
        self.groups = self.metadata_df.groupby(self.group_col).groups # dict: group_key -> index list
        self.group_keys = list(self.groups.keys())

        # Calculate total number of batches
        self.num_batches = 0
        for group_key in self.group_keys:
            self.num_batches += (len(self.groups[group_key]) + self.batch_size - 1) // self.batch_size

        logging.info(f"GroupedBatchSampler initialized. Found groups: {self.group_keys}. Total batches: {self.num_batches}")

    def __iter__(self):
        all_batches = []
        group_keys = self.group_keys
        if self.shuffle:
            random.shuffle(group_keys) # Shuffle the order of groups

        for group_key in group_keys:
            indices = self.groups[group_key].tolist() # Get indices for this group
            if self.shuffle:
                random.shuffle(indices) # Shuffle indices within the group

            # Create batches for the current group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                # Optional: Drop last if not full? For now, keep partial batches.
                # if len(batch_indices) == self.batch_size: # Uncomment to drop last partial batch
                all_batches.append(batch_indices)

        # Optional: Shuffle the order of all generated batches across groups
        if self.shuffle:
             random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        """Returns the total number of batches."""
        return self.num_batches


# --- Main Execution Block (Modified) ---
if __name__ == "__main__":

    logging.info("--- Starting Data Preparation Stage 1.3 - 1.5 (With Grouped Sampler) ---")
    logging.info(f"Number of segmentation classes (including walls/doors): {NUM_CLASSES}")

    # --- Load Metadata ---
    try:
        metadata = pd.read_csv(METADATA_FILENAME)
        logging.info(f"Loaded metadata from {METADATA_FILENAME}. Shape: {metadata.shape}")
        if 'RelativePath' not in metadata.columns or 'TargetWidth' not in metadata.columns or 'PlotSize' not in metadata.columns:
             raise ValueError("Metadata CSV missing required columns like 'RelativePath', 'TargetWidth', 'PlotSize'.")
        missing_count_cols = [col for col in ROOM_COUNT_COLS if col not in metadata.columns]
        if missing_count_cols:
            logging.warning(f"Metadata CSV is missing expected count columns: {missing_count_cols}. Counts treated as 0.")
            for col in missing_count_cols: metadata[col] = 0

    except FileNotFoundError: logging.error(f"FATAL: Metadata file not found: {METADATA_FILENAME}."); exit()
    except Exception as e: logging.error(f"FATAL: Error loading metadata: {e}"); exit()

    # --- Create Dataset ---
    floor_plan_dataset = FloorPlanDataset(
        metadata_df=metadata,
        base_dir=BASE_DATA_DIR,
        color_to_id_map=COLOR_TO_ID_MAP,
        plot_size_categories=PLOT_SIZE_CATEGORIES,
        room_count_cols=ROOM_COUNT_COLS,
        transform=None
    )

    # --- Create Sampler ---
    # Group by 'PlotSize' to ensure batches have consistent dimensions
    grouped_sampler = GroupedBatchSampler(
        metadata_df=metadata,
        group_col='PlotSize',
        batch_size=BATCH_SIZE,
        shuffle=True # Shuffle within groups and the order of groups/batches
    )

    # --- Create DataLoader (Using the Custom Sampler) ---
    if len(floor_plan_dataset) > 0:
        # When using a batch_sampler, batch_size, shuffle, sampler, drop_last are ignored.
        data_loader = DataLoader(
            floor_plan_dataset,
            batch_sampler=grouped_sampler, # Use the custom batch sampler
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            # collate_fn can still be customized if needed for other reasons
        )
        logging.info(f"DataLoader created using GroupedBatchSampler. Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")

        # --- Test DataLoader (Optional) ---
        logging.info("Testing DataLoader by fetching a few batches...")
        try:
            batch_count = 0
            for batch in tqdm(data_loader, desc="DataLoader Test", total=min(5, len(data_loader))): # Check first 5 batches
                 masks, conditions = batch
                 logging.info(f"Batch {batch_count + 1}:")
                 # Check that all masks in this batch have the same shape
                 first_mask_shape = masks[0].shape
                 all_same_shape = all(m.shape == first_mask_shape for m in masks)
                 logging.info(f"  Masks shape: {masks.shape} (Batch Size, H, W), All shapes identical in batch: {all_same_shape}")
                 if not all_same_shape:
                     logging.error("  ---> ERROR: Masks in batch have different shapes! Sampler failed?")
                 logging.info(f"  Unique mask values (sample 0): {torch.unique(masks[0])}")
                 logging.info(f"  Conditions shape: {conditions.shape}, Conditions dtype: {conditions.dtype}")
                 logging.info(f"  Condition vector (sample 0): {conditions[0]}")
                 # Optional: Verify plot size consistency using condition vector
                 plot_size_indices = conditions[:, :len(PLOT_SIZE_CATEGORIES)].argmax(dim=1)
                 if not torch.all(plot_size_indices == plot_size_indices[0]):
                      logging.error("  ---> ERROR: Batch contains multiple plot sizes based on condition vector!")

                 batch_count += 1
                 if batch_count >= 5: break # Stop after checking a few batches

            if batch_count > 0:
                logging.info("DataLoader test completed successfully.")
            else:
                 logging.warning("DataLoader test ran but yielded 0 batches.")

        except Exception as e:
            logging.error(f"Error occurred during DataLoader testing: {e}", exc_info=True)

    else:
        logging.warning("Dataset is empty. DataLoader not created.")

    logging.info("--- Data Preparation Stage 1.3 - 1.5 Finished ---")