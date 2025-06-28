import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple

# Configuration for architectural standards (adapted from cganFloorPlanGenerator.ts)
ARCHITECTURAL_PATTERNS = {
    'adjacency_matrix': {
        'bedroom_bathroom': 0.95,
        'kitchen_dining': 0.90,
        'lounge_drawingRoom': 0.85,
        # Add other adjacency rules as needed
    },
    'room_size_ratios': {
        'bedroom': {'min_area': 100, 'ratio': 1.0},
        'bathroom': {'min_area': 25, 'ratio': 0.75},
        'kitchen': {'min_area': 80, 'ratio': 1.2},
        'lounge': {'min_area': 252, 'ratio': 1.29},
        # Add other room types as needed
    }
}

# Room class to mimic TypeScript Room type
class Room:
    def __init__(self, room_type: str, position: Tuple[float, float], size: Tuple[float, float]):
        self.type = room_type
        self.position = {'x': position[0], 'y': position[1]}
        self.size = {'width': size[0], 'height': size[1]}

# GeneratedFloorPlan class to mimic TypeScript structure
class GeneratedFloorPlan:
    def __init__(self, rooms: List[Room]):
        self.rooms = rooms

class CNNValidator:
    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        self.initialize_model()

    def initialize_model(self):
        """Initialize a U-Net-based CNN model for layout validation."""
        try:
            self.model = self.create_cnn_model()
            # Optionally load pre-trained weights
            # self.model.load_weights('path/to/pretrained/weights.h5')
            self.is_model_loaded = True
        except Exception as e:
            print(f"Failed to initialize CNN model: {e}")
            self.is_model_loaded = False

    def create_cnn_model(self) -> tf.keras.Model:
        """Create a U-Net CNN model for validating 10x10 floor plan grids."""
        input_shape = (10, 10, 1)  # 10x10 grid with 1 channel (room type encoding)
        inputs = tf.keras.Input(shape=input_shape)

        # Encoder: Downsampling path
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        # Bottleneck
        bottleneck = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)

        # Decoder: Upsampling path
        u1 = tf.keras.layers.UpSampling2D((2, 2))(bottleneck)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
        u2 = tf.keras.layers.UpSampling2D((2, 2))(c3)
        c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)

        # Output: Binary classification (valid/invalid)
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c4)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def preprocess_grid(self, grid: np.ndarray) -> tf.Tensor:
        """Preprocess the 10x10 CGAN grid for CNN input."""
        # Ensure grid is 10x10
        grid = np.array(grid, dtype=np.float32)
        if grid.shape != (10, 10):
            raise ValueError("Grid must be 10x10")
        
        # Reshape to [1, 10, 10, 1] and normalize to [0, 1]
        grid = grid.reshape(1, 10, 10, 1) / 255.0
        return tf.convert_to_tensor(grid)

    def validate_layout(self, grid: np.ndarray, floor_plan: GeneratedFloorPlan) -> bool:
        """Validate the CGAN-generated grid using the CNN."""
        if not self.is_model_loaded or not self.model:
            print("CNN model not loaded, using rule-based validation")
            return self.rule_based_validation(floor_plan)

        try:
            # Preprocess and predict
            input_tensor = self.preprocess_grid(grid)
            prediction = self.model.predict(input_tensor, verbose=0)
            validity_score = prediction[0, 0, 0, 0]  # Extract scalar probability

            # Threshold for validity
            is_valid = validity_score >= 0.7
            if not is_valid:
                print("CNN validation failed, checking with rule-based logic")
                return self.rule_based_validation(floor_plan)
            
            return True
        except Exception as e:
            print(f"CNN validation error: {e}")
            return self.rule_based_validation(floor_plan)

    def rule_based_validation(self, floor_plan: GeneratedFloorPlan) -> bool:
        """Fallback rule-based validation for floor plan correctness."""
        rooms = floor_plan.rooms

        # Check for overlaps
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if self.has_overlap(rooms[i], rooms[j]):
                    print(f"Overlap detected between {rooms[i].type} and {rooms[j].type}")
                    return False

        # Check room sizes
        for room in rooms:
            min_area = ARCHITECTURAL_PATTERNS['room_size_ratios'].get(room.type, {'min_area': 25})['min_area']
            area = room.size['width'] * room.size['height']
            if area < min_area:
                print(f"Room {room.type} area ({area}) below minimum ({min_area})")
                return False

        # Check adjacency rules
        for room in rooms:
            adjacent_rooms = self.find_adjacent_rooms(room, rooms)
            for key, score in ARCHITECTURAL_PATTERNS['adjacency_matrix'].items():
                room_type1, room_type2 = key.split('_')
                if room.type == room_type1 and score > 0.8:
                    has_required_adjacent = any(adj_room.type == room_type2 for adj_room in adjacent_rooms)
                    if not has_required_adjacent:
                        print(f"Missing required adjacency for {room.type} to {room_type2}")
                        return False

        return True

    def has_overlap(self, room1: Room, room2: Room) -> bool:
        """Check if two rooms overlap (adapted from floorPlanGenerator.ts)."""
        buffer = 1.0  # 1-unit buffer for spacing
        r1x1, r1y1 = room1.position['x'], room1.position['y']
        r1x2 = r1x1 + room1.size['width']
        r1y2 = r1y1 + room1.size['height']
        r2x1, r2y1 = room2.position['x'], room2.position['y']
        r2x2 = r2x1 + room2.size['width']
        r2y2 = r2y1 + room2.size['height']

        return (r1x1 < r2x2 + buffer and
                r1x2 > r2x1 - buffer and
                r1y1 < r2y2 + buffer and
                r1y2 > r2y1 - buffer)

    def find_adjacent_rooms(self, room: Room, rooms: List[Room]) -> List[Room]:
        """Find rooms adjacent to the given room."""
        adjacent = []
        for other_room in rooms:
            if other_room != room and self.has_shared_wall(room, other_room):
                adjacent.append(other_room)
        return adjacent

    def has_shared_wall(self, room1: Room, room2: Room) -> bool:
        """Check if two rooms share a wall (adapted from floorPlanGenerator.ts)."""
        tolerance = 1.0
        r1x1, r1y1 = room1.position['x'], room1.position['y']
        r1x2 = r1x1 + room1.size['width']
        r1y2 = r1y1 + room1.size['height']
        r2x1, r2y1 = room2.position['x'], room2.position['y']
        r2x2 = r2x1 + room2.size['width']
        r2y2 = r2y1 + room2.size['height']

        return (
            (abs(r1x2 - r2x1) <= tolerance and r1y1 < r2y2 and r1y2 > r2y1) or
            (abs(r1x1 - r2x2) <= tolerance and r1y1 < r2y2 and r1y2 > r2y1) or
            (abs(r1y2 - r2y1) <= tolerance and r1x1 < r2x2 and r1x2 > r2x1) or
            (abs(r1y1 - r2y2) <= tolerance and r1x1 < r2x2 and r1x2 > r2x1)
        )

# Example usage
def integrate_cnn_with_cgan(grid: np.ndarray, floor_plan: GeneratedFloorPlan) -> bool:
    validator = CNNValidator()
    return validator.validate_layout(grid, floor_plan)

# Example test
if __name__ == "__main__":
    # Sample 10x10 grid (mock CGAN output)
    sample_grid = np.random.rand(10, 10) * 255  # Mock room type encodings
    # Sample floor plan
    sample_rooms = [
        Room('bedroom', (2, 2), (10, 10)),
        Room('bathroom', (12, 2), (5, 5)),
        Room('kitchen', (2, 12), (8, 8)),
    ]
    sample_floor_plan = GeneratedFloorPlan(sample_rooms)

    # Validate
    is_valid = integrate_cnn_with_cgan(sample_grid, sample_floor_plan)
    print(f"Floor plan is {'valid' if is_valid else 'invalid'}")