# Hand landmarker config
CAMERA_INDEX = 0
NUM_HANDS = 2
WINDOW_NAME = "Hand Landmarker"
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

# Landmark feature engineering
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
WRIST_INDEX = 0

# Numerical stability
EPSILON = 1e-8

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_ESTIMATORS = 200

# Inference config
MIN_CONFIDENCE = 0.6
