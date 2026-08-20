"""Project-wide paths and reproducibility defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_CROP_COLUMN = "chestnut_kg"
DEFAULT_FEATURES = (
    "avg_temp",
    "humidity",
    "precipitation",
    "soil_depth_type",
    "soil_texture_code",
)
