from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "abalone.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "web_service" / "local_objects" / "model.pkl"

CATEGORICAL_COLS = ["Sex"]
