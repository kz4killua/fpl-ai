import pickle
from pathlib import Path

from sklearn.base import BaseEstimator

MODELS_DIR = Path("models")


def save_model(model: BaseEstimator, name: str):
    """Save the model to a file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(name: str) -> BaseEstimator:
    """Load the model from a file."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model {name} not found at {path}.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
