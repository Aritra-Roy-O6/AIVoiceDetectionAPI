import pickle
import os
import sys

# Ensure present directory is in path so we can import utils
sys.path.append(os.getcwd())

from utils.model_classes import DummyModel

# Create model directory
os.makedirs("model", exist_ok=True)

if __name__ == "__main__":
    model = DummyModel()
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Dummy model saved to model/model.pkl")
