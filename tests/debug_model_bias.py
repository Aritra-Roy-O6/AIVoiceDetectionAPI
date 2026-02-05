import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to test the Real Preprocessor + Real Model (loaded from pickle).
# If model.pkl exists, real predictor will use it.
from predictor import VoicePredictor

def run_tests():
    print("Initializing Predictor...")
    try:
        predictor = VoicePredictor.get_instance()
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        return

    sr = 16000
    duration = 3 # seconds
    
    print("\n--- Test 1: Random Noise (Input A) ---")
    # Gaussian noise
    audio_noise = np.random.randn(sr * duration).astype(np.float32)
    label, conf = predictor.predict(audio_noise, sr)
    print(f"Result: {label}, {conf}")

    print("\n--- Test 2: Sine Wave (Input B) ---")
    # 440Hz Sine wave
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    audio_sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    label, conf = predictor.predict(audio_sine, sr)
    print(f"Result: {label}, {conf}")
    
    print("\n--- Test 3: Silence (Input C) ---")
    audio_silence = np.zeros(sr * duration, dtype=np.float32)
    label, conf = predictor.predict(audio_silence, sr)
    print(f"Result: {label}, {conf}")

if __name__ == "__main__":
    run_tests()
