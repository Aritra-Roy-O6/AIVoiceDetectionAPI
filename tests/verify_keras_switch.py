import sys
import os
import numpy as np

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"

try:
    from predictor import VoicePredictor
    
    print("Initializing VoicePredictor...")
    predictor = VoicePredictor.get_instance()
    
    if predictor.model:
        print("[PASS] Model loaded successfully.")
    else:
        print("[FAIL] Model instance is None.")
        sys.exit(1)

    print("Running Prediction Test on Synthetic Audio...")
    sr = 16000
    duration = 3
    # Synthetic Sine Wave
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    label, conf = predictor.predict(audio, sr)
    print(f"[PASS] Prediction Result: Label={label}, Confidence={conf}")
    
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
