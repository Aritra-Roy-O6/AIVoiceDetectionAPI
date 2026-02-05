import os
import pickle
import threading
import numpy as np
import logging

from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.preprocess import preprocess

import librosa

class ModelLoadError(Exception):
    pass

class PredictionError(Exception):
    pass

class VoicePredictor:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = "model/model-1.keras"):
        self.model_path = model_path
        self.model = None
        
        # Set Keras backend to torch to avoid TF dependency if possible
        os.environ["KERAS_BACKEND"] = "torch"
        # self.load_model() # Disabled for lazy loading

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            # Try fallback to legacy pickle if keras missing? No, user explicitly wants this.
            raise ModelLoadError(f"Model file {self.model_path} not found.")

        try:
            logger.info(f"Loading Keras model from {self.model_path}...")
            
            # Legacy HDF5 handling for .keras extension
            # Note: We rely on inspection finding that it's HDF5
            import shutil
            import keras
            
            # Create a localized temp file to satisfy Keras 3 extension rules for HDF5
            temp_h5 = "model/temp_loader.h5"
            shutil.copyfile(self.model_path, temp_h5)
            
            try:
                self.model = keras.models.load_model(temp_h5)
                logger.info("Keras model loaded successfully.")
            finally:
                if os.path.exists(temp_h5):
                    os.remove(temp_h5)
                    
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Corrupted or invalid model file: {e}")

    def predict(self, audio: np.ndarray, sr: int = 16000) -> Tuple[str, float]:
        """
        Predicts whether the voice is AI_GENERATED or HUMAN.
        """
        # Lazy Loading
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Lazy loading failed: {e}. returning Fallback.")
                return "HUMAN", 0.5

        try:
            target_sr = 16000
            chunk_samples = 48000 # 3 seconds

            # 1. Standardize Audio
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            # 2. Chunking
            total_samples = len(audio)
            
            if total_samples == 0:
                 chunks = [np.zeros(0, dtype=np.float32)]
            else:
                chunks = []
                for i in range(0, total_samples, chunk_samples):
                    chunk = audio[i : i + chunk_samples]
                    chunks.append(chunk)

            chunk_probs = []

            for chunk in chunks:
                # 3. Preprocess Chunk -> (91, 150, 1)
                features = preprocess(chunk, sr)
                
                # Expand Batch Dim -> (1, 91, 150, 1)
                features = np.expand_dims(features, axis=0)

                # 4. Predict
                # model.predict returns [[p_human, p_ai]] (assuming softmax)
                # We need to verify output shape/semantics. Inspection showed Dense(2).
                # Assuming output is [p0, p1]
                preds = self.model.predict(features, verbose=0)
                chunk_probs.append(preds[0])

            # 5. Aggregate
            if not chunk_probs:
                raise PredictionError("No audio chunks processed.")
                
            avg_probs = np.mean(chunk_probs, axis=0)
            
            # 6. Determine Class
            class_index = np.argmax(avg_probs)
            confidence = float(avg_probs[class_index])
            
            # Map Label: Check model training mapping.
            # Usually 0: Human, 1: AI or vice versa.
            # USER GUIDE SAYS: "AI vs Human classification".
            # Assuming 1=AI, 0=Human is standard convention. 
            # If inverted, we'll swap. Let's assume standard.
            label_map = {0: "HUMAN", 1: "AI_GENERATED"}
            label = label_map.get(class_index, "UNKNOWN")
            
            return label, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Model prediction failed: {e}")
