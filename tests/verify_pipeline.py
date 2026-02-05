import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import preprocess
from predictor import VoicePredictor

class TestPipeline(unittest.TestCase):
    def test_preprocess_output_shape(self):
        print("\nTesting preprocessing shape...")
        # Create dummy audio (1 second at 16k)
        audio = np.random.rand(16000).astype(np.float32)
        sr = 16000
        
        features = preprocess(audio, sr)
        
        print(f"Features shape: {features.shape}")
        self.assertEqual(features.shape, (128,), "Features should be shape (128,)")
        self.assertEqual(features.dtype, np.float32, "Features should be float32")

    def test_preprocess_resampling_padding(self):
        print("\nTesting resampling and padding...")
        # Create dummy audio (0.5 second at 44.1k)
        sr = 44100
        audio = np.random.rand(int(0.5 * sr)).astype(np.float32)
        
        features = preprocess(audio, sr)
        self.assertEqual(features.shape, (128,))

    @patch('predictor.VoicePredictor.load_model')
    def test_predictor_integration(self, mock_load_model):
        print("\nTesting predictor integration...")
        
        # Reset singleton to ensure fresh init
        with VoicePredictor._lock:
            VoicePredictor._instance = None
        
        # Initialize predictor
        predictor = VoicePredictor(model_path="dummy_path")
        
        # --- Subtest 1: Test AI_GENERATED prediction ---
        print("  Subtest 1: AI_GENERATED (Class 1)")
        mock_model_ai = MagicMock()
        mock_model_ai.predict.return_value = np.array([1])
        mock_model_ai.predict_proba.return_value = np.array([[0.05, 0.95]])
        
        predictor.model = mock_model_ai
        
        audio = np.random.rand(16000).astype(np.float32)
        label, confidence = predictor.predict(audio)
        
        self.assertEqual(label, "AI_GENERATED", "Class 1 should map to AI_GENERATED")
        self.assertEqual(confidence, 0.95, "Confidence should be max probability")
        
        # --- Subtest 2: Test HUMAN prediction ---
        print("  Subtest 2: HUMAN (Class 0)")
        mock_model_human = MagicMock()
        mock_model_human.predict.return_value = np.array([0])
        mock_model_human.predict_proba.return_value = np.array([[0.88, 0.12]])
        
        predictor.model = mock_model_human
        
        label, confidence = predictor.predict(audio)
        
        self.assertEqual(label, "HUMAN", "Class 0 should map to HUMAN")
        self.assertEqual(confidence, 0.88, "Confidence should be max probability")

        print("Integration test passed.")
        
        # --- Subtest 3: Test Long Audio Chunking ---
        print("  Subtest 3: Long Audio Chunking (7 seconds)")
        # Create 7 seconds of audio (approx 2 full chunks + 1 partial)
        # 16000 * 7 = 112000 samples
        audio_long = np.random.rand(16000 * 7).astype(np.float32)
        
        # Mock model to return different probabilities for each chunk
        # Chunk 1: Human (0.9, 0.1)
        # Chunk 2: Human (0.8, 0.2)
        # Chunk 3: AI (0.4, 0.6) -> Partial chunk, maybe noisy?
        
        # Average:
        # Human: (0.9 + 0.8 + 0.4) / 3 = 2.1 / 3 = 0.7
        # AI:    (0.1 + 0.2 + 0.6) / 3 = 0.9 / 3 = 0.3
        # Result should be HUMAN, conf 0.7
        
        mock_model_chunked = MagicMock()
        # side_effect allows different returns for consecutive calls
        mock_model_chunked.predict_proba.side_effect = [
            np.array([[0.9, 0.1]]), # Chunk 1
            np.array([[0.8, 0.2]]), # Chunk 2
            np.array([[0.4, 0.6]])  # Chunk 3
        ]
        mock_model_chunked.predict.return_value = np.array([0]) # Fallback if needed, but we check proba first
        
        predictor.model = mock_model_chunked
        
        label, confidence = predictor.predict(audio_long)
        
        print(f"    Result: {label}, Confidence: {confidence:.4f}")
        
        self.assertEqual(label, "HUMAN", "Aggregated result should be HUMAN")
        self.assertAlmostEqual(confidence, 0.7, places=4, msg="Confidence should be averaged probability")
        
        # Verify predict_proba was called 3 times
        self.assertEqual(mock_model_chunked.predict_proba.call_count, 3)

if __name__ == '__main__':
    unittest.main()
