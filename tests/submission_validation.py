import sys
import os
import threading
import time
import requests
import base64
import json
import uvicorn
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch Singletons before App Import
# We want to mock the MODELS but verify the API Logic
from utils.language_detector import LanguageDetector
from predictor import VoicePredictor

# Language Mock
mock_lang_detector = MagicMock()
mock_lang_detector.SUPPORTED_LANGUAGES = {
    "ta": "Tamil", "en": "English", "hi": "Hindi", "ml": "Malayalam", "te": "Telugu"
}
# Default behavior: English
mock_lang_detector.detect.return_value = ("en", 0.99)
mock_lang_detector.is_supported.side_effect = lambda lang: lang in mock_lang_detector.SUPPORTED_LANGUAGES

# Predictor Mock
mock_predictor = MagicMock()
mock_predictor.predict.return_value = ("HUMAN", 0.88)

# Apply patches
LanguageDetector._instance = mock_lang_detector
VoicePredictor._instance = mock_predictor

from app import app

PORT = 8002 # Different port
BASE_URL = f"http://127.0.0.1:{PORT}"

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")

def check_environment():
    print("Checking Environment...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  [PASS] FFMPEG found: {ffmpeg_path}")
    else:
        print("  [FAIL] FFMPEG NOT found!")
        # return False # Soft fail for dev check if local ffmpeg missing? No, required.
        
    # Check for correct model file (Keras)
    possible_models = ["model/model-1.keras", "model/model.pkl"]
    found = False
    for m in possible_models:
        if os.path.exists(m):
            print(f"  [PASS] Model file found: {m}")
            found = True
            break
    
    if not found:
        print(f"  [FAIL] No model file found (checked {possible_models})")
        return False
    return True

def run_tests():
    print(f"\nStarting API Validation on {BASE_URL}...")
    
    # 1. Availability
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            print("  [PASS] Health Check (200 OK)")
        else:
            print(f"  [FAIL] Health Check: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("  [FAIL] Health Check: Connection Refused")
        return

    # 2. Authentication (Assuming no API key set in this env for fail test, or we mock header)
    # The default app.py warns but doesn't crash if API_KEY not set, assuming dev mode?
    # Actually logic: if API_KEY and header != API_KEY -> 403.
    # If API_KEY not set, then key check passes (default depends). 
    # Let's see: `if API_KEY and api_key_header != API_KEY`
    # We didn't set API_KEY env var in this script.
    # So auth might be open. We'll skip strict auth fail test unless we enforce it.
    
    # 3. Valid JSON Request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "any_key_since_env_is_unset" 
    }
    payload = {"audio_base64": base64.b64encode(b"dummyaudiobytes").decode("utf-8")}
    
    # We patch convert/load to avoid needing real audio
    with patch("app.convert_mp3_to_wav_bytes") as mock_convert, \
         patch("app.load_audio_waveform") as mock_load:
         
        mock_convert.return_value = "wav_bytes"
        mock_load.return_value = np.zeros(16000*3, dtype=np.float32)

        resp = requests.post(f"{BASE_URL}/detect", json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            # 4. Response Format
            required_fields = ["status", "detected_language", "classification", "confidenceScore", "messages"]
            if all(f in data for f in required_fields):
                print("  [PASS] Response Schema Valid")
            else:
                print(f"  [FAIL] Response Schema Missing Fields: {data.keys()}")
            
            # 5. Language Name Check
            if data["detected_language"] == "English":
                print(f"  [PASS] Language detected as full name: {data['detected_language']}")
            else:
                print(f"  [FAIL] Language detected as: {data['detected_language']} (Expected 'English')")

            if data["classification"] == "HUMAN":
                 print("  [PASS] Classification verified")
        else:
            print(f"  [FAIL] Valid JSON Request: {resp.status_code} {resp.text}")

        # 6. Invalid Request (Empty JSON)
        resp = requests.post(f"{BASE_URL}/detect", json={}, headers=headers)
        if resp.status_code in [400, 422]:
             print(f"  [PASS] Invalid JSON (Missing Field) Rejected ({resp.status_code})")
        else:
             print(f"  [FAIL] Invalid JSON Code: {resp.status_code}")

        # 7. Hardening: Missing Content-Type
        headers_no_ct = {"x-api-key": headers["x-api-key"]}
        resp = requests.post(f"{BASE_URL}/detect", data='{"audio_base64": "x"}', headers=headers_no_ct)
        # 415 is also acceptable here if strict media type check
        if resp.status_code in [400, 415, 422]:
            print(f"  [PASS] Missing Content-Type Rejected ({resp.status_code})")
        else:
            print(f"  [FAIL] Missing Content-Type Code: {resp.status_code}")

        # 8. Hardening: Malformed Body (Valid Content-Type)
        resp = requests.post(f"{BASE_URL}/detect", data='{broken_json', headers=headers)
        if resp.status_code in [400, 422]:
            print(f"  [PASS] Malformed JSON Body Rejected ({resp.status_code})")
        else:
            print(f"  [FAIL] Malformed JSON Body Code: {resp.status_code}")

    print("\nValidation Complete.")

if __name__ == "__main__":
    if not check_environment():
        print("Environment Validation Failed. Stopping.")
        sys.exit(1)
        
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(3)
    run_tests()
