import os
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from utils.audio import decode_base64_audio, convert_mp3_to_wav_bytes, load_audio_waveform, AudioProcessingError
from utils.language_detector import LanguageDetector, LanguageDetectionError
from predictor import VoicePredictor, ModelLoadError, PredictionError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_api")

app = FastAPI(title="Voice AI Detection API", version="1.1.0")

# Configuration
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY")
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024 # 10MB limit (Adjusted from 5MB if needed, or keep 5MB)

if not API_KEY:
    logger.warning("No API_KEY environment variable set! API is vulnerable.")

# Models
class Message(BaseModel):
    stage: str
    message: str

class DetectRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio string")

class DetectResponse(BaseModel):
    status: str
    detected_language: Optional[str] = None
    classification: Optional[str] = None
    confidenceScore: Optional[float] = None
    messages: List[Message] = []

# Dependencies
async def get_api_key(api_key_header: str = Header(..., alias=API_KEY_NAME)):
    if API_KEY and api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return api_key_header

@app.on_event("startup")
async def startup_event():
    # Initialize Predictor
    try:
        VoicePredictor.get_instance()
    except ModelLoadError:
        logger.critical("Model file missing or corrupted at startup.")
    
    # Initialize Language Detector
    try:
        LanguageDetector.get_instance()
    except LanguageDetectionError:
        logger.critical("Language detection model failed to load.")

@app.post("/detect", response_model=DetectResponse)
async def detect_voice(request_body: DetectRequest, api_key: str = Depends(get_api_key)):
    """
    Detects whether the provided voice sample is AI-generated or Human.
    Accepts:
    - JSON: {"audio_base64": "..."}
    """
    messages = []
    
    try:
        # 1. Parse Input
        messages.append(Message(stage="input_validation", message="Received JSON request"))
        
        audio_base64 = request_body.audio_base64
        if not audio_base64:
             # Should be caught by Pydantic but double check
             raise HTTPException(status_code=400, detail="Missing 'audio_base64' field")
        
        messages.append(Message(stage="input_validation", message="Received Base64 audio input"))
        audio_bytes = decode_base64_audio(audio_base64)


        if not audio_bytes:
             # Should be caught above, but safety check
             raise HTTPException(status_code=400, detail="No audio data extracted.")
             
        if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
             raise AudioProcessingError(f"Audio size exceeds limit of {MAX_AUDIO_SIZE_BYTES} bytes")

        # 2. Convert to WAV & Load Waveform
        wav_io = convert_mp3_to_wav_bytes(audio_bytes)
        waveform = load_audio_waveform(wav_io)
        messages.append(Message(stage="preprocessing", message=f"Audio loaded successfully. Duration: {len(waveform)/16000:.2f}s"))

        # 3. Language Detection
        lang_detector = LanguageDetector.get_instance()
        detected_lang, lang_conf = lang_detector.detect(waveform)
        
        # Validate Support
        if not lang_detector.is_supported(detected_lang):
             raise HTTPException(status_code=400, detail=f"Unsupported language detected: {detected_lang}")
             
        full_lang_name = lang_detector.SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
        messages.append(Message(stage="language_detection", message=f"Detected language {full_lang_name} ({detected_lang}) with confidence {lang_conf:.2f}"))

        # 4. Classification
        predictor = VoicePredictor.get_instance()
        label, confidence = predictor.predict(waveform)
        
        messages.append(Message(stage="classification", message=f"Classified as {label} with confidence {confidence:.2f}"))
        
        # 5. Clamp confidence
        confidence = max(0.0, min(1.0, float(confidence)))

        return DetectResponse(
            status="success",
            detected_language=full_lang_name,
            classification=label,
            confidenceScore=confidence,
            messages=messages
        )

    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "error": str(e), "messages": [m.dict() for m in messages]}
        )
    except LanguageDetectionError as e:
        logger.error(f"Language detection error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Language detection failed", "messages": [m.dict() for m in messages]}
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Prediction failed", "messages": [m.dict() for m in messages]}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Internal Processing Error", "messages": [m.dict() for m in messages]}
        )

@app.get("/health")
def health_check():
    return {"status": "ok"}
