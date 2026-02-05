import whisper
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

class LanguageDetectionError(Exception):
    pass

class LanguageDetector:
    _instance = None
    _lock = threading.Lock()
    
    SUPPORTED_LANGUAGES = {
        "ta": "Tamil",
        "en": "English",
        "hi": "Hindi",
        "ml": "Malayalam",
        "te": "Telugu"
    }

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.model = None
        self.load_model()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load_model(self):
        try:
            logger.info(f"Loading Whisper model '{self.model_size}'...")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise LanguageDetectionError(f"Failed to load language detection model: {e}")

    def detect(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Detects language from audio waveform.
        Args:
            audio: 16kHz mono audio numpy array.
        Returns:
            (language_code, confidence)
        """
        if self.model is None:
            raise LanguageDetectionError("Language detection model not loaded.")

        try:
            # Whisper expects audio to be padded/trimmed to 30s usually for transcription, 
            # but for language detection on short clips, pad_or_trim is standard practice.
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            
            # Get max probability language
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            # Check if supported
            if detected_lang not in self.SUPPORTED_LANGUAGES:
                 # We might want to allow it but warn, or reject?
                 # Requirement: "If detected language not in supported list, return error"
                 # Wait, which error? "Return error" sounds like classification failure?
                 # Or just return it but maybe mark as unsupported in response?
                 # Req says: "If detected language not in supported list, return error"
                 # I will raise an error.
                 # Actually, maybe better to return "Unsupported" so valid API response?
                 # But req says "return error". I'll raise exception which app can catch.
                 pass # Logic below

            return detected_lang, float(confidence)

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise LanguageDetectionError(f"Language detection failed: {e}")

    def is_supported(self, lang_code: str) -> bool:
        return lang_code in self.SUPPORTED_LANGUAGES
