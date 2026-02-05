import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class LanguageDetectionError(Exception):
    pass

class LanguageDetector:
    _instance = None
    
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ta": "Tamil", 
        "hi": "Hindi", 
        "ml": "Malayalam", 
        "te": "Telugu"
    }

    def __init__(self):
        self.model = None
        logger.info("LanguageDetector initialized (Lightweight Mode - Whisper removed)")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Detects language. 
        In lightweight mode (no Whisper), allows defaulting or external heuristics.
        Currently defaults to 'en' (English) as we removed the STT engine.
        """
        # Placeholder for actual lightweight detection if text were available
        detected_lang = "en"
        confidence = 1.0
        
        return detected_lang, confidence

    def is_supported(self, lang_code: str) -> bool:
        return True # Relaxed for lightweight mode
