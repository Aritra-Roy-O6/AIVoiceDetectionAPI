import base64
import io
import sys
import os
from pydub import AudioSegment
import librosa
import numpy as np

# Set max audio size (e.g., 5MB)
MAX_AUDIO_SIZE_BYTES = 5 * 1024 * 1024
# Max duration in seconds
MAX_DURATION_SECONDS = 30

class AudioProcessingError(Exception):
    """Custom exception for audio validation/processing errors."""
    pass

def decode_base64_audio(audio_base64: str) -> bytes:
    """
    Decodes a Base64 string into bytes.
    """
    try:
        # Check if header exists (e.g., "data:audio/mp3;base64,...") and strip it
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
            
        audio_bytes = base64.b64decode(audio_base64)
        
        if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
            raise AudioProcessingError(f"Audio file content exceeds maximum size of {MAX_AUDIO_SIZE_BYTES} bytes.")
            
        return audio_bytes
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise e
        raise AudioProcessingError(f"Invalid Base64 input: {str(e)}")

def convert_mp3_to_wav_bytes(mp3_bytes: bytes) -> io.BytesIO:
    """
    Converts MP3 bytes to WAV bytes in-memory using pydub.
    Requires FFMPEG to be installed and available in PATH.
    """
    try:
        # Load MP3 from bytes
        audio_segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        
        # Check duration
        if audio_segment.duration_seconds > MAX_DURATION_SECONDS:
             raise AudioProcessingError(f"Audio duration ({audio_segment.duration_seconds:.2f}s) exceeds limit of {MAX_DURATION_SECONDS}s.")

        # Export to WAV in-memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise e
        # Catch pydub specific errors (often related to ffmpeg)
        raise AudioProcessingError(f"Audio decoding failed (ensure FFMPEG is installed): {str(e)}")

def load_audio_waveform(wav_io: io.BytesIO, target_sr: int = 16000) -> np.ndarray:
    """
    Loads WAV audio into a numpy array using librosa.
    """
    try:
        # Load with librosa
        # sr=None loads at original sampling rate, but we want to resample if needed.
        # However, specifications say 'Resample to 16kHz' in preprocessing, but librosa.load can do it directly.
        # We will use target_sr here for efficiency.
        y, sr = librosa.load(wav_io, sr=target_sr, mono=True)
        return y
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio waveform: {str(e)}")
