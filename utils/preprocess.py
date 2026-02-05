import numpy as np
import librosa
import librosa.feature

def preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocesses the raw audio waveform for the model.
    Steps:
    1. Resample to 16000 Hz
    2. Convert to mono
    3. Normalize amplitude
    4. Trim/Pad to exactly 3 seconds
    5. Extract MFCC features (n_mfcc=128)
    6. Mean pool across time
    """
    target_sr = 16000
    target_duration = 3  # seconds
    target_length = target_sr * target_duration
    
    # 1. Resample to 16000 Hz if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
    # 2. Ensure mono (librosa.load default is mono, but good to ensure)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
        
    # 3. Normalize amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    # 4. Trim or Pad to exactly 3 seconds
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    else:
        audio = audio[:target_length]

    # 5. Extract Log Mel Spectrogram
    # Target Shape: (91, 150, 1)
    # n_mels=91
    # Time steps=150 (for 3s duration). 48000 / 150 = 320 hop_length
    
    n_fft = 1024
    hop_length = 320
    n_mels = 91
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=target_sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    # Shape: (n_mels, time_steps) -> (91, 151) likely due to padding?
    # Ensure exact 150 width
    
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Trim or calculate exact slice
    if log_mel_spec.shape[1] > 150:
        log_mel_spec = log_mel_spec[:, :150]
    elif log_mel_spec.shape[1] < 150:
        pad_width = 150 - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')

    # Add channel dimension: (91, 150, 1)
    # Keras expects (Batch, Height, Width, Channels) usually for 2D Conv
    # But check input shape (91, 150, 1) -> (Height, Width, Channel)
    
    features = np.expand_dims(log_mel_spec, axis=-1)
    # Shape: (91, 150, 1)

    return features.astype(np.float32)
