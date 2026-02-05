# Voice AI Detection API

A production-ready REST API for detecting AI-generated voices vs Human speech, featuring automatic language detection.

## Features

- **Voice Classification**: Distinguishes between "HUMAN" and "AI_GENERATED" voices using a deep learning model (Keras/TensorFlow).
- **Language Detection**: Automatically identifies the spoken language (English, Tamil, Hindi, Malayalam, Telugu) using OpenAI Whisper.
- **Robust Input**: Supports:
    - **Multipart/Form-Data**: Upload MP3/WAV files directly.
    - **Base64 JSON**: Send audio as a Base64 encoded string.
    - **Variable Length**: Handles long audio files via intelligent chunking and probability averaging.
- **Secure**: API Key authentication (`x-api-key`).

## Tech Stack

- **Framework**: FastAPI (Python)
- **ML Engine**: Keras (Torch Backend), Librosa (Audio Processing)
- **Language Model**: OpenAI Whisper (Tiny)
- **Dependencies**: Numpy, Scikit-learn, SoundFile, FFMPEG

## Setup & Installation

### Prerequisites
- Python 3.10+
- FFMPEG installed and added to system PATH.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aritra-Roy-O6/AIVoiceDetectionAPI.git
   cd AIVoiceDetectionAPI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   API_KEY=your_secure_api_key_here
   KERAS_BACKEND=torch
   ```

## Running the API

Start the server using Uvicorn:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

## API Usage

### Endpoint: `POST /detect`
**Headers:** `x-api-key: <YOUR_KEY>`

#### 1. File Upload (Multipart)
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "x-api-key: your_key" \
     -F "file=@/path/to/audio.mp3"
```

#### 2. Base64 Input (JSON)
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "x-api-key: your_key" \
     -H "Content-Type: application/json" \
     -d '{
           "audio_base64": "<BASE64_STRING>"
         }'
```

### Response Format
```json
{
  "status": "success",
  "detected_language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.98,
  "messages": [
    {
      "stage": "language_detection",
      "message": "Detected language English (en) with confidence 0.99"
    },
    {
      "stage": "classification", 
      "message": "Classified as HUMAN with confidence 0.98"
    }
  ]
}
```

## validation

To run the submission-day validation suite:

```bash
python tests/submission_validation.py
```
