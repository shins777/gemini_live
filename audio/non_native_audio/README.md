# Audio examples for gemini-live

This directory contains example Python scripts that demonstrate streaming microphone audio to Google Gemini Live and receiving either text or audio responses.

## Files
- [audio/stream_audio_text.py](audio/stream_audio_text.py) — Audio(real-time microphone streaming) with TEXT responses. 
- [audio/stream_audio_audio.py](audio/stream_audio_audio.py) — Audio(real-time microphone streaming) with AUDIO responses (plays model audio). .


## Prerequisites
- Python 3.13 (see `.python-version`).
- PortAudio development headers:
  - Debian/Ubuntu: sudo apt-get install portaudio19-dev
  - macOS: brew install portaudio

## Install Python dependencies
- Using pip:
  - pip install google-genai pyaudio
- Or use your preferred tooling that reads [pyproject.toml](pyproject.toml).

## Usage

First, authenticate to GCP 
```
gcloud auth application-default login
```

Run the examples with the following command. 

- Stream microphone -> text responses:
```
uv run -m audio.stream_audio_text
```

- Stream microphone -> audio responses :
```
uv run -m audio.stream_audio_audio
```

## Notes
- Edit the MODEL constant and the client configuration directly in the example scripts if needed.
- The examples use PortAudio via PyAudio and async I/O to send and receive real-time messages.
- If you need to inspect or modify behavior, edit the async `main` functions in the linked scripts above.

## License
This project follows the Apache License 2.0. All code and content copyright **ForusOne** (shins777@gmail.com).