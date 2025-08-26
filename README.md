# gemini_live

Example code demonstrating real-time microphone streaming to Google Gemini Live. The repository contains small example scripts that stream microphone audio and receive either text or audio responses from the model.

## Contents
- audio/ — Example scripts for streaming microphone audio to Gemini Live.
  - audio/stream_audio_text.py — real-time microphone streaming with TEXT responses.
  - audio/stream_audio_audio.py — real-time microphone streaming with AUDIO responses (plays model audio).

## Notes
- Ensure your environment has access to the Google Gemini Live credentials/API keys required by `google-genai`.
- If you encounter audio device issues on macOS, grant microphone permission to the terminal/IDE and check PortAudio installation.

## License
This project follows the Apache License 2.0. All code and content copyright ForusOne
