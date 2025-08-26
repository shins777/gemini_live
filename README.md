# gemini_live

Example code demonstrating real-time microphone streaming to Google Gemini Live. The repository contains small example scripts that stream microphone audio and receive either text or audio responses from the model.

## Useful links
* Gemini Live API Manual : https://cloud.google.com/vertex-ai/generative-ai/docs/live-api
* Live API Document : https://googleapis.github.io/python-genai/genai.html#module-genai.live
* [intro_multimodal_live_api_genai_sdk.ipynb](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/multimodal-live-api/intro_multimodal_live_api_genai_sdk.ipynb)
* [intro_multimodal_live_api_websockets.ipynb](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/multimodal-live-api/intro_multimodal_live_api.ipynb)
* https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig

## Contents
- audio/ — Example scripts for streaming microphone audio to Gemini Live.
  - audio/stream_audio_text.py — real-time microphone streaming with TEXT responses.
  - audio/stream_audio_audio.py — real-time microphone streaming with AUDIO responses (plays model audio).

## Notes
- Ensure your environment has access to the Google Gemini Live credentials/API keys required by `google-genai`.
- If you encounter audio device issues on macOS, grant microphone permission to the terminal/IDE and check PortAudio installation.

## License
This project follows the Apache License 2.0. All code and content copyright ForusOne
