# Audio examples for gemini-live


## Install Python dependencies
```
uv add google-genai google-cloud-speech google-cloud-texttospeech playsound==1.2.2 pyobjc
```
### references
google-genai : Gemini Live API
google-cloud-texttospeech: Chirp3
google-cloud-speech : STT
playsound==1.2.2,  pyobjc : PlaySound
pyaudio : Audio


Check the pyproject.toml

```
    "google-cloud-speech>=2.34.0",
    "google-cloud-texttospeech>=2.33.0",
    "google-genai>=1.31.0",
    "numpy>=2.3.4",
    "playsound==1.2.2",
    "pyaudio>=0.2.14",
    "pyobjc>=12.0",
```

## Usage

Authenticate your environment. 
```
gcloud config set project {PROJECT_ID}
gcloud auth application-default set-quota-project {PROJECT_ID}
gcloud auth application-default login
```

Run the examples with the following command. 
```
gemini_live$ uv run -m audio.stt_liveapi_tts.stt_livet2t_tts

```
