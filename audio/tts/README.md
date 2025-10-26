# Audio examples for gemini-live


## Install Python dependencies
- Using pip:
  - install google-cloud-texttospeech, playsound==1.2.2,  pyobjc


## Usage

First, authenticate your environment. 
```
gcloud config set project {PROJECT_ID}
gcloud auth application-default set-quota-project {PROJECT_ID}
gcloud auth application-default login
```

Run the examples with the following command. 
```
uv run -m audio.tts.chirp3
uv run -m audio.tts.gemini_tts
```

## License
This project follows the Apache License 2.0. All code and content copyright **ForusOne** (shins777@gmail.com).