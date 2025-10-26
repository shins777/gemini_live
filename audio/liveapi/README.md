# Live API Text to Text 

## Install Python dependencies
```
uv add google-genai
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
gemini_live$ uv run -m audio.liveapi.text2text
```
