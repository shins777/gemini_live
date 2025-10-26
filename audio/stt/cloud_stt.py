
from google.cloud import speech_v1p1beta1 as speech

client = speech.SpeechClient()

speech_file = "./output.wav"

with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=24000,
    language_code="en-US",
    use_enhanced=True,
    # A model must be specified to use enhanced model.
    model="phone_call",
    audio_channel_count = 1
)

response = client.recognize(config=config, audio=audio)

for i, result in enumerate(response.results):
    alternative = result.alternatives[0]
    #print(f"First alternative of result {i}")
    print(f"Transcript: {alternative.transcript}")

# return response.results