import os
from IPython.display import Audio, display
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google import genai

#--------------- Configuration and Initialization ---------------#
def init():
    # Load configuration from environment variables
    LOCATION = "global"
    PROJECT_ID = "ai-hangsik"
    
    API_ENDPOINT = (
        f"{LOCATION}-texttospeech.googleapis.com"
        if LOCATION != "global"
        else "texttospeech.googleapis.com"
    )

    client = texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )

    return client

#--------------- TTS Function ---------------#
def configure_tts():
    
    MODEL = "gemini-2.5-flash-tts"  
    VOICE = "Aoede"  # @param ["Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe", "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi"]

    LANGUAGE_CODE = "en-us" 
    voice = texttospeech.VoiceSelectionParams(
        name=VOICE, 
        language_code=LANGUAGE_CODE, 
        model_name=MODEL
    )

    return voice

#--------------- Main Execution ---------------#

def synthesize_speech(client, voice):
    # fmt: off
    PROMPT = "You should answer the customer's questions as politely as possible."
    # fmt: on
    TEXT = "I apologize for any inconvenience caused by our product. How can I help you?" 

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=TEXT, prompt=PROMPT),
        voice=voice,
        # Select the type of audio file you want returned
        audio_config=texttospeech.AudioConfig(
            # https://cloud.google.com/text-to-speech/docs/reference/rest/v1/AudioEncoding
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
    )
    return response

#--------------- Run ---------------#
if __name__ == "__main__":
    
    import playsound
    import time

    start_time = time.perf_counter()

    tts_client = init()
    voice_params = configure_tts()
    response = synthesize_speech(tts_client, voice_params)
    # Save the response to an audio file
    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')

    end_time = time.perf_counter()
    print(f"Elapsed Time for synthesize_speech : {(end_time - start_time):.6f} seconds")

    # Play the audio file
    playsound.playsound("./output.wav")
    print("Audio playback finished.")

    