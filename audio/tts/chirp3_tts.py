from collections.abc import Iterator
import re

from IPython.display import Audio, display
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
import numpy as np
     


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

def text_generator(text: str) -> Iterator[str]:
    """Split text into sentences to simulate streaming"""

    # Use regex with positive lookahead to find sentence boundaries
    # without consuming the space after the punctuation
    sentences: list[str] = re.findall(r"[^.!?]+[.!?](?:\s|$)", text + " ")

    # Yield each complete sentence
    for sentence in sentences:
        yield sentence.strip()

    # Check if there's remaining text not caught by the regex
    # (text without ending punctuation)
    last_char_pos: int = 0
    for sentence in sentences:
        last_char_pos += len(sentence)

    if last_char_pos < len(text.strip()):
        remaining: str = text.strip()[last_char_pos:]
        if remaining:
            yield remaining.strip()


def process_streaming_audio(
    text: str,
    voice: texttospeech.VoiceSelectionParams,
    display_individual_chunks: bool = False,
) -> np.ndarray:
    """Process text into speech using streaming TTS"""

    # Generate sentences from text
    sentences: list[str] = list(text_generator(text))

    # Get streaming audio
    print("Streaming audio processing...")
    audio_iterator: Iterator[bytes] = synthesize_streaming(iter(sentences), voice=voice)

    # Process audio chunks
    final_audio_data: np.ndarray = np.array([], dtype=np.int16)

    for idx, audio_content in enumerate(audio_iterator):
        audio_chunk: np.ndarray = np.frombuffer(audio_content, dtype=np.int16)

        # Concatenate to final audio
        final_audio_data = np.concatenate((final_audio_data, audio_chunk))

        # Optionally display individual chunks
        if display_individual_chunks and len(audio_chunk) > 0:
            print(f"Processed chunk # {idx}")
            display(Audio(audio_chunk, rate=24000))

    print("Streaming audio processing complete!")
    return final_audio_data


def synthesize_streaming(
    text_iterator: Iterator[str],
    voice: texttospeech.VoiceSelectionParams,
) -> Iterator[bytes]:
    """Synthesizes speech from an iterator of text inputs and yields audio content as an iterator.

    This function demonstrates how to use the Google Cloud Text-to-Speech API
    to synthesize speech from a stream of text inputs provided by an iterator.
    It yields the audio content from each response as an iterator of bytes.

    """

    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=texttospeech.StreamingSynthesizeConfig(
            voice=voice,
        )
    )

    def request_generator() -> Iterator[texttospeech.StreamingSynthesizeRequest]:
        yield config_request
        for text in text_iterator:
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

    streaming_responses: Iterator[texttospeech.StreamingSynthesizeResponse] = (
        client.streaming_synthesize(request_generator())
    )

    for response in streaming_responses:
        yield response.audio_content
     

def synthesize_chirp3(client, prompt: str) -> texttospeech.SynthesizeSpeechResponse:
    """Synthesize speech using Chirp 3 voices"""
    
    voice = "Aoede"  # @param ["Aoede", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Zephyr"]
    language_code = "en-US" 

    voice_name = f"{language_code}-Chirp3-HD-{voice}"
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code=language_code,
    )
    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=prompt),
        voice=voice,
        # Select the type of audio file you want returned
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
    )

    return response

#--------------- Run ---------------#

if __name__ == "__main__":
    
    import playsound
    import time

    start_time = time.perf_counter()
    client = init()

    prompt = "I apologize for any inconvenience caused by our product. How can I help you?" 

    response = synthesize_chirp3(client, prompt)

    # Save the response to an audio file
    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')

    end_time = time.perf_counter()
    print(f"Elapsed Time for synthesize_speech : {(end_time - start_time):.6f} seconds")

    # Play the audio file
    playsound.playsound("./output.wav")
    print("Audio playback finished.")

    