
import re
import sys
import asyncio

from google.cloud import speech
from audio.live_native.microphone import MicrophoneStream
from audio.live_native import live_text2text 
from audio.live_native import chirp3_tts
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.api_core.client_options import ClientOptions


async def write_audio_stream(input_text) -> None:
    import playsound
    import time

    LOCATION = "global"
    
    API_ENDPOINT = (
        f"{LOCATION}-texttospeech.googleapis.com"
        if LOCATION != "global"
        else "texttospeech.googleapis.com"
    )

    client = texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )

    start_time = time.perf_counter()
    response = chirp3_tts.synthesize_chirp3(client, input_text)

    # Save the response to an audio file
    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')

    end_time = time.perf_counter()
    print(f"Elapsed Time for synthesize_speech : {(end_time - start_time):.6f} seconds")

    # Play the audio file
    playsound.playsound("./output.wav")
    print("Audio playback finished.")


async def listen_print_loop(responses: object) -> str:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Args:
        responses: List of server responses

    Returns:
        The transcribed text.
    """
    num_chars_printed = 0

    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            #sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)
            
        else:
            print(transcript + overwrite_chars)
            received_final = transcript + overwrite_chars
            response = await live_text2text.call_live(query=received_final)
            print(f"AI Agent Response: {response.get('output_transcription')}")
            await write_audio_stream(response.get('output_transcription'))

            # Exit recognition if any of the transcribed phrases could be one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

    return transcript

async def main() -> None:

    """Transcribe speech from audio file."""

    # Audio recording parameters
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms

    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
        use_enhanced=True,
        model="telephony", # or "phone_call"
        audio_channel_count = 1
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        await listen_print_loop(responses)

        #print("Final Transcript:", transcript)


if __name__ == "__main__":
    asyncio.run(main())