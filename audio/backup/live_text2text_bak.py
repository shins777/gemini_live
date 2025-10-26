import os
from google import genai
import numpy as np
import asyncio
import time

from typing import Any, Dict, List, Optional

# from IPython.display import Audio, Markdown, display
from google.genai.types import (
    AudioTranscriptionConfig,
    Content,
    LiveConnectConfig,
    Part,
    ProactivityConfig,
)

#--------------- Configuration and Initialization ---------------#
def init():
    # Load configuration from environment variables
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-hangsik")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return client

#--------------- Session configuration ---------------#

def configure_session(
    system_instruction: Optional[str] = None,
    enable_transcription: bool = True,
    enable_proactivity: bool = False,
    enable_affective_dialog: bool = False,
) -> LiveConnectConfig:
    """
    Creates a configuration object for the Live Connect session.
    """
    input_transcription = AudioTranscriptionConfig() if enable_transcription else None
    output_transcription = AudioTranscriptionConfig() if enable_transcription else None
    proactivity = (ProactivityConfig(proactive_audio=True) if enable_proactivity else None)

    config = LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        input_audio_transcription=input_transcription,
        output_audio_transcription=output_transcription,
        proactivity=proactivity,
        enable_affective_dialog=enable_affective_dialog,
    )

    return config

#--------------- Send and receieve ---------------#

async def send_and_receive_turn(
    session: genai.live.AsyncSession, text_input: str
) -> Dict[str, Any]:
    """
    Sends a single text turn to the Live Connect session and processes the streaming response.
    """
    print("\n---")
    print(f"**Input:** {text_input}")

    # 1. Send the user's content
    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    input_transcriptions = []
    output_transcriptions = []

    start_time = time.perf_counter()

    # 2. Process the streaming response messages
    async for message in session.receive():

        # Collect input transcription (what the model heard the user say)
        if (message.server_content.input_transcription and message.server_content.input_transcription.text):
            input_transcriptions.append(message.server_content.input_transcription.text)

        # Collect output transcription (the model's spoken response text)
        if (message.server_content.output_transcription and message.server_content.output_transcription.text):
            output_transcriptions.append(message.server_content.output_transcription.text)

    # 3. Display the results
    results = {
        "input_transcription": "".join(input_transcriptions),
        "output_transcription": "".join(output_transcriptions),
    }

    if results["input_transcription"]:
        print(f"**Input transcription >** {results['input_transcription']}")

    if results["output_transcription"]:
        print(f"**Output transcription >** {results['output_transcription']}")

    end_time = time.perf_counter()
    print(f"Elapsed Time : {(end_time - start_time):.6f} seconds")

    return results

#--------------- Main execution ---------------#

async def run_live_session(
    model_id: str,
    config: LiveConnectConfig,
    query: str,
):
    """
    Establishes the Live Connect session and runs a series of conversational turns.
    """
    print("## Starting Live Connect Session...")
    system_instruction = config.system_instruction
    print(f"**System Instruction:** *{system_instruction}*")
    
    client = init()

    try:
        # Use an asynchronous context manager to establish and manage the session lifecycle
        async with client.aio.live.connect(
            model=model_id,
            config=config,
        ) as session:
            print(f"**Status:** Session established with model: `{model_id}`")

            # all_results = []
            # for turn in turns:
                # Send each user input sequentially
            result = await send_and_receive_turn(session, query)
            # all_results.append(result)

            print("\n---")
            print("**Status:** All turns complete. Session closed.")
            return result
    except Exception as e:
        print(f"**Error:** Failed to connect or run session: {e}")
        return []

#--------------- Main execution ---------------#

async def call_live(query: str):
    """
    Runs a Live Connect session with affective dialog enabled.
    """
    
    affective_config = configure_session(
        enable_transcription=True,
        enable_proactivity=True,
        enable_affective_dialog=True,
        system_instruction="You are a assistant to help the customer with their questions about products and services.",
    )

    # affective_dialog_turns = [
    #     "My refrigerator isn't working properly. Can you help me troubleshoot the issue?",
    #     "I tried adjusting the temperature settings, but it still doesn't cool effectively.",
    # ]

    MODEL_ID = os.environ.get("GOOGLE_GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025")
    response = await run_live_session(MODEL_ID, affective_config, query)
    
    return response

