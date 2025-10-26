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
    RealtimeInputConfig,
    TurnCoverage,
    ActivityHandling,
    AutomaticActivityDetection,
    StartSensitivity,
    EndSensitivity,
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
    
    realtime_input_config=RealtimeInputConfig(
        
        # https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling
        activity_handling = ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        
        # https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage
        turn_coverage= TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,

        # https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection
        automatic_activity_detection=AutomaticActivityDetection(
            disabled=False,  # default is False
            start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW, # Either START_SENSITIVITY_LOW or START_SENSITIVITY_HIGH
            end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW, # Either END_SENSITIVITY_LOW or END_SENSITIVITY_HIGH
            prefix_padding_ms=20,
            silence_duration_ms=1500,
        )
    )

    # https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig
    config = LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        input_audio_transcription=input_transcription,
        output_audio_transcription=output_transcription,
        proactivity=proactivity,
        enable_affective_dialog=enable_affective_dialog,
        realtime_input_config=realtime_input_config,

    )

    return config

# --------------- Live session manager (new) ---------------#

class LiveSessionManager:
    """
    Manages a single long-lived Async Live Connect session.
    - start(model_id, config): opens the session
    - send(query): sends a user turn and waits for the response (reuses same session)
    - stop(): closes the session
    """

    def __init__(self):
        self.client = None
        self._cm = None  # context manager returned by client.aio.live.connect(...)
        self.session = None  # active async session
        self._lock = asyncio.Lock()

    async def start(self, model_id: str, config: LiveConnectConfig):
        if self.session:
            return  # already started
        self.client = init()
        self._cm = self.client.aio.live.connect(model=model_id, config=config)
        # Enter the async context to get the session object
        self.session = await self._cm.__aenter__()
        return self.session

    async def send(self, query: str) -> Dict[str, Any]:
        """
        Sends a single query using the active session and returns transcriptions.
        Serializes access with a lock so multiple callers won't interleave turns.
        """
        if not self.session:
            raise RuntimeError("Session not started. Call start(...) first.")

        async with self._lock:
            await self.session.send_client_content(
                turns=Content(role="user", parts=[Part(text=query)])
            )

            input_transcriptions = []
            output_transcriptions = []

            start_time = time.perf_counter()

            # Collect messages for this turn. This mirrors the original logic:
            async for message in self.session.receive():
                if (message.server_content.input_transcription and message.server_content.input_transcription.text):
                    input_transcriptions.append(message.server_content.input_transcription.text)

                if (message.server_content.output_transcription and message.server_content.output_transcription.text):
                    output_transcriptions.append(message.server_content.output_transcription.text)

            results = {
                "input_transcription": "".join(input_transcriptions),
                "output_transcription": "".join(output_transcriptions),
            }

            if results["input_transcription"]:
                print(f"[Live Model] Input transcription : {results['input_transcription']}")

            if results["output_transcription"]:
                print(f"[Live Model] Output transcription : {results['output_transcription']}")

            end_time = time.perf_counter()
            print(f"[Live Model] Elapsed Time : {(end_time - start_time):.6f} seconds")

            return results

    async def stop(self):
        if self.session and self._cm:
            await self._cm.__aexit__(None, None, None)
            self.session = None
            self._cm = None
            self.client = None

# Module-level manager instance you can reuse across calls
_live_manager: Optional[LiveSessionManager] = None

def _get_manager() -> LiveSessionManager:
    global _live_manager
    if _live_manager is None:
        _live_manager = LiveSessionManager()
    return _live_manager

# --------------- Send and receive  ---------------#
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

# --------------- Main execution (modified to reuse manager) ---------------#

async def run_live_session(
    model_id: str,
    config: LiveConnectConfig,
    query: str,
):
    """
    (Kept for compatibility) Starts a session, sends a single query, then closes.
    Prefer using the LiveSessionManager methods if you want a persistent session.
    """
    print("## Starting Live Connect Session (one-shot)...")
    client = init()
    try:
        async with client.aio.live.connect(
            model=model_id,
            config=config,
        ) as session:
            result = await send_and_receive_turn(session, query)
            return result
    except Exception as e:
        print(f"**Error:** Failed to connect or run session: {e}")
        return []

# Convenience functions to start/send/stop using the persistent manager

async def start_session_if_needed(model_id: str, config: LiveConnectConfig):
    manager = _get_manager()
    if manager.session is None:
        print("## Starting persistent Live Connect Session...")
        await manager.start(model_id, config)
        print("**Status:** Persistent session started.")

async def stop_session():
    manager = _get_manager()
    if manager.session:
        print("## Stopping persistent Live Connect Session...")
        await manager.stop()
        print("**Status:** Session stopped.")

async def call_live(query: str):
    """
    Starts the persistent session if not started, sends the query using that session,
    and returns the response. The same session will be reused on subsequent calls.
    """
    affective_config = configure_session(
        enable_transcription=True,
        enable_proactivity=True,
        enable_affective_dialog=True,
        system_instruction="You are a assistant to help the customer with their questions about products and services.",
    )

    MODEL_ID = os.environ.get("GOOGLE_GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025")

    # Ensure persistent session exists
    await start_session_if_needed(MODEL_ID, affective_config)

    manager = _get_manager()
    response = await manager.send(query)
    return response

