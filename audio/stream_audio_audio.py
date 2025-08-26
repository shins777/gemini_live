"""
# Installation
# on linux
sudo apt-get install portaudio19-dev

# on mac
brew install portaudio

python3 -m venv env
source env/bin/activate
pip install google-genai
"""

import asyncio
import pyaudio
from google import genai
from google.genai import types

CHUNK=4200
FORMAT=pyaudio.paInt16
CHANNELS=1
RECORD_SECONDS=5
MODEL = 'gemini-live-2.5-flash'
# MODEL = 'gemini-live-2.5-flash-preview-native-audio'
INPUT_RATE=16000
OUTPUT_RATE=24000

client = genai.Client(
    vertexai=True,
    project='ai-hangsik',
    location='us-central1',
)

instruction = """ You are a helpful assistant that helps the user. 
                Always answer in a very concise manner."""

# https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig
config = types.LiveConnectConfig(
    
    system_instruction= instruction,

    enable_affective_dialog=True,

    response_modalities=["AUDIO"],
    input_audio_transcription={}, 
    output_audio_transcription={}, 

    # tools configuration
    # tools=[types.Tool(google_search=types.GoogleSearch())],
    # tools=[get_current_weather],

    # https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfig
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Achernar",
            )
        ),
        language_code="en-US",
    ),    
    
    
    realtime_input_config=types.RealtimeInputConfig(
        
        # https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling
        activity_handling = types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        
        # https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage
        turn_coverage= types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,

        # https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=False,  # default is False
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW, # Either START_SENSITIVITY_LOW or START_SENSITIVITY_HIGH
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW, # Either END_SENSITIVITY_LOW or END_SENSITIVITY_HIGH
            prefix_padding_ms=20,
            silence_duration_ms=1500,
        )
    ),
)

async def main():
    
    print(MODEL)
    p = pyaudio.PyAudio()

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        #exit()
        async def send():
            
            stream = p.open(
                format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK)
            
            try:
                loop = asyncio.get_running_loop()
                while True:
                    # pyaudio.read는 블로킹이므로 executor에서 호출
                    frame = await loop.run_in_executor(None, stream.read, CHUNK, False)
                    await session.send_realtime_input(
                        media=types.Blob(data=frame, mime_type="audio/pcm"),
                    )
                    await asyncio.sleep(1e-12)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print("send() error:", repr(e))
                # 필요시 session.close() / 이벤트 설정 등 추가

        async def receive():
            
            output_stream = p.open(
                format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=CHUNK)
            
            try:
                while True:
                    
                    async for message in session.receive():
                        # 들어오는 메시지 로그 (디버깅)
                        # print("received message:", message)
                        if message.server_content.input_transcription:
                            print("input_transcription:", message.server_content.model_dump(mode="json", exclude_none=True))
                        if message.server_content.output_transcription:
                            print("output_transcription:", message.server_content.model_dump(mode="json", exclude_none=True))

                        if message.server_content.model_turn:
                            for part in message.server_content.model_turn.parts:
                                if part.inline_data.data:
                                    audio_data = part.inline_data.data
                                    output_stream.write(audio_data)
                                    await asyncio.sleep(1e-12)

                    await asyncio.sleep(1)  
                                                      
                # # async for가 정상 종료했을 때
                # print("session.receive() iterator finished normally")

            except asyncio.CancelledError:
                print("receive() cancelled")
                raise
            except Exception as e:
                print("receive() error:", repr(e))

        send_task = asyncio.create_task(send())
        receive_task = asyncio.create_task(receive())

        # 한쪽 예외로 다른 태스크가 자동 취소되는 것을 원치 않으면 return_exceptions=True로 처리
        results = await asyncio.gather(send_task, receive_task, return_exceptions=True)
        
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"task[{i}] exception:", repr(r))

asyncio.run(main())