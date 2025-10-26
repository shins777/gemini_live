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

# -------------------------------------------------------------------------
# 파일 목적 (요약)
# - Google Gemini Live Connect 세션을 생성하고 관리하는 예제 코드입니다.
# - 단발성(one-shot) 세션 실행 함수와, 동일한 프로세스 내에서 세션을
#   재사용(reuse)할 수 있는 LiveSessionManager를 제공합니다.
# - 텍스트 기반의 요청을 보내고 서버에서 스트리밍으로 반환되는
#   transcription(입력/출력 텍스트)을 수집하여 반환합니다.
#
# 주요 특징:
# - asyncio 기반 비동기 처리
# - 세션 재사용을 위한 컨텍스트 매니저 진입/종료 (__aenter__/__aexit__)
# - 동시 접근 방지를 위한 asyncio.Lock 사용 (호출 직렬화)
# - 세션 구성 옵션(전사, proactivity, realtime input detection 등) 포함
# -------------------------------------------------------------------------

#--------------- Configuration and Initialization ---------------#

def init():
    """
    genai 클라이언트를 초기화하여 반환합니다.
    - 환경 변수로부터 프로젝트 ID와 리전을 읽어옵니다.
    - genai.Client를 생성할 때 vertexai=True 옵션을 전달하여 Vertex AI 연동 모드로 동작합니다.
    반환값:
      genai.Client 인스턴스
    주의:
      - 환경변수 GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION 사용.
      - 이 함수는 세션을 새로 생성할 때마다 호출되며, LiveSessionManager 내부에서 재사용됩니다.
    """
    # 환경변수에서 REGION, PROJECT 읽기(없으면 기본값 사용)
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-hangsik")

    # genai 클라이언트 생성 (Vertex AI 통합)
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
    Live Connect 세션을 위한 설정 객체(LiveConnectConfig)를 생성하여 반환합니다.
    주요 옵션:
      - system_instruction: 모델에게 전달할 시스템 지시문(컨텍스트)
      - enable_transcription: 입력/출력 오디오 전사 활성화 여부
      - enable_proactivity: 모델의 proactive audio(모델 쪽에서 선행 음성 생성) 활성화 여부
      - enable_affective_dialog: 정서적 대화(affective dialog) 활성화 여부
    반환값:
      LiveConnectConfig 인스턴스
    설명:
      - realtime_input_config는 자동 음성 감지(AutomaticActivityDetection) 등
        실시간 입력 감지에 관련된 파라미터를 포함합니다.
      - TurnCoverage, ActivityHandling 등은 모델이 입력에서 활동(activity)을
        어떻게 해석하고, 사용자가 말하는 동안 모델이 어떻게 반응할지 제어합니다.
    """
    # 입력/출력 전사를 활성화하면 AudioTranscriptionConfig 객체를 전달
    input_transcription = AudioTranscriptionConfig() if enable_transcription else None
    output_transcription = AudioTranscriptionConfig() if enable_transcription else None
    proactivity = (ProactivityConfig(proactive_audio=True) if enable_proactivity else None)
    
    # 실시간 입력 관련 구성: 자동 음성 탐지 및 민감도 설정
    realtime_input_config=RealtimeInputConfig(
        # ActivityHandling: 활동 시작(START_OF_ACTIVITY)에서 인터럽트 허용 설정
        activity_handling = ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        
        # TurnCoverage: 모델의 턴이 활동만 포함하는지 여부 설정
        turn_coverage= TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,

        # AutomaticActivityDetection: 음성 시작/종료 검출 민감도 등 설정
        automatic_activity_detection=AutomaticActivityDetection(
            disabled=False,  # 자동 감지 사용(기본 False)
            # 음성 시작 검출 민감도: LOW/HIGH 선택
            start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
            # 음성 종료 검출 민감도: LOW/HIGH 선택
            end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
            # 말 시작시(prefix) 패딩(밀리초)
            prefix_padding_ms=20,
            # 무음 지속시간을 종료로 간주할 최소 밀리초
            silence_duration_ms=1500,
        )
    )

    # LiveConnectConfig 생성: 응답 modality, 시스템 지시문, 전사 옵션 등 포함
    config = LiveConnectConfig(
        response_modalities=["AUDIO"],  # 서버 응답 modality 예: AUDIO, TEXT 등
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
    장기 실행(Lifecycle 재사용 가능한) Live Connect 세션을 관리하는 클래스입니다.
    목적:
      - 세션을 한 번 열어 여러 번의 질의를 같은 세션에서 처리할 수 있게 함.
      - 세션 진입/종료 처리를 캡슐화하고, 동시 호출이 들어와도 턴이 섞이지 않도록 직렬화(lock) 처리.
    주요 속성:
      - client: genai.Client 인스턴스
      - _cm: client.aio.live.connect(...) 호출로 얻은 context manager 객체
      - session: context manager 진입 후 반환되는 AsyncSession 객체
      - _lock: asyncio.Lock으로 send 호출 직렬화
    사용법:
      - start(model_id, config): 세션을 열고 내부 session 필드에 저장
      - send(query): 이미 열린 session에 사용자 발화(turn)로 텍스트 전송 및 응답 수집
      - stop(): 세션을 닫고 리소스 해제
    주의:
      - 동일 프로세스 내에서 session을 유지해야 하므로, 세션을 유지하는 동안 프로세스가 종료되면 안됩니다.
      - send()는 비동기 함수이며 내부적으로 receive()를 순회하여 스트리밍 메시지를 모두 처리합니다.
    """

    def __init__(self):
        # genai 클라이언트 인스턴스(세션 시작 시 init()으로 생성)
        self.client = None
        # context manager 객체(client.aio.live.connect) 저장
        self._cm = None
        # 실제 활성 세션 객체
        self.session = None
        # 동시 접근 방지용 락(여러 코루틴이 동시에 send()를 호출하면 턴이 섞일 수 있으므로 직렬화)
        self._lock = asyncio.Lock()

    async def start(self, model_id: str, config: LiveConnectConfig):
        """
        세션을 새로 시작합니다. 이미 시작된 경우 아무 동작도 하지 않습니다.
        - client.aio.live.connect(...)에서 반환된 context manager를 __aenter__로 진입하여 session을 획득합니다.
        반환값:
          session 객체 (genai.live.AsyncSession 예상)
        """
        if self.session:
            # 이미 세션이 열려있으면 재사용
            return
        self.client = init()
        # connect()는 비동기 context manager를 반환하므로 직접 보관
        self._cm = self.client.aio.live.connect(model=model_id, config=config)
        # context manager로 진입하여 실제 session을 얻음
        self.session = await self._cm.__aenter__()
        return self.session

    async def send(self, query: str) -> Dict[str, Any]:
        """
        이미 열려있는 session에 텍스트 턴(query)을 전송하고,
        해당 턴에 관련된 스트리밍 메시지를 모두 수신하여 결과를 반환합니다.
        동작:
          - 내부 락으로 여러 호출을 직렬화하여 각 호출이 독립된 턴으로 처리되도록 보장합니다.
          - session.send_client_content(...)로 사용자 역할(role="user")의 텍스트를 전송합니다.
          - session.receive()로부터 오는 스트리밍 메시지를 비동기 반복하면서
            message.server_content.input_transcription / output_transcription을 수집합니다.
          - 수집된 전사들을 문자열로 결합하여 반환합니다.
        반환값:
          dict { "input_transcription": str, "output_transcription": str }
        예외:
          - 세션이 시작되지 않았을 경우 RuntimeError 발생
        """
        if not self.session:
            raise RuntimeError("Session not started. Call start(...) first.")

        # 동일 세션에 대해 동시 send 호출이 들어와도 턴이 섞이지 않게 직렬화
        async with self._lock:
            # 사용자 턴 전송: Content(role="user", parts=[Part(text=query)])
            await self.session.send_client_content(
                turns=Content(role="user", parts=[Part(text=query)])
            )

            input_transcriptions = []
            output_transcriptions = []

            start_time = time.perf_counter()

            # 스트리밍 메시지 수신: 메시지의 타임라인이 끝날 때까지(서버가 전송을 마칠 때까지) 반복
            # session.receive()는 서버에서 전달되는 다양한 이벤트/메시지를 yield 합니다.
            async for message in self.session.receive():
                # 각 메시지에서 input_transcription.text (모델이 인식한 입력) 수집
                if (message.server_content.input_transcription and message.server_content.input_transcription.text):
                    input_transcriptions.append(message.server_content.input_transcription.text)

                # output_transcription.text (모델의 발화/응답 텍스트) 수집
                if (message.server_content.output_transcription and message.server_content.output_transcription.text):
                    output_transcriptions.append(message.server_content.output_transcription.text)

            # 스트리밍 수신이 끝나면 수집된 조각들을 하나의 문자열로 결합
            results = {
                "input_transcription": "".join(input_transcriptions),
                "output_transcription": "".join(output_transcriptions),
            }

            # 로그 출력: 디버깅/모니터링 목적으로 전사 출력
            if results["input_transcription"]:
                print(f"[Live Model] Input transcription : {results['input_transcription']}")

            if results["output_transcription"]:
                print(f"[Live Model] Output transcription : {results['output_transcription']}")

            end_time = time.perf_counter()
            print(f"[Live Model] Elapsed Time : {(end_time - start_time):.6f} seconds")

            return results

    async def stop(self):
        """
        열려있는 session을 안전하게 종료합니다.
        - context manager의 __aexit__를 호출하여 세션 자원을 해제합니다.
        - 내부 상태(self.session, self._cm, self.client)를 None으로 초기화합니다.
        """
        if self.session and self._cm:
            # __aexit__ 호출로 context manager 종료(세션 닫기)
            await self._cm.__aexit__(None, None, None)
            self.session = None
            self._cm = None
            self.client = None

# Module-level manager instance you can reuse across calls
# - 모듈 전역에 하나의 LiveSessionManager 인스턴스를 보관하여 재사용하도록 함
_live_manager: Optional[LiveSessionManager] = None

def _get_manager() -> LiveSessionManager:
    """
    전역 LiveSessionManager 인스턴스를 반환합니다. 없으면 새로 생성합니다.
    용도:
      - 여러 모듈(또는 여러 호출)에서 동일한 세션 매니저를 재사용하도록 보장
    """
    global _live_manager
    if _live_manager is None:
        _live_manager = LiveSessionManager()
    return _live_manager

# --------------- Send and receive  ---------------#
async def send_and_receive_turn(
    session: genai.live.AsyncSession, text_input: str
) -> Dict[str, Any]:
    """
    기존(호환성 유지용) 한 턴 전송/수신 함수.
    - 세션 객체를 인자로 받아, 하나의 텍스트 턴을 전송하고 해당 턴에 대한 스트리밍 응답을 수집하여 반환.
    - LiveSessionManager.send와 로직이 유사하므로 재사용 가능하지만, 외부에서 직접 session을 얻어 사용하고자 할 때 유용합니다.
    반환값:
      dict { "input_transcription": str, "output_transcription": str }
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
    # - session.receive()를 비동기 반복하여 서버가 보내는 모든 이벤트를 처리합니다.
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
    한 번의 세션을 열어 단발성 질의를 처리하고 세션을 닫는 호환성 유지용 함수입니다.
    - 내부에서 새 클라이언트를 초기화하고 client.aio.live.connect 컨텍스트를 사용해 세션을 생성한 뒤
      send_and_receive_turn을 호출하여 결과를 수집하고 컨텍스트를 빠져나오며 세션을 닫습니다.
    사용 시나리오:
      - 간단한 스크립트 또는 세션을 매번 새로 열고 닫아도 되는 경우.
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
        # 예외 발생 시 에러 로그 출력 후 빈 리스트 반환 (호출부에서 에러 처리 가능)
        print(f"**Error:** Failed to connect or run session: {e}")
        return []

# Convenience functions to start/send/stop using the persistent manager

async def start_session_if_needed(model_id: str, config: LiveConnectConfig):
    """
    모듈 전역의 LiveSessionManager를 이용해 세션이 열려있지 않다면 세션을 시작합니다.
    - 처음 호출될 때만 실제 세션을 개설하며, 이후 호출은 아무 동작을 하지 않습니다.
    """
    manager = _get_manager()
    if manager.session is None:
        print("## Starting persistent Live Connect Session...")
        await manager.start(model_id, config)
        print("**Status:** Persistent session started.")

async def stop_session():
    """
    전역 LiveSessionManager가 관리하는 세션이 열려있다면 이를 종료합니다.
    - 명시적으로 세션을 닫고 자원을 해제할 때 사용합니다.
    """
    manager = _get_manager()
    if manager.session:
        print("## Stopping persistent Live Connect Session...")
        await manager.stop()
        print("**Status:** Session stopped.")

async def call_live(query: str):
    """
    편의 함수:
    - 내부적으로 configure_session으로 세션 설정을 만들고,
      start_session_if_needed로 세션을 초기화(또는 재사용),
      _get_manager().send(query)로 동일 세션에 여러 번 요청을 보낼 수 있게 함.
    동작 흐름:
      1) affective_config 구성(전사/프로액티비티/감정대화 옵션 포함)
      2) MODEL_ID 환경 변수 확인(없으면 기본 모델 사용)
      3) 세션이 없으면 시작
      4) manager.send(query)로 요청 전송 및 결과 반환
    반환값:
      manager.send가 반환하는 dict {"input_transcription": ..., "output_transcription": ...}
    사용 예:
      - 동일 프로세스에서 반복적으로 await call_live("some query") 호출하여 동일한 Live 세션 재사용
      - 작업 종료 후 await stop_session() 호출하여 세션 종료
    """
    affective_config = configure_session(
        enable_transcription=True,
        enable_proactivity=True,
        enable_affective_dialog=True,
        system_instruction="You are a assistant to help the customer with their questions about products and services.",
    )

    # 환경변수로 모델 ID를 지정할 수 있음. 기본값은 예시 모델명.
    MODEL_ID = os.environ.get("GOOGLE_GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025")

    # Ensure persistent session exists (start on first call)
    await start_session_if_needed(MODEL_ID, affective_config)

    manager = _get_manager()
    response = await manager.send(query)
    return response

