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

# ---------------------------------------------------------------------
# 파일 목적 (요약)
# - 이 스크립트는 Google Gemini Live 를 사용하여
#   텍스트 기반의 대화를 모델과 실시간(스트리밍)으로 주고받는 예제입니다.
# - 주요 기능:
#   1) LiveConnectConfig 생성 (전사, proactivity 등 옵션 포함)
#   2) 비동기 세션을 열고(Async context manager) 텍스트 턴을 전송
#   3) session.receive()로부터 오는 스트리밍 메시지를 처리하여
#      입력 전사(input_transcription)와 모델 출력 전사(output_transcription)를 수집
#   4) 여러 턴을 순차적으로 같은 세션 내에서 실행할 수 있음
#
# 주의 사항:
# - genai.Client(vertexai=True, ...)를 사용하므로 Google Cloud 인증/환경 변수 설정 필요
#   (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION 또는 적절한 Application Default Credentials).
# - session.receive()는 서버가 끝을 알릴 때까지 스트리밍 메시지를 yield 하므로
#   메시지 수집 루프는 서버 측 이벤트 흐름에 따라 블록될 수 있음.
# - 예제는 비동기 함수(async/await)를 사용하므로 asyncio 이벤트 루프 내에서 실행해야 함.
# ---------------------------------------------------------------------

# 기본 환경/모델 설정(프로젝트/리전/모델 아이디)
# 실제 배포 시에는 환경변수 또는 구성 파일로 관리가능.
LOCATION = "us-central1"
PROJECT_ID = "ai-hangsik"
MODEL_ID = "gemini-live-2.5-flash-preview-native-audio-09-2025"

#--------------- Session configuration ---------------#

def configure_session(
    system_instruction: Optional[str] = None,
    enable_transcription: bool = True,
    enable_proactivity: bool = False,
    enable_affective_dialog: bool = False,
) -> LiveConnectConfig:
    """
    Live Connect 세션에서 사용할 설정(LiveConnectConfig)을 생성하여 반환합니다.

    파라미터 설명:
      - system_instruction: 모델에 전달할 시스템 레벨 지시문(대화 전반의 행동 지침)
      - enable_transcription: 입력/출력 오디오 전사 기능을 활성화할지 여부
      - enable_proactivity: 모델이 proactive audio(선행 음성) 등을 생성하게 할지 여부
      - enable_affective_dialog: 감성 대화 관련 기능을 활성화할지 여부

    동작:
      - enable_transcription이 True이면 AudioTranscriptionConfig 객체를 생성하여
        input/output 전사 옵션으로 설정합니다.
      - proactivity 옵션은 ProactivityConfig(proactive_audio=True)로 매핑됩니다.
      - LiveConnectConfig 객체를 구성하여 반환합니다.
    """
    
    # 오디오 전사 설정 생성 (활성화된 경우)
    input_transcription = AudioTranscriptionConfig() if enable_transcription else None
    output_transcription = AudioTranscriptionConfig() if enable_transcription else None

    # 모델 측의 proactive audio(선제 발화) 활성화 설정
    proactivity = (ProactivityConfig(proactive_audio=True) if enable_proactivity else None)

    # LiveConnectConfig 생성: 여기서는 응답 modality로 AUDIO를 사용하도록 설정
    # 현재 AUDIO 모달리티만 지원됨.

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
    한 턴의 사용자 텍스트를 모델에 전송하고, 해당 턴에 대한 스트리밍 응답을 처리합니다.

    동작 흐름:
      1) session.send_client_content(...)로 Content(role="user", parts=[Part(text=...)]) 전송
      2) session.receive()로부터 오는 스트리밍 메시지를 반복(비동기) 수신
         - 메시지 내 server_content.input_transcription: 모델이 들었다고 판단한 입력 텍스트 조각
         - 메시지 내 server_content.output_transcription: 모델의 발화/응답 텍스트 조각
      3) 수집된 조각들을 결합하여 최종 input/output 전사 문자열 생성
      4) 전사 문자열을 반환

    주의:
      - session.receive()의 반복은 서버가 해당 턴의 스트리밍을 종료할 때까지 계속됩니다.
      - 네트워크 오류/예외 처리가 필요한 경우 호출부에서 try/except로 감싸 처리하세요.
    반환값:
      dict {
        "input_transcription": <string>,   # 모델이 인식한 사용자 발화(조각 결합)
        "output_transcription": <string>,  # 모델의 응답 텍스트(조각 결합)
      }
    """
    print("\n---")
    print(f"**Input:** {text_input}")

    # 1. 사용자 발화 전송: Content(role="user", parts=[Part(text=text_input)])
    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    # 스트리밍으로 수신되는 전사 조각을 임시 리스트에 수집
    input_transcriptions = []
    output_transcriptions = []

    # 시간 측정 시작(성능/디버깅 목적)
    start_time = time.perf_counter()

    # 2. 서버의 스트리밍 응답 메시지 처리
    #    session.receive()는 여러 종류의 이벤트/메시지를 yield 할 수 있음.
    async for message in session.receive():

        # 모델이 인식한 입력 전사 조각 수집 (있을 때만)
        if (message.server_content.input_transcription and message.server_content.input_transcription.text):
            input_transcriptions.append(message.server_content.input_transcription.text)

        # 모델의 출력 전사(음성 출력에 대한 텍스트) 조각 수집
        if (message.server_content.output_transcription and message.server_content.output_transcription.text):
            output_transcriptions.append(message.server_content.output_transcription.text)

    # 3. 수집된 전사 조각들을 결합하여 최종 문자열 구성
    results = {
        "input_transcription": "".join(input_transcriptions),
        "output_transcription": "".join(output_transcriptions),
    }

    # 디버깅/모니터링 용도로 전사 결과 출력
    if results["input_transcription"]:
        print(f"**Input transcription >** {results['input_transcription']}")

    if results["output_transcription"]:
        print(f"**Output transcription >** {results['output_transcription']}")

    # 소요 시간 출력
    end_time = time.perf_counter()
    print(f"Elapsed Time : {(end_time - start_time):.6f} seconds")

    return results

#--------------- Main execution ---------------#

async def run_live_session(
    model_id: str,
    config: LiveConnectConfig,
    turns: List[str],
):
    """
    Live Connect 세션을 생성하고 여러 턴(turns)을 순차적으로 처리합니다.

    동작 요약:
      - genai.Client를 생성하고, 비동기 컨텍스트 매니저(client.aio.live.connect)를 통해
        세션을 열고(session) 닫습니다.
      - 한 세션 내에서 전달된 turns 리스트의 각 요소를 send_and_receive_turn로 sequential하게 전송.
      - 모든 턴 처리 후 세션은 컨텍스트 종료와 함께 닫힙니다.

    파라미터:
      - model_id: 사용할 모델 ID
      - config: LiveConnectConfig (configure_session으로 생성)
      - turns: 순차적으로 전송할 사용자 텍스트 목록

    반환값:
      - 각 턴에 대해 send_and_receive_turn이 반환한 결과 리스트 (각 항목은 dict)
    예외 처리:
      - 연결 실패나 런타임 에러 발생 시 빈 리스트 반환(호출부에서 추가 처리 가능)
    """
    print("## Starting Live Connect Session...")
    system_instruction = config.system_instruction
    print(f"**System Instruction:** *{system_instruction}*")
    
    # genai Client 인스턴스 생성 (Vertex AI 모드)
    # 실제 환경에서는 genai.Client 생성 시 인증/환경변수 확인 필요
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


    try:
        # 비동기 컨텍스트 매니저로 세션 수명 관리
        async with client.aio.live.connect(
            model=model_id,
            config=config,
        ) as session:
            print(f"**Status:** Session established with model: `{model_id}`")

            all_results = []
            # turns 리스트를 순회하며 각 텍스트를 순차적으로 전송하고 응답을 수집
            for turn in turns:
                result = await send_and_receive_turn(session, turn)
                all_results.append(result)

            print("\n---")
            print("**Status:** All turns complete. Session closed.")
            return all_results
    except Exception as e:
        # 연결 또는 세션 실행 중 에러 발생 시 에러 로그 출력
        # 운영 환경에서는 로깅 프레임워크를 사용하고 에러를 재시도/대체 흐름으로 처리하세요.
        print(f"**Error:** Failed to connect or run session: {e}")
        return []

#--------------- Main execution ---------------#

async def main():
    """
    Runs a Live Connect session with affective dialog enabled.
    """
    
    affective_config = configure_session(
        enable_transcription=True,
        enable_proactivity=True,
        enable_affective_dialog=True,
        system_instruction="""You are a assistant to help the customer with their questions about products and services. 

        """,
    )

    affective_dialog_turns = [
        "My refrigerator isn't working properly. Can you help me troubleshoot the issue?",
        "I tried adjusting the temperature settings, but it still doesn't cool effectively.",
    ]

    results = await run_live_session(MODEL_ID, affective_config, affective_dialog_turns)

asyncio.run(main())