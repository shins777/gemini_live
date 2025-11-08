# 이 파일은 마이크 입력을 실시간으로 캡처하여 Google Cloud Speech-to-Text의
# 스트리밍 인식으로 전사하고, 최종 전사(확정된 발화)를 동일 프로세스 내에서 재사용 가능한
# Gemini Live 세션으로 전송하여 AI 응답을 얻고, 그 응답을 Text-to-Speech로 합성·재생하는
# 전체 워크플로우 예제입니다.
#
# 아래 주석은 각 부분의 목적과 동작 흐름, 비동기/블로킹 I/O 관련 주의사항을 설명합니다.

import re
import sys
import asyncio
import time

# Google Cloud Speech-to-Text (동기/스트리밍 클라이언트)
from google.cloud import speech

# 로컬 모듈: 마이크 입력 제너레이터 및 Live Connect / TTS 유틸리티
# 프로젝트 구조에 따라 상대 경로로 임포트합니다.
from audio.stt_liveapi_tts.microphone import MicrophoneStream
from audio.stt_liveapi_tts import live_text2text 
from audio.stt_liveapi_tts import chirp3_tts

# Text-to-Speech 클라이언트(단발 합성에 사용)
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.api_core.client_options import ClientOptions


async def write_audio_stream(input_text) -> None:
    """
    입력 텍스트(input_text)를 Chirp3 기반 TTS로 합성하고 로컬 파일로 저장한 뒤 재생합니다.

    동작 세부:
      1) TextToSpeech 클라이언트 초기화: LOCATION 기반 endpoint 구성
         - production/리전 환경에 맞게 LOCATION을 변경할 수 있습니다.
      2) chirp3_tts.synthesize_chirp3 호출: 단일 프롬프트를 한 번에 합성
      3) 응답(response.audio_content)을 output.wav로 기록
      4) playsound로 파일 재생(로컬 확인용)

    중요 사항:
      - 파일 쓰기와 playsound 재생은 블로킹 작업입니다. 이 함수를 이벤트 루프상에서 직접 호출하면
        이벤트 루프가 일시 중단될 수 있으므로 asyncio.to_thread로 감싸거나 별도 스레드에서 실행하세요.
      - 실제 애플리케이션에서는 비동기 재생 라이브러리(예: sounddevice, aiofiles 등)를 고려하세요.
    """
    import playsound
    import time

    # TTS 서비스 엔드포인트(리전 기반) 구성: 기본은 global
    LOCATION = "global"
    
    API_ENDPOINT = (
        f"{LOCATION}-texttospeech.googleapis.com"
        if LOCATION != "global"
        else "texttospeech.googleapis.com"
    )

    # TextToSpeech 클라이언트 생성 (환경 변수로 인증 수행)
    client = texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )

    # 합성 시간 측정 시작
    start_time = time.perf_counter()
    # 단일 프롬프트를 Chirp3 음성으로 합성
    response = chirp3_tts.synthesize_chirp3(client, input_text)

    # 반환된 PCM/LINEAR16 바이트를 파일로 저장
    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')

    # 합성 시간 측정 및 출력
    end_time = time.perf_counter()
    print(f"Elapsed Time for synthesize_speech : {(end_time - start_time):.6f} seconds")

    # 로컬에서 재생 (간단한 확인용 — 블로킹)
    playsound.playsound("./output.wav")
    print("Audio playback finished.")


async def listen_print_loop(responses: object) -> str:
    """
    Speech-to-Text 스트리밍 응답(responses)을 순회하면서 중간/최종 전사 결과를 처리합니다.

    동작 원리:
      - responses는 client.streaming_recognize(...)에서 반환된 이터레이터/제너레이터입니다.
      - 각 response는 여러 result를 포함할 수 있지만 스트리밍에서는 보통 첫 번째 result가 현재 발화.
      - result.alternatives[0].transcript는 가장 가능성 높은 전사 문자열.
      - result.is_final이 False인 동안은 중간(interim) 전사이며, True가 되면 최종 전사로 확정됩니다.
      - 최종 전사가 나올 때마다 그 텍스트를 live_text2text.call_live로 전송하여 AI 응답을 얻고,
        해당 응답 텍스트를 TTS로 합성 후 재생합니다.

    반환값:
      마지막으로 확정된(transcript is_final True) 전사 문자열을 반환합니다.
    """
    # 이전에 콘솔에 표시한 문자 수를 추적하여 덮어쓰기(overwrite) 구현에 사용
    num_chars_printed = 0

    # responses는 blocking 제너레이터일 수 있으므로 순회하면서 처리
    for response in responses:
        

        start_time = time.perf_counter()

        # response.results가 비어있으면 이벤트(예: 서버 상태)만 온 것이므로 건너뜀
        if not response.results:
            continue

        # 일반적으로 스트리밍에서 첫 번째 result가 현재 발화의 누적 전사
        result = response.results[0]
        if not result.alternatives:
            continue

        # 최상위 대체안(가장 높은 확률)
        transcript = result.alternatives[0].transcript

        # 이전에 출력한 길이보다 현재가 짧으면 덮어쓰기 시 남는 칸을 공백으로 채움
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            # 중간 결과: 진행 표시를 위해 콘솔 덮어쓰기 가능 (여기서는 실제 출력 주석 처리)
            # sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            # 현재 출력 길이 갱신
            num_chars_printed = len(transcript)
            
        else:
            # 최종 결과 확정 시 처리 흐름
            # 1) 콘솔 출력 (확정 전사)
            print(transcript + overwrite_chars)
            received_final = transcript + overwrite_chars

            end_time = time.perf_counter()
            print(f"Elapsed Time for STT streaming response : {(end_time - start_time):.6f} seconds")

            # 2) Gemini Live 세션에 최종 전사 전달(동일 프로세스 내 재사용 세션)
            #    live_text2text.call_live은 내부에서 세션을 시작하거나 기존 세션을 재사용함
            response = await live_text2text.call_live(query=received_final)

            # AI의 응답(출력 전사)을 로그로 출력
            print(f"AI Agent Response: {response.get('output_transcription')}")

            # 3) AI 응답 텍스트를 TTS로 합성하여 재생
            #    write_audio_stream은 블로킹 재생을 수행하므로 필요 시 쓰레드로 옮겨 실행해야 함
            await write_audio_stream(response.get('output_transcription'))

            # 4) 종료 키워드 감지: "exit" 또는 "quit" 포함 시 루프 종료
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            # 다음 발화를 위해 출력 길이 초기화
            num_chars_printed = 0

    # 루프 종료 시 마지막 확보된 transcript 반환
    return transcript


async def main() -> None:
    """
    전체 파이프라인 엔트리포인트:
      - 마이크로부터 실시간 오디오를 취득
      - Google Speech-to-Text 스트리밍으로 전사(interim 및 final 수신)
      - final 전사를 Gemini Live로 전송하여 응답 획득
      - 응답을 TTS로 합성 및 재생
    주의:
      - 마이크 캡처, STT 스트리밍, TTS 재생이 혼합되어 있어 블로킹/비동기 경계에 유의해야 함.
      - 로컬 재생(playsound 등)은 블로킹이므로 이벤트 루프를 막지 않도록 별도 스레드 사용 고려.
    """
    # 로컬 마이크 캡처 설정: 샘플레이트 및 청크 크기
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms 단위

    # Google Cloud Speech-to-Text 클라이언트 생성
    client = speech.SpeechClient()

    # RecognitionConfig 설정:
    #  - encoding: LINEAR16 (PCM 16-bit)
    #  - sample_rate_hertz: 마이크의 실제 샘플레이트와 일치시킬 것
    #  - language_code: BCP-47 포맷 (예: "en-US")
    #  - use_enhanced: 향상된 모델 사용 여부 (True로 설정하면 더 나은 모델 사용 가능)
    #  - model: "telephony" 등 용도에 맞는 모델 선택
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
        use_enhanced=True,
        model="telephony", # 필요에 따라 "phone_call" 등으로 조정
        audio_channel_count = 1
    )

    # StreamingRecognitionConfig: 중간 결과(interim_results)를 활성화하여
    # 실시간 피드백(부분 전사)을 받을 수 있게 설정
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    # 마이크를 컨텍스트 매니저로 열어 안전하게 시작/종료 처리
    # MicrophoneStream.generator()는 PCM16 바이트 청크를 순차적으로 yield
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()

        # streaming_recognize에 넘길 요청 제너레이터 정의
        # 각 요청은 audio_content 필드에 마이크에서 읽은 바이트 청크를 담음
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        # streaming_recognize 호출: 서버에서 오는 responses 제너레이터를 반환
        responses = client.streaming_recognize(streaming_config, requests)

        # responses를 처리하여 전사 -> 모델 호출 -> TTS 재생의 루프 수행
        await listen_print_loop(responses)

        # (선택) listen_print_loop에서 반환된 최종 전사를 여기서 추가로 처리 가능
        # 예: 저장, 로그 전송 등
        # print("Final Transcript:", transcript)


if __name__ == "__main__":
    # asyncio 이벤트 루프에서 main 실행
    # 주의: write_audio_stream 내의 블로킹 재생이 있을 경우 이벤트 루프가 중단될 수 있음
    asyncio.run(main())