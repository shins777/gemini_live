# 이 파일은 Google Cloud Text-to-Speech(오디오 합성) 기능을 활용하여
# - 단일 프롬프트로 Chirp3 음성 합성 (synthesize_chirp3)
# - 텍스트를 문장 단위로 분할하여 스트리밍 합성 시뮬레이션 (text_generator, synthesize_streaming, process_streaming_audio)
# 등을 보여주는 예제입니다.
# 아래 주석은 코드 로직과 각 함수의 역할, 입력/출력, 동작 흐름을 자세히 설명합니다.

from collections.abc import Iterator
import re

from IPython.display import Audio, display
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
import numpy as np

#--------------- Configuration and Initialization ---------------#

def init():
    """
    클라이언트 초기화 함수.
    - 환경 또는 코드에서 지정한 위치(LOCATION)를 기반으로 Text-to-Speech 엔드포인트를 구성.
    - global인 경우 기본 엔드포인트(texttospeech.googleapis.com)를 사용.
    - ClientOptions에 api_endpoint를 넘겨 TextToSpeechClient를 생성하여 반환.
    반환값:
        texttospeech.TextToSpeechClient 인스턴스
    사용처:
        스크립트 시작 시 한 번 호출하여 client를 얻어 음성 합성 요청에 사용.
    """
    # 기본 지역 설정: "global" 또는 특정 리전(예: "us-central1") 가능
    LOCATION = "global"
    
    # LOCATION이 global이 아닌 경우 리전 기반 엔드포인트를 구성
    API_ENDPOINT = (
        f"{LOCATION}-texttospeech.googleapis.com"
        if LOCATION != "global"
        else "texttospeech.googleapis.com"
    )

    # TextToSpeechClient 생성: 내부적으로 인증(환경변수 등)을 사용
    client = texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )

    return client


def text_generator(text: str) -> Iterator[str]:
    """
    긴 텍스트를 문장 단위로 분할해서 순차적으로 반환하는 제너레이터.
    목적:
      - 스트리밍 TTS를 시뮬레이션하거나, 텍스트를 작은 청크로 나누어 점진적으로 합성할 때 사용.
    동작:
      1) 정규식을 사용해 마침표(.), 물음표(?), 느낌표(!)로 끝나는 문장 단위를 캡처.
         - r"[^.!?]+[.!?](?:\s|$)" : 문장 끝의 구두점까지 포함하여 추출.
      2) 캡처된 각 문장을 순회하며 strip() 처리 후 yield.
      3) 정규식으로 잡히지 않은 마지막 미완성 문장이 있으면 그것도 yield.
    반환값:
      각 문장(혹은 남은 텍스트) 문자열을 순서대로 반환하는 Iterator[str].
    """
    # 정규식으로 문장 단위 추출, text + " "로 끝 공백을 보장하여 정규식이 마지막 문장까지 잡도록 함
    sentences: list[str] = re.findall(r"[^.!?]+[.!?](?:\s|$)", text + " ")

    # 추출된 각 문장을 공백 제거하여 순차적으로 반환
    for sentence in sentences:
        yield sentence.strip()

    # 정규식으로 끝 구두점을 가지지 않은 잔여 텍스트(마지막 문장)를 탐지해서 반환
    last_char_pos: int = 0
    for sentence in sentences:
        last_char_pos += len(sentence)

    # 원본 텍스트의 길이와 추출된 길이를 비교하여 미포함 부분이 있으면 반환
    if last_char_pos < len(text.strip()):
        remaining: str = text.strip()[last_char_pos:]
        if remaining:
            yield remaining.strip()


def process_streaming_audio(
    text: str,
    voice: texttospeech.VoiceSelectionParams,
    display_individual_chunks: bool = False,
) -> np.ndarray:
    """
    스트리밍 합성 시뮬레이션을 수행하는 상위 래퍼 함수.
    - text_generator로 텍스트를 문장 단위로 분할
    - synthesize_streaming을 호출하여 각 문장에 대한 오디오 청크(바이트)를 순차적으로 수신
    - 받은 바이트를 numpy int16 배열로 변환하여 하나의 오디오 배열로 연결(concatenate)
    - (옵션) 각 청크를 IPython.display.Audio로 재생하여 디버깅/검증 가능
    인수:
      text: 합성할 전체 텍스트
      voice: VoiceSelectionParams (음성 이름, 언어 등)
      display_individual_chunks: True일 경우 각 청크를 즉시 출력/재생
    반환값:
      합성된 전체 오디오 데이터를 담은 numpy.ndarray(dtype=np.int16)
    주의:
      - 실제 스트리밍 API는 비동기/스트리밍 응답을 사용할 수 있으며, 이 함수는 간단한 시뮬레이션/동기 처리용.
      - 샘플 레이트는 synthesize_streaming과 호출부에서 일관되게 사용해야 함 (여기서는 24000으로 가정).
    """
    # 텍스트를 문장 단위로 분할하여 리스트로 준비
    sentences: list[str] = list(text_generator(text))

    # 합성 스트리밍(제너레이터) 얻기
    print("Streaming audio processing...")
    audio_iterator: Iterator[bytes] = synthesize_streaming(iter(sentences), voice=voice)

    # 최종 합성 오디오(빈 numpy int16 배열로 시작)
    final_audio_data: np.ndarray = np.array([], dtype=np.int16)

    # 수신된 각 오디오 청크(바이트)를 numpy로 변환하여 누적
    for idx, audio_content in enumerate(audio_iterator):
        # audio_content는 bytes로, int16 PCM 로 가정하여 numpy로 변환
        audio_chunk: np.ndarray = np.frombuffer(audio_content, dtype=np.int16)

        # 기존 오디오에 이어붙임
        final_audio_data = np.concatenate((final_audio_data, audio_chunk))

        # 디버깅용: 각 청크를 개별적으로 재생/출력
        if display_individual_chunks and len(audio_chunk) > 0:
            print(f"Processed chunk # {idx}")
            display(Audio(audio_chunk, rate=24000))

    print("Streaming audio processing complete!")
    return final_audio_data


def synthesize_streaming(
    text_iterator: Iterator[str],
    voice: texttospeech.VoiceSelectionParams,
) -> Iterator[bytes]:
    """
    StreamingSynthesize API를 사용하여 텍스트 제너레이터로부터 순차적으로 오디오 바이트를 생성해 반환하는 제너레이터.
    - Google Cloud Text-to-Speech의 streaming_synthesize 호출을 래핑하여 사용자가 제공한 텍스트 청크를 순차적으로 전송.
    동작:
      1) 가장 먼저 StreamingSynthesizeConfig(voice 포함)를 포함하는 요청을 전송하여 스트리밍 세션 설정.
      2) 이후 각 텍스트 청크를 StreamingSynthesizeRequest의 input에 담아 순차적으로 전송.
      3) client.streaming_synthesize(...)에서 반환되는 StreamingSynthesizeResponse들을 순회하면서
         response.audio_content(바이트)를 yield.
    인수:
      text_iterator: 텍스트 청크(문장)를 순회하는 Iterator
      voice: VoiceSelectionParams (voice.name, language_code 등)
    반환값:
      오디오 바이트를 순차적으로 yield하는 Iterator[bytes]
    주의:
      - streaming_synthesize의 반환 타입/동작은 라이브러리 버전에 따라 상이할 수 있으므로 실제 환경에서 문서 확인 필요.
    """
    # 스트리밍 설정을 포함한 최초 요청 구성
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=texttospeech.StreamingSynthesizeConfig(
            voice=voice,
        )
    )

    # 내부 요청 생성기: 첫 번째로 설정 요청을 보내고 뒤이어 텍스트 입력들을 보냄
    def request_generator() -> Iterator[texttospeech.StreamingSynthesizeRequest]:
        # 설정 요청(voice 등)
        yield config_request
        # 각 텍스트 청크를 StreamingSynthesisInput으로 래핑하여 전송
        for text in text_iterator:
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

    # 클라이언트의 streaming_synthesize에 요청 제너레이터 전달하여 응답 이터레이터 획득
    streaming_responses: Iterator[texttospeech.StreamingSynthesizeResponse] = (
        client.streaming_synthesize(request_generator())
    )

    # 각 응답의 audio_content를 바이트로 반환
    for response in streaming_responses:
        yield response.audio_content
     

def synthesize_chirp3(client, prompt: str) -> texttospeech.SynthesizeSpeechResponse:
    """
    Chirp 3 계열 음성(HD)로 단일 텍스트를 합성하는 함수.
    - 'Aoede' 등 미리 정의된 음성 중 하나를 선택해 voice name을 구성.
    - SynthesizeSpeech API를 호출하여 전체 문장을 한 번에 합성하고 응답을 반환.
    인수:
      client: TextToSpeechClient (초기화된 클라이언트)
      prompt: 합성할 텍스트 문자열
    반환값:
      texttospeech.SynthesizeSpeechResponse (audio_content 등 포함)
    비고:
      - audio_config에서 LINEAR16 형식을 지정하므로 반환되는 audio_content는 PCM int16 바이트임.
      - 파일로 저장하거나 재생할 때 이를 그대로 WAV 파일로 쓰면 일반적으로 재생 가능.
    """
    
    # 사용할 음성 이름(사전 설정 또는 선택 가능)
    voice = "Aoede"  # @param ["Aoede", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Zephyr"]
    language_code = "en-US" 

    # 음성 이름을 API의 voice name 포맷에 맞추어 구성
    voice_name = f"{language_code}-Chirp3-HD-{voice}"
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code=language_code,
    )
    # SynthesizeSpeech 요청: 입력, voice, audio_config을 지정
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=prompt),
        voice=voice,
        # 반환 오디오 형식: LINEAR16 (PCM 16-bit)
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
    )

    return response

#--------------- Run ---------------#
# 이 모듈이 스크립트로 직접 실행될 때 아래 블록이 동작합니다.
# - 클라이언트를 초기화하고, 샘플 프롬프트를 합성하여 output.wav에 저장 후 재생합니다.
# - 실제 서비스/프로덕션에서는 이 블록을 호출부에서 직접 사용하거나 테스트용으로만 사용하세요.

if __name__ == "__main__":
    
    import playsound
    import time

    # 실행 시간 측정 시작
    start_time = time.perf_counter()
    # Text-to-Speech 클라이언트 초기화
    client = init()

    # 합성할 프롬프트(예제)
    prompt = "I apologize for any inconvenience caused by our product. How can I help you?" 

    # 단일 합성 호출(Chirp3 음성)
    response = synthesize_chirp3(client, prompt)

    # 반환된 audio_content(바이트)를 output.wav 파일로 기록
    with open("output.wav", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')

    # 실행 시간 측정 종료 및 출력
    end_time = time.perf_counter()
    print(f"Elapsed Time for synthesize_speech : {(end_time - start_time):.6f} seconds")

    # 저장한 파일을 로컬에서 재생 (playsound 사용)
    playsound.playsound("./output.wav")
    print("Audio playback finished.")

