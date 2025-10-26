# 이 스크립트는 Google Cloud Speech-to-Text (v1p1beta1)를 사용하여
# 로컬 WAV 파일("./output.wav")을 동기적으로 STT(음성->텍스트) 처리하는 예제입니다.
# 주요 단계:
# 1) 클라이언트 생성 (환경 변수 또는 기본 자격증명 사용)
# 2) 로컬 오디오 파일을 바이너리로 읽기
# 3) RecognitionConfig로 인식 옵션 설정 (인코딩, 샘플링 레이트, 언어 등)
# 4) client.recognize() 호출하여 동기적 인식 수행
# 5) 결과에서 최상위 대체(transcript)를 출력
#
# 참고:
# - 긴 오디오(약 1분 이상)는 long_running_recognize() 사용을 권장합니다.
# - GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되어 있어야 합니다.

from google.cloud import speech_v1p1beta1 as speech

# Speech-to-Text 클라이언트 생성.
# 클라이언트는 기본 자격증명(GOOGLE_APPLICATION_CREDENTIALS 등)을 사용합니다.
client = speech.SpeechClient()

# 변환할 로컬 오디오 파일 경로 (WAV, LINEAR16을 가정).
speech_file = "./output.wav"

# 파일을 바이너리 모드로 열어 전체 내용을 읽음.
# 매우 큰 파일의 경우 스트리밍이나 long_running_recognize() 사용을 고려하세요.
with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

# RecognitionAudio 객체 생성 (content에 바이트 전달).
audio = speech.RecognitionAudio(content=content)

# 인식 설정:
# - encoding: 오디오 인코딩 형식 (LINEAR16 등)
# - sample_rate_hertz: 샘플링 레이트(오디오 파일과 일치시킬 것)
# - language_code: BCP-47 언어 코드 (예: "en-US")
# - use_enhanced: 향상된 모델 사용 여부
# - model: 특정 모델 지정(향상된 모델 사용 시 필수)
# - audio_channel_count: 오디오 채널 수
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=24000,
    language_code="en-US",
    use_enhanced=True,
    # 향상된 모델을 사용하려면 model을 지정해야 합니다.
    model="phone_call",
    audio_channel_count = 1
)

# 동기 인식 호출. 짧은 오디오에 적합합니다.
response = client.recognize(config=config, audio=audio)

# response.results에는 여러 청크가 들어올 수 있으며,
# 각 결과에는 여러 대체안(alternatives)이 포함됩니다.
# 여기서는 각 결과의 첫 번째(가장 높은 확신) 대체를 출력합니다.
for i, result in enumerate(response.results):
    alternative = result.alternatives[0]
    print(f"Transcript: {alternative.transcript}")

# 필요 시 response.results를 반환하거나 저장할 수 있습니다.
# return response.results