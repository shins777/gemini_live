# Audio examples for gemini-live


## Install Python dependencies
```
uv add google-genai google-cloud-speech google-cloud-texttospeech playsound==1.2.2 pyobjc
```
### references
google-genai : Gemini Live API
google-cloud-texttospeech: Chirp3
google-cloud-speech : STT
playsound==1.2.2,  pyobjc : PlaySound
pyaudio : Audio


Check the pyproject.toml

```
    "google-cloud-speech>=2.34.0",
    "google-cloud-texttospeech>=2.33.0",
    "google-genai>=1.31.0",
    "numpy>=2.3.4",
    "playsound==1.2.2",
    "pyaudio>=0.2.14",
    "pyobjc>=12.0",
```

## Usage

Authenticate your environment. 
```
gcloud config set project {PROJECT_ID}
gcloud auth application-default set-quota-project {PROJECT_ID}
gcloud auth application-default login
```

Run the examples with the following command. 
```
gemini_live$ uv run -m audio.stt_liveapi_tts.stt_livet2t_tts

```


## Flow

아래는 stt_liveapi_tts 폴더 내 파이썬 파일들 간의 호출 관계와 자세한 실행 흐름(시퀀스 다이어그램 포함)입니다. 핵심 포인트: stt_livet2t_tts.py가 엔트리 포인트이며, 마이크→STT→Gemini Live→TTS의 파이프라인을 조율합니다.

### 파일별 역할요약

```
microphone.py
MicrophoneStream 클래스: PyAudio 콜백으로 마이크 PCM 바이트를 큐에 저장하고 generator()로 바이트 청크를 제공.
마이크 캡처만 담당(입출력/네트워크 호출 없음).
chirp3_tts.py
Chirp3 기반 TTS 유틸리티: synthesize_chirp3(), streaming 합성 시뮬레이션 함수들(text_generator, synthesize_streaming, process_streaming_audio).
실제 TTS API 호출 및 PCM 바이트 반환 담당.
live_text2text.py
Gemini Live 관련: genai.Client 초기화(init), LiveConnectConfig 생성(configure_session), LiveSessionManager(세션 재사용), send_and_receive_turn(), call_live() 등.
Gemini Live 세션을 열고 텍스트 턴을 보내며 스트리밍 응답을 수집/반환.
stt_livet2t_tts.py
전체 워크플로우 엔트리 포인트:
MicrophoneStream로 오디오를 캡처해 Google Speech-to-Text(Streaming)로 전송.
listen_print_loop()에서 final 전사 수신 시 live_text2text.call_live() 호출.
Gemini 응답을 받아 chirp3_tts.synthesize_chirp3()로 TTS 합성 후 재생(write_audio_stream).
init.py / README.md: 패키지/문서용(호출 흐름에 영향 없음).
```

###  상세 시퀀스 다이어그램 

```


화살표는 호출/데이터 흐름을 나타냄.
User (speaks)
↓
MicrophoneStream (microphone.py)

PyAudio 콜백으로 in_data → 내부 Queue
generator() → PCM 바이트 청크 제공
↓ 
audio chunk generator
stt_livet2t_tts.main()
requests 생성 (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in generator)
↓ 
(streaming)

Google Speech-to-Text (client.streaming_recognize)
↓ 
streaming responses
stt_livet2t_tts.listen_print_loop(responses)
interim 처리(콘솔)
final 전사 확인 시:
↓ 

call_live(final_transcript)
live_text2text.call_live(query)
↓ 

start_session_if_needed -> LiveSessionManager.start() (세션이 없을 때)
live_text2text.LiveSessionManager.start() → genai.Client(...).aio.live.connect(...).aenter() => session
↓ 

manager.send(query) 또는 send_and_receive_turn(session, text)
session.send_client_content(Content(role="user", parts=[Part(text=query)]))
async for message in session.receive(): collect input/output transcription
↑ 

results dict returned (input_transcription, output_transcription)
↑ 

call_live returns results to listen_print_loop
↓ 

받은 output_transcription
stt_livet2t_tts.write_audio_stream(output_transcription)
TextToSpeech client init (chirp3_tts uses texttospeech client)
chirp3_tts.synthesize_chirp3(client, text) 호출
↓ 

call
Google Cloud Text-to-Speech API
↑ 

audio_content (LINEAR16 PCM bytes)
write to output.wav, playsound.playsound()로 재생 (블로킹)
세션 재사용 포인트

LiveSessionManager는 프로세스 레벨로 세션을 열어 여러 call_live 호출에서 동일 세션 재사용.
manager._lock으로 send 호출 직렬화(동시성 방지).
비동기 / 블로킹 경계 (주의)

Gemini Live 관련: 비동기(async/await)로 구현(live_text2text, session.receive 등).
Google Speech 스트리밍: blocking generator 형태의 responses 사용(stt_livet2t_tts.listen_print_loop는 동기적 iteration 내에서 await 호출 포함).
TTS 재생(playsound) 및 파일 I/O: 블로킹. 이벤트 루프 내에서 직접 호출하면 루프가 중단됨 → asyncio.to_thread나 별도 스레드에서 실행 권장.
MicrophoneStream.generator()는 블로킹(큐 기반)으로 동작하므로 비동기 환경에서는 별도 쓰레드/작업으로 실행 고려.
핵심 연계 포인트 (호출 관계 요약)

stt_livet2t_tts.py → (uses) microphone.MicrophoneStream
stt_livet2t_tts.py → (calls) Google Speech streaming API
stt_livet2t_tts.py → (on final) live_text2text.call_live()
live_text2text.py → (manages) genai.Client, LiveSessionManager, session.send/receive
stt_livet2t_tts.py → (to synthesize) chirp3_tts.synthesize_chirp3()
chirp3_tts.py → (calls) Google Text-to-Speech API
MicrophoneStream은 누구에게도 직접 호출되지 않음(엔트리 포인트가 사용)
추천 고려사항 (간단)

TTS 재생을 asyncio.to_thread로 감싸서 이벤트 루프 블로킹 방지.
LiveSessionManager.stop()으로 프로세스 종료 시 세션 정리.
동시성 필요 시 LiveSessionManager.send()의 lock이 호출 직렬화를 수행하므로 다중 요청 시 대기 발생함(요구사항에 따라 병렬 전략 필요).

```
------------------------------------------------------------------------------------------------

```
sequenceDiagram
    participant User
    participant MicrophoneStream as Mic (microphone.py)
    participant STT_Client as Google STT
    participant App as stt_livet2t_tts.py
    participant LiveModule as live_text2text.py
    participant LiveMgr as LiveSessionManager
    participant Gemini as genai.live session
    participant TTS_Module as chirp3_tts.py
    participant TTS_Service as Google TTS
    participant Player as playsound / output.wav


    User ->> Mic: speak (PCM)
    Mic ->> App: yield audio chunks (generator)
    App ->> STT_Client: streaming_recognize(requests from generator)
    STT_Client ->> App: streaming responses (interim / final)

    alt interim
        App ->> App: display interim transcript
    else final
        App ->> LiveModule: call_live(final_transcript)
        LiveModule ->> LiveMgr: start_session_if_needed(model, config)
        alt session not started
            LiveMgr ->> Gemini: client.aio.live.connect().__aenter__() (open session)
        end
        LiveModule ->> LiveMgr: send(query)
        LiveMgr ->> Gemini: send_client_content(Content(user text))
        Gemini ->> LiveMgr: stream messages (input/output transcription)
        LiveMgr -->> LiveModule: return {input_transcription, output_transcription}
        LiveModule -->> App: return result

        App ->> TTS_Module: synthesize_chirp3(output_transcription)
        TTS_Module ->> TTS_Service: synthesize_speech(request)
        TTS_Service -->> TTS_Module: audio_content (LINEAR16 PCM)
        TTS_Module ->> Player: write output.wav
        Player ->> User: play audio
    end

    Note over LiveMgr,Gemini: LiveSessionManager keeps session open\nand serializes sends with a lock\nso multiple call_live reuse same session

```