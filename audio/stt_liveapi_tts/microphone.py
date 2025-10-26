# 이 파일은 마이크 입력을 PyAudio로 캡처하여 바이트 청크를 생성하는 보조 모듈입니다.
# 주요 목적:
# - 마이크 입력을 별도의 콜백 스레드에서 읽어 안전하게 큐에 저장
# - 컨텍스트 매니저(interface)로 열고 닫을 수 있도록 구현
# - generator()를 통해 연속된 PCM 바이트 청크를 제공 (서버 전송 또는 파일 저장용)
# 사용 예:
#   with MicrophoneStream(rate, chunk) as mic:
#       for audio_chunk in mic.generator():
#           # audio_chunk는 PCM16 바이트 -> STT나 전송에 사용
#           process(audio_chunk)
#
# 주의:
# - PyAudio와 시스템의 마이크 권한이 필요합니다 (macOS의 경우 터미널/IDE에 마이크 권한 허용).
# - RATE와 CHUNK를 오디오 API(예: STT 서비스) 요구사항에 맞게 설정해야 합니다.
# - generator는 블로킹 동작을 하므로 비동기 환경에서는 별도 스레드/태스크로 실행 고려.
import queue
import pyaudio

# 오디오 녹음 파라미터 (기본값)
# RATE: 샘플링 레이트(Hz). 많은 STT/TTS 서비스가 16000 또는 24000을 권장.
# CHUNK: 한 번에 읽어오는 프레임 수. 여기서는 100ms 단위로 구성(RATE / 10).
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream:
    """PyAudio를 사용해 마이크 스트림을 열고, 안전하게 오디오 청크(bytes)를 생성하는 클래스.

    역할 요약:
    - PyAudio 스트림을 비동기 콜백으로 열어 들어오는 오디오를 내부 Queue에 저장.
    - 컨텍스트 매니저 인터페이스(__enter__/__exit__)를 제공하여 사용 후 리소스 자동 정리.
    - generator() 메서드로 소비자가 큐에서 바이트 청크를 연속적으로 읽게 함.

    내부 동작 요약:
    - _fill_buffer 콜백은 PyAudio가 제공하는 in_data를 큐에 put().
    - generator()는 큐에서 블로킹으로 데이터를 꺼내고, 가능하면 추가로 버퍼된 조각들도 합쳐서 반환.
    - 스트림 종료 시 None을 큐에 넣어 generator가 종료되도록 신호를 보냄.
    """

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """초기화.
        인자:
            rate: 샘플링 레이트(Hz)
            chunk: 프레임당 버퍼 크기 (bytes 단위가 아닌 프레임 수)
        내부 상태:
            _buff: 스레드 안전 큐 (PyAudio 콜백 스레드 -> 메인 스레드로 데이터 전달)
            closed: 스트림이 닫혀 있는지 여부 표시 플래그
        """
        self._rate = rate
        self._chunk = chunk

        # 스레드 안전 큐: 콜백으로 들어온 바이트 데이터를 보관
        self._buff = queue.Queue()
        # 스트림 초기 상태는 닫혀있음. __enter__에서 열림
        self.closed = True

    def __enter__(self: object) -> object:
        """컨텍스트 진입 시 PyAudio 인터페이스와 스트림을 초기화하고 콜백 등록.

        동작:
        - PyAudio() 인스턴스 생성
        - open()으로 입력 스트림 시작. stream_callback에 _fill_buffer 등록하여 비동기로 큐에 데이터 저장
        - closed 플래그를 False로 설정
        반환값:
        - self (컨텍스트 내에서 MicrophoneStream 객체 사용 가능)
        """
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # 현재 구현은 모노(1채널)을 가정합니다. 많은 STT 서비스가 모노를 요구합니다.
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # 콜백 기반으로 스트림을 비동기 처리하여 입력 장치 버퍼 오버플로우 방지
            stream_callback=self._fill_buffer,
        )

        # 정상적으로 스트림을 연 후 closed 상태 해제
        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """컨텍스트 종료 시 호출되어 스트림과 PyAudio 인터페이스를 안전하게 종료.

        동작:
        - 스트림 정지 및 닫기
        - closed 플래그 설정
        - 소비자(generator)에 종료 신호로 None을 큐에 넣음 (generator는 이를 받아 종료)
        - PyAudio 인터페이스 terminate()
        """
        # 스트림을 중지하고 닫음
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # generator가 대기하고 있을 수 있으므로 None을 넣어 종료를 알림
        self._buff.put(None)
        # PyAudio 리소스 해제
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """PyAudio 콜백: 마이크로 들어온 raw bytes(in_data)를 큐에 넣음.

        이 콜백은 별도의 스레드(또는 PyAudio 내부 스레드)에서 실행됩니다.
        큐에 데이터를 넣는 것 외에는 다른 작업을 수행하지 않도록 가볍게 유지해야 합니다.

        인자:
            in_data: 마이크에서 읽은 raw PCM bytes
            frame_count: 캡처된 프레임 수
            time_info: 시간 정보(플랫폼 의존)
            status_flags: 상태 플래그

        반환값:
            (None, pyaudio.paContinue)를 반환하여 스트림을 계속 유지
        """
        # 들어온 바이트 데이터를 큐에 저장 (consumer가 읽을 때까지 보관)
        self._buff.put(in_data)
        # pyaudio에 스트림 계속 실행 지시
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """큐에서 오디오 청크(bytes)를 읽어 합쳐서 순차적으로 yield하는 제너레이터.

        동작 상세:
        - closed가 False인 동안 블로킹 get()으로 큐에서 청크를 읽음
        - None을 읽으면 스트림 종료 신호로 판단하고 generator 종료
        - 하나의 yield 단위는 여러 큐 항목을 합친 바이트 블록(b"".join(data))으로 반환
          (이렇게 하면 작은 프레임들을 하나의 네트워크 패킷/요청 단위로 묶기 쉬움)
        사용 예:
            for chunk in mic.generator():
                send_to_server(chunk)
        주의:
        - 이 메서드는 블로킹 동작을 하므로 메인 스레드에서 장시간 사용 시 UI/이벤트 루프 정지에 주의.
        - 비동기 환경에서는 별도의 쓰레드나 asyncio.to_thread로 실행하는 것을 권장.
        """
        while not self.closed:
            # 큐에서 블로킹으로 한 항목을 읽음. None이면 종료 신호.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # 추가로 버퍼에 남아있는 데이터들을 함께 소비하여 한 번에 묶음으로 반환
            # 이렇게 하면 네트워크 전송 단위를 더 크게 만들 수 있음
            while True:
                try:
                    # non-blocking으로 남은 조각들 빠르게 읽음
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        # 중간에 None이 온 경우 종료
                        return
                    data.append(chunk)
                except queue.Empty:
                    # 더 이상 버퍼에 데이터가 없으면 루프 종료
                    break

            # 합쳐진 바이트 블록을 yield
            yield b"".join(data)
