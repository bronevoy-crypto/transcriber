"""Захват системного звука через WASAPI loopback (Windows)."""
import queue
import threading

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


class AudioCapture:

    def __init__(self, sample_rate: int = _SAMPLE_RATE, chunk_ms: int = 500):
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("AudioCapture: запущен", sample_rate=self._sample_rate, chunk_ms=self._chunk_ms)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("AudioCapture: остановлен")

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            logger.error("pyaudiowpatch не установлен. Установите: pip install pyaudiowpatch")
            raise

        pa = pyaudio.PyAudio()

        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            # Ищем loopback версию устройства вывода
            if not default_speakers.get("isLoopbackDevice", False):
                for loopback in pa.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break

            logger.info("AudioCapture: захват с устройства", device=default_speakers["name"])

            device_rate = int(default_speakers["defaultSampleRate"])
            device_channels = default_speakers["maxInputChannels"]
            frames_per_chunk = int(device_rate * self._chunk_ms / 1000)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=device_channels,
                rate=device_rate,
                input=True,
                input_device_index=default_speakers["index"],
                frames_per_buffer=frames_per_chunk,
            )

            try:
                while not self._stop_event.is_set():
                    raw = stream.read(frames_per_chunk, exception_on_overflow=False)
                    audio = np.frombuffer(raw, dtype=np.int16)

                    # Стерео → моно
                    if device_channels > 1:
                        samples = (len(audio) // device_channels) * device_channels
                        audio = audio[:samples].reshape(-1, device_channels).mean(axis=1).astype(np.int16)

                    # Ресемплинг если частота устройства отличается от целевой
                    if device_rate != self._sample_rate:
                        audio = _resample(audio, device_rate, self._sample_rate)

                    if not self._queue.full():
                        self._queue.put(audio)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    import scipy.signal
    resampled = scipy.signal.resample_poly(audio, to_rate, from_rate)
    return resampled.astype(np.int16)
