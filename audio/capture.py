"""Захват системного звука (loopback) и микрофона (Windows)."""
import queue
import threading
import time

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
            logger.error("pyaudiowpatch не установлен: pip install pyaudiowpatch")
            raise

        pa = pyaudio.PyAudio()
        loopback_stream = None
        mic_stream = None

        try:
            # Loopback
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not speakers.get("isLoopbackDevice", False):
                for lb in pa.get_loopback_device_info_generator():
                    if speakers["name"] in lb["name"]:
                        speakers = lb
                        break

            lb_rate = int(speakers["defaultSampleRate"])
            lb_channels = max(1, speakers["maxInputChannels"])
            lb_frames = int(lb_rate * self._chunk_ms / 1000)

            loopback_stream = pa.open(
                format=pyaudio.paInt16,
                channels=lb_channels,
                rate=lb_rate,
                input=True,
                input_device_index=speakers["index"],
                frames_per_buffer=lb_frames,
            )
            logger.info("AudioCapture: loopback", device=speakers["name"])

            # Микрофон
            mic_info = self._find_mic(pa)
            mic_rate, mic_channels, mic_frames = None, None, None
            if mic_info:
                try:
                    mic_rate = int(mic_info["defaultSampleRate"])
                    mic_channels = max(1, mic_info["maxInputChannels"])
                    mic_frames = int(mic_rate * self._chunk_ms / 1000)
                    mic_stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=mic_channels,
                        rate=mic_rate,
                        input=True,
                        input_device_index=mic_info["index"],
                        frames_per_buffer=mic_frames,
                    )
                    logger.info("AudioCapture: микрофон", device=mic_info["name"])
                except Exception as e:
                    logger.warning("AudioCapture: не удалось открыть микрофон", error=str(e))
                    mic_stream = None
            else:
                logger.warning("AudioCapture: микрофон не найден, только loopback")

            while not self._stop_event.is_set():
                # Loopback чанк
                lb_raw = loopback_stream.read(lb_frames, exception_on_overflow=False)
                lb_audio = _to_mono(_normalize(lb_raw, lb_channels), lb_channels)
                if lb_rate != self._sample_rate:
                    lb_audio = _resample(lb_audio, lb_rate, self._sample_rate)

                # Микрофон чанк
                if mic_stream:
                    try:
                        mic_raw = mic_stream.read(mic_frames, exception_on_overflow=False)
                        mic_audio = _to_mono(_normalize(mic_raw, mic_channels), mic_channels)
                        if mic_rate != self._sample_rate:
                            mic_audio = _resample(mic_audio, mic_rate, self._sample_rate)

                        # Микшируем
                        min_len = min(len(lb_audio), len(mic_audio))
                        mixed = lb_audio[:min_len].astype(np.int32) + mic_audio[:min_len].astype(np.int32)
                        audio = np.clip(mixed // 2, -32768, 32767).astype(np.int16)
                    except Exception:
                        audio = lb_audio
                else:
                    audio = lb_audio

                if not self._queue.full():
                    self._queue.put(audio)

        finally:
            if loopback_stream:
                loopback_stream.stop_stream()
                loopback_stream.close()
            if mic_stream:
                mic_stream.stop_stream()
                mic_stream.close()
            pa.terminate()

    def _find_mic(self, pa) -> dict | None:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice", False):
                    return info
            except Exception:
                continue
        return None


def _normalize(raw: bytes, channels: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16)


def _to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels > 1:
        n = (len(audio) // channels) * channels
        return audio[:n].reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    import scipy.signal
    return scipy.signal.resample_poly(audio, to_rate, from_rate).astype(np.int16)
