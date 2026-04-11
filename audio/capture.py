"""Захват системного звука и микрофона (Windows)."""
import queue
import threading
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


class AudioCapture:
    # Пишем одновременно loopback (то что играет в колонках — собеседник
    # в звонке) и микрофон (свой голос), мержим в один поток 16kHz моно.
    # Буферы — обычные list'ы, а не Queue: PortAudio-колбэки дергают
    # append, а он под GIL атомарен. Queue.put может залочиться на
    # переполнении, в колбэке это смерть.

    def __init__(self, sample_rate: int = _SAMPLE_RATE, chunk_ms: int = 500):
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("AudioCapture: запущен", sample_rate=self._sample_rate, chunk_ms=self._chunk_ms)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info("AudioCapture: остановлен")

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        import pyaudiowpatch as pyaudio

        lb_buf: list[bytes] = []
        mic_buf: list[bytes] = []
        buf_lock = threading.Lock()  # защищает pop в main loop от конкурентного append

        def lb_callback(in_data, frame_count, time_info, status):
            lb_buf.append(in_data)  # GIL делает append атомарным, lock здесь вызовет deadlock
            return (None, pyaudio.paContinue)

        pa = pyaudio.PyAudio()
        lb_stream = None
        mic_stream = None
        mic_thread = None

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
            lb_ch = max(1, speakers["maxInputChannels"])
            lb_frames = int(lb_rate * self._chunk_ms / 1000)

            lb_stream = pa.open(
                format=pyaudio.paInt16,
                channels=lb_ch,
                rate=lb_rate,
                input=True,
                input_device_index=speakers["index"],
                frames_per_buffer=lb_frames,
                stream_callback=lb_callback,
            )
            logger.info("AudioCapture: loopback", device=speakers["name"])

            # Микрофон
            mic_info = _find_mic(pa)
            mic_rate, mic_ch, mic_frames = None, None, None
            if mic_info:
                try:
                    mic_rate = int(mic_info["defaultSampleRate"])
                    mic_ch = max(1, mic_info["maxInputChannels"])
                    mic_frames = int(mic_rate * self._chunk_ms / 1000)
                    mic_stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=mic_ch,
                        rate=mic_rate,
                        input=True,
                        input_device_index=mic_info["index"],
                        frames_per_buffer=mic_frames,
                    )
                    logger.info("AudioCapture: микрофон", device=mic_info["name"])

                    def _mic_reader():
                        while not self._stop_event.is_set():
                            try:
                                data = mic_stream.read(mic_frames, exception_on_overflow=False)
                                mic_buf.append(data)  # GIL делает append атомарным
                            except Exception:
                                break

                    mic_thread = threading.Thread(target=_mic_reader, daemon=True)
                except Exception as e:
                    logger.warning("AudioCapture: микрофон не открылся", error=str(e))
                    mic_stream = None
            else:
                logger.warning("AudioCapture: микрофон не найден")

            lb_stream.start_stream()
            if mic_stream:
                mic_stream.start_stream()
            if mic_thread:
                mic_thread.start()

            push_interval = self._chunk_ms / 1000.0
            _last_push = time.monotonic()

            while not self._stop_event.is_set():
                elapsed = time.monotonic() - _last_push
                if elapsed < push_interval * 0.9:
                    time.sleep(0.005)
                    continue

                pushed = False
                with buf_lock:
                    lb_chunk = lb_buf.pop(0) if lb_buf else None
                    mic_chunk = mic_buf.pop(0) if (mic_stream and mic_buf) else None

                if lb_chunk is not None or mic_chunk is not None:
                    lb_audio = _process(lb_chunk, lb_ch, lb_rate, self._sample_rate) if lb_chunk is not None else None
                    mic_audio = _process(mic_chunk, mic_ch, mic_rate, self._sample_rate) if mic_chunk is not None else None

                    if lb_audio is not None and mic_audio is not None:
                        n = min(len(lb_audio), len(mic_audio))
                        mixed = np.clip(
                            (lb_audio[:n].astype(np.float32) + mic_audio[:n].astype(np.float32)) * 0.5,
                            -32768, 32767,
                        ).astype(np.int16)
                        audio = mixed
                    else:
                        audio = lb_audio if lb_audio is not None else mic_audio

                    if not self._queue.full():
                        self._queue.put(audio)
                        pushed = True
                    else:
                        logger.warning("AudioCapture: очередь переполнена, чанк потерян")

                _last_push = time.monotonic()

        finally:
            if mic_thread and mic_thread.is_alive():
                mic_thread.join(timeout=2.0)
            if mic_stream:
                try:
                    mic_stream.stop_stream()
                    mic_stream.close()
                except Exception:
                    pass
            if lb_stream:
                try:
                    lb_stream.stop_stream()
                    lb_stream.close()
                except Exception:
                    pass
            try:
                pa.terminate()
            except Exception:
                pass


def _process(raw: bytes, channels: int, from_rate: int, to_rate: int) -> np.ndarray:
    # .copy() обязателен — frombuffer держит указатель на буфер pyaudio,
    # после закрытия стрима получим мусор в последних чанках.
    audio = np.frombuffer(raw, dtype=np.int16).copy()
    if channels > 1:
        n = (len(audio) // channels) * channels
        mixed = audio[:n].reshape(-1, channels).mean(axis=1)
        audio = np.clip(mixed, -32768, 32767).astype(np.int16)
    if from_rate != to_rate:
        import scipy.signal
        resampled = scipy.signal.resample_poly(audio.astype(np.float32), to_rate, from_rate)
        audio = np.clip(resampled, -32768, 32767).astype(np.int16)
    return audio


def _find_mic(pa) -> dict | None:
    import pyaudiowpatch as pyaudio

    wasapi_host_index = None
    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_host_index = wasapi_info["index"]
    except Exception:
        pass

    skip = ("sound mapper", "microsoft", "primary", "mapper", "stereo mix", "стерео микшер", "what u hear", "wave out")

    wasapi_mic = None
    fallback_mic = None

    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if (info["maxInputChannels"] > 0
                    and not info.get("isLoopbackDevice", False)
                    and not any(s in info["name"].lower() for s in skip)):
                if wasapi_host_index is not None and info.get("hostApi") == wasapi_host_index:
                    if wasapi_mic is None:
                        wasapi_mic = info
                elif fallback_mic is None:
                    fallback_mic = info
        except Exception:
            continue

    return wasapi_mic or fallback_mic
