"""Запуск: python main.py  |  Остановка: Ctrl+C"""
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import structlog
import yaml
from dotenv import load_dotenv

from audio.capture import AudioCapture
from audio.vad import VADProcessor
from transcribe.factory import create_transcriber
from transcribe.diarizer import Diarizer
from output.writer import JSONWriter

load_dotenv()

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Файл конфига не найден: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(auto_stop_sec: float | None = None) -> None:
    config = load_config()

    audio_cfg = config.get("audio", {})
    vad_cfg = config.get("vad", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})
    diar_cfg = config.get("diarization", {})

    sample_rate: int = audio_cfg.get("sample_rate", 16000)
    chunk_ms: int = audio_cfg.get("chunk_ms", 500)
    silence_duration: float = audio_cfg.get("silence_duration", 1.5)

    logger.info("Инициализация компонентов...")

    vad = VADProcessor(
        threshold=vad_cfg.get("threshold", 0.5),
        min_speech_ms=vad_cfg.get("min_speech_ms", 250),
    )
    transcriber = create_transcriber(model_cfg)

    # Токен HuggingFace: приоритет у переменной окружения (из .env),
    # fallback на значение из config.yaml для обратной совместимости.
    hf_token = os.environ.get("HF_TOKEN") or diar_cfg.get("hf_token", "")

    diarizer: Diarizer | None = None
    if diar_cfg.get("enabled", False) and hf_token.startswith("hf_"):
        try:
            diarizer = Diarizer(
                hf_token=hf_token,
                device=diar_cfg.get("device", "cpu"),
            )
            diarizer.load()
            print("Диаризация: инициализирована")
        except Exception as e:
            logger.warning("Диаризация: ошибка инициализации, продолжаем без неё", error=str(e))
            print(f"Диаризация отключена (ошибка инициализации: {e})")
            diarizer = None
    else:
        logger.info("Диаризация отключена или токен не задан")
        print("Диаризация отключена (enabled: false или токен не задан)")

    capture = AudioCapture(sample_rate=sample_rate, chunk_ms=chunk_ms)
    writer = JSONWriter(output_dir=output_cfg.get("dir", "meetings"))
    writer.start_meeting(config={
        "model": model_cfg.get("type"),
        "sample_rate": sample_rate,
        "chunk_ms": chunk_ms,
        "silence_duration": silence_duration,
        "diarization": diar_cfg.get("enabled", False),
    })

    max_segment_chunks = int(180_000 / chunk_ms)  # принудительная нарезка каждые 180 сек

    # speech_buffer копит активную речь, pending_silence — тишину ПОСЛЕ речи
    # (короткую паузу не отдаём сразу, чтобы не обрезать хвост слова —
    # склеим со следующим чанком если пауза меньше silence_duration).
    # Когда тишина набежала на silence_duration — финализируем сегмент.
    # max_segment_chunks режет монолог в 180с насильно, иначе VAD залипает.
    #
    # _diar_slots — параллельный буфер для пост-диаризации: pyannote
    # работает на всём аудио разом в конце, ей нужен сплошной поток.
    # Ключ — номер слота по chunk_ms от старта, два источника усредняем
    # если пришли в один слот (loopback и mic могут опередить друг друга).
    speech_buffer: list[np.ndarray] = []
    pending_silence: list[np.ndarray] = []
    silence_start: float | None = None
    segment_start: float | None = None
    meeting_start = time.monotonic()
    _diar_slots: dict[int, np.ndarray] = {}

    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        print("\nЗавершение записи...")
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Запись началась. Нажмите Ctrl+C для остановки.\n")
    vad.reset()  # сбрасываем LSTM-состояние Silero VAD перед новой записью
    capture.start()

    if auto_stop_sec:
        import threading
        threading.Timer(auto_stop_sec, lambda: handle_signal(None, None)).start()
        print(f"[test] авто-остановка через {auto_stop_sec}s", flush=True)

    try:
        while not stop:
            chunk = capture.get_chunk(timeout=1.0)
            if chunk is None:
                continue

            now = time.monotonic() - meeting_start

            slot = int((time.monotonic() - meeting_start) * 1000 / chunk_ms)
            chunk_f = chunk.astype(np.float32) / 32768.0
            if slot not in _diar_slots:
                _diar_slots[slot] = chunk_f
            else:
                n = min(len(_diar_slots[slot]), len(chunk_f))
                mixed = (_diar_slots[slot][:n] + chunk_f[:n]) * 0.5
                _diar_slots[slot] = np.clip(mixed, -1.0, 1.0)

            try:
                is_speech = vad.is_speech(chunk)
            except Exception as e:
                logger.warning("VAD ошибка, пропускаем чанк", error=str(e))
                speech_buffer = []
                pending_silence = []
                silence_start = None
                segment_start = None
                continue

            if is_speech:
                if pending_silence:
                    speech_buffer.extend(pending_silence)
                    pending_silence = []
                if not speech_buffer:
                    segment_start = now
                speech_buffer.append(chunk)
                silence_start = None

                if len(speech_buffer) >= max_segment_chunks:
                    audio_segment = np.concatenate(speech_buffer)
                    try:
                        result = transcriber.transcribe(audio_segment)
                        if result.text and segment_start is not None:
                            writer.write_segment(start=segment_start, end=now, text=result.text, speaker="SPEAKER_?", words=result.words)
                            print(f"[{segment_start:.1f}s] {result.text}")
                    except Exception as e:
                        logger.warning("Ошибка транскрибации", error=str(e))
                    segment_start = None
                    speech_buffer = []
                    pending_silence = []
                    silence_start = None
            else:
                if speech_buffer:
                    if silence_start is None:
                        silence_start = now
                    pending_silence.append(chunk)

                    if now - silence_start >= silence_duration:
                        audio_segment = np.concatenate(speech_buffer)
                        try:
                            result = transcriber.transcribe(audio_segment)
                            if result.text and segment_start is not None:
                                writer.write_segment(start=segment_start, end=now, text=result.text, speaker="SPEAKER_?", words=result.words)
                                print(f"[{segment_start:.1f}s] {result.text}")
                        except Exception as e:
                            logger.warning("Ошибка транскрибации", error=str(e))

                        speech_buffer = []
                        pending_silence = []
                        silence_start = None
                        segment_start = None
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # игнорируем Ctrl+C во время пост-обработки
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        if speech_buffer and segment_start is not None:
            _end_now = time.monotonic() - meeting_start
            audio_segment = np.concatenate(speech_buffer)
            try:
                result = transcriber.transcribe(audio_segment)
                if result.text:
                    writer.write_segment(start=segment_start, end=_end_now, text=result.text, speaker="SPEAKER_?", words=result.words)
                    print(f"[{segment_start:.1f}s] {result.text}")
            except Exception as e:
                logger.warning("Ошибка финальной транскрибации", error=str(e))

        print("Остановка захвата...", flush=True)
        try:
            capture.stop()  # PortAudio завершается здесь (включает sleep 1.0s внутри)
        except Exception as e:
            print(f"[capture.stop] ошибка: {e}", flush=True)

        print("Сохранение файла...", flush=True)
        output_path = writer.finish()
        if output_path:
            print(f"\nЗапись сохранена: {output_path}", flush=True)

        if _diar_slots and output_cfg.get("save_debug_wav", False):
            try:
                import scipy.io.wavfile as wavfile
                all_chunks = [_diar_slots[s] for s in sorted(_diar_slots.keys())]
                full_audio_f = np.concatenate(all_chunks)
                full_audio = (np.clip(full_audio_f, -1.0, 1.0) * 32767).astype(np.int16)
                debug_wav = str(output_path).replace(".json", "_debug.wav") if output_path else "debug_loopback.wav"
                wavfile.write(debug_wav, sample_rate, full_audio)
                print(f"Debug WAV сохранён: {debug_wav}", flush=True)
            except Exception as e:
                print(f"[debug] Ошибка сохранения WAV: {e}", flush=True)

        if diarizer and _diar_slots and output_path:
            print("Диаризация полного аудио...", flush=True)
            try:
                all_chunks = [_diar_slots[s] for s in sorted(_diar_slots.keys())]
                full_audio_f = np.concatenate(all_chunks)
                full_audio = (np.clip(full_audio_f, -1.0, 1.0) * 32767).astype(np.int16)

                timeline = diarizer.build_timeline(
                    full_audio, sample_rate,
                    min_speakers=diar_cfg.get("min_speakers"),
                    max_speakers=diar_cfg.get("max_speakers"),
                )
                if timeline:
                    with open(output_path, encoding="utf-8") as f:
                        data = json.load(f)

                    # Если транскрибер даёт пословные тайминги — привязываем
                    # спикера к каждому слову, это точнее на границах реплик.
                    # Иначе — ставим одного доминирующего спикера на весь сегмент.
                    if getattr(transcriber, "supports_word_timestamps", False):
                        data["segments"] = diarizer.assign_speakers_by_word(
                            data.get("segments", []), timeline
                        )
                    else:
                        updated = []
                        for seg in data.get("segments", []):
                            spk = diarizer.speaker_at(timeline, seg["start"], seg["end"])
                            updated.append({**seg, "speaker": spk})
                        data["segments"] = updated

                    for seg in data["segments"]:
                        seg.pop("words", None)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    speakers = len(set(t["speaker"] for t in timeline))
                    print(f"Диаризация завершена. Найдено дикторов: {speakers}, интервалов: {len(timeline)}", flush=True)
                else:
                    print("Диаризация: спикеры не найдены (запись слишком короткая или один голос). Метки не обновлены.", flush=True)
            except Exception as e:
                print(f"[Диаризация] ОШИБКА: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                logger.warning("Ошибка post-recording диаризации", error=str(e))


if __name__ == "__main__":
    _duration = None
    for _arg in sys.argv[1:]:
        if _arg.startswith("--duration="):
            _duration = float(_arg.split("=")[1])
    main(auto_stop_sec=_duration)
