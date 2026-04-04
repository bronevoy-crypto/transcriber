"""Запуск: python main.py  |  Остановка: Ctrl+C"""
import signal
import time
from pathlib import Path

import numpy as np
import structlog
import yaml

from audio.capture import AudioCapture
from audio.vad import VADProcessor
from transcribe.factory import create_transcriber
from transcribe.diarizer import Diarizer
from output.writer import JSONWriter

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Файл конфига не найден: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
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

    diarizer: Diarizer | None = None
    if diar_cfg.get("enabled", False) and diar_cfg.get("hf_token", "").startswith("hf_"):
        diarizer = Diarizer(
            hf_token=diar_cfg["hf_token"],
            device=diar_cfg.get("device", "cpu"),
        )
        diarizer.load()
    else:
        logger.info("Диаризация отключена или токен не задан")
        print("Диаризация отключена (enabled: false или токен не задан)")

    capture = AudioCapture(sample_rate=sample_rate, chunk_ms=chunk_ms)
    writer = JSONWriter(output_dir=output_cfg.get("dir", "meetings"))
    writer.start_meeting()

    # Сколько тихих чанков подряд = конец сегмента
    silence_threshold = max(1, int(silence_duration * 1000 / chunk_ms))
    max_segment_chunks = int(180_000 / chunk_ms)  # принудительная нарезка каждые 180 сек

    speech_buffer: list[np.ndarray] = []
    silence_count = 0
    segment_start: float | None = None
    meeting_start = time.monotonic()
    # Полный аудио-буфер для post-recording диаризации.
    # Ключ — временной слот (индекс 500мс окна), значение — список чанков.
    # Loopback и mic попадают в один слот и микшируются перед диаризацией.
    _diar_slots: dict[int, list[np.ndarray]] = {}

    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        print("\nЗавершение записи...")
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Запись началась. Нажмите Ctrl+C для остановки.\n")
    capture.start()

    try:
        while not stop:
            chunk = capture.get_chunk(timeout=1.0)
            if chunk is None:
                continue

            now = time.monotonic() - meeting_start

            slot = int((time.monotonic() - meeting_start) * 1000 / chunk_ms)
            if slot not in _diar_slots:
                _diar_slots[slot] = []
            _diar_slots[slot].append(chunk)

            try:
                is_speech = vad.is_speech(chunk)
            except Exception as e:
                logger.warning("VAD ошибка, пропускаем чанк", error=str(e))
                speech_buffer = []
                silence_count = 0
                segment_start = None
                continue

            if is_speech:
                if not speech_buffer:
                    segment_start = now
                speech_buffer.append(chunk)
                silence_count = 0

                if len(speech_buffer) >= max_segment_chunks:
                    audio_segment = np.concatenate(speech_buffer)
                    try:
                        result = transcriber.transcribe(audio_segment)
                        if result.text and segment_start is not None:
                            # Диктор будет назначен после записи через post-recording диаризацию
                            writer.write_segment(start=segment_start, end=now, text=result.text, speaker="SPEAKER_?")
                            print(f"[{segment_start:.1f}s] {result.text}")
                    except Exception as e:
                        logger.warning("Ошибка транскрибации", error=str(e))
                    segment_start = None
                    speech_buffer = []
                    silence_count = 0
            else:
                if speech_buffer:
                    silence_count += 1
                    speech_buffer.append(chunk)

                    if silence_count >= silence_threshold:
                        audio_segment = np.concatenate(speech_buffer)
                        try:
                            result = transcriber.transcribe(audio_segment)
                            if result.text and segment_start is not None:
                                writer.write_segment(start=segment_start, end=now, text=result.text, speaker="SPEAKER_?")
                                print(f"[{segment_start:.1f}s] {result.text}")
                        except Exception as e:
                            logger.warning("Ошибка транскрибации", error=str(e))

                        speech_buffer = []
                        silence_count = 0
                        segment_start = None
    finally:
        capture.stop()
        output_path = writer.finish()
        if output_path:
            print(f"\nЗапись сохранена: {output_path}")

        # Post-recording диаризация на полном аудио
        if diarizer and _diar_slots and output_path:
            print("Диаризация полного аудио...", flush=True)
            try:
                print(f"[Diarizer] миксуем аудио из {len(_diar_slots)} слотов...", flush=True)
                # Микшируем слоты: loopback + mic в одно аудио правильной длины
                mixed_chunks = []
                slot_keys = sorted(_diar_slots.keys())
                print(f"[Diarizer] sorted keys OK, первый={slot_keys[0]}, последний={slot_keys[-1]}", flush=True)
                for i, slot in enumerate(slot_keys):
                    chunks = _diar_slots[slot]
                    print(f"[Diarizer] слот {i}: {len(chunks)} чанков, dtype={chunks[0].dtype}, shape={chunks[0].shape}", flush=True)
                    if len(chunks) == 1:
                        mixed_chunks.append(chunks[0])
                    else:
                        max_len = max(len(c) for c in chunks)
                        mix = np.zeros(max_len, dtype=np.float32)
                        for c in chunks:
                            mix[:len(c)] += c.astype(np.float32)
                        mixed_chunks.append(np.clip(mix, -32768, 32767).astype(np.int16))
                print(f"[Diarizer] цикл завершён, {len(mixed_chunks)} чанков", flush=True)
                full_audio = np.concatenate(mixed_chunks)
                print(f"[Diarizer] аудио готово: {len(full_audio)/sample_rate:.1f}s, dtype={full_audio.dtype}", flush=True)
                timeline = diarizer.build_timeline(
                    full_audio, sample_rate,
                    min_speakers=diar_cfg.get("min_speakers"),
                    max_speakers=diar_cfg.get("max_speakers"),
                )
                if timeline:
                    # Обновляем JSON с правильными метками дикторов
                    import json
                    with open(output_path, encoding="utf-8") as f:
                        data = json.load(f)
                    for seg in data.get("segments", []):
                        seg["speaker"] = diarizer.speaker_at(timeline, seg["start"], seg["end"])
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    speakers = len(set(t["speaker"] for t in timeline))
                    print(f"Диаризация завершена. Найдено дикторов: {speakers}, интервалов: {len(timeline)}", flush=True)
                else:
                    print("Диаризация: спикеры не найдены (запись слишком короткая или один голос). Метки не обновлены.", flush=True)
            except BaseException as e:
                import traceback, sys
                print(f"[Диаризация] ОШИБКА: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                logger.warning("Ошибка post-recording диаризации", error=str(e))


if __name__ == "__main__":
    main()
