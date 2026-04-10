"""Диаризация говорящих через pyannote-audio."""
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import wave
import structlog
import torch

logger = structlog.get_logger(__name__)


def _fix_torch_load_compat():
    # PyTorch 2.6+ ломает старые чекпоинты pyannote — отключаем weights_only
    try:
        import lightning_fabric.utilities.cloud_io as _cloud_io
        import torch
        _orig_load = _cloud_io._load

        def _patched_load(path_or_url, map_location=None, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(path_or_url, map_location=map_location, **kwargs)

        _cloud_io._load = _patched_load
    except Exception:
        pass


def _fix_pyannote_compat():
    # pyannote 3.3.x передаёт use_auth_token, новый huggingface_hub ждёт token
    try:
        import huggingface_hub
        orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, use_auth_token=None, token=None, **kwargs):
            if use_auth_token is not None and token is None:
                token = use_auth_token
            return orig_download(*args, token=token, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download
        import pyannote.audio.core.pipeline as _pp
        _pp.hf_hub_download = _patched_download
    except Exception:
        pass


def _fix_speechbrain_k2():
    # Python 3.13 + speechbrain LazyModule (k2) + inspect.stack() = ImportError
    # возвращаем None для служебных атрибутов вместо падения
    try:
        import speechbrain.utils.importutils as _sb_importutils
        if not hasattr(_sb_importutils, "LazyModule"):
            return
        LazyModule = _sb_importutils.LazyModule
        if getattr(LazyModule, "_k2_patched", False):
            return
        _orig_getattr = LazyModule.__getattr__

        def _safe_getattr(self, attr):
            if attr in ("__file__", "__spec__", "__loader__", "__package__", "__path__"):
                try:
                    return _orig_getattr(self, attr)
                except (ImportError, AttributeError):
                    return None
            return _orig_getattr(self, attr)

        LazyModule.__getattr__ = _safe_getattr
        LazyModule._k2_patched = True
    except Exception:
        pass


def _fix_windows_multiprocessing():
    # на случай если pyannote начнёт использовать num_workers — форсируем 0

    try:
        from pyannote.audio.core.inference import Inference
        import inspect as _inspect
        sig = _inspect.signature(Inference.__init__)
        if "num_workers" not in sig.parameters:
            return  # эта версия не использует workers — ничего не делаем
        _orig_inference_init = Inference.__init__

        def _patched_inference_init(self, model, *args, num_workers=0, **kwargs):
            _orig_inference_init(self, model, *args, num_workers=num_workers, **kwargs)

        Inference.__init__ = _patched_inference_init
    except Exception:
        pass


def _fix_torchaudio_compat():
    # torchaudio 2.x убрал ряд атрибутов которые ждёт pyannote 3.x
    try:
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            from dataclasses import dataclass
            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str
            torchaudio.AudioMetaData = AudioMetaData

        if not hasattr(torchaudio, "info"):
            import soundfile as sf
            def _info(path, **kwargs):
                info = sf.info(str(path))
                return torchaudio.AudioMetaData(
                    sample_rate=info.samplerate,
                    num_frames=info.frames,
                    num_channels=info.channels,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )
            torchaudio.info = _info

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

        if not hasattr(torchaudio, "load"):
            import soundfile as sf
            import numpy as np
            def _load(path, **kwargs):
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T), sr
            torchaudio.load = _load

    except Exception:
        pass


class Diarizer:
    def __init__(self, hf_token: str, device: str = "cpu"):
        self._hf_token = hf_token
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Diarizer: CUDA недоступна, используем CPU")
            device = "cpu"
        self._device = torch.device(device)
        self._pipeline = None
        self._speaker_map: dict[str, str] = {}  # pyannote_id -> наш SPEAKER_XX
        self._next_id = 0

    def load(self) -> None:
        if not self._hf_token or not self._hf_token.startswith("hf_"):
            raise RuntimeError(
                "hf_token не задан или некорректен. "
                "Укажите токен HuggingFace в config.yaml (diarization.hf_token)."
            )
        logger.info("Diarizer: токен установлен, модель загрузится при первой диаризации")

    def build_timeline(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict]:
        """Запустить диаризацию на всём аудио, вернуть таймлайн."""
        duration_s = len(audio) / sample_rate
        logger.info("Diarizer: диаризация полного аудио", duration_s=round(duration_s, 1))
        if duration_s < 1.0:
            logger.warning("Diarizer: аудио слишком короткое, пропускаем", duration_s=round(duration_s, 1))
            return []

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            audio_bytes = np.ascontiguousarray(audio, dtype=np.int16).tobytes()
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            worker = os.path.join(os.path.dirname(__file__), "diarize_worker.py")
            cmd = [sys.executable, worker, tmp.name, self._hf_token, str(self._device)]
            if min_speakers is not None or max_speakers is not None:
                cmd += [
                    str(min_speakers) if min_speakers is not None else "none",
                    str(max_speakers) if max_speakers is not None else "none",
                ]

            print("[Diarizer] запуск воркера (при первом запуске загрузка модели ~10 мин)...", flush=True)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            try:
                stdout_raw, _ = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                logger.warning("Diarizer: таймаут диаризации (10 мин)")
                return []

            enc = sys.stdout.encoding or "utf-8"
            stdout_text = stdout_raw.decode(enc, errors="replace")

            for line in stdout_text.splitlines():
                if line.strip():
                    print(line, flush=True)

            if proc.returncode != 0:
                logger.warning("Diarizer: воркер завершился с ошибкой", code=proc.returncode)
                return []

            lines = stdout_text.strip().splitlines()
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("["):
                    try:
                        timeline = json.loads(line)
                        unique = len(set(t["speaker"] for t in timeline))
                        logger.info("Diarizer: таймлайн готов", intervals=len(timeline), speakers=unique)
                        return timeline
                    except json.JSONDecodeError:
                        continue

            logger.warning("Diarizer: не удалось распарсить таймлайн из stdout")
            return []

        except Exception as e:
            import traceback
            print(f"[Diarizer] ОШИБКА: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return []
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def speaker_at(self, timeline: list[dict], start: float, end: float) -> str:
        """Кто говорил больше всего в интервале [start, end]."""
        speaker_times: dict[str, float] = {}
        for item in timeline:
            overlap_start = max(item["start"], start)
            overlap_end = min(item["end"], end)
            if overlap_end > overlap_start:
                dur = overlap_end - overlap_start
                speaker_times[item["speaker"]] = speaker_times.get(item["speaker"], 0) + dur
        if not speaker_times:
            return "SPEAKER_00"
        return max(speaker_times, key=speaker_times.get)

    def split_segments_by_speakers(
        self, segments: list[dict], timeline: list[dict], min_turn: float = 1.0
    ) -> list[dict]:
        if not timeline:
            return segments

        result = []
        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            text = seg.get("text", "")

            turns = []
            for item in timeline:
                o_start = max(item["start"], seg_start)
                o_end = min(item["end"], seg_end)
                if o_end > o_start:
                    turns.append({"start": o_start, "end": o_end, "speaker": item["speaker"]})

            if len(turns) > 1:
                merged = [turns[0]]
                for t in turns[1:]:
                    dur = t["end"] - t["start"]
                    if dur < min_turn:
                        merged[-1]["end"] = t["end"]
                    elif t["speaker"] == merged[-1]["speaker"]:
                        merged[-1]["end"] = t["end"]
                    else:
                        merged.append(t)
                turns = merged

            if len(turns) <= 1:
                if turns:
                    speaker = turns[0]["speaker"]
                else:
                    speaker = self.speaker_at(timeline, seg_start, seg_end)
                result.append({**seg, "speaker": speaker})
                continue

            total_dur = sum(t["end"] - t["start"] for t in turns)
            words = text.split()
            if not words or total_dur <= 0:
                result.append(seg)
                continue

            word_idx = 0
            for i, turn in enumerate(turns):
                turn_dur = turn["end"] - turn["start"]
                if i == len(turns) - 1:
                    turn_words = words[word_idx:]
                else:
                    n_words = max(1, round(len(words) * turn_dur / total_dur))
                    turn_words = words[word_idx:word_idx + n_words]
                    word_idx += n_words

                if turn_words:
                    result.append({
                        "start": round(turn["start"], 2),
                        "end": round(turn["end"], 2),
                        "speaker": turn["speaker"],
                        "text": " ".join(turn_words),
                    })

        if len(result) > 1:
            merged = [result[0]]
            for seg in result[1:]:
                if seg["speaker"] == merged[-1]["speaker"]:
                    merged[-1]["end"] = seg["end"]
                    merged[-1]["text"] += " " + seg["text"]
                else:
                    merged.append(seg)
            result = merged

        return result

    def assign_speakers_by_word(
        self, segments: list[dict], timeline: list[dict]
    ) -> list[dict]:
        """Назначение спикеров по словам через word timestamps GigaAM."""
        if not timeline:
            return segments

        result: list[dict] = []
        for seg in segments:
            words = seg.get("words")
            if not words:
                speaker = self.speaker_at(timeline, seg["start"], seg["end"])
                result.append({**seg, "speaker": speaker})
                continue

            seg_offset = seg["start"]
            current_speaker = None
            current_words: list[str] = []
            current_start = seg["start"]
            current_end = seg["end"]

            for w in words:
                abs_start = seg_offset + w["start"]
                abs_end = seg_offset + w["end"]
                # окно ±1.0s — pyannote иногда запаздывает с границей на 1-2с
                WINDOW = 1.0
                new_speaker = self.speaker_at(timeline, abs_start - WINDOW, abs_end + WINDOW)

                speaker_changed = new_speaker != current_speaker and current_speaker is not None

                if speaker_changed:
                    # Переключаем спикера только на границе предложения.
                    prev_word = current_words[-1] if current_words else ""
                    at_sentence_end = bool(prev_word) and prev_word[-1] in ".?!…"
                    if at_sentence_end:
                        if current_words:
                            result.append({
                                "start": round(current_start, 2),
                                "end": round(abs_start, 2),
                                "speaker": current_speaker,
                                "text": " ".join(current_words),
                            })
                        current_speaker = new_speaker
                        current_start = abs_start
                        current_words = [w["text"]]
                    else:
                        # Середина предложения — слово остаётся у текущего спикера
                        current_words.append(w["text"])
                        current_end = abs_end
                elif current_speaker is None:
                    current_speaker = new_speaker
                    current_start = abs_start
                    current_words = [w["text"]]
                else:
                    current_words.append(w["text"])
                    current_end = abs_end

            if current_words and current_speaker:
                result.append({
                    "start": round(current_start, 2),
                    "end": round(current_end, 2),
                    "speaker": current_speaker,
                    "text": " ".join(current_words),
                })

        # Поглощаем очень короткие фрагменты (< 1.0s) соседом с большим перекрытием
        MIN_SEG_DUR = 1.0
        if len(result) > 1:
            absorbed = [result[0]]
            for seg in result[1:]:
                dur = seg["end"] - seg["start"]
                if dur < MIN_SEG_DUR and absorbed:
                    # присваиваем к предыдущему сегменту
                    absorbed[-1]["end"] = seg["end"]
                    absorbed[-1]["text"] += " " + seg["text"]
                else:
                    absorbed.append(seg)
            result = absorbed

        # Мержим соседние сегменты одного спикера
        if len(result) > 1:
            merged = [result[0]]
            for seg in result[1:]:
                prev = merged[-1]
                if seg["speaker"] == prev["speaker"]:
                    prev["end"] = seg["end"]
                    prev["text"] += " " + seg["text"]
                else:
                    merged.append(seg)
            result = merged

        return result

