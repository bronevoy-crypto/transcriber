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
    """PyTorch 2.6+ defaults to weights_only=True, breaking old pyannote checkpoints.

    Patches lightning_fabric's _load to use weights_only=False for trusted sources.
    """
    try:
        import lightning_fabric.utilities.cloud_io as _cloud_io
        import torch
        _orig_load = _cloud_io._load

        def _patched_load(path_or_url, map_location=None, **kwargs):
            kwargs["weights_only"] = False  # force: pyannote checkpoints need full unpickling
            return _orig_load(path_or_url, map_location=map_location, **kwargs)

        _cloud_io._load = _patched_load
    except Exception:
        pass


def _fix_pyannote_compat():
    """Monkey-patch: pyannote 3.3.x + huggingface_hub 0.24+ используют разные имена параметра.

    pyannote 3.3.x передаёт use_auth_token в hf_hub_download,
    но новый huggingface_hub ждёт token.
    """
    try:
        import huggingface_hub
        orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, use_auth_token=None, token=None, **kwargs):
            # normalize: use_auth_token → token
            if use_auth_token is not None and token is None:
                token = use_auth_token
            return orig_download(*args, token=token, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download
        # Also patch the import that pyannote already did
        import pyannote.audio.core.pipeline as _pp
        _pp.hf_hub_download = _patched_download
    except Exception:
        pass


def _fix_speechbrain_k2():
    """Python 3.13 inspect.stack() вызывает hasattr(mod, '__file__') на всех модулях.

    speechbrain регистрирует k2_fsa как LazyModule. При __file__ запросе LazyModule
    пытается импортировать k2 — которого нет. Патчим сам класс LazyModule:
    для служебных атрибутов inspect (__, __file__, __spec__ и т.п.) возвращаем None вместо ошибки.
    Патчим ДО загрузки pipeline, чтобы покрыть все экземпляры.
    """
    try:
        import speechbrain.utils.importutils as _sb_importutils
        if not hasattr(_sb_importutils, "LazyModule"):
            return
        LazyModule = _sb_importutils.LazyModule
        if getattr(LazyModule, "_k2_patched", False):
            return
        _orig_getattr = LazyModule.__getattr__

        def _safe_getattr(self, attr):
            # Атрибуты, которые inspect.py запрашивает на любом ModuleType-объекте.
            # Если k2 не установлен — вернём None вместо ImportError.
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
    """На Windows после Ctrl+C возможны проблемы с multiprocessing в pyannote.

    В pyannote 3.3.x Inference не использует DataLoader workers (нет num_workers),
    поэтому патч не требуется. Функция оставлена для совместимости на случай
    обновления pyannote до версии с num_workers.
    """
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
    """Monkey-patch для совместимости torchaudio 2.x с pyannote 3.x."""
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
        # Fallback to CPU if CUDA not available in this torch build
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Diarizer: CUDA недоступна, используем CPU")
            device = "cpu"
        self._device = torch.device(device)
        self._pipeline = None
        self._speaker_map: dict[str, str] = {}  # pyannote_id -> наш SPEAKER_XX
        self._next_id = 0

    def load(self) -> None:
        """Проверка токена и доступности модели."""
        try:
            import huggingface_hub
            info = huggingface_hub.model_info(
                "pyannote/speaker-diarization-3.1",
                token=self._hf_token,
            )
            logger.info("Diarizer: токен валиден, модель доступна", model=info.id)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось получить доступ к pyannote/speaker-diarization-3.1: {e}\n"
                "Проверьте hf_token в config.yaml и примите условия использования на "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            ) from e

    def build_timeline(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict]:
        """Диаризировать ПОЛНОЕ аудио, вернуть таймлайн со стабильными ID дикторов."""
        duration_s = len(audio) / sample_rate
        print(f"[DIAR] build_timeline called, duration={duration_s:.1f}s", flush=True)
        logger.info("Diarizer: диаризация полного аудио", duration_s=round(duration_s, 1))
        if duration_s < 1.0:
            logger.warning("Diarizer: аудио слишком короткое, пропускаем", duration_s=round(duration_s, 1))
            return []

        # Сохраняем аудио во временный WAV (wave stdlib — не зависит от scipy)
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

            # stderr=subprocess.STDOUT — мержим stderr в stdout, избегаем race condition
            # при ручном потоке + communicate() на одном pipe
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

            # Печатаем весь вывод воркера (прогресс + ошибки)
            for line in stdout_text.splitlines():
                if line.strip():
                    print(line, flush=True)

            if proc.returncode != 0:
                logger.warning("Diarizer: воркер завершился с ошибкой", code=proc.returncode)
                return []

            # Парсим JSON из последней непустой строки stdout
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
        """Найти доминирующего диктора в заданном временном интервале по таймлайну."""
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

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """[УСТАРЕЛО] Определить доминирующего говорящего в одном сегменте.

        Не рекомендуется: работает некорректно для разных сегментов одной встречи.
        Используйте build_timeline() + speaker_at() вместо этого.
        """
        if self._pipeline is None:
            return "SPEAKER_00"

        waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0).to(self._device)
        try:
            diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
        except Exception as e:
            logger.warning("Diarizer: ошибка диаризации", error=str(e))
            return "SPEAKER_00"

        speaker_times: dict[str, float] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        if not speaker_times:
            return "SPEAKER_00"

        dominant = max(speaker_times, key=speaker_times.get)
        if dominant not in self._speaker_map:
            self._speaker_map[dominant] = f"SPEAKER_{self._next_id:02d}"
            self._next_id += 1
            logger.info("Diarizer: новый спикер", id=self._speaker_map[dominant])

        return self._speaker_map[dominant]
