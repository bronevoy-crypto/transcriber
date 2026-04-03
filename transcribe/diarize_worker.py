"""Воркер диаризации — запускается как subprocess из diarizer.py.

Использование:
    python -m transcribe.diarize_worker <wav_file> <hf_token> <device>

Выводит JSON-таймлайн в stdout. Ошибки — в stderr.
"""
import sys
import os
import json
import numpy as np
import scipy.io.wavfile as wavfile

# Добавляем корень проекта в sys.path чтобы найти пакет transcribe
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    if len(sys.argv) not in (4, 6):
        print("Usage: diarize_worker.py <wav_file> <hf_token> <device> [min_speakers max_speakers]", file=sys.stderr)
        sys.exit(1)

    wav_file, hf_token, device = sys.argv[1], sys.argv[2], sys.argv[3]
    min_speakers = int(sys.argv[4]) if len(sys.argv) == 6 and sys.argv[4] != "none" else None
    max_speakers = int(sys.argv[5]) if len(sys.argv) == 6 and sys.argv[5] != "none" else None

    # Загружаем аудио
    rate, audio = wavfile.read(wav_file)
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    # Применяем все патчи совместимости
    from transcribe.diarizer import (
        _fix_torch_load_compat,
        _fix_torchaudio_compat,
        _fix_pyannote_compat,
        _fix_windows_multiprocessing,
        _fix_speechbrain_k2,
    )
    _fix_torch_load_compat()
    _fix_torchaudio_compat()
    _fix_pyannote_compat()
    _fix_windows_multiprocessing()
    _fix_speechbrain_k2()

    import torch
    from pyannote.audio import Pipeline

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[worker] загрузка pipeline на {device}...", file=sys.stderr, flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if pipeline is None:
        print("[worker] pipeline is None — проверь токен HF", file=sys.stderr)
        sys.exit(2)

    pipeline.to(torch.device(device))
    spk_hint = ""
    if min_speakers is not None or max_speakers is not None:
        spk_hint = f" (min={min_speakers}, max={max_speakers})"
    print(f"[worker] pipeline загружен, диаризация {len(audio)/rate:.1f}s{spk_hint}...", file=sys.stderr, flush=True)

    waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
    diar_kwargs = {}
    if min_speakers is not None:
        diar_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diar_kwargs["max_speakers"] = max_speakers
    diarization = pipeline({"waveform": waveform, "sample_rate": rate}, **diar_kwargs)

    speaker_map: dict[str, str] = {}
    next_id = 0
    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        timeline.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker_map[speaker],
        })

    print(f"[worker] готово: {len(timeline)} интервалов, {next_id} дикторов", file=sys.stderr, flush=True)
    # Результат — в stdout
    print(json.dumps(timeline, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
