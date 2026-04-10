"""
Тест диаризации: точность разделения спикеров.

Метрики:
  - Speaker Count Accuracy: найдено vs ожидалось
  - DER (Diarization Error Rate) - только для файлов с временными метками в эталоне
  - WER с диаризацией: транскрипция по каждому спикеру

Использование:
    python run_diar_suite.py
"""
import os
import sys
import re
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

FFMPEG_DIR = r"D:\AI\RVC\Workspace\ffmpeg\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]

TTS_DIR = "tts_tests"

# Файлы для теста диаризации
# (wav_name, expected_speakers, ref_file_or_none, label, has_timestamps)
DIAR_TESTS = [
    ("bench2_dialogue.wav",   2, "bench_ref2.txt",             "bench2 / диалог 2 спик",  False),
    ("bench3_3speakers.wav",  3, "bench_ref3.txt",             "bench3 / 3 спикера",      False),
    ("dense_dialogue.wav",    2, "bench_ref_dense.json",       "dense / 2 спикера",       True),   # DER
    ("meeting_it_sprint.wav", 3, "bench_ref_it_sprint.json",   "meeting / IT sprint",     False),
    ("meeting_product.wav",   4, "bench_ref_product.json",     "meeting / product",       False),
    ("meeting_sales.wav",     2, "bench_ref_sales.json",       "meeting / sales",         False),
    ("client_test2.wav",      3, "bench_ref_client_test2.json", "client_test2 (3 спик)",  False),
    ("real_negotiations.wav", None, None,                      "real_negotiations",       False),
    ("real_standup.wav",      None, None,                      "real_standup",            False),
]


def load_config() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def convert_to_16k_mono(wav_path: str) -> np.ndarray:
    import scipy.io.wavfile as wavfile
    tmp_path = wav_path + "_16k_tmp.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", "-ac", "1", tmp_path],
        capture_output=True
    )
    if result.returncode == 0:
        rate, data = wavfile.read(tmp_path)
        os.remove(tmp_path)
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        return data
    rate, data = wavfile.read(wav_path)
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)
    return data


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer_simple(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref)][len(hyp)] / len(ref)


def compute_der(ref_timeline: list[dict], hyp_timeline: list[dict], total_duration: float) -> dict:
    """
    Упрощённый DER: только speaker_error + missed_speech + false_alarm.
    ref_timeline: [{start, end, speaker}]
    hyp_timeline: [{start, end, speaker}]  — из pyannote
    """
    # Сначала делаем сетку с шагом 10мс
    step = 0.01
    frames = int(total_duration / step) + 1

    ref_labels = ["silence"] * frames
    hyp_labels = ["silence"] * frames

    for seg in ref_timeline:
        start_f = int(seg["start"] / step)
        end_f = int(seg["end"] / step)
        for f in range(start_f, min(end_f, frames)):
            ref_labels[f] = seg["speaker"]

    for seg in hyp_timeline:
        start_f = int(seg["start"] / step)
        end_f = int(seg["end"] / step)
        for f in range(start_f, min(end_f, frames)):
            hyp_labels[f] = seg["speaker"]

    # Подсчёт ошибок
    total_speech_frames = sum(1 for l in ref_labels if l != "silence")
    missed = 0
    false_alarm = 0

    for rf, hf in zip(ref_labels, hyp_labels):
        if rf != "silence" and hf == "silence":
            missed += 1
        elif rf == "silence" and hf != "silence":
            false_alarm += 1

    # Speaker error: оптимальное сопоставление спикеров (простая версия)
    # Строим confusion matrix
    ref_speakers = list(set(l for l in ref_labels if l != "silence"))
    hyp_speakers = list(set(l for l in hyp_labels if l != "silence"))

    overlap = {}
    for rs in ref_speakers:
        for hs in hyp_speakers:
            cnt = sum(1 for rf, hf in zip(ref_labels, hyp_labels)
                      if rf == rs and hf == hs)
            overlap[(rs, hs)] = cnt

    # Жадное сопоставление
    assignment = {}
    used_hyp = set()
    for rs in ref_speakers:
        best_hs = None
        best_cnt = -1
        for hs in hyp_speakers:
            if hs not in used_hyp and overlap.get((rs, hs), 0) > best_cnt:
                best_cnt = overlap[(rs, hs)]
                best_hs = hs
        if best_hs:
            assignment[rs] = best_hs
            used_hyp.add(best_hs)

    # Фреймы речи где оба не тишина — считаем speaker error
    speaker_error = 0
    for rf, hf in zip(ref_labels, hyp_labels):
        if rf != "silence" and hf != "silence":
            mapped_hyp = {v: k for k, v in assignment.items()}.get(hf, hf)
            if rf != mapped_hyp:
                speaker_error += 1

    if total_speech_frames == 0:
        return {"der": 0.0, "missed": 0.0, "false_alarm": 0.0, "speaker_error": 0.0}

    der = (missed + false_alarm + speaker_error) / total_speech_frames
    return {
        "der": round(der * 100, 2),
        "missed_pct": round(missed / total_speech_frames * 100, 2),
        "false_alarm_pct": round(false_alarm / total_speech_frames * 100, 2),
        "speaker_error_pct": round(speaker_error / total_speech_frames * 100, 2),
    }


def run_vad_transcribe(audio: np.ndarray, config: dict) -> list[dict]:
    """Запустить VAD + GigaAM, вернуть сегменты с временными метками."""
    from audio.vad import VADProcessor
    from transcribe.factory import create_transcriber

    audio_cfg = config.get("audio", {})
    vad_cfg = config.get("vad", {})
    model_cfg = config.get("model", {})

    chunk_ms = audio_cfg.get("chunk_ms", 500)
    silence_duration = audio_cfg.get("silence_duration", 0.2)
    sample_rate = audio_cfg.get("sample_rate", 16000)
    silence_threshold = max(1, int(silence_duration * 1000 / chunk_ms))

    vad = VADProcessor(
        threshold=vad_cfg.get("threshold", 0.3),
        min_speech_ms=vad_cfg.get("min_speech_ms", 250),
    )
    transcriber = create_transcriber(model_cfg)

    chunk_size = int(sample_rate * chunk_ms / 1000)
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    speech_buffer = []
    silence_count = 0
    speech_start_chunk = 0
    segments = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < chunk_size // 2:
            continue
        try:
            is_speech = vad.is_speech(chunk)
        except Exception:
            speech_buffer = []
            silence_count = 0
            continue

        if is_speech:
            if not speech_buffer:
                speech_start_chunk = i
            speech_buffer.append(chunk)
            silence_count = 0
        else:
            if speech_buffer:
                silence_count += 1
                speech_buffer.append(chunk)
                if silence_count >= silence_threshold:
                    seg = np.concatenate(speech_buffer)
                    r = transcriber.transcribe(seg)
                    if r.text:
                        segments.append({
                            "start": speech_start_chunk * chunk_ms / 1000,
                            "end": (i + 1) * chunk_ms / 1000,
                            "text": r.text,
                            "speaker": "?",
                        })
                    speech_buffer = []
                    silence_count = 0

    if speech_buffer:
        seg = np.concatenate(speech_buffer)
        r = transcriber.transcribe(seg)
        if r.text:
            segments.append({
                "start": speech_start_chunk * chunk_ms / 1000,
                "end": len(audio) / 16000,
                "text": r.text,
                "speaker": "?",
            })

    return segments


def diarize(audio: np.ndarray, config: dict) -> list[dict]:
    """Запустить pyannote, вернуть таймлайн [{start, end, speaker}]."""
    from transcribe.diarizer import _fix_torch_load_compat, _fix_torchaudio_compat, _fix_pyannote_compat
    _fix_torch_load_compat()
    _fix_torchaudio_compat()
    _fix_pyannote_compat()

    import torch
    from pyannote.audio import Pipeline

    diar_cfg = config.get("diarization", {})
    hf_token = diar_cfg.get("hf_token", "")
    device = diar_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    min_sp = diar_cfg.get("min_speakers")
    max_sp = diar_cfg.get("max_speakers")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)

    kwargs = {}
    if min_sp is not None:
        kwargs["min_speakers"] = min_sp
    if max_sp is not None:
        kwargs["max_speakers"] = max_sp

    diarization = pipeline({"waveform": waveform, "sample_rate": 16000}, **kwargs)

    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return timeline


def get_speaker_at(timeline: list[dict], start: float, end: float) -> str:
    speaker_times: dict[str, float] = {}
    for item in timeline:
        ov_start = max(item["start"], start)
        ov_end = min(item["end"], end)
        if ov_end > ov_start:
            dur = ov_end - ov_start
            speaker_times[item["speaker"]] = speaker_times.get(item["speaker"], 0) + dur
    if not speaker_times:
        return "UNKNOWN"
    return max(speaker_times, key=speaker_times.get)


def load_ref(ref_file: str) -> tuple[str, list[dict]]:
    """Вернуть (full_text, segments_with_speaker_if_any)."""
    if ref_file is None:
        return "", []
    if not os.path.exists(ref_file):
        return "", []
    if ref_file.endswith(".json"):
        with open(ref_file, encoding="utf-8") as f:
            data = json.load(f)
        full_text = " ".join(s["text"] for s in data)
        return full_text, data
    else:
        with open(ref_file, encoding="utf-8") as f:
            text = f.read().strip()
        return text, []


def run_suite():
    config = load_config()
    sample_rate = 16000

    print("=" * 70)
    print(f"ТЕСТ ДИАРИЗАЦИИ — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Загружаем pyannote один раз
    print("\nЗагрузка pyannote pipeline...")
    t0 = time.time()
    from transcribe.diarizer import _fix_torch_load_compat, _fix_torchaudio_compat, _fix_pyannote_compat
    _fix_torch_load_compat()
    _fix_torchaudio_compat()
    _fix_pyannote_compat()
    import torch
    from pyannote.audio import Pipeline

    diar_cfg = config.get("diarization", {})
    hf_token = diar_cfg.get("hf_token", "")
    device = diar_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))
    print(f"Pyannote загружена за {time.time()-t0:.1f}s (device={device})\n")

    # Загружаем VAD + GigaAM
    print("Загрузка VAD + GigaAM...")
    from audio.vad import VADProcessor
    from transcribe.factory import create_transcriber
    _vad_warm = VADProcessor(threshold=0.3, min_speech_ms=250)
    _tr_warm = create_transcriber(config["model"])
    print("VAD + GigaAM загружены.\n")

    all_results = []

    for wav_name, expected_speakers, ref_file, label, has_timestamps in DIAR_TESTS:
        wav_path = os.path.join(TTS_DIR, wav_name)
        if not os.path.exists(wav_path):
            print(f"[SKIP] {label}: {wav_path} не найден")
            continue

        print(f"\n{'='*60}")
        print(f"Тест: {label}")
        if expected_speakers:
            print(f"Ожидается спикеров: {expected_speakers}")

        ref_text, ref_segs = load_ref(ref_file)

        t_start = time.time()

        try:
            print(f"  Конвертация аудио...")
            audio = convert_to_16k_mono(wav_path)
            audio_sec = len(audio) / sample_rate
            print(f"  Длина: {audio_sec:.1f}s")

            # Диаризация
            print(f"  Диаризация...")
            waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            found_speakers = len(set(t["speaker"] for t in timeline))
            print(f"  Найдено спикеров: {found_speakers} (интервалов: {len(timeline)})")

            # VAD + транскрибация с сопоставлением
            print(f"  Транскрибация + сопоставление со спикерами...")
            segments = run_vad_transcribe(audio, config)
            print(f"  Сегментов транскрипции: {len(segments)}")

            # Сопоставление сегментов с диаризацией
            speaker_map = {}
            next_id = [0]

            def normalize_sp(raw_id):
                if raw_id not in speaker_map:
                    speaker_map[raw_id] = f"SP_{next_id[0]:02d}"
                    next_id[0] += 1
                return speaker_map[raw_id]

            for seg in segments:
                raw_sp = get_speaker_at(timeline, seg["start"], seg["end"])
                seg["speaker"] = normalize_sp(raw_sp)

            elapsed = time.time() - t_start

            # WER (если есть эталон)
            wer_val = None
            if ref_text:
                hyp_text = " ".join(s["text"] for s in segments)
                wer_val = wer_simple(ref_text, hyp_text)

            # DER (если есть временные метки в эталоне)
            der_result = None
            if has_timestamps and ref_segs and all("start" in s and "end" in s for s in ref_segs):
                print(f"  Подсчёт DER...")
                der_result = compute_der(ref_segs, timeline, audio_sec)
                print(f"  DER: {der_result['der']}%")

            # Точность по числу спикеров
            sp_ok = (expected_speakers is None) or (found_speakers == expected_speakers)
            sp_close = (expected_speakers is None) or (abs(found_speakers - expected_speakers) <= 1)

            result = {
                "label": label,
                "audio_sec": round(audio_sec, 1),
                "expected_speakers": expected_speakers,
                "found_speakers": found_speakers,
                "speaker_count_ok": sp_ok,
                "speaker_count_close": sp_close,
                "intervals": len(timeline),
                "transcription_segments": len(segments),
                "elapsed_sec": round(elapsed, 2),
                "wer_pct": round(wer_val * 100, 2) if wer_val is not None else None,
                "der": der_result,
                "segments_preview": [
                    {"speaker": s["speaker"], "text": s["text"][:80]}
                    for s in segments[:5]
                ],
            }
            all_results.append(result)

            print(f"\n  Транскрипция с дикторами (первые 5 сегментов):")
            for s in segments[:5]:
                print(f"    [{s['speaker']}] {s['text'][:70]}")

            if wer_val is not None:
                print(f"\n  WER: {wer_val*100:.1f}%")
            sp_status = "OK" if sp_ok else (f"~OK (разница 1)" if sp_close else "FAIL")
            print(f"  Спикеры: {found_speakers}/{expected_speakers if expected_speakers else '?'} -> {sp_status}")
            if der_result:
                print(f"  DER: {der_result['der']}% (missed={der_result['missed_pct']}%, fa={der_result['false_alarm_pct']}%, sp_err={der_result['speaker_error_pct']}%)")

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"label": label, "error": str(e), "elapsed_sec": round(elapsed, 2)})

    # Итоговая статистика
    print(f"\n\n{'='*70}")
    print("ИТОГОВАЯ СТАТИСТИКА ДИАРИЗАЦИИ")
    print(f"{'='*70}")

    ok_results = [r for r in all_results if "error" not in r]

    sp_ok_cnt = sum(1 for r in ok_results if r.get("speaker_count_ok"))
    sp_close_cnt = sum(1 for r in ok_results if r.get("speaker_count_close"))
    total = len(ok_results)

    print(f"\n  Тестов выполнено: {total}/{len(all_results)}")
    print(f"  Точное кол-во спикеров: {sp_ok_cnt}/{total}")
    print(f"  С точностью ±1:         {sp_close_cnt}/{total}")

    wer_vals = [r["wer_pct"] for r in ok_results if r.get("wer_pct") is not None]
    if wer_vals:
        print(f"\n  WER транскрипции (с диаризацией):")
        print(f"    Средний: {sum(wer_vals)/len(wer_vals):.1f}%")
        print(f"    Лучший:  {min(wer_vals):.1f}%")
        print(f"    Худший:  {max(wer_vals):.1f}%")

    der_vals = [r["der"]["der"] for r in ok_results if r.get("der")]
    if der_vals:
        print(f"\n  DER (с временными метками):")
        for r in ok_results:
            if r.get("der"):
                print(f"    {r['label']}: DER={r['der']['der']}%")

    print(f"\n  Полная таблица:")
    print(f"  {'Тест':<30} {'Ожид':>5} {'Найд':>5} {'Спик':>6}  {'WER':>7}  {'Время':>6}")
    print(f"  {'-'*65}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['label']:<30} {'?':>5} {'?':>5} {'ERR':>6}  {'---':>7}  {r['elapsed_sec']:>5.1f}s")
        else:
            sp_str = "OK" if r["speaker_count_ok"] else ("~1" if r["speaker_count_close"] else "FAIL")
            exp = str(r["expected_speakers"]) if r["expected_speakers"] else "?"
            wer_str = f"{r['wer_pct']:.1f}%" if r["wer_pct"] is not None else "---"
            print(f"  {r['label']:<30} {exp:>5} {r['found_speakers']:>5} {sp_str:>6}  {wer_str:>7}  {r['elapsed_sec']:>5.1f}s")

    # Сохраняем
    out_path = f"diar_suite_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(all_results),
            "speaker_count_exact": sp_ok_cnt,
            "speaker_count_close": sp_close_cnt,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_suite()
