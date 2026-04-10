"""
Полный тестовый сюит: 40+ тестов на реальных данных.

Тестирует все аудиофайлы с эталонами при разных конфигурациях:
  - silence_duration: 0.2, 1.0, 2.5
  - vad_threshold: 0.3, 0.5

Использование:
    python run_full_suite.py [--quick]
    --quick : только 1 конфиг (текущий)
"""
import os
import sys
import re
import json
import copy
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Форсируем UTF-8 для вывода в консоль Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import yaml
import scipy.io.wavfile as wavfile

FFMPEG_DIR = r"D:\AI\RVC\Workspace\ffmpeg\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]

TTS_DIR = "tts_tests"

# Маппинг: wav -> (ref_file, label, категория)
TESTS = [
    ("bench1_aidar.wav",       "bench_ref1_tts.txt",           "bench1 / aidar (муж)",        "TTS голоса"),
    ("bench1_baya.wav",        "bench_ref1_tts.txt",           "bench1 / baya (жен)",         "TTS голоса"),
    ("bench1_kseniya.wav",     "bench_ref1_tts.txt",           "bench1 / kseniya (жен)",      "TTS голоса"),
    ("bench1_xenia.wav",       "bench_ref1_tts.txt",           "bench1 / xenia (жен)",        "TTS голоса"),
    ("bench2_dialogue.wav",    "bench_ref2.txt",               "bench2 / диалог 2 спик",      "Диалоги"),
    ("bench3_3speakers.wav",   "bench_ref3.txt",               "bench3 / 3 спикера",          "Диалоги"),
    ("bench4_long.wav",        "bench_ref4.txt",               "bench4 / длинный (8мин)",     "Длинные"),
    ("bench_domain_casual.wav","bench_ref_domain_casual.txt",  "domain / разговорный",        "Домены"),
    ("bench_domain_legal.wav", "bench_ref_domain_legal.txt",   "domain / юридический",        "Домены"),
    ("bench_domain_medical.wav","bench_ref_domain_medical.txt","domain / медицинский",        "Домены"),
    ("bench_domain_numbers.wav","bench_ref_domain_numbers.txt","domain / числа",              "Домены"),
    ("client_test2.wav",       "bench_ref_client_test2.txt",   "client_test2",                "Клиент"),
    ("meeting_it_sprint.wav",  "bench_ref_meeting_it.txt",     "meeting / IT sprint",         "Митинги"),
    ("meeting_product.wav",    "bench_ref_meeting_product.txt","meeting / product",           "Митинги"),
    ("meeting_sales.wav",      "bench_ref_meeting_sales.txt",  "meeting / sales",             "Митинги"),
    ("dense_dialogue.wav",     None,                           "dense / диалог",              "Спец"),  # JSON ref
]
DENSE_REF_JSON = "bench_ref_dense.json"

# Конфиги для вариации
CONFIGS = [
    {"silence_duration": 0.2, "vad_threshold": 0.3, "label": "s=0.2 vad=0.3"},
    {"silence_duration": 1.0, "vad_threshold": 0.3, "label": "s=1.0 vad=0.3"},
    {"silence_duration": 2.5, "vad_threshold": 0.3, "label": "s=2.5 vad=0.3"},
    {"silence_duration": 0.2, "vad_threshold": 0.5, "label": "s=0.2 vad=0.5"},
]


def load_base_config() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_config(base: dict, silence: float, vad_thr: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["audio"]["silence_duration"] = silence
    cfg["vad"]["threshold"] = vad_thr
    cfg.setdefault("diarization", {})["enabled"] = False
    return cfg


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0, 0, 0, 0
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
    return d[len(ref)][len(hyp)] / len(ref), d[len(ref)][len(hyp)], len(ref), len(hyp)


def convert_to_16k_mono(wav_path: str) -> np.ndarray:
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


def transcribe_file(wav_path: str, config: dict) -> tuple[str, int, float]:
    """Вернуть (текст, кол-во сегментов, длительность_сек)."""
    from audio.vad import VADProcessor
    from transcribe.factory import create_transcriber

    audio_cfg = config.get("audio", {})
    vad_cfg = config.get("vad", {})
    model_cfg = config.get("model", {})

    chunk_ms = audio_cfg.get("chunk_ms", 500)
    silence_duration = audio_cfg.get("silence_duration", 0.2)
    sample_rate = audio_cfg.get("sample_rate", 16000)
    silence_threshold = max(1, int(silence_duration * 1000 / chunk_ms))
    max_segment_chunks = int(180_000 / chunk_ms)

    vad = VADProcessor(
        threshold=vad_cfg.get("threshold", 0.3),
        min_speech_ms=vad_cfg.get("min_speech_ms", 250),
    )
    transcriber = create_transcriber(model_cfg)

    audio = convert_to_16k_mono(wav_path)
    audio_sec = len(audio) / sample_rate
    chunk_size = int(sample_rate * chunk_ms / 1000)
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    speech_buffer = []
    silence_count = 0
    all_text = []
    seg_count = 0

    for chunk in chunks:
        if len(chunk) < chunk_size // 2:
            continue
        try:
            is_speech = vad.is_speech(chunk)
        except Exception:
            speech_buffer = []
            silence_count = 0
            continue

        if is_speech:
            speech_buffer.append(chunk)
            silence_count = 0
            if len(speech_buffer) >= max_segment_chunks:
                seg = np.concatenate(speech_buffer)
                r = transcriber.transcribe(seg)
                if r.text:
                    all_text.append(r.text)
                    seg_count += 1
                speech_buffer = []
                silence_count = 0
        else:
            if speech_buffer:
                silence_count += 1
                speech_buffer.append(chunk)
                if silence_count >= silence_threshold:
                    seg = np.concatenate(speech_buffer)
                    r = transcriber.transcribe(seg)
                    if r.text:
                        all_text.append(r.text)
                        seg_count += 1
                    speech_buffer = []
                    silence_count = 0

    if speech_buffer:
        seg = np.concatenate(speech_buffer)
        r = transcriber.transcribe(seg)
        if r.text:
            all_text.append(r.text)
            seg_count += 1

    return " ".join(all_text), seg_count, audio_sec


def load_reference(ref_file: str, wav_name: str) -> str:
    """Загрузить эталон из txt или json."""
    if ref_file is None:
        # dense_dialogue.wav → bench_ref_dense.json
        if not os.path.exists(DENSE_REF_JSON):
            return ""
        with open(DENSE_REF_JSON, encoding="utf-8") as f:
            segs = json.load(f)
        return " ".join(s["text"] for s in segs)

    if not os.path.exists(ref_file):
        return ""
    with open(ref_file, encoding="utf-8") as f:
        return f.read().strip()


def run_suite(quick: bool = False):
    base_config = load_base_config()

    configs_to_run = CONFIGS[:1] if quick else CONFIGS

    print(f"{'='*70}")
    print(f"ПОЛНЫЙ ТЕСТОВЫЙ СЮИТ — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Конфигов: {len(configs_to_run)}, Файлов: ~{len(TESTS)}")
    print(f"{'='*70}\n")

    # Загружаем модели один раз
    print("Инициализация модели (VAD + GigaAM)...")
    from audio.vad import VADProcessor
    from transcribe.factory import create_transcriber
    _vad = VADProcessor(threshold=0.3, min_speech_ms=250)  # прогрев
    _tr = create_transcriber(base_config["model"])
    print("Модель загружена.\n")

    all_results = []
    test_num = 0

    for cfg_params in configs_to_run:
        cfg = make_config(base_config, cfg_params["silence_duration"], cfg_params["vad_threshold"])
        cfg_label = cfg_params["label"]

        print(f"\n{'-'*70}")
        print(f"КОНФИГ: {cfg_label}")
        print(f"{'-'*70}")

        for wav_name, ref_file, label, category in TESTS:
            wav_path = os.path.join(TTS_DIR, wav_name)

            if not os.path.exists(wav_path):
                print(f"  [SKIP] {label}: файл не найден")
                continue

            reference = load_reference(ref_file, wav_name)
            if not reference:
                print(f"  [SKIP] {label}: эталон не найден")
                continue

            test_num += 1
            print(f"\n  [{test_num:02d}] {label} | {cfg_label}")

            t0 = time.time()
            try:
                hypothesis, seg_count, audio_sec = transcribe_file(wav_path, cfg)
                elapsed = time.time() - t0

                w_score, w_dist, ref_len, hyp_len = wer(reference, hypothesis)
                ok = w_score < 0.15

                rtf = elapsed / audio_sec if audio_sec > 0 else 0

                print(f"       WER={w_score*100:.1f}% | segs={seg_count} | RTF={rtf:.2f}x | {'OK' if ok else 'FAIL'}")

                all_results.append({
                    "test_num": test_num,
                    "label": label,
                    "category": category,
                    "config": cfg_label,
                    "silence_duration": cfg_params["silence_duration"],
                    "vad_threshold": cfg_params["vad_threshold"],
                    "wer_pct": round(w_score * 100, 2),
                    "wer_dist": w_dist,
                    "ref_words": ref_len,
                    "hyp_words": hyp_len,
                    "segments": seg_count,
                    "audio_sec": round(audio_sec, 1),
                    "elapsed_sec": round(elapsed, 2),
                    "rtf": round(rtf, 3),
                    "ok": ok,
                    "hypothesis_preview": hypothesis[:150],
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"       ОШИБКА: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "test_num": test_num,
                    "label": label,
                    "category": category,
                    "config": cfg_label,
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 2),
                    "ok": False,
                })

    # ─── Статистика ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*70}")

    ok_tests = [r for r in all_results if r.get("ok") and "error" not in r]
    fail_tests = [r for r in all_results if not r.get("ok")]
    err_tests = [r for r in all_results if "error" in r]

    print(f"\nВсего тестов: {len(all_results)}")
    print(f"Прошли (WER<15%): {len(ok_tests)}")
    print(f"Провалили:        {len(fail_tests)}")
    print(f"Ошибки:           {len(err_tests)}")

    if ok_tests:
        wers = [r["wer_pct"] for r in ok_tests]
        print(f"\nДля прошедших тестов (WER<15%):")
        print(f"  Средний WER:  {sum(wers)/len(wers):.1f}%")
        print(f"  Лучший WER:   {min(wers):.1f}%")
        print(f"  Худший WER:   {max(wers):.1f}%")

    all_wers = [r["wer_pct"] for r in all_results if "error" not in r]
    if all_wers:
        print(f"\nПо всем тестам (без ошибок):")
        print(f"  Средний WER:  {sum(all_wers)/len(all_wers):.1f}%")
        print(f"  Медиана WER:  {sorted(all_wers)[len(all_wers)//2]:.1f}%")

    # По категориям
    categories = {}
    for r in all_results:
        if "error" in r:
            continue
        cat = r.get("category", "?")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["wer_pct"])

    print(f"\nПо категориям:")
    print(f"  {'Категория':<20} {'Ср.WER':>8}  {'Мин':>7}  {'Макс':>7}  {'N':>4}")
    print(f"  {'-'*55}")
    for cat, wers in sorted(categories.items()):
        avg = sum(wers) / len(wers)
        print(f"  {cat:<20} {avg:>7.1f}%  {min(wers):>6.1f}%  {max(wers):>6.1f}%  {len(wers):>4}")

    # По конфигам
    cfg_stats = {}
    for r in all_results:
        if "error" in r:
            continue
        c = r.get("config", "?")
        if c not in cfg_stats:
            cfg_stats[c] = []
        cfg_stats[c].append(r["wer_pct"])

    print(f"\nПо конфигам:")
    print(f"  {'Конфиг':<20} {'Ср.WER':>8}  {'Прошло':>8}")
    print(f"  {'-'*45}")
    for cfglbl, wers in cfg_stats.items():
        ok_cnt = sum(1 for w in wers if w < 15)
        print(f"  {cfglbl:<20} {sum(wers)/len(wers):>7.1f}%  {ok_cnt}/{len(wers)}")

    # Топ худших
    sorted_by_wer = sorted([r for r in all_results if "error" not in r], key=lambda x: x["wer_pct"], reverse=True)
    print(f"\nТоп-5 ХУДШИХ:")
    for r in sorted_by_wer[:5]:
        print(f"  {r['wer_pct']:>6.1f}%  [{r['config']}]  {r['label']}")

    # Топ лучших
    print(f"\nТоп-5 ЛУЧШИХ:")
    for r in sorted_by_wer[-5:][::-1]:
        print(f"  {r['wer_pct']:>6.1f}%  [{r['config']}]  {r['label']}")

    # Полная таблица
    print(f"\n{'='*70}")
    print("ПОЛНАЯ ТАБЛИЦА:")
    print(f"  {'#':>3}  {'Тест':<35} {'Конфиг':<18} {'WER':>6}  {'RTF':>6}  {'OK':>4}")
    print(f"  {'-'*78}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['test_num']:>3}  {r['label']:<35} {r['config']:<18} {'ERROR':>6}  {'---':>6}  {'НЕТ':>4}")
        else:
            ok_str = "ДА" if r["ok"] else "НЕТ"
            rtf_str = f"{r.get('rtf', 0):.2f}x"
            print(f"  {r['test_num']:>3}  {r['label']:<35} {r['config']:<18} {r['wer_pct']:>5.1f}%  {rtf_str:>6}  {ok_str:>4}")

    # Сохраняем
    out_path = f"full_suite_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(all_results),
            "passed": len(ok_tests),
            "failed": len(fail_tests),
            "errors": len(err_tests),
            "avg_wer_all": round(sum(all_wers)/len(all_wers), 2) if all_wers else None,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_suite(quick=quick)
