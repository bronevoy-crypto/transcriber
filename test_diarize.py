"""Тест диаризации в изоляции. Запуск: python test_diarize.py"""
import sys
import os
import subprocess
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import yaml

CONFIG_PATH = "config.yaml"

def main():
    # Читаем токен из конфига
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    diar_cfg = cfg.get("diarization", {})
    hf_token = diar_cfg.get("hf_token", "")
    device = diar_cfg.get("device", "cpu")

    if not hf_token.startswith("hf_"):
        print("ОШИБКА: hf_token не задан в config.yaml")
        sys.exit(1)

    print(f"Токен: {hf_token[:8]}..., устройство: {device}")

    # Генерируем 30 секунд тестового аудио (тишина + тон)
    sr = 16000
    duration = 30
    audio = np.zeros(sr * duration, dtype=np.int16)
    # Добавляем 2 "голоса" — тон 440 Гц и 880 Гц
    t = np.arange(sr * 10) / sr
    audio[sr*2 : sr*12] = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    audio[sr*15 : sr*25] = (np.sin(2 * np.pi * 880 * t) * 8000).astype(np.int16)

    # Пишем во временный WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wavfile.write(tmp.name, sr, audio)
    print(f"Тестовый WAV: {tmp.name} ({duration}s, {sr}Hz)")

    # Запускаем воркер напрямую
    worker = os.path.join("transcribe", "diarize_worker.py")
    cmd = [sys.executable, worker, tmp.name, hf_token, device]
    print(f"Запуск: {' '.join(cmd[:3])} ... {device}")
    print("-" * 60)

    result = subprocess.run(cmd, timeout=600)

    print("-" * 60)
    print(f"Код завершения: {result.returncode}")
    os.unlink(tmp.name)

if __name__ == "__main__":
    main()
