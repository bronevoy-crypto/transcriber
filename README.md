# Meeting Transcriber

Локальный транскрибер совещаний для Windows. Пишет системный звук + микрофон, режет по паузам, транскрибирует, определяет кто говорит. Результат — JSON с таймкодами и метками дикторов.

Всё работает на локальной машине, никуда не отправляется.

## Требования

- Windows 10/11
- Python 3.10+
- NVIDIA GPU — желательно, но не обязательно (GigaAM работает и на CPU)
- Аккаунт на HuggingFace — нужен для диаризации

## Установка

```bash
git clone https://github.com/HRYNdev/transcriber
cd transcriber
pip install -r requirements.txt
pip install --no-deps git+https://github.com/salute-developers/GigaAM.git
copy config.example.yaml config.yaml
```

## Настройка токена для диаризации

1. Зарегистрироваться на [huggingface.co](https://huggingface.co)
2. Принять условия использования:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Создать токен: https://huggingface.co/settings/tokens → New token → Read
4. Вставить в `config.yaml`:

```yaml
diarization:
  enabled: true
  hf_token: "hf_ВАШ_ТОКЕН"
  device: "cuda"  # cpu если нет GPU
```

Если диаризация не нужна — `enabled: false`.

## Запуск

```bash
python main.py
```

Запускать до митинга. Остановить — Ctrl+C. После остановки автоматически запустится диаризация всей записи.

Результат в папке `meetings/`.

## Формат вывода

`meetings/2026-03-31_08-19-47.json`:

```json
{
  "meeting_date": "2026-03-31T08:19:47",
  "segments": [
    {
      "start": 3.18,
      "end": 12.69,
      "speaker": "SPEAKER_00",
      "text": "Добрый день, начинаем."
    },
    {
      "start": 13.75,
      "end": 28.40,
      "speaker": "SPEAKER_01",
      "text": "Велосити за спринт — 35 поинтов."
    }
  ]
}
```

## Конфиг

Основные параметры в `config.yaml`:

```yaml
model:
  type: "gigaam_e2e"  # gigaam_e2e / gigaam / whisper

audio:
  silence_duration: 2.5  # секунд тишины для конца сегмента

vad:
  threshold: 0.5  # чувствительность, 0.1–0.9
```

Модели:

| Тип | Описание |
|-----|----------|
| `gigaam_e2e` | GigaAM v3 — рекомендуется для русского. Числа цифрами, есть пунктуация. |
| `gigaam` | GigaAM v3 без нормализации. |
| `whisper` | Whisper large-v3. Нужна GPU с 8+ GB. |
