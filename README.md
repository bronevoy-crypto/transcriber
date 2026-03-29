# Meeting Transcriber

Консольное приложение для локальной транскрибации совещаний на лету. Захватывает системный звук, нарезает по паузам, транскрибирует и пишет в JSON с таймкодами.

Работает полностью офлайн, никуда ничего не отправляет.

## Что нужно

- Windows 10/11
- Python 3.10+ — скачать с [python.org](https://python.org)
- NVIDIA GPU (8+ GB VRAM)
- ~500 MB свободного места для модели

## Установка

Открыть терминал (Win+R → ввести `cmd` → Enter) и выполнить по очереди:

**1. Скачать проект**
```bash
git clone https://github.com/HRYNdev/transcriber
cd transcriber
```

**2. Установить зависимости**
```bash
pip install -r requirements.txt
```

**3. Скачать модель GigaAM (~500 MB)**
```bash
python download_models.py
```

## Запуск

```bash
python main.py
```

Запускать до начала встречи. Приложение само захватывает системный звук — VB-Cable не нужен.

Остановить — **Ctrl+C**. Результат появится в папке `meetings/`.

## Как выглядит результат

Файл называется по дате и времени: `2024-01-15_10-30-00.json`

```json
{
  "meeting_date": "2024-01-15T10:30:00",
  "segments": [
    {
      "start": 5.48,
      "end": 12.30,
      "speaker": "SPEAKER_00",
      "text": "Добрый день, начинаем совещание"
    },
    {
      "start": 14.10,
      "end": 20.55,
      "speaker": "SPEAKER_01",
      "text": "Да, давайте начнём с первого пункта"
    }
  ]
}
```

## Настройки

Все параметры меняются в файле `config.yaml`. После изменений перезапустить приложение.

```yaml
model:
  type: "gigaam"   # gigaam или whisper

vad:
  threshold: 0.5          # чувствительность к голосу (0.1–0.9, ниже = чувствительнее)
  silence_duration: 1.5   # секунд тишины для завершения сегмента
```

## Смена модели

По умолчанию стоит GigaAM v3 — русская модель, WER ~9%. Для переключения на Whisper изменить в `config.yaml`:

```yaml
model:
  type: "whisper"
```

Для более точной версии GigaAM (WER ~8%):
```bash
python download_models.py --rnnt
```
Затем в `config.yaml` изменить `gigaam.type: "transducer"`.
