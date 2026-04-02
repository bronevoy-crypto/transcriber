# Meeting Transcriber

Консольное приложение для локальной транскрибации совещаний. Захватывает системный звук и микрофон, нарезает по паузам, транскрибирует речь, определяет дикторов и сохраняет результат в JSON с таймкодами.

Работает полностью офлайн — интернет нужен только при первом запуске для загрузки моделей.

## Что нужно

- Windows 10/11
- Python 3.10+
- NVIDIA GPU (рекомендуется) — ускоряет транскрибацию и диаризацию. GigaAM e2e работает и на CPU.
- Аккаунт на [huggingface.co](https://huggingface.co) — для диаризации (определения дикторов)

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
pip install --no-deps git+https://github.com/salute-developers/GigaAM.git
```

**3. Скопировать конфиг**
```bash
copy config.example.yaml config.yaml
```

## Настройка диаризации (определение дикторов)

Диаризация разделяет запись по участникам совещания (SPEAKER_00, SPEAKER_01 и т.д.).

**Шаг 1.** Зарегистрироваться на [huggingface.co](https://huggingface.co)

**Шаг 2.** Принять условия использования моделей — открыть каждую ссылку и нажать **Agree**:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Шаг 3.** Создать токен: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Role: Read

**Шаг 4.** Вставить токен в `config.yaml`:
```yaml
diarization:
  enabled: true
  hf_token: "hf_ВАШ_ТОКЕН_ЗДЕСЬ"
  device: "cuda"   # или "cpu" если нет GPU
```

Без токена диаризация не запустится. Чтобы отключить — поставить `enabled: false`.

## Запуск

```bash
python main.py
```

Запускать до начала встречи. Приложение захватывает системный звук и микрофон одновременно.

Остановить — **Ctrl+C**. После остановки автоматически запустится диаризация полной записи (если включена).

Результат появится в папке `meetings/`.

## Как выглядит результат

Файл называется по дате и времени: `2026-03-31_08-19-47.json`

```json
{
  "meeting_date": "2026-03-31T08:19:47",
  "segments": [
    {
      "start": 3.18,
      "end": 12.69,
      "speaker": "SPEAKER_00",
      "text": "Добрый день, коллеги. Начинаем планирование спринта."
    },
    {
      "start": 13.75,
      "end": 28.40,
      "speaker": "SPEAKER_01",
      "text": "Велосити за последние три спринта — в среднем 35 стори поинтов."
    }
  ]
}
```

## Настройки

Все параметры в `config.yaml`. После изменений — перезапустить приложение.

### Выбор модели транскрибации

```yaml
model:
  type: "gigaam_e2e"   # gigaam_e2e, gigaam или whisper
```

| Модель | Описание |
|--------|----------|
| `gigaam_e2e` | GigaAM v3 e2e — рекомендуется. Высокая точность на русском, нормализует текст (числа цифрами, пунктуация). |
| `gigaam` | GigaAM v3 CTC/RNNT — без нормализации текста. |
| `whisper` | OpenAI Whisper large-v3 — требует NVIDIA GPU с 8+ GB VRAM. |

### Чувствительность распознавания речи

```yaml
vad:
  threshold: 0.5          # 0.1–0.9. Ниже = чувствительнее

audio:
  silence_duration: 2.5   # Секунд тишины для завершения сегмента
```

### GPU / CPU

```yaml
diarization:
  device: "cuda"   # "cuda" — GPU (быстрее), "cpu" — без GPU
```

Для транскрибации устройство выбирается автоматически по наличию GPU.
