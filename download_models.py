"""
Скачивание моделей GigaAM v3.

Запуск:
    python download_models.py            # gigaam_e2e CTC (дефолт)
    python download_models.py --rnnt     # gigaam_e2e RNNT
    python download_models.py --sherpa   # sherpa-onnx CTC (альтернатива)
    python download_models.py --all      # все модели
"""
import argparse
from pathlib import Path


def download_e2e_ctc() -> None:
    import gigaam
    models_dir = Path("models/gigaam")
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Скачивание GigaAM v3 e2e CTC...")
    gigaam.load_model("v3_e2e_ctc", download_root=str(models_dir))
    print("GigaAM v3 e2e CTC скачана: models/gigaam/")


def download_e2e_rnnt() -> None:
    import gigaam
    models_dir = Path("models/gigaam")
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Скачивание GigaAM v3 e2e RNNT...")
    gigaam.load_model("v3_e2e_rnnt", download_root=str(models_dir))
    print("GigaAM v3 e2e RNNT скачана: models/gigaam/")


def download_sherpa_ctc() -> None:
    from huggingface_hub import hf_hub_download
    print("Скачивание GigaAM v3 sherpa-onnx CTC...")
    for filename in ["model.int8.onnx", "tokens.txt"]:
        hf_hub_download(
            repo_id="csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16",
            filename=filename,
            local_dir="models/gigaam-v3",
        )
    print("GigaAM v3 sherpa-onnx CTC скачана: models/gigaam-v3/")


def download_sherpa_rnnt() -> None:
    from huggingface_hub import hf_hub_download
    print("Скачивание GigaAM v3 sherpa-onnx RNNT...")
    for filename in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]:
        hf_hub_download(
            repo_id="csukuangfj/sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16",
            filename=filename,
            local_dir="models/gigaam-v3-rnnt",
        )
    print("GigaAM v3 sherpa-onnx RNNT скачана: models/gigaam-v3-rnnt/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnnt", action="store_true", help="Скачать e2e RNNT вместо CTC")
    parser.add_argument("--sherpa", action="store_true", help="Скачать sherpa-onnx версию")
    parser.add_argument("--all", action="store_true", help="Скачать все модели")
    args = parser.parse_args()

    if args.all:
        download_e2e_ctc()
        download_e2e_rnnt()
        download_sherpa_ctc()
        download_sherpa_rnnt()
    elif args.sherpa and args.rnnt:
        download_sherpa_rnnt()
    elif args.sherpa:
        download_sherpa_ctc()
    elif args.rnnt:
        download_e2e_rnnt()
    else:
        download_e2e_ctc()
