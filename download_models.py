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


_CDN = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"


def _download_file(url: str, dest: Path) -> None:
    import urllib.request
    from tqdm import tqdm
    if dest.exists():
        print(f"  уже скачан: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  скачивание {dest.name}...")
    with urllib.request.urlopen(url) as src, open(dest, "wb") as out:
        total = int(src.info().get("Content-Length", 0))
        with tqdm(total=total, unit="iB", unit_scale=True, ncols=70) as bar:
            while chunk := src.read(8192):
                out.write(chunk)
                bar.update(len(chunk))


def download_e2e_ctc() -> None:
    models_dir = Path("models/gigaam")
    print("Скачивание GigaAM v3 e2e CTC...")
    _download_file(f"{_CDN}/v3_e2e_ctc.ckpt", models_dir / "v3_e2e_ctc.ckpt")
    _download_file(f"{_CDN}/v3_e2e_ctc_tokenizer.model", models_dir / "v3_e2e_ctc_tokenizer.model")
    print("GigaAM v3 e2e CTC скачана: models/gigaam/")


def download_e2e_rnnt() -> None:
    models_dir = Path("models/gigaam")
    print("Скачивание GigaAM v3 e2e RNNT...")
    _download_file(f"{_CDN}/v3_e2e_rnnt.ckpt", models_dir / "v3_e2e_rnnt.ckpt")
    _download_file(f"{_CDN}/v3_e2e_rnnt_tokenizer.model", models_dir / "v3_e2e_rnnt_tokenizer.model")
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
