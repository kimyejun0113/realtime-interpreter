"""Downloads the local models into models/ (run once).

    python download_models.py            # everything
    python download_models.py en mt      # only some
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).with_name("models")

# streaming speech recognition (sherpa-onnx release archives)
ASR_MODELS = {
    "en": "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    "ko": "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
}
ASR_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

# live translation: NLLB-200 distilled 600M, converted for CTranslate2 (int8, CPU). License: CC-BY-NC-4.0
MT_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
MT_DIR = "nllb-200-distilled-600M-ct2-int8"
MT_FILES = ["config.json", "model.bin", "sentencepiece.bpe.model", "shared_vocabulary.txt"]


def _progress(blocks, block_size, total):
    if total > 0:
        print(f"\r  {min(100, blocks * block_size * 100 // total)}%", end="", flush=True)


def download_asr(name: str) -> None:
    target = MODELS_DIR / name
    if target.exists():
        print(f"이미 있음: {target}")
        return
    archive = MODELS_DIR / f"{name}.tar.bz2"
    print(f"다운로드 중: {name} …")
    urllib.request.urlretrieve(ASR_BASE_URL + archive.name, archive, _progress)
    print("\n압축 해제 중 …")
    with tarfile.open(archive) as tar:
        tar.extractall(MODELS_DIR, filter="data")
    archive.unlink()
    print(f"완료: {target}")


def download_mt() -> None:
    target = MODELS_DIR / MT_DIR
    target.mkdir(exist_ok=True)
    for name in MT_FILES:
        path = target / name
        if path.exists():
            continue
        print(f"다운로드 중: {MT_DIR}/{name} …")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"https://huggingface.co/{MT_REPO}/resolve/main/{name}", tmp, _progress)
        tmp.rename(path)
        print()
    print(f"완료: {target}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    for key in sys.argv[1:] or ["en", "ko", "mt"]:
        if key == "mt":
            download_mt()
        else:
            download_asr(ASR_MODELS[key])
