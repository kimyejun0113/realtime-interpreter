"""Downloads the local streaming speech-recognition model into models/ (run once)."""
import sys
import tarfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).with_name("models")
MODELS = {
    "en": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    "ko": "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
}
BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"


def download(name: str) -> None:
    target = MODELS_DIR / name
    if target.exists():
        print(f"이미 있음: {target}")
        return
    MODELS_DIR.mkdir(exist_ok=True)
    archive = MODELS_DIR / f"{name}.tar.bz2"
    print(f"다운로드 중: {name} …")

    def progress(blocks, block_size, total):
        if total > 0:
            print(f"\r  {min(100, blocks * block_size * 100 // total)}%", end="", flush=True)

    urllib.request.urlretrieve(BASE_URL + archive.name, archive, progress)
    print("\n압축 해제 중 …")
    with tarfile.open(archive) as tar:
        tar.extractall(MODELS_DIR, filter="data")
    archive.unlink()
    print(f"완료: {target}")


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["en"]:
        download(MODELS[lang])
