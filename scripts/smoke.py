"""Smoke test: launch the real app on recorded speech and make sure it translates without crashing.

Unit tests can't catch everything (audio threads, Qt, model loading), so this starts main.py as a
separate process exactly like the desktop shortcut does, feeds it a WAV instead of the microphone,
and checks the exit code.

    .venv\\Scripts\\python scripts\\smoke.py              # tests/data/en.wav and ko.wav, offline
    .venv\\Scripts\\python scripts\\smoke.py --online     # also use the configured API engines
    .venv\\Scripts\\python scripts\\smoke.py my.wav       # any 16 kHz mono 16-bit WAV
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAVS = [ROOT / "tests" / "data" / "en.wav", ROOT / "tests" / "data" / "ko.wav"]


def run(wav: Path, online: bool) -> bool:
    cmd = [sys.executable, str(ROOT / "main.py"), "--smoke", str(wav)] + ([] if online else ["--offline"])
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=180)
    except subprocess.TimeoutExpired:
        print(f"FAIL {wav.name}: did not exit within 180 s")
        return False
    verdict = [line for line in proc.stdout.splitlines() if line.startswith("SMOKE")]
    ok = proc.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'} {wav.name} ({time.perf_counter() - t0:.0f} s): {verdict[-1] if verdict else '(no verdict)'}")
    if not ok:
        print((proc.stdout + proc.stderr)[-3000:])
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="*", type=Path)
    parser.add_argument("--online", action="store_true", help="use the API engines from .env (uses quota)")
    args = parser.parse_args()
    results = [run(w, args.online) for w in (args.wavs or DEFAULT_WAVS)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
