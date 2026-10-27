# 실시간 동시통역 (Windows, Python)

마이크·PC 소리·특정 앱 소리를 듣고 영어→한국어(또는 한국어→영어) 자막을 띄우는 데스크톱 앱.

## 구조
- `main.py`: Qt 앱 시작, 이벤트 연결, `--smoke WAV [--offline]` 스모크 모드
- `pipeline.py`: 오디오 → 로컬 스트리밍 음성인식 → 문장 분할(`_commit_point`), 짧은 조각 합치기(`_next_final`) → 임시 번역(로컬 NLLB) / 최종 번역(엔진 목록) → 기록
- `audio/capture.py`: 마이크, PC 소리(WASAPI 루프백 + 자동 음량), 앱 소리(`app_loopback.py`, 프로세스 루프백), WAV 파일 입력
- `stt/streaming.py`: sherpa-onnx 스트리밍 인식(영어 Kroko, 한국어 zipformer). `DualRecognizer`가 자동 모드에서 두 모델을 동시에 돌려 최근 토큰 확신도(`ys_probs`)로 언어를 고른다. 임계값은 tests/test_dual.py와 실제 녹음으로 맞춘 값이라 바꾸면 둘 다 다시 확인
- `translate/`: `base.py` 프롬프트(KO_STYLE, Context), `engines.py` 엔진 목록(.env `TRANSLATE_ENGINES`), `openai_compat.py` LLM, `mt_services.py` DeepL/Azure, `local_mt.py` NLLB, `memo.py` 대화 메모
- `quota.py`: 응답 헤더로 무료 한도를 추적해 미리 속도를 줄임
- `ui/overlay.py`: 자막 창(문장 유지 시간, 창에 맞춰 줄이기), `ui/settings.py` 메뉴

## 실행 환경 (중요)
- 가상환경은 `.venv`, 파이썬은 프로젝트 안 `.python`(standalone 3.12). **Anaconda 기반 파이썬으로 만들지 말 것**: 오래된 MSVC 런타임 때문에 PySide6 6.8+, ctranslate2, sentencepiece가 죽는다.
- 폴더 이름에 한글(`동시번역`)이 있다. sentencepiece처럼 경로를 못 여는 라이브러리는 bytes로 넘긴다.
- Claude 데스크톱 앱에서 `%APPDATA%`에 쓰면 샌드박스 가상 폴더로 간다. 실행에 필요한 것은 프로젝트 폴더 안에 둔다. 실제 실행 확인은 `explorer.exe` 로 바탕화면 바로가기를 연다.
- 모델은 `python download_models.py`(models/, git 제외). API 키는 `.env`(git 제외).

## 커밋 전
```
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\smoke.py
```
둘 다 통과해야 한다. 스모크는 실제 앱을 WAV 입력으로 띄워 번역이 나오고 예외가 없는지 본다(기본 오프라인, `--online`이면 API 사용). 사람이 확인할 항목은 `SMOKE.md`.
