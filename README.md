# 동시통역 (영어 → 한국어 실시간 자막)

마이크로 들어오는 말을 **단어 단위로 바로 화면에 띄우고**, 말하는 도중에도 번역을 계속 다시 조합해 보여주다가, 문장이 끝나면 고품질 번역으로 확정합니다.
- 영어 → 한국어 (기본), 한국어 → 영어 (메뉴에서 전환)
- 무료: 음성인식과 실시간 임시 번역은 이 PC에서(한도 없음), 문장 최종 번역만 Groq 무료 티어

```
마이크 ─▶ 로컬 스트리밍 음성인식 (Kroko, 문장부호까지) ─▶ 단어가 들리는 즉시 표시
              │ 말하는 중 (0.6초마다)  ─▶ 로컬 번역 NLLB-600M 로 임시 번역 (흐린 글씨, 한도 없음)
              │ 문장 끝 (마침표 후 다음 문장 시작, 또는 0.6초 쉼)
              ▼
        Groq gpt-oss-120b 최종 번역 (흰 글씨) ─ 한도가 차면 20b → Gemini → 로컬 번역으로 대체
```

## 설치

1. https://console.groq.com/keys 에서 무료 API 키 발급
2. 설치 — **Anaconda가 아닌 일반 Python 3.12**가 필요합니다. Anaconda에 들어 있는 오래된 MSVC 런타임 때문에 PySide6 최신판, CTranslate2, sentencepiece가 충돌합니다. 아래처럼 `uv`로 독립 Python을 받아 쓰면 시스템은 그대로 둘 수 있어요.
   ```bash
   pip install uv
   uv python install 3.12
   uv venv .venv --python 3.12 --python-preference only-managed
   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
   .venv\Scripts\python download_models.py
   ```
   `download_models.py`는 로컬 모델 약 1.1GB를 `models/`에 받습니다: 영어 음성인식(Kroko), 한국어 음성인식, 번역(NLLB-200 600M).
3. `.env.example`을 `.env`로 복사한 뒤 `GROQ_API_KEY=` 뒤에 키 입력

## 실행

바탕화면 바로가기 만들기 (한 번만, 이후엔 바탕화면의 "동시통역"을 더블클릭):
```bash
powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
```
바로가기로 실행하면 터미널 창 없이 뜨고, 로그는 `logs/app.log`에 남습니다.

터미널에서 직접 실행 (프로젝트 폴더에서):
```bash
.venv\Scripts\python main.py
```

### 자막 창 사용법
- 오른쪽 위 버튼: **일시정지/시작**, **⋯ 설정**(언어, 듣기, 글자 크기, 배경, 기록 폴더), **✕ 닫기** (Esc 도 닫기)
- **⋯ → 듣기 → PC 소리 (영상·회의)**: 유튜브·넷플릭스·줌처럼 컴퓨터에서 나오는 소리를 통역합니다 (스테레오 믹스 설정 불필요). 선택은 다음 실행 때도 유지됩니다. 에어팟·이어폰을 연결하는 등 Windows 기본 출력 장치가 바뀌면 1초 안에 자동으로 따라갑니다 (상태 줄에 현재 장치 표시). PC 볼륨이 작아도 자동으로 음량을 보정하지만, 음소거나 볼륨 0이면 들리지 않습니다
- 드래그로 이동, 오른쪽 아래 모서리로 크기 조절. 위치·크기·글자 크기는 다음 실행 때 그대로 유지
- 위쪽에 오늘 남은 무료 요청 수가 표시됩니다 (`번역 950/1000 · 실시간 990/1000 · 인식 1900/2000`)

## 무료 한도 관리

음성인식과 실시간 임시 번역은 이 PC에서 돌아가서 **한도를 전혀 쓰지 않습니다.**
Groq는 문장이 끝날 때 최종 번역 1회(+ 말이 멈췄을 때 Whisper로 문장 다듬기 1회)만 씁니다.

- Groq 무료 티어는 모델마다 한도가 따로 있습니다 (대략 번역 모델 하루 1000회·분당 30회, Whisper 하루 2000회·분당 20회)
- 앱은 응답 헤더로 남은 한도를 추적해 미리 속도를 줄이고, 자막 창 위쪽에 남은 횟수를 표시합니다
- 최종 번역은 120b → 20b → (Gemini 키가 있으면) Gemini → 로컬 번역 순으로 시도합니다. 한도가 다 차도 번역은 계속 나옵니다

더 아끼려면 `.env`에서 `FINAL_STT=local`로 바꾸면 Whisper를 쓰지 않습니다.

## 통역 기록 (로그)

모든 통역 결과는 `logs/날짜.jsonl`에 한 줄씩 저장됩니다.
```json
{"time": "...", "src_lang": "en", "tgt_lang": "ko", "source": "...", "source_local": "...", "translation": "...", "latency_ms": 950, "model": "openai/gpt-oss-120b"}
```
`.env`에서 `SAVE_AUDIO=1`로 설정하면 음성(wav)도 `logs/audio/`에 같이 저장됩니다.
나중에 번역 품질 개선용 데이터(용어집, 예시 문장, 파인튜닝)로 쓸 수 있습니다.

## 참고

- 로컬 모델 라이선스: NLLB-200은 CC-BY-NC-4.0 (비상업용), Kroko는 https://huggingface.co/Banafo/Kroko-ASR 참고
