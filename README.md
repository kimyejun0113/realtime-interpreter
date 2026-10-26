# 동시통역 (영어 → 한국어 실시간 자막)

마이크로 들어오는 말을 **단어 단위로 바로 화면에 띄우고**, 말하는 도중에도 번역을 계속 다시 조합해 보여주다가, 문장이 끝나면 고품질 번역으로 확정합니다.
- 영어 → 한국어 (기본), 한국어 → 영어 (메뉴에서 전환)
- 무료: 음성인식은 로컬, 번역은 Groq 무료 티어

```
마이크 ─▶ 로컬 스트리밍 음성인식 (sherpa-onnx, 무료·무제한) ─▶ 단어가 들리는 즉시 표시
              │ 말하는 중 (2초마다, 새 단어 4개 이상)  ─▶ gpt-oss-20b 로 임시 번역 (흐린 글씨)
              │ 문장 끝 (0.6초 쉼)
              ▼
        Groq Whisper 로 문장 다듬기 (한도 여유 있을 때만) ─▶ gpt-oss-120b 최종 번역 (흰 글씨)
```

## 설치

1. https://console.groq.com/keys 에서 무료 API 키 발급
2. 설치
   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python download_models.py en ko
   ```
   `download_models.py`는 로컬 음성인식 모델(영어 ~330MB, 한국어)을 `models/`에 받습니다.
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
- 오른쪽 위 버튼: **일시정지/시작**, **⋯ 설정**(언어, 마이크, 글자 크기, 배경, 기록 폴더), **✕ 닫기** (Esc 도 닫기)
- 드래그로 이동, 오른쪽 아래 모서리로 크기 조절. 위치·크기·글자 크기는 다음 실행 때 그대로 유지
- 위쪽에 오늘 남은 무료 요청 수가 표시됩니다 (`번역 950/1000 · 실시간 990/1000 · 인식 1900/2000`)

## 무료 한도 관리

Groq 무료 티어는 **모델마다 따로** 한도가 있습니다 (대략 번역 모델 하루 1000회·분당 30회, Whisper 하루 2000회·분당 20회).
앱은 응답 헤더로 남은 한도를 추적해서 미리 속도를 줄입니다.

- 음성인식은 로컬이라 한도를 쓰지 않음. Whisper는 문장이 끝날 때 한 번, 분당 한도에 여유가 있을 때만 사용
- 실시간 임시 번역은 최종 번역과 **다른 모델**(20b)을 써서 한도를 나눠 씀. 하루 한도가 30% 아래로 떨어지면 자동으로 멈추고 최종 번역에 양보
- 최종 번역은 120b → 20b → (Gemini 키가 있으면) Gemini 순으로 시도. 한도 초과(429) 시 안내된 시간만큼 해당 모델을 쉬게 함
- 모두 막히면 영어 원문만 계속 표시

더 아끼려면 `.env`에서:
- `PARTIAL_INTERVAL_SEC` / `PARTIAL_MIN_NEW_WORDS`를 늘리기, 또는 `LIVE_TRANSLATION=0` (문장 끝에만 번역)
- `FINAL_STT=local` (Whisper 사용 안 함)

## 통역 기록 (로그)

모든 통역 결과는 `logs/날짜.jsonl`에 한 줄씩 저장됩니다.
```json
{"time": "...", "src_lang": "en", "tgt_lang": "ko", "source": "...", "source_local": "...", "translation": "...", "latency_ms": 950, "model": "openai/gpt-oss-120b"}
```
`.env`에서 `SAVE_AUDIO=1`로 설정하면 음성(wav)도 `logs/audio/`에 같이 저장됩니다.
나중에 번역 품질 개선용 데이터(용어집, 예시 문장, 파인튜닝)로 쓸 수 있습니다.

## 참고

- Anaconda가 설치된 PC에서는 PySide6 6.8 이상이 DLL 충돌을 일으켜 6.7.3으로 고정했습니다.
- 유튜브/줌 소리를 통역하려면 스피커 소리가 마이크로 들어가게 하거나, 설정의 마이크에서 "스테레오 믹스" 장치를 선택하세요.
