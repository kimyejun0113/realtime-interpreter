# 동시통역 (영어 → 한국어 실시간 자막)

마이크로 들어오는 말을 듣고 언어를 자동 감지해 화면 자막으로 바로 통역합니다.
- 영어(및 한국어가 아닌 언어) → 한국어
- 한국어 → 영어
- 무료: Groq 무료 티어 사용 (카드 등록 불필요)

```
마이크 → VAD(말 끊김 감지, 로컬) → Groq Whisper(음성인식+언어감지) → Groq LLM(스트리밍 번역) → 자막 창
```
말이 끊기고 약 1~1.5초 안에 번역 자막이 뜨기 시작합니다.

## 설치

1. https://console.groq.com/keys 에서 무료 API 키 발급
2. 설치
   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. `.env.example`을 `.env`로 복사한 뒤 `GROQ_API_KEY=` 뒤에 키 입력

## 실행

```bash
.venv\Scripts\python main.py
```

- 자막 창은 **왼쪽 드래그로 이동**, **오른쪽 클릭으로 메뉴**(시작/일시정지, 마이크 선택, 글자 크기, 투명도, 종료)
- 터미널에는 문장마다 단계별 지연시간(ms)이 출력됩니다

## 통역 기록 (로그)

모든 통역 결과는 `logs/날짜.jsonl`에 한 줄씩 저장됩니다.
```json
{"time": "...", "src_lang": "en", "tgt_lang": "ko", "source": "...", "translation": "...", "stt_ms": 410, "latency_ms": 1180, ...}
```
`.env`에서 `SAVE_AUDIO=1`로 설정하면 음성(wav)도 `logs/audio/`에 같이 저장됩니다.
나중에 번역 품질 개선용 데이터(용어집, 예시 문장, 파인튜닝)로 쓸 수 있습니다.

## 설정 (.env)

| 항목 | 설명 |
|---|---|
| `TRANSLATE_MODEL` | 번역 모델 (기본 `openai/gpt-oss-120b`) |
| `TRANSLATE_PROVIDER` | `groq` 또는 `gemini` (Gemini 무료 키 사용 시) |
| `VAD_SILENCE_MS` | 이만큼 조용하면 끊어서 통역 (기본 400) |
| `MAX_SEGMENT_SEC` | 계속 말해도 이 길이에서 강제 분할 (기본 5) |
| `LOCAL_STT_FALLBACK` | `1`이면 Groq 한도 초과 시 로컬 음성인식 사용 (`pip install faster-whisper` 필요) |

## 참고

- Anaconda가 설치된 PC에서는 PySide6 6.8 이상이 DLL 충돌을 일으켜 6.7.3으로 고정했습니다.
- 유튜브/줌 소리를 통역하려면 스피커 소리가 마이크로 들어가게 하거나, 메뉴에서 "스테레오 믹스" 장치를 선택하세요.
