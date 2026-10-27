"""Local machine translation (NLLB-200 600M via CTranslate2). Free, unlimited, ~0.2-0.6 s on CPU."""
from pathlib import Path
from typing import Iterator

from .base import Translator

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "nllb-200-distilled-600M-ct2-int8"

NLLB_CODES = {
    "en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans", "es": "spa_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl", "vi": "vie_Latn", "th": "tha_Thai",
    "id": "ind_Latn", "pt": "por_Latn", "it": "ita_Latn", "ar": "arb_Arab", "hi": "hin_Deva",
}


def available() -> bool:
    return (MODEL_DIR / "model.bin").is_file()


class LocalTranslator(Translator):
    model = name = "local-nllb"

    def __init__(self, threads: int = 4):
        import ctranslate2
        import sentencepiece

        self.translator = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8", intra_threads=threads)
        # pass bytes: sentencepiece can't open paths with non-ASCII characters (e.g. "동시번역") on Windows
        self.sp = sentencepiece.SentencePieceProcessor(model_proto=(MODEL_DIR / "sentencepiece.bpe.model").read_bytes())

    def stream(self, text, src, tgt, context, partial=False) -> Iterator[str]:
        source = [NLLB_CODES.get(src, "eng_Latn"), *self.sp.encode(text, out_type=str), "</s>"]
        target = NLLB_CODES.get(tgt, "kor_Hang")
        result = self.translator.translate_batch(
            [source], target_prefix=[[target]], beam_size=1, max_decoding_length=len(source) * 2 + 8, repetition_penalty=1.1
        )
        yield self.sp.decode(result[0].hypotheses[0][1:])
