"""Pluggable translation backends for the multilingual attack.

Every backend implements ``Translator.translate(text, source, target)`` using
ISO-639-1 language codes (``en``, ``hi``, ``zu`` ...). ``translate_batch`` has a
default loop implementation that backends may override for efficiency.

``CachedTranslator`` wraps any backend with a JSON disk cache so each text is
translated exactly once — runs are reproducible and repeat work is avoided.

Backends
--------
* ``nllb``   — Meta NLLB-200 running locally (default; offline, deterministic)
* ``google`` — Google Translate via ``deep-translator`` (no key; may be blocked)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path


class TranslationError(RuntimeError):
    """Raised when a backend cannot translate a text."""


class Translator(ABC):
    """Abstract translation backend."""

    name: str = "base"

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Translate ``text`` from ISO-639-1 ``source`` to ``target``."""

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        return [self.translate(t, source, target) for t in texts]


# --------------------------------------------------------------------------- #
# Text segmentation (shared helper)
# --------------------------------------------------------------------------- #
# Split after . ! ? or the Devanagari danda (।), but not after initialisms such as "U.S." / "U.K."
# (otherwise "a U.S. president" is translated as two broken sentences).
_SENTENCE_END = re.compile(r"(?<=[.!?।])(?<!\b[A-Z]\.[A-Z]\.)\s+")


def split_into_segments(text: str) -> list[tuple[str, str]]:
    """Split text into ``(segment, separator)`` pairs.

    Paragraph/line breaks are preserved as separators so the translated text
    keeps its original layout. Joining ``seg + sep`` for every pair rebuilds
    the structure of the input.
    """
    pairs: list[tuple[str, str]] = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        line_sep = "\n" if li < len(lines) - 1 else ""
        sentences = [s for s in _SENTENCE_END.split(line.strip()) if s]
        if not sentences:
            pairs.append(("", line_sep))
            continue
        for si, sent in enumerate(sentences):
            pairs.append((sent, " " if si < len(sentences) - 1 else line_sep))
    return pairs


# --------------------------------------------------------------------------- #
# NLLB-200 (local)
# --------------------------------------------------------------------------- #
class NLLBBackend(Translator):
    """Meta NLLB-200 running locally via ``transformers``.

    Long inputs are split into sentences (NLLB quality degrades beyond ~200
    tokens), translated in batches with deterministic beam search, then
    re-joined preserving line breaks.
    """

    name = "nllb"

    LANG_CODES: dict[str, str] = {
        "en": "eng_Latn", "hi": "hin_Deva", "zu": "zul_Latn", "fr": "fra_Latn",
        "gd": "gla_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "ur": "urd_Arab",
        "es": "spa_Latn", "de": "deu_Latn", "zh": "zho_Hans", "ar": "arb_Arab",
    }

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        num_beams: int = 4,
        batch_size: int = 8,
        max_input_tokens: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.name = f"nllb:{model_name.split('/')[-1]}"
        self._tokenizer = None
        self._model = None

    # Lazy loading: importing this module must not pull in torch.
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()

    @classmethod
    def to_nllb_code(cls, lang: str) -> str:
        if lang == "auto":
            raise ValueError("NLLB needs an explicit source language; 'auto' is not supported.")
        if lang in cls.LANG_CODES.values():
            return lang
        try:
            return cls.LANG_CODES[lang]
        except KeyError:
            raise ValueError(f"Language '{lang}' not mapped for NLLB. Add it to LANG_CODES.") from None

    def _generate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        import torch

        tok, model = self._tokenizer, self._model
        tok.src_lang = src
        outputs: list[str] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_input_tokens).to(self.device)
            max_new = min(512, int(enc["input_ids"].shape[1] * 2) + 16)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                    num_beams=self.num_beams,
                    do_sample=False,
                    max_new_tokens=max_new,
                    max_length=None,
                )
            outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return outputs

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        src, tgt = self.to_nllb_code(source), self.to_nllb_code(target)
        if src == tgt:
            return list(texts)
        self._load()

        # Flatten all texts into sentences, translate together, then rebuild.
        structure = [split_into_segments(t) for t in texts]
        flat = [seg for pairs in structure for seg, _ in pairs if seg]
        translated = iter(self._generate(flat, src, tgt) if flat else [])

        results = []
        for pairs in structure:
            parts = [(next(translated) if seg else "") + sep for seg, sep in pairs]
            results.append("".join(parts).strip())
        return results

    def translate(self, text: str, source: str, target: str) -> str:
        return self.translate_batch([text], source, target)[0]


# --------------------------------------------------------------------------- #
# Google Translate (online)
# --------------------------------------------------------------------------- #
class GoogleTranslateBackend(Translator):
    """Google Translate via ``deep-translator`` (no API key).

    Same backend as Yong et al. (2023). Note: Google may block some networks
    with a CAPTCHA (HTTP 429) — use the ``nllb`` backend in that case.
    """

    name = "google"

    def __init__(self, max_retries: int = 3, backoff_s: float = 2.0) -> None:
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def translate(self, text: str, source: str, target: str) -> str:
        from deep_translator import GoogleTranslator  # lazy: optional dependency

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = GoogleTranslator(source=source, target=target).translate(text)
                if not result:
                    raise TranslationError("empty translation")
                return result
            except Exception as exc:  # network / rate-limit / captcha
                last_exc = exc
                time.sleep(self.backoff_s * (2**attempt))
        raise TranslationError(f"Google translation failed: {last_exc}") from last_exc


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
class CachedTranslator(Translator):
    """Disk-cached wrapper around another ``Translator``."""

    def __init__(self, backend: Translator, cache_path: str | Path) -> None:
        self.backend = backend
        self.name = backend.name
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, text: str, source: str, target: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        return f"{self.backend.name}:{source}>{target}:{digest}"

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(self.cache_path)

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        keys = [self._key(t, source, target) for t in texts]
        with self._lock:
            missing = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if missing:
            new = self.backend.translate_batch(list(missing.values()), source, target)
            with self._lock:
                self._cache.update(zip(missing.keys(), new))
                self._save()
        with self._lock:
            return [self._cache[k] for k in keys]

    def translate(self, text: str, source: str, target: str) -> str:
        return self.translate_batch([text], source, target)[0]

    def is_cached(self, text: str, source: str, target: str) -> bool:
        return self._key(text, source, target) in self._cache


BACKENDS: dict[str, type[Translator]] = {
    "nllb": NLLBBackend,
    "google": GoogleTranslateBackend,
}


def build_translator(
    backend: str = "nllb",
    cache_path: str | Path | None = None,
    **backend_kwargs,
) -> Translator:
    """Factory used by the strategies / evaluator."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown translation backend '{backend}'. Options: {list(BACKENDS)}")
    translator: Translator = BACKENDS[backend](**backend_kwargs)
    return CachedTranslator(translator, cache_path) if cache_path else translator
