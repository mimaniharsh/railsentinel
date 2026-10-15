"""Hallucination detection for generated answers.

Two independent checks, both against the evidence the answer was supposed to
use (tool results, retrieved policy passages and the user's own message):

1. Numeric grounding: every amount, percentage, duration or count stated in the
   answer must appear in the evidence. Invented numbers are the most harmful
   hallucinations in a refund or schedule answer and are easy to verify.
2. Sentence support: each factual sentence must be close (TF-IDF cosine or
   content-word overlap) to some evidence sentence.

An optional LLM judge (Gemini) adds a faithfulness rating on a sample of
answers when an API key is configured.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NUMBER_RE = re.compile(r"(?<![\w.])(?:₹\s*|rs\.?\s*|inr\s*)?(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?!\w)",
                       re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
META_PATTERNS = re.compile(
    r"^(?:i can help|i could not|i couldn't|i don't|i do not|please (?:confirm|check|share|tell)|sources?:|"
    r"would you like|let me know|you can (?:ask|also)|here is|here's|sorry|note:|for the latest)", re.IGNORECASE)


def normalise_number(whole: str, frac: str | None) -> str:
    value = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
    return str(int(value)) if value == int(value) else f"{value:.2f}".rstrip("0").rstrip(".")


def numbers_in(text: str) -> set[str]:
    return {normalise_number(m.group(1), m.group(2)) for m in NUMBER_RE.finditer(text or "")}


def _sentences(text: str) -> list[str]:
    parts = [s.strip(" -•*\t") for s in SENTENCE_SPLIT.split(text or "")]
    return [s for s in parts if len(s.split()) >= 4]


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2}


def check_grounding(answer: str, evidence: list[str], question: str = "", support_threshold: float = 0.3) -> dict:
    evidence_text = "\n".join(evidence)
    allowed_numbers = numbers_in(evidence_text) | numbers_in(question)
    # Requests and pleasantries ("please share your 10-digit PNR") are not factual claims.
    claim_text = " ".join(s for s in (x.strip(" -•*\t") for x in SENTENCE_SPLIT.split(answer or ""))
                          if s and not META_PATTERNS.match(s) and not s.endswith("?"))
    answer_numbers = numbers_in(claim_text)
    # Single-digit counts ("1 alternative", "2 legs") are rarely the risky claims and appear in
    # ordinary phrasing, so they are not required to be grounded.
    unsupported_numbers = sorted(n for n in answer_numbers - allowed_numbers if not (n.isdigit() and int(n) < 10))
    claims = [s for s in _sentences(answer) if not META_PATTERNS.match(s) and not s.endswith("?")]
    ev_sentences = [s for s in _sentences(evidence_text)] or [evidence_text or " "]
    unsupported_sentences: list[str] = []
    if claims and evidence_text.strip():
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(ev_sentences + claims)
        sims = cosine_similarity(vec.transform(claims), vec.transform(ev_sentences))
        ev_words = [_content_words(s) for s in ev_sentences]
        for i, claim in enumerate(claims):
            words = _content_words(claim)
            overlap = max((len(words & ew) / len(words) for ew in ev_words), default=0) if words else 1.0
            if sims[i].max() < support_threshold and overlap < 0.6:
                unsupported_sentences.append(claim)
    elif claims:
        unsupported_sentences = list(claims)
    supported = len(claims) - len(unsupported_sentences)
    score = 1.0 if not claims else supported / len(claims)
    if unsupported_numbers:
        score = min(score, 0.5)
    relevance = 0.0
    if question.strip() and answer.strip():
        try:
            vec = TfidfVectorizer(stop_words="english").fit([question, answer])
            relevance = float(cosine_similarity(vec.transform([question]), vec.transform([answer]))[0, 0])
        except ValueError:
            relevance = 0.0
    return {"grounding_score": round(score, 3), "unsupported_numbers": unsupported_numbers,
            "unsupported_sentences": unsupported_sentences[:5], "claims_checked": len(claims),
            "hallucination": bool(unsupported_numbers) or score < 0.6, "relevance": round(relevance, 3)}


JUDGE_SYSTEM = ("You audit a railway customer-support answer for faithfulness. Use only the EVIDENCE. "
                "Reply with JSON only: {\"faithful\": true|false, \"score\": 1-5, \"unsupported_claims\": [..]}. "
                "Score 5 means every claim is supported by the evidence.")


def llm_judge(llm, question: str, answer: str, evidence: list[str]) -> dict | None:
    if llm is None or not llm.enabled:
        return None
    prompt = (f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nEVIDENCE:\n" + "\n---\n".join(evidence)[:12000])
    result = llm.json(JUDGE_SYSTEM, prompt, model=llm.settings.llm_judge_model)
    if not isinstance(result, dict) or "faithful" not in result:
        return None
    return {"faithful": bool(result.get("faithful")), "score": result.get("score"),
            "unsupported_claims": result.get("unsupported_claims", [])[:5],
            "judge_model": llm.settings.llm_judge_model}
