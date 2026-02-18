from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Doc

from models.schema import ModalityResult


CERTAINTY_MODAL_TERMS = {
    "must",
    "will",
    "cannot",
    "certainly",
    "definitely",
    "undoubtedly",
    "necessarily",
}

HEDGE_MODAL_TERMS = {
    "may",
    "might",
    "could",
    "perhaps",
    "possibly",
    "probably",
    "seem",
    "appears",
}


@dataclass(frozen=True)
class ModalityStats:
    certainty_count: int
    hedge_count: int


class ModalityAnalyzer:
    def analyze(self, doc: Doc) -> ModalityResult:
        certainty_count = 0
        hedge_count = 0

        for token in doc:
            lemma = token.lemma_.lower()
            lower = token.text.lower()
            candidate = lemma if lemma else lower

            if candidate in CERTAINTY_MODAL_TERMS or lower in CERTAINTY_MODAL_TERMS:
                certainty_count += 1
            if candidate in HEDGE_MODAL_TERMS or lower in HEDGE_MODAL_TERMS:
                hedge_count += 1

        denominator = certainty_count + hedge_count
        certainty_index = certainty_count / denominator if denominator > 0 else 0.5

        return ModalityResult(
            certainty_count=certainty_count,
            hedge_count=hedge_count,
            certainty_index=float(certainty_index),
        )


__all__ = [
    "ModalityAnalyzer",
    "CERTAINTY_MODAL_TERMS",
    "HEDGE_MODAL_TERMS",
]
