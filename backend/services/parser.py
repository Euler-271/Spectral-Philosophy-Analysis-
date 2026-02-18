from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from models.schema import AssumptionRecord, AssumptionType


@dataclass(frozen=True)
class ParsedSentence:
    id: str
    text: str
    doc: Doc
    pos_signature: Dict[str, float]


ESSENTIALIST_TERMS = {
    "essentially",
    "inherently",
    "fundamentally",
    "necessarily",
    "intrinsically",
    "naturally",
}

UNIVERSAL_QUANTIFIERS = {
    "all",
    "every",
    "always",
    "none",
    "never",
    "any",
    "everything",
    "nothing",
}

NORMATIVE_TERMS = {
    "should",
    "must",
    "ought",
    "good",
    "bad",
    "right",
    "wrong",
    "ethical",
    "moral",
    "duty",
}

CAUSAL_TERMS = {
    "because",
    "therefore",
    "thus",
    "hence",
    "causes",
    "leads",
    "results",
    "drives",
}

EPISTEMIC_TERMS = {
    "know",
    "believe",
    "assume",
    "evidence",
    "proof",
    "certain",
    "uncertain",
    "justify",
}

TELEOLOGICAL_TERMS = {
    "goal",
    "purpose",
    "optimize",
    "maximize",
    "efficient",
    "utility",
    "instrument",
}

MODAL_TERMS = {
    "must",
    "should",
    "might",
    "may",
    "could",
    "cannot",
    "always",
    "necessarily",
}


@lru_cache(maxsize=2)
def _load_nlp(model_name: str = "en_core_web_sm") -> Language:
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install with: python -m spacy download en_core_web_sm"
        ) from exc


class ParserService:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp = _load_nlp(model_name)

    def parse(self, text: str, min_sentence_chars: int = 6, max_sentences: int = 2048) -> List[ParsedSentence]:
        doc = self.nlp(text)
        sentences: List[ParsedSentence] = []

        for idx, sent in enumerate(doc.sents):
            cleaned = sent.text.strip()
            if len(cleaned) < min_sentence_chars:
                continue
            signature = self._pos_signature(sent)
            sentences.append(
                ParsedSentence(
                    id=f"sent_{idx}",
                    text=cleaned,
                    doc=sent.as_doc(),
                    pos_signature=signature,
                )
            )
            if len(sentences) >= max_sentences:
                break

        return sentences

    def extract_assumptions(self, parsed_sentence: ParsedSentence) -> List[AssumptionRecord]:
        sentence_doc = parsed_sentence.doc
        assumptions: List[AssumptionRecord] = []
        assumption_index = 0

        if self._has_copula_claim(sentence_doc):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.ONTOLOGICAL,
                    trigger="copula_claim",
                    confidence=0.85,
                )
            )
            assumption_index += 1

        lex_lower = {tok.lemma_.lower() for tok in sentence_doc if tok.is_alpha}

        if lex_lower.intersection(NORMATIVE_TERMS):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.NORMATIVE,
                    trigger="normative_lexeme",
                    confidence=0.82,
                )
            )
            assumption_index += 1

        if lex_lower.intersection(CAUSAL_TERMS):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.CAUSAL,
                    trigger="causal_lexeme",
                    confidence=0.78,
                )
            )
            assumption_index += 1

        if lex_lower.intersection(EPISTEMIC_TERMS):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.EPISTEMIC,
                    trigger="epistemic_lexeme",
                    confidence=0.76,
                )
            )
            assumption_index += 1

        if lex_lower.intersection(TELEOLOGICAL_TERMS):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.TELEOLOGICAL,
                    trigger="teleological_lexeme",
                    confidence=0.73,
                )
            )
            assumption_index += 1

        if lex_lower.intersection(MODAL_TERMS):
            assumptions.append(
                self._assumption(
                    parsed_sentence,
                    assumption_index,
                    AssumptionType.MODAL,
                    trigger="modal_lexeme",
                    confidence=0.7,
                )
            )

        # Deduplicate by (type, trigger) to keep deterministic compact output.
        unique: Dict[tuple[AssumptionType, str], AssumptionRecord] = {}
        for item in assumptions:
            unique[(item.type, item.trigger)] = item

        return list(unique.values())

    def _assumption(
        self,
        sentence: ParsedSentence,
        idx: int,
        assumption_type: AssumptionType,
        trigger: str,
        confidence: float,
    ) -> AssumptionRecord:
        return AssumptionRecord(
            id=f"asm_{sentence.id}_{idx}",
            sentence_id=sentence.id,
            text=sentence.text,
            trigger=trigger,
            type=assumption_type,
            confidence=confidence,
        )

    @staticmethod
    def _pos_signature(sent: Span) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        token_total = 0
        for tok in sent:
            if tok.is_space or tok.is_punct:
                continue
            token_total += 1
            counts[tok.pos_] = counts.get(tok.pos_, 0) + 1

        if token_total == 0:
            return {}
        return {k: v / token_total for k, v in counts.items()}

    @staticmethod
    def _has_copula_claim(doc: Doc) -> bool:
        for token in doc:
            if token.dep_ == "cop" and token.lemma_.lower() == "be":
                has_subject = any(child.dep_.startswith("nsubj") for child in token.head.children)
                has_predicate = token.head.pos_ in {"NOUN", "ADJ", "PROPN"}
                if has_subject and has_predicate:
                    return True
        return False


__all__ = [
    "ParserService",
    "ParsedSentence",
    "ESSENTIALIST_TERMS",
    "UNIVERSAL_QUANTIFIERS",
]
