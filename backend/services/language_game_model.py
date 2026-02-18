from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from models.schema import LanguageGameName
from services.embedding_engine import EmbeddingEngine
from services.parser import ParsedSentence


@dataclass(frozen=True)
class LanguageGameDefinition:
    name: LanguageGameName
    lexical_patterns: Sequence[str]
    prototypes: Sequence[str]
    pos_signature: Dict[str, float]


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x)
    if denom == 0:
        return np.full_like(exp_x, fill_value=1.0 / len(exp_x))
    return exp_x / denom


class LanguageGameModel:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        lexical_weight: float = 0.35,
        pos_weight: float = 0.20,
    ) -> None:
        self.embedding_engine = embedding_engine
        self.lexical_weight = lexical_weight
        self.pos_weight = pos_weight
        self.definitions = self._build_definitions()
        self.centroids = {
            d.name: self.embedding_engine.centroid(d.prototypes) for d in self.definitions
        }
        self.pos_tags = self._collect_pos_tags(self.definitions)

    def analyze(
        self,
        sentences: Sequence[ParsedSentence],
        sentence_embeddings: np.ndarray,
    ) -> tuple[np.ndarray, Dict[LanguageGameName, float]]:
        if len(sentences) == 0:
            k = len(self.definitions)
            return np.empty((0, k), dtype=np.float32), {
                d.name: 0.0 for d in self.definitions
            }

        distributions = np.vstack(
            [
                self.sentence_distribution(sent, sentence_embeddings[i])
                for i, sent in enumerate(sentences)
            ]
        ).astype(np.float32)

        mean_distribution = distributions.mean(axis=0)
        summary = {
            definition.name: float(mean_distribution[idx])
            for idx, definition in enumerate(self.definitions)
        }
        return distributions, summary

    def sentence_distribution(self, sentence: ParsedSentence, embedding: np.ndarray) -> np.ndarray:
        scores: List[float] = []
        lower_text = sentence.text.lower()

        for definition in self.definitions:
            centroid = self.centroids[definition.name]
            emb_similarity = float(np.dot(embedding, centroid))
            lexical_score = self._lexical_match_score(lower_text, definition.lexical_patterns)
            pos_score = self._pos_signature_score(sentence.pos_signature, definition.pos_signature)
            score = emb_similarity + self.lexical_weight * lexical_score + self.pos_weight * pos_score
            scores.append(score)

        return _softmax(np.asarray(scores, dtype=np.float32))

    @staticmethod
    def _lexical_match_score(text: str, patterns: Sequence[str]) -> float:
        if not patterns:
            return 0.0
        matches = sum(1 for pattern in patterns if pattern in text)
        return matches / len(patterns)

    def _pos_signature_score(
        self,
        sentence_signature: Dict[str, float],
        game_signature: Dict[str, float],
    ) -> float:
        if not sentence_signature:
            return 0.0
        sent_vec = np.array([sentence_signature.get(tag, 0.0) for tag in self.pos_tags], dtype=np.float32)
        game_vec = np.array([game_signature.get(tag, 0.0) for tag in self.pos_tags], dtype=np.float32)

        sent_norm = np.linalg.norm(sent_vec)
        game_norm = np.linalg.norm(game_vec)
        if sent_norm == 0 or game_norm == 0:
            return 0.0
        return float(np.dot(sent_vec, game_vec) / (sent_norm * game_norm))

    @staticmethod
    def _collect_pos_tags(definitions: Sequence[LanguageGameDefinition]) -> List[str]:
        tags = sorted({k for d in definitions for k in d.pos_signature.keys()})
        return tags

    @staticmethod
    def _build_definitions() -> List[LanguageGameDefinition]:
        return [
            LanguageGameDefinition(
                name=LanguageGameName.MEASUREMENT,
                lexical_patterns=("measure", "metric", "quantify", "evidence", "data", "statistical"),
                prototypes=(
                    "The validity of the claim depends on measurable indicators.",
                    "Empirical data and operational definitions are central to this argument.",
                    "The model is evaluated through quantitative metrics.",
                ),
                pos_signature={"NOUN": 0.30, "VERB": 0.20, "ADJ": 0.12, "NUM": 0.08},
            ),
            LanguageGameDefinition(
                name=LanguageGameName.CAUSAL,
                lexical_patterns=("because", "therefore", "causes", "leads to", "drives", "mechanism"),
                prototypes=(
                    "The observed effect follows from a specific causal mechanism.",
                    "One condition produces another through stable dependence.",
                    "The explanation identifies causes rather than descriptions.",
                ),
                pos_signature={"NOUN": 0.26, "VERB": 0.24, "SCONJ": 0.06, "ADP": 0.08},
            ),
            LanguageGameDefinition(
                name=LanguageGameName.NORMATIVE,
                lexical_patterns=("should", "must", "ought", "good", "bad", "ethical", "responsibility"),
                prototypes=(
                    "The argument evaluates what agents ought to do.",
                    "Moral obligation guides the conclusion.",
                    "The text frames practice in terms of value and duty.",
                ),
                pos_signature={"NOUN": 0.24, "VERB": 0.20, "AUX": 0.10, "ADJ": 0.14},
            ),
            LanguageGameDefinition(
                name=LanguageGameName.ONTOLOGICAL,
                lexical_patterns=("is", "are", "reality", "exists", "essence", "being"),
                prototypes=(
                    "The statement claims what entities are in themselves.",
                    "Existence and essence are treated as primary categories.",
                    "The discourse asserts an ontology of stable kinds.",
                ),
                pos_signature={"NOUN": 0.30, "VERB": 0.18, "AUX": 0.10, "ADJ": 0.10},
            ),
            LanguageGameDefinition(
                name=LanguageGameName.INSTRUMENTAL,
                lexical_patterns=("optimize", "efficient", "goal", "utility", "cost", "performance"),
                prototypes=(
                    "The central criterion is instrumental efficiency.",
                    "Actions are ranked by optimization toward explicit goals.",
                    "Utility maximization structures the recommendation.",
                ),
                pos_signature={"NOUN": 0.28, "VERB": 0.22, "ADJ": 0.12, "ADV": 0.08},
            ),
            LanguageGameDefinition(
                name=LanguageGameName.CONSTRUCTIVIST,
                lexical_patterns=("constructed", "framing", "discourse", "socially", "narrative", "interpretation"),
                prototypes=(
                    "Meaning is produced through social and historical framing.",
                    "Categories are constructed within language and practice.",
                    "Interpretation is treated as constitutive of the object.",
                ),
                pos_signature={"NOUN": 0.27, "VERB": 0.20, "ADJ": 0.12, "ADV": 0.08},
            ),
        ]


__all__ = ["LanguageGameModel", "LanguageGameDefinition"]
