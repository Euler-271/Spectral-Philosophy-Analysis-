from __future__ import annotations

from typing import List, Sequence

import numpy as np

from models.schema import ModalityResult, RigidityFeatures
from services.modality_analyzer import ModalityAnalyzer
from services.parser import ESSENTIALIST_TERMS, UNIVERSAL_QUANTIFIERS, ParsedSentence


class RigidityDetector:
    def __init__(self, modality_analyzer: ModalityAnalyzer | None = None) -> None:
        self.modality_analyzer = modality_analyzer or ModalityAnalyzer()

    def analyze(
        self,
        sentences: Sequence[ParsedSentence],
    ) -> tuple[List[float], List[RigidityFeatures], List[ModalityResult], float]:
        if not sentences:
            return [], [], [], 0.0

        scores: List[float] = []
        feature_rows: List[RigidityFeatures] = []
        modalities: List[ModalityResult] = []

        for sentence in sentences:
            doc = sentence.doc
            token_count = max(sum(1 for tok in doc if tok.is_alpha), 1)

            copula_hits = sum(1 for tok in doc if tok.dep_ == "cop" and tok.lemma_.lower() == "be")
            essentialist_hits = sum(
                1 for tok in doc if tok.lemma_.lower() in ESSENTIALIST_TERMS or tok.text.lower() in ESSENTIALIST_TERMS
            )
            universal_hits = sum(
                1
                for tok in doc
                if tok.lemma_.lower() in UNIVERSAL_QUANTIFIERS or tok.text.lower() in UNIVERSAL_QUANTIFIERS
            )

            modality = self.modality_analyzer.analyze(doc)
            certainty_density = modality.certainty_count / token_count
            hedge_density = modality.hedge_count / token_count

            features = RigidityFeatures(
                copula_density=min(copula_hits / token_count, 1.0),
                essentialist_density=min(essentialist_hits / token_count, 1.0),
                universal_quantifier_density=min(universal_hits / token_count, 1.0),
                certainty_modal_density=min(certainty_density, 1.0),
                hedge_modal_density=min(hedge_density, 1.0),
            )

            score = (
                0.35 * features.copula_density
                + 0.25 * features.essentialist_density
                + 0.20 * features.universal_quantifier_density
                + 0.20 * features.certainty_modal_density
                - 0.15 * features.hedge_modal_density
            )
            score = max(0.0, min(score, 1.0))

            scores.append(float(score))
            feature_rows.append(features)
            modalities.append(modality)

        rigidity_variance = float(np.var(np.asarray(scores, dtype=np.float32)))
        return scores, feature_rows, modalities, rigidity_variance


__all__ = ["RigidityDetector"]
