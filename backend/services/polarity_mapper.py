from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from models.schema import PolarityVector
from services.embedding_engine import EmbeddingEngine


@dataclass(frozen=True)
class PolarityDefinition:
    key: str
    positive_label: str
    negative_label: str
    positive_prototypes: Sequence[str]
    negative_prototypes: Sequence[str]


class PolarityMapper:
    def __init__(self, embedding_engine: EmbeddingEngine) -> None:
        self.embedding_engine = embedding_engine
        self.definitions = self._definitions()
        self.positive_centroids = {
            d.key: self.embedding_engine.centroid(d.positive_prototypes) for d in self.definitions
        }
        self.negative_centroids = {
            d.key: self.embedding_engine.centroid(d.negative_prototypes) for d in self.definitions
        }

    def analyze(self, sentence_embeddings: np.ndarray) -> tuple[List[PolarityVector], Dict[str, float], float]:
        if len(sentence_embeddings) == 0:
            empty_variance = {d.key: 0.0 for d in self.definitions}
            return [], empty_variance, 0.0

        axis_scores: Dict[str, List[float]] = {d.key: [] for d in self.definitions}
        vectors: List[PolarityVector] = []

        for embedding in sentence_embeddings:
            row: Dict[str, float] = {}
            for definition in self.definitions:
                pos = float(np.dot(embedding, self.positive_centroids[definition.key]))
                neg = float(np.dot(embedding, self.negative_centroids[definition.key]))
                score = pos - neg
                row[definition.key] = score
                axis_scores[definition.key].append(score)

            vectors.append(
                PolarityVector(
                    reductionism_vs_holism=row["reductionism_vs_holism"],
                    realism_vs_constructivism=row["realism_vs_constructivism"],
                    determinism_vs_agency=row["determinism_vs_agency"],
                    instrumental_vs_intrinsic=row["instrumental_vs_intrinsic"],
                )
            )

        variances = {
            key: float(np.var(np.asarray(values, dtype=np.float32)))
            for key, values in axis_scores.items()
        }
        mean_variance = float(np.mean(list(variances.values()))) if variances else 0.0
        return vectors, variances, mean_variance

    @staticmethod
    def _definitions() -> List[PolarityDefinition]:
        return [
            PolarityDefinition(
                key="reductionism_vs_holism",
                positive_label="Reductionism",
                negative_label="Holism",
                positive_prototypes=(
                    "Complex systems can be explained by their elementary parts.",
                    "Lower-level mechanisms fully determine higher-level behavior.",
                ),
                negative_prototypes=(
                    "The whole has properties not reducible to isolated parts.",
                    "Relational organization is primary in explanation.",
                ),
            ),
            PolarityDefinition(
                key="realism_vs_constructivism",
                positive_label="Realism",
                negative_label="Constructivism",
                positive_prototypes=(
                    "Truth tracks a mind-independent reality.",
                    "Entities exist independently of conceptual schemes.",
                ),
                negative_prototypes=(
                    "Knowledge objects are constituted by social discourse.",
                    "Categories arise through interpretive construction.",
                ),
            ),
            PolarityDefinition(
                key="determinism_vs_agency",
                positive_label="Determinism",
                negative_label="Agency",
                positive_prototypes=(
                    "Outcomes are fixed by prior causal conditions.",
                    "Human behavior follows deterministic laws.",
                ),
                negative_prototypes=(
                    "Agents can intervene and choose among alternatives.",
                    "Action expresses deliberative autonomy.",
                ),
            ),
            PolarityDefinition(
                key="instrumental_vs_intrinsic",
                positive_label="Instrumental",
                negative_label="Intrinsic",
                positive_prototypes=(
                    "Value is judged by utility toward external goals.",
                    "Practical efficiency is the decisive criterion.",
                ),
                negative_prototypes=(
                    "Some goods are valuable in themselves.",
                    "Worth can be intrinsic rather than instrumental.",
                ),
            ),
        ]


__all__ = ["PolarityMapper", "PolarityDefinition"]
