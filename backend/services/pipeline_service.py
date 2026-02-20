from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

from models.schema import (
    AggregateMetrics,
    AnalyzeRequest,
    AnalyzeResponse,
    AssumptionRecord,
    DistributionSummary,
    LanguageGameName,
    SentenceAnalysis,
)
from services.divergence_calculator import DivergenceCalculator
from services.embedding_engine import EmbeddingEngine
from services.graph_constructor import GraphConstructor
from services.instability_functional import InstabilityFunctional
from services.language_game_model import LanguageGameModel
from services.modality_analyzer import ModalityAnalyzer
from services.parser import ParsedSentence, ParserService
from services.polarity_mapper import PolarityMapper
from services.rigidity_detector import RigidityDetector
from services.spectral_analyzer import SpectralAnalyzer


class PipelineService:
    def __init__(self) -> None:
        embedding_model = os.getenv("SPECTRAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        self.parser = ParserService()
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.language_game_model = LanguageGameModel(self.embedding_engine)
        self.polarity_mapper = PolarityMapper(self.embedding_engine)
        self.modality_analyzer = ModalityAnalyzer()
        self.rigidity_detector = RigidityDetector(self.modality_analyzer)
        self.divergence_calculator = DivergenceCalculator()
        self.graph_constructor = GraphConstructor()
        self.spectral_analyzer = SpectralAnalyzer()
        self.instability_functional = InstabilityFunctional()

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        sentences = self.parser.parse(
            request.text,
            min_sentence_chars=request.config.min_sentence_chars,
            max_sentences=request.config.max_sentences,
        )
        if not sentences:
            raise ValueError("No analyzable sentences were detected.")

        sentence_texts = [s.text for s in sentences]
        sentence_embeddings = self.embedding_engine.encode(
            sentence_texts,
            batch_size=request.config.batch_size,
        )

        language_distributions, game_summary = self.language_game_model.analyze(sentences, sentence_embeddings)
        local_kl_values, mean_kl, _ = self.divergence_calculator.analyze(language_distributions)

        polarity_vectors, _, polarity_variance = self.polarity_mapper.analyze(sentence_embeddings)
        rigidity_scores, rigidity_features, modalities, rigidity_variance = self.rigidity_detector.analyze(sentences)

        assumptions_by_sentence: Dict[str, List[AssumptionRecord]] = defaultdict(list)
        all_assumptions: List[AssumptionRecord] = []
        for sentence in sentences:
            extracted = self.parser.extract_assumptions(sentence)
            assumptions_by_sentence[sentence.id].extend(extracted)
            all_assumptions.extend(extracted)

        language_game_names = [definition.name for definition in self.language_game_model.definitions]

        nx_graph, graph_payload = self.graph_constructor.build(
            sentences=sentences,
            language_game_names=language_game_names,
            language_distributions=language_distributions,
            polarity_vectors=polarity_vectors,
            assumptions_by_sentence=assumptions_by_sentence,
            rigidity_scores=rigidity_scores,
        )

        spectral_metrics = self.spectral_analyzer.analyze(nx_graph)

        instability_result = self.instability_functional.compute(
            mean_kl_divergence=mean_kl,
            polarity_variance=polarity_variance,
            spectral_entropy=spectral_metrics.spectral_entropy,
            rigidity_variance=rigidity_variance,
            weights=request.weights,
        )

        sentence_rows = self._build_sentence_rows(
            sentences=sentences,
            language_distributions=language_distributions,
            language_game_names=language_game_names,
            local_kl_values=local_kl_values,
            polarity_vectors=polarity_vectors,
            rigidity_scores=rigidity_scores,
            modalities=modalities,
            rigidity_features=rigidity_features,
            assumptions_by_sentence=assumptions_by_sentence,
        )

        distribution_summary = DistributionSummary(
            language_game_mean={k: float(v) for k, v in game_summary.items()},
            mean_kl_divergence=float(mean_kl),
        )

        metrics = AggregateMetrics(
            mean_kl_divergence=float(mean_kl),
            polarity_variance=float(polarity_variance),
            rigidity_variance=float(rigidity_variance),
            spectral_entropy=float(spectral_metrics.spectral_entropy),
            instability_score=float(instability_result.score),
        )

        return AnalyzeResponse(
            distribution=distribution_summary,
            spectral=spectral_metrics,
            instability=instability_result,
            metrics=metrics,
            sentences=sentence_rows,
            assumptions=all_assumptions,
            graph=graph_payload,
        )

    @staticmethod
    def _build_sentence_rows(
        sentences: List[ParsedSentence],
        language_distributions,
        language_game_names: List[LanguageGameName],
        local_kl_values,
        polarity_vectors,
        rigidity_scores,
        modalities,
        rigidity_features,
        assumptions_by_sentence: Dict[str, List[AssumptionRecord]],
    ) -> List[SentenceAnalysis]:
        rows: List[SentenceAnalysis] = []
        for idx, sentence in enumerate(sentences):
            distribution = {
                game: float(language_distributions[idx][game_idx])
                for game_idx, game in enumerate(language_game_names)
            }
            row = SentenceAnalysis(
                id=sentence.id,
                text=sentence.text,
                language_game_distribution=distribution,
                local_kl_divergence=float(local_kl_values[idx]),
                polarity_vector=polarity_vectors[idx],
                rigidity_score=float(rigidity_scores[idx]),
                modality=modalities[idx],
                rigidity_features=rigidity_features[idx],
                assumptions=assumptions_by_sentence.get(sentence.id, []),
            )
            rows.append(row)
        return rows


__all__ = ["PipelineService"]
