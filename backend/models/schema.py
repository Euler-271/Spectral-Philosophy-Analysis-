from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LanguageGameName(str, Enum):
    MEASUREMENT = "Measurement Game"
    CAUSAL = "Causal Explanation Game"
    NORMATIVE = "Normative Game"
    ONTOLOGICAL = "Ontological Assertion Game"
    INSTRUMENTAL = "Instrumental Optimization Game"
    CONSTRUCTIVIST = "Constructivist Framing Game"


class AssumptionType(str, Enum):
    ONTOLOGICAL = "ontological"
    NORMATIVE = "normative"
    CAUSAL = "causal"
    EPISTEMIC = "epistemic"
    MODAL = "modal"
    TELEOLOGICAL = "teleological"


class AnalysisWeights(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=1.0, ge=0.0)
    delta: float = Field(default=1.0, ge=0.0)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_sentences: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=64, ge=1)
    min_sentence_chars: int = Field(default=6, ge=1)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    weights: AnalysisWeights = Field(default_factory=AnalysisWeights)
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Input text is empty after stripping whitespace.")
        return normalized


class LanguageGameActivation(BaseModel):
    game: LanguageGameName
    probability: float = Field(..., ge=0.0, le=1.0)


class PolarityVector(BaseModel):
    reductionism_vs_holism: float
    realism_vs_constructivism: float
    determinism_vs_agency: float
    instrumental_vs_intrinsic: float


class RigidityFeatures(BaseModel):
    copula_density: float = Field(..., ge=0.0, le=1.0)
    essentialist_density: float = Field(..., ge=0.0, le=1.0)
    universal_quantifier_density: float = Field(..., ge=0.0, le=1.0)
    certainty_modal_density: float = Field(..., ge=0.0, le=1.0)
    hedge_modal_density: float = Field(..., ge=0.0, le=1.0)


class ModalityResult(BaseModel):
    certainty_count: int = Field(..., ge=0)
    hedge_count: int = Field(..., ge=0)
    certainty_index: float = Field(..., ge=0.0, le=1.0)


class AssumptionRecord(BaseModel):
    id: str
    sentence_id: str
    text: str
    trigger: str
    type: AssumptionType
    confidence: float = Field(..., ge=0.0, le=1.0)


class SentenceAnalysis(BaseModel):
    id: str
    text: str
    language_game_distribution: Dict[LanguageGameName, float]
    local_kl_divergence: float = Field(..., ge=0.0)
    polarity_vector: PolarityVector
    rigidity_score: float = Field(..., ge=0.0, le=1.0)
    modality: ModalityResult
    rigidity_features: RigidityFeatures
    assumptions: List[AssumptionRecord]


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str
    weight: float = Field(default=1.0, ge=0.0)
    centrality: float = Field(default=0.0, ge=0.0)
    metadata: Dict[str, float | str | int] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = Field(default=1.0, ge=0.0)


class GraphPayload(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class SpectralMetrics(BaseModel):
    eigenvalues: List[float]
    spectral_entropy: float = Field(..., ge=0.0)
    algebraic_connectivity: float = Field(..., ge=0.0)


class DistributionSummary(BaseModel):
    language_game_mean: Dict[LanguageGameName, float]
    mean_kl_divergence: float = Field(..., ge=0.0)


class InstabilityBreakdown(BaseModel):
    mean_kl_divergence: float = Field(..., ge=0.0)
    polarity_variance: float = Field(..., ge=0.0)
    spectral_entropy: float = Field(..., ge=0.0)
    rigidity_variance: float = Field(..., ge=0.0)


class InstabilityResult(BaseModel):
    score: float = Field(..., ge=0.0)
    weights: AnalysisWeights
    components: InstabilityBreakdown


class AggregateMetrics(BaseModel):
    mean_kl_divergence: float = Field(..., ge=0.0)
    polarity_variance: float = Field(..., ge=0.0)
    rigidity_variance: float = Field(..., ge=0.0)
    spectral_entropy: float = Field(..., ge=0.0)
    instability_score: float = Field(..., ge=0.0)


class AnalyzeResponse(BaseModel):
    distribution: DistributionSummary
    spectral: SpectralMetrics
    instability: InstabilityResult
    metrics: AggregateMetrics
    sentences: List[SentenceAnalysis]
    assumptions: List[AssumptionRecord]
    graph: GraphPayload


class ErrorResponse(BaseModel):
    detail: str
