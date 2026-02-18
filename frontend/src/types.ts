export interface AnalysisWeights {
  alpha: number;
  beta: number;
  gamma: number;
  delta: number;
}

export interface AnalysisConfig {
  embedding_model: string;
  max_sentences: number;
  batch_size: number;
  min_sentence_chars: number;
}

export interface AnalyzeRequest {
  text: string;
  weights: AnalysisWeights;
  config: AnalysisConfig;
}

export interface AssumptionRecord {
  id: string;
  sentence_id: string;
  text: string;
  trigger: string;
  type: string;
  confidence: number;
}

export interface ModalityResult {
  certainty_count: number;
  hedge_count: number;
  certainty_index: number;
}

export interface RigidityFeatures {
  copula_density: number;
  essentialist_density: number;
  universal_quantifier_density: number;
  certainty_modal_density: number;
  hedge_modal_density: number;
}

export interface PolarityVector {
  reductionism_vs_holism: number;
  realism_vs_constructivism: number;
  determinism_vs_agency: number;
  instrumental_vs_intrinsic: number;
}

export interface SentenceAnalysis {
  id: string;
  text: string;
  language_game_distribution: Record<string, number>;
  local_kl_divergence: number;
  polarity_vector: PolarityVector;
  rigidity_score: number;
  modality: ModalityResult;
  rigidity_features: RigidityFeatures;
  assumptions: AssumptionRecord[];
}

export interface GraphNode {
  id: string;
  label: string;
  kind: string;
  weight: number;
  centrality: number;
  metadata: Record<string, string | number>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SpectralMetrics {
  eigenvalues: number[];
  spectral_entropy: number;
  algebraic_connectivity: number;
}

export interface DistributionSummary {
  language_game_mean: Record<string, number>;
  mean_kl_divergence: number;
}

export interface InstabilityBreakdown {
  mean_kl_divergence: number;
  polarity_variance: number;
  spectral_entropy: number;
  rigidity_variance: number;
}

export interface InstabilityResult {
  score: number;
  weights: AnalysisWeights;
  components: InstabilityBreakdown;
}

export interface AggregateMetrics {
  mean_kl_divergence: number;
  polarity_variance: number;
  rigidity_variance: number;
  spectral_entropy: number;
  instability_score: number;
}

export interface AnalyzeResponse {
  distribution: DistributionSummary;
  spectral: SpectralMetrics;
  instability: InstabilityResult;
  metrics: AggregateMetrics;
  sentences: SentenceAnalysis[];
  assumptions: AssumptionRecord[];
  graph: GraphPayload;
}
