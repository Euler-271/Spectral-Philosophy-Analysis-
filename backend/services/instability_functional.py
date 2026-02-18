from __future__ import annotations

from models.schema import AnalysisWeights, InstabilityBreakdown, InstabilityResult


class InstabilityFunctional:
    def compute(
        self,
        mean_kl_divergence: float,
        polarity_variance: float,
        spectral_entropy: float,
        rigidity_variance: float,
        weights: AnalysisWeights,
    ) -> InstabilityResult:
        score = (
            weights.alpha * mean_kl_divergence
            + weights.beta * polarity_variance
            + weights.gamma * spectral_entropy
            + weights.delta * rigidity_variance
        )

        components = InstabilityBreakdown(
            mean_kl_divergence=float(mean_kl_divergence),
            polarity_variance=float(polarity_variance),
            spectral_entropy=float(spectral_entropy),
            rigidity_variance=float(rigidity_variance),
        )
        return InstabilityResult(score=max(0.0, float(score)), weights=weights, components=components)


__all__ = ["InstabilityFunctional"]
