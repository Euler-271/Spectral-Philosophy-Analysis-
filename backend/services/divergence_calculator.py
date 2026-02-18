from __future__ import annotations

from typing import List

import numpy as np


class DivergenceCalculator:
    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon

    def analyze(self, sentence_distributions: np.ndarray) -> tuple[List[float], float, np.ndarray]:
        if sentence_distributions.size == 0:
            return [], 0.0, np.array([], dtype=np.float32)

        mean_distribution = sentence_distributions.mean(axis=0)
        mean_distribution = self._normalize(mean_distribution)

        local_kls = [
            float(self._kl_divergence(dist, mean_distribution))
            for dist in sentence_distributions
        ]
        mean_kl = float(np.mean(np.asarray(local_kls, dtype=np.float32))) if local_kls else 0.0

        return local_kls, mean_kl, mean_distribution

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p_s = self._normalize(p)
        q_s = self._normalize(q)
        p_safe = np.clip(p_s, self.epsilon, 1.0)
        q_safe = np.clip(q_s, self.epsilon, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        total = float(np.sum(values))
        if total <= 0:
            return np.full_like(values, 1.0 / len(values))
        return values / total


__all__ = ["DivergenceCalculator"]
