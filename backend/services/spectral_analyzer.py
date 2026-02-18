from __future__ import annotations

import numpy as np
import networkx as nx

from models.schema import SpectralMetrics


class SpectralAnalyzer:
    def analyze(self, graph: nx.DiGraph) -> SpectralMetrics:
        if graph.number_of_nodes() == 0:
            return SpectralMetrics(eigenvalues=[], spectral_entropy=0.0, algebraic_connectivity=0.0)

        nodes = list(graph.nodes())
        node_index = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)

        adjacency = np.zeros((n, n), dtype=np.float64)
        for source, target, data in graph.edges(data=True):
            i = node_index[source]
            j = node_index[target]
            weight = float(data.get("weight", 1.0))
            adjacency[i, j] += weight

        # Use symmetrized weighted adjacency for stable spectral interpretation.
        adjacency = 0.5 * (adjacency + adjacency.T)
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency

        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        eigenvalues.sort()

        spectral_entropy = self._spectral_entropy(eigenvalues)
        algebraic_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        return SpectralMetrics(
            eigenvalues=[float(v) for v in eigenvalues.tolist()],
            spectral_entropy=float(spectral_entropy),
            algebraic_connectivity=max(0.0, algebraic_connectivity),
        )

    @staticmethod
    def _spectral_entropy(eigenvalues: np.ndarray, epsilon: float = 1e-12) -> float:
        total = float(np.sum(eigenvalues))
        if total <= epsilon:
            return 0.0
        probs = eigenvalues / total
        probs = probs[probs > epsilon]
        return float(-np.sum(probs * np.log(probs)))


__all__ = ["SpectralAnalyzer"]
