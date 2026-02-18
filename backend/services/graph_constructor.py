from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np

from models.schema import AssumptionRecord, GraphEdge, GraphNode, GraphPayload, LanguageGameName, PolarityVector
from services.parser import ParsedSentence


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


class GraphConstructor:
    def build(
        self,
        sentences: Sequence[ParsedSentence],
        language_game_names: Sequence[LanguageGameName],
        language_distributions: np.ndarray,
        polarity_vectors: Sequence[PolarityVector],
        assumptions_by_sentence: Dict[str, List[AssumptionRecord]],
        rigidity_scores: Sequence[float],
    ) -> tuple[nx.DiGraph, GraphPayload]:
        graph = nx.DiGraph()

        # Language game nodes
        for game in language_game_names:
            node_id = f"lg_{_slug(game.value)}"
            graph.add_node(node_id, kind="language_game", label=game.value, weight=1.0)

        polarity_axes = [
            "reductionism_vs_holism",
            "realism_vs_constructivism",
            "determinism_vs_agency",
            "instrumental_vs_intrinsic",
        ]

        for axis in polarity_axes:
            pos_id = f"pol_{axis}_pos"
            neg_id = f"pol_{axis}_neg"
            graph.add_node(pos_id, kind="polarity", label=f"{axis}:+", weight=1.0)
            graph.add_node(neg_id, kind="polarity", label=f"{axis}:-", weight=1.0)

        for idx, sentence in enumerate(sentences):
            sent_id = sentence.id
            graph.add_node(
                sent_id,
                kind="sentence",
                label=sentence.text[:120],
                weight=float(rigidity_scores[idx]) if idx < len(rigidity_scores) else 0.0,
            )

            distribution = language_distributions[idx]
            max_game_idx = int(np.argmax(distribution))
            max_game_name = language_game_names[max_game_idx]
            max_game_node_id = f"lg_{_slug(max_game_name.value)}"

            for game_idx, game_name in enumerate(language_game_names):
                weight = float(distribution[game_idx])
                if weight < 0.05:
                    continue
                graph.add_edge(sent_id, f"lg_{_slug(game_name.value)}", relation="activates", weight=weight)

            if idx < len(polarity_vectors):
                vector = polarity_vectors[idx]
                axis_scores = {
                    "reductionism_vs_holism": vector.reductionism_vs_holism,
                    "realism_vs_constructivism": vector.realism_vs_constructivism,
                    "determinism_vs_agency": vector.determinism_vs_agency,
                    "instrumental_vs_intrinsic": vector.instrumental_vs_intrinsic,
                }
                for axis, score in axis_scores.items():
                    sign = "pos" if score >= 0 else "neg"
                    target = f"pol_{axis}_{sign}"
                    graph.add_edge(sent_id, target, relation="activates", weight=abs(float(score)))

            sentence_assumptions = assumptions_by_sentence.get(sent_id, [])
            for assumption in sentence_assumptions:
                graph.add_node(
                    assumption.id,
                    kind="assumption",
                    label=f"{assumption.type.value}:{assumption.trigger}",
                    weight=float(assumption.confidence),
                )
                graph.add_edge(sent_id, assumption.id, relation="derived_from", weight=float(assumption.confidence))
                graph.add_edge(assumption.id, max_game_node_id, relation="depends_on", weight=float(distribution[max_game_idx]))

        self._add_sentence_contradictions(graph, sentences, polarity_vectors)
        self._add_assumption_contradictions(graph, assumptions_by_sentence)

        nodes_payload, edges_payload = self._to_payload(graph)
        payload = GraphPayload(nodes=nodes_payload, edges=edges_payload)
        return graph, payload

    def _add_sentence_contradictions(
        self,
        graph: nx.DiGraph,
        sentences: Sequence[ParsedSentence],
        polarity_vectors: Sequence[PolarityVector],
    ) -> None:
        if len(sentences) < 2 or len(polarity_vectors) < 2:
            return

        for idx in range(len(sentences) - 1):
            v1 = polarity_vectors[idx]
            v2 = polarity_vectors[idx + 1]
            flips = 0
            axis1 = self._vector_to_axis(v1)
            axis2 = self._vector_to_axis(v2)
            for key, val in axis1.items():
                if np.sign(val) != np.sign(axis2[key]):
                    flips += 1

            if flips >= 2:
                weight = flips / 4.0
                graph.add_edge(
                    sentences[idx].id,
                    sentences[idx + 1].id,
                    relation="contradicts",
                    weight=float(weight),
                )

    def _add_assumption_contradictions(
        self,
        graph: nx.DiGraph,
        assumptions_by_sentence: Dict[str, List[AssumptionRecord]],
    ) -> None:
        all_assumptions = [a for bucket in assumptions_by_sentence.values() for a in bucket]
        by_type: Dict[str, List[AssumptionRecord]] = {}
        for assumption in all_assumptions:
            by_type.setdefault(assumption.type.value, []).append(assumption)

        for _, group in by_type.items():
            tokens_by_id: Dict[str, set[str]] = {a.id: self._content_tokens(a.text) for a in group}
            id_to_assumption: Dict[str, AssumptionRecord] = {a.id: a for a in group}

            term_index: Dict[str, set[str]] = {}
            for assumption in group:
                for term in tokens_by_id[assumption.id]:
                    term_index.setdefault(term, set()).add(assumption.id)

            # Use frozenset pairs so deduplication is order-independent and not
            # sensitive to lexicographic ID ordering (which breaks for UUIDs / timestamps).
            seen_pairs: set[frozenset[str]] = set()
            for assumption in group:
                a_id = assumption.id
                candidate_ids: set[str] = set()
                for term in tokens_by_id[a_id]:
                    candidate_ids.update(term_index.get(term, set()))

                for b_id in candidate_ids:
                    if a_id == b_id:
                        continue
                    pair: frozenset[str] = frozenset({a_id, b_id})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    contradiction = self._contradiction_strength(
                        id_to_assumption[a_id].text,
                        id_to_assumption[b_id].text,
                    )
                    if contradiction > 0:
                        graph.add_edge(a_id, b_id, relation="contradicts", weight=contradiction)

    @staticmethod
    def _contradiction_strength(text_a: str, text_b: str) -> float:
        neg_terms = {"not", "never", "no", "none", "cannot"}
        tokens_a = GraphConstructor._content_tokens(text_a)
        tokens_b = GraphConstructor._content_tokens(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        overlap = len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))
        has_negation_a = bool(tokens_a.intersection(neg_terms))
        has_negation_b = bool(tokens_b.intersection(neg_terms))

        if overlap >= 0.35 and has_negation_a != has_negation_b:
            return min(1.0, 0.4 + overlap)
        return 0.0

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {tok for tok in re.findall(r"[a-zA-Z]+", text.lower()) if len(tok) > 2}

    @staticmethod
    def _vector_to_axis(vector: PolarityVector) -> Dict[str, float]:
        return {
            "reductionism_vs_holism": vector.reductionism_vs_holism,
            "realism_vs_constructivism": vector.realism_vs_constructivism,
            "determinism_vs_agency": vector.determinism_vs_agency,
            "instrumental_vs_intrinsic": vector.instrumental_vs_intrinsic,
        }

    def _to_payload(self, graph: nx.DiGraph) -> tuple[List[GraphNode], List[GraphEdge]]:
        weighted_degree = self._weighted_degree_centrality(graph)

        nodes = [
            GraphNode(
                id=node,
                label=str(data.get("label", node)),
                kind=str(data.get("kind", "unknown")),
                weight=float(data.get("weight", 1.0)),
                centrality=float(weighted_degree.get(node, 0.0)),
                metadata={},
            )
            for node, data in graph.nodes(data=True)
        ]

        edges = [
            GraphEdge(
                source=u,
                target=v,
                relation=str(data.get("relation", "related_to")),
                weight=float(data.get("weight", 1.0)),
            )
            for u, v, data in graph.edges(data=True)
        ]

        return nodes, edges

    @staticmethod
    def _weighted_degree_centrality(graph: nx.DiGraph) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}

        strengths: Dict[str, float] = {node: 0.0 for node in graph.nodes()}

        for node in graph.nodes():
            out_strength = sum(data.get("weight", 1.0) for _, _, data in graph.out_edges(node, data=True))
            in_strength = sum(data.get("weight", 1.0) for _, _, data in graph.in_edges(node, data=True))
            strengths[node] = float(out_strength + in_strength)

        max_strength = max(strengths.values()) if strengths else 1.0
        if max_strength <= 0:
            return {k: 0.0 for k in strengths}

        return {k: v / max_strength for k, v in strengths.items()}


__all__ = ["GraphConstructor"]
