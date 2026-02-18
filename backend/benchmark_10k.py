from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from main import system
from models.schema import AnalysisConfig, AnalysisWeights, AnalyzeRequest, AnalyzeResponse


DETERMINISTIC_SENTENCE_BANK = [
    "All explanatory frameworks rely on prior ontological commitments.",
    "A measurement is meaningful only within a rule-governed language-game.",
    "Causal accounts are persuasive when mechanism and evidence cohere.",
    "Normative judgments should distinguish instrumental value from intrinsic value.",
    "Constructed categories can stabilize social reality through institutional uptake.",
    "Deterministic narratives often understate the practical significance of agency.",
    "A model is accepted because it predicts well and because it frames relevance.",
    "Universal claims require explicit modal qualification and evidential scope.",
    "Reductionist analysis isolates components, while holism emphasizes relational structure.",
    "Philosophical disagreement frequently arises from incompatible inferential grammars.",
]


@dataclass
class BenchmarkSummary:
    mode: str
    words: int
    sentences: int
    runs: int
    warmup_seconds: float
    run_seconds: List[float]
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    max_seconds: float
    threshold_seconds: float
    threshold_passed: bool
    instability_score: float
    mean_kl_divergence: float
    polarity_variance: float
    spectral_entropy: float
    rigidity_variance: float
    assumptions: int


def build_deterministic_text(target_words: int) -> str:
    if target_words < 50:
        raise ValueError("target_words must be at least 50.")

    words: List[str] = []
    sentence_index = 0

    while len(words) < target_words:
        sentence = DETERMINISTIC_SENTENCE_BANK[sentence_index % len(DETERMINISTIC_SENTENCE_BANK)]
        words.extend(sentence.split())
        sentence_index += 1

    trimmed = words[:target_words]
    text = " ".join(trimmed)

    # Re-segment deterministically into sentences to keep parser behavior stable.
    chunk_size = 22
    chunks = [trimmed[i : i + chunk_size] for i in range(0, len(trimmed), chunk_size)]
    sentence_chunks = [" ".join(chunk).strip() for chunk in chunks if chunk]
    return ". ".join(sentence_chunks) + "."


def run_analysis(text: str, weights: AnalysisWeights, config: AnalysisConfig) -> AnalyzeResponse:
    request = AnalyzeRequest(text=text, weights=weights, config=config)
    return system.analyze(request)


def percentile_95(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    idx = int(round(0.95 * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def benchmark(
    text: str,
    runs: int,
    threshold_seconds: float,
    weights: AnalysisWeights,
    config: AnalysisConfig,
) -> tuple[BenchmarkSummary, AnalyzeResponse]:
    warmup_start = time.perf_counter()
    run_analysis(text=text[: min(len(text), 2500)], weights=weights, config=config)
    warmup_elapsed = time.perf_counter() - warmup_start

    elapsed: List[float] = []
    last_response: AnalyzeResponse | None = None

    for _ in range(runs):
        start = time.perf_counter()
        last_response = run_analysis(text=text, weights=weights, config=config)
        elapsed.append(time.perf_counter() - start)

    if last_response is None:
        raise RuntimeError("Benchmark failed to produce an analysis response.")

    summary = BenchmarkSummary(
        mode="deterministic_local_cpu",
        words=len(text.split()),
        sentences=len(last_response.sentences),
        runs=runs,
        warmup_seconds=warmup_elapsed,
        run_seconds=elapsed,
        mean_seconds=statistics.mean(elapsed),
        median_seconds=statistics.median(elapsed),
        p95_seconds=percentile_95(elapsed),
        max_seconds=max(elapsed),
        threshold_seconds=threshold_seconds,
        threshold_passed=max(elapsed) <= threshold_seconds,
        instability_score=last_response.metrics.instability_score,
        mean_kl_divergence=last_response.metrics.mean_kl_divergence,
        polarity_variance=last_response.metrics.polarity_variance,
        spectral_entropy=last_response.metrics.spectral_entropy,
        rigidity_variance=last_response.metrics.rigidity_variance,
        assumptions=len(last_response.assumptions),
    )
    return summary, last_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic CPU benchmark for Spectral and Information-Geometric Modeling of "
            "Philosophical Assumption Structures in Text."
        )
    )
    parser.add_argument("--words", type=int, default=10000, help="Target word count for synthetic benchmark text.")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed benchmark runs.")
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        default=20.0,
        help="Per-run maximum allowed latency for pass/fail.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file path. If provided, --words is ignored.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output file for full benchmark JSON report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if threshold is exceeded.",
    )
    return parser.parse_args()


def read_input_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    return text


def host_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def main() -> int:
    args = parse_args()

    try:
        if args.input_file:
            text = read_input_text(args.input_file)
        else:
            text = build_deterministic_text(args.words)

        weights = AnalysisWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
        config = AnalysisConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            max_sentences=4096,
            batch_size=64,
            min_sentence_chars=6,
        )

        summary, response = benchmark(
            text=text,
            runs=max(1, args.runs),
            threshold_seconds=max(0.1, args.threshold_seconds),
            weights=weights,
            config=config,
        )

        report = {
            "benchmark": asdict(summary),
            "host": host_metadata(),
            "distribution": {
                key.value if hasattr(key, "value") else str(key): value
                for key, value in response.distribution.language_game_mean.items()
            },
        }

        print(json.dumps(report, indent=2))

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if args.strict and not summary.threshold_passed:
            return 2
        return 0
    except Exception as exc:
        print(f"benchmark_error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
