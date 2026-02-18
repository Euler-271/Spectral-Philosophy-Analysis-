import type { SentenceAnalysis } from '../types';

interface HeatmapPanelProps {
  sentences: SentenceAnalysis[];
}

interface HeatRow {
  id: string;
  text: string;
  value: number;
  normalized: number;
}

export function HeatmapPanel({ sentences }: HeatmapPanelProps) {
  if (sentences.length === 0) {
    return (
      <section className="panel panel-bottom">
        <div className="panel-header">Sentence Instability Heatmap</div>
        <div className="placeholder">No sentence-level measurements yet.</div>
      </section>
    );
  }

  const rows = buildRows(sentences);

  return (
    <section className="panel panel-bottom">
      <div className="panel-header">Sentence Instability Heatmap</div>
      <div className="heatmap-table">
        {rows.map((row, index) => (
          <div className="heatmap-row" key={row.id}>
            <div className="heatmap-index">{index + 1}</div>
            <div className="heatmap-text" title={row.text}>
              {row.text}
            </div>
            <div className="heatmap-bar-wrap">
              <div
                className="heatmap-bar"
                style={{
                  width: `${Math.max(4, row.normalized * 100)}%`,
                  background: heatColor(row.normalized)
                }}
              />
            </div>
            <code className="heatmap-value">{row.value.toFixed(4)}</code>
          </div>
        ))}
      </div>
    </section>
  );
}

function buildRows(sentences: SentenceAnalysis[]): HeatRow[] {
  const values = sentences.map((sentence) => {
    const polarity = sentence.polarity_vector;
    const polarityMagnitude =
      (Math.abs(polarity.reductionism_vs_holism) +
        Math.abs(polarity.realism_vs_constructivism) +
        Math.abs(polarity.determinism_vs_agency) +
        Math.abs(polarity.instrumental_vs_intrinsic)) /
      4;

    return sentence.local_kl_divergence + 0.6 * sentence.rigidity_score + 0.25 * polarityMagnitude;
  });

  const max = Math.max(...values, 1e-6);
  const min = Math.min(...values, 0);
  const range = Math.max(max - min, 1e-6);

  return sentences.map((sentence, index) => {
    const value = values[index];
    const normalized = (value - min) / range;

    return {
      id: sentence.id,
      text: sentence.text,
      value,
      normalized
    };
  });
}

function heatColor(value: number): string {
  const hue = 200 - value * 200;
  return `hsl(${hue}, 78%, 58%)`;
}
