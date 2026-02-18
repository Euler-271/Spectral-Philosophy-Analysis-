import type { AnalyzeResponse } from '../types';

interface MetricsPanelProps {
  result?: AnalyzeResponse;
}

export function MetricsPanel({ result }: MetricsPanelProps) {
  const eigenvalues = result?.spectral.eigenvalues ?? [];

  return (
    <aside className="panel panel-right">
      <div className="panel-header">Metrics</div>
      {!result ? (
        <div className="placeholder">No metrics available yet.</div>
      ) : (
        <>
          <div className="metric-grid">
            <MetricCard label="Mean KL Divergence" value={result.metrics.mean_kl_divergence} />
            <MetricCard label="Polarity Variance" value={result.metrics.polarity_variance} />
            <MetricCard label="Spectral Entropy" value={result.metrics.spectral_entropy} />
            <MetricCard label="Rigidity Variance" value={result.metrics.rigidity_variance} />
            <MetricCard label="Instability Score" value={result.metrics.instability_score} isPrimary />
          </div>

          <div className="distribution-block">
            <div className="field-label">Language-Game Mean Distribution</div>
            <ul>
              {Object.entries(result.distribution.language_game_mean).map(([game, prob]) => (
                <li key={game}>
                  <span>{game}</span>
                  <code>{format(prob)}</code>
                </li>
              ))}
            </ul>
          </div>

          <div className="spectrum-block">
            <div className="field-label">Eigenvalue Spectrum</div>
            <SpectrumChart values={eigenvalues} />
          </div>
        </>
      )}
    </aside>
  );
}

function MetricCard({ label, value, isPrimary = false }: { label: string; value: number; isPrimary?: boolean }) {
  return (
    <div className={`metric-card${isPrimary ? ' primary' : ''}`}>
      <div className="metric-label">{label}</div>
      <code className="metric-value">{format(value)}</code>
    </div>
  );
}

function SpectrumChart({ values }: { values: number[] }) {
  if (values.length === 0) {
    return <div className="placeholder small">No spectral data.</div>;
  }

  const width = 320;
  const height = 160;
  const padding = 20;

  const maxX = Math.max(values.length - 1, 1);
  const maxY = Math.max(...values, 1e-6);

  const points = values
    .map((value, index) => {
      const x = padding + (index / maxX) * (width - padding * 2);
      const y = height - padding - (value / maxY) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg className="spectrum-chart" viewBox={`0 0 ${width} ${height}`}>
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#4B5563" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#4B5563" />
      <polyline points={points} fill="none" stroke="#00D4FF" strokeWidth={2} />
    </svg>
  );
}

function format(value: number): string {
  if (!Number.isFinite(value)) {
    return '0.0000';
  }
  return value.toFixed(4);
}
