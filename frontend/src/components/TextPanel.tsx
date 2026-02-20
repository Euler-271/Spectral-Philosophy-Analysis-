import { ChangeEvent } from 'react';

import type { AnalysisWeights } from '../types';

interface TextPanelProps {
  text: string;
  weights: AnalysisWeights;
  isLoading: boolean;
  hasResult: boolean;
  onTextChange: (value: string) => void;
  onWeightsChange: (next: AnalysisWeights) => void;
  onAnalyze: () => void;
  onExportJson: () => void;
  onExportCsv: () => void;
  onExportGraphJson: () => void;
  onExportSummary: () => void;
  onExportPng: () => void;
}

export function TextPanel(props: TextPanelProps) {
  const {
    text,
    weights,
    isLoading,
    hasResult,
    onTextChange,
    onWeightsChange,
    onAnalyze,
    onExportJson,
    onExportCsv,
    onExportGraphJson,
    onExportSummary,
    onExportPng
  } = props;

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const content = await file.text();
    onTextChange(content);
  };

  const setWeight = (key: keyof AnalysisWeights, value: string) => {
    const numeric = Number(value);
    if (Number.isNaN(numeric) || numeric < 0) {
      return;
    }
    onWeightsChange({ ...weights, [key]: numeric });
  };

  return (
    <aside className="panel panel-left">
      <div className="panel-header">Corpus Input</div>
      <label className="field-label" htmlFor="text-input">
        Text
      </label>
      <textarea
        id="text-input"
        className="input-area"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Paste discourse for analysis..."
      />

      <div className="preset-row">
        <button
          className="preset-pill"
          onClick={() => onTextChange("All scientific models are simplifications of reality. Therefore, explanation should prioritize causal mechanisms over surface correlations. Yet social categories are often constructed through institutional discourse.")}
        >
          Preset: Realism
        </button>
        <button
          className="preset-pill"
          onClick={() => onTextChange("Policy decisions must optimize outcomes, but persons also possess intrinsic worth. If agency is illusory, moral responsibility becomes conceptually unstable.")}
        >
          Preset: Duty
        </button>
        <button
          className="preset-pill clear-btn"
          onClick={() => onTextChange("")}
        >
          Clear
        </button>
      </div>

      <label className="field-label" htmlFor="upload">
        Upload .txt
      </label>
      <input id="upload" type="file" accept=".txt,.md" onChange={handleFileUpload} className="file-input" />

      <div className="weights-grid">
        <div className="field-label">Instability Weights</div>
        <label>
          α
          <input type="number" step="0.1" min="0" value={weights.alpha} onChange={(e) => setWeight('alpha', e.target.value)} />
        </label>
        <label>
          β
          <input type="number" step="0.1" min="0" value={weights.beta} onChange={(e) => setWeight('beta', e.target.value)} />
        </label>
        <label>
          γ
          <input type="number" step="0.1" min="0" value={weights.gamma} onChange={(e) => setWeight('gamma', e.target.value)} />
        </label>
        <label>
          δ
          <input type="number" step="0.1" min="0" value={weights.delta} onChange={(e) => setWeight('delta', e.target.value)} />
        </label>
      </div>

      <button className="analyze-button" onClick={onAnalyze} disabled={isLoading || text.trim().length === 0}>
        {isLoading ? 'Analyzing...' : 'Analyze'}
      </button>

      <div className="export-section">
        <div className="field-label">Export</div>
        <button onClick={onExportJson} disabled={!hasResult}>
          JSON report
        </button>
        <button onClick={onExportCsv} disabled={!hasResult}>
          CSV assumptions
        </button>
        <button onClick={onExportGraphJson} disabled={!hasResult}>
          Graph JSON
        </button>
        <button onClick={onExportSummary} disabled={!hasResult}>
          Numerical summary
        </button>
        <button onClick={onExportPng} disabled={!hasResult}>
          PNG graph snapshot
        </button>
      </div>
    </aside>
  );
}
