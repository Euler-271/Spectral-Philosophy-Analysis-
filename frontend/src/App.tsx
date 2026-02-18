import { useMemo, useRef, useState } from 'react';

import { analyzeText } from './api';
import { GraphPanel } from './components/GraphPanel';
import { HeatmapPanel } from './components/HeatmapPanel';
import { MetricsPanel } from './components/MetricsPanel';
import { TextPanel } from './components/TextPanel';
import type { AnalysisConfig, AnalysisWeights, AnalyzeResponse } from './types';

const DEFAULT_WEIGHTS: AnalysisWeights = {
  alpha: 1.0,
  beta: 1.0,
  gamma: 1.0,
  delta: 1.0
};

const DEFAULT_CONFIG: AnalysisConfig = {
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  max_sentences: 2048,
  batch_size: 64,
  min_sentence_chars: 6
};

const SAMPLE_TEXT = `All scientific models are simplifications of reality.
Therefore, explanation should prioritize causal mechanisms over surface correlations.
Yet social categories are often constructed through institutional discourse.
Policy decisions must optimize outcomes, but persons also possess intrinsic worth.
If agency is illusory, moral responsibility becomes conceptually unstable.`;

export default function App() {
  const [text, setText] = useState<string>(SAMPLE_TEXT);
  const [weights, setWeights] = useState<AnalysisWeights>(DEFAULT_WEIGHTS);
  const [result, setResult] = useState<AnalyzeResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const graphRef = useRef<SVGSVGElement | null>(null);

  const hasResult = Boolean(result);
  const sentences = useMemo(() => result?.sentences ?? [], [result]);

  const runAnalysis = async () => {
    setLoading(true);
    setError(undefined);
    try {
      const payload = {
        text,
        weights,
        config: DEFAULT_CONFIG
      };
      const next = await analyzeText(payload);
      setResult(next);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const exportJson = () => {
    if (!result) {
      return;
    }
    downloadBlob('spectral_report.json', JSON.stringify(result, null, 2), 'application/json');
  };

  const exportCsv = () => {
    if (!result) {
      return;
    }
    const header = ['id', 'sentence_id', 'type', 'trigger', 'confidence', 'text'];
    const rows = result.assumptions.map((a) => [
      csvEscape(a.id),
      csvEscape(a.sentence_id),
      csvEscape(a.type),
      csvEscape(a.trigger),
      a.confidence.toFixed(4),
      csvEscape(a.text)
    ]);
    const csv = [header.join(','), ...rows.map((row) => row.join(','))].join('\n');
    downloadBlob('assumptions.csv', csv, 'text/csv');
  };

  const exportGraphJson = () => {
    if (!result) {
      return;
    }
    downloadBlob('graph.json', JSON.stringify(result.graph, null, 2), 'application/json');
  };

  const exportSummary = () => {
    if (!result) {
      return;
    }
    const summary = {
      metrics: result.metrics,
      instability: result.instability
    };
    downloadBlob('instability_summary.json', JSON.stringify(summary, null, 2), 'application/json');
  };

  const exportPng = async () => {
    if (!graphRef.current) {
      return;
    }

    let url: string | undefined;
    try {
      const svg = graphRef.current;
      const serializer = new XMLSerializer();
      const source = serializer.serializeToString(svg);
      const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
      url = URL.createObjectURL(blob);

      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error('Could not render graph snapshot.'));
        image.src = url;
      });

      const viewBox = svg.viewBox.baseVal;
      const width = Math.max(1, Math.floor(viewBox.width || 1200));
      const height = Math.max(1, Math.floor(viewBox.height || 700));

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Canvas context unavailable.');
      }

      context.fillStyle = '#0F1117';
      context.fillRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);

      const pngBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!pngBlob) {
        throw new Error('PNG encoding failed.');
      }
      downloadBlob('graph_snapshot.png', pngBlob, 'image/png', true);
    } catch (snapshotError) {
      setError(snapshotError instanceof Error ? snapshotError.message : 'Graph snapshot export failed.');
    } finally {
      if (url) {
        URL.revokeObjectURL(url);
      }
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Spectral and Information-Geometric Modeling of Philosophical Assumption Structures in Text</h1>
        <p>
          Deterministic local-model pipeline for language-game simplex mapping, information geometry, spectral graph
          analysis, and instability functional evaluation.
        </p>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="top-grid">
        <TextPanel
          text={text}
          weights={weights}
          isLoading={loading}
          hasResult={hasResult}
          onTextChange={setText}
          onWeightsChange={setWeights}
          onAnalyze={runAnalysis}
          onExportJson={exportJson}
          onExportCsv={exportCsv}
          onExportGraphJson={exportGraphJson}
          onExportSummary={exportSummary}
          onExportPng={exportPng}
        />

        <GraphPanel ref={graphRef} graph={result?.graph} />
        <MetricsPanel result={result} />
      </main>

      <HeatmapPanel sentences={sentences} />
    </div>
  );
}

function downloadBlob(fileName: string, data: string | Blob, mimeType: string, isBlob = false) {
  const blob = isBlob ? (data as Blob) : new Blob([data as string], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value: string): string {
  const escaped = value.replace(/"/g, '""');
  return `"${escaped}"`;
}
