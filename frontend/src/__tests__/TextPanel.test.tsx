import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TextPanel } from '../components/TextPanel';

describe('TextPanel', () => {
    it('renders analyze button', () => {
        const defaultWeights = { alpha: 1, beta: 1, gamma: 1, delta: 1 };

        render(
            <TextPanel
                text=""
                weights={defaultWeights}
                isLoading={false}
                hasResult={false}
                onTextChange={vi.fn()}
                onWeightsChange={vi.fn()}
                onAnalyze={vi.fn()}
                onExportJson={vi.fn()}
                onExportCsv={vi.fn()}
                onExportGraphJson={vi.fn()}
                onExportSummary={vi.fn()}
                onExportPng={vi.fn()}
            />
        );

        const analyzeBtn = screen.getByRole('button', { name: /analyze/i });
        expect(analyzeBtn).toBeInTheDocument();
        expect(analyzeBtn).toBeDisabled(); // empty text
    });
});
