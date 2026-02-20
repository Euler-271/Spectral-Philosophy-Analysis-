import { ForwardedRef, MutableRefObject, forwardRef, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';

import type { GraphPayload } from '../types';

interface GraphPanelProps {
  graph?: GraphPayload;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  kind: string;
  centrality: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
  relation: string;
  weight: number;
}

function mergeRefs(refA: MutableRefObject<SVGSVGElement | null>, refB: ForwardedRef<SVGSVGElement>) {
  return (node: SVGSVGElement | null) => {
    refA.current = node;
    if (typeof refB === 'function') {
      refB(node);
    } else if (refB && typeof refB === 'object') {
      refB.current = node;
    }
  };
}

export const GraphPanel = forwardRef<SVGSVGElement, GraphPanelProps>(function GraphPanel(
  { graph },
  forwardedRef
) {
  const internalSvgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 920, height: 560 });

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = Math.max(500, Math.floor(entry.contentRect.width));
        const height = Math.max(420, Math.floor(entry.contentRect.height));
        setSize((prev) => {
          if (prev.width === width && prev.height === height) return prev;
          return { width, height };
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const mergedRef = useMemo(() => mergeRefs(internalSvgRef, forwardedRef), [forwardedRef]);

  // Ref to store the simulation instance so it persists across renders
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  // Effect to handle Size updates (update center force and viewBox only — no restart to avoid ResizeObserver loop)
  useEffect(() => {
    const svgElement = internalSvgRef.current;
    if (!svgElement) return;

    const svg = d3.select(svgElement);
    const { width, height } = size;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    if (simulationRef.current) {
      simulationRef.current.force('x', d3.forceX(width / 2).strength(0.08));
      simulationRef.current.force('y', d3.forceY(height / 2).strength(0.08));
      // Only nudge the simulation if it has already cooled down to avoid a
      // ResizeObserver → restart → node movement → container resize → ResizeObserver loop.
      if (simulationRef.current.alpha() < 0.05) {
        simulationRef.current.alpha(0.1).restart();
      }
    }
  }, [size]);

  // Effect to handle Graph Data updates (initialize simulation)
  useEffect(() => {
    const svgElement = internalSvgRef.current;
    if (!svgElement || !graph) return;

    const svg = d3.select(svgElement);
    // Clear previous elements
    svg.selectAll('*').remove();


    const nodes: SimNode[] = graph.nodes.map((node) => ({
      id: node.id,
      label: node.label,
      kind: node.kind,
      centrality: node.centrality
    }));

    const links: SimLink[] = graph.edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      relation: edge.relation,
      weight: edge.weight
    }));

    // Stop old simulation if it exists
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    // Snapshot current size from the SVG element itself to avoid stale state
    const svgRect = svgElement.getBoundingClientRect();
    const initWidth = svgRect.width > 0 ? svgRect.width : size.width;
    const initHeight = svgRect.height > 0 ? svgRect.height : size.height;

    // Initialize new simulation
    const simulation = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance((d) => 60 + (1 - Math.min(d.weight, 1)) * 100)
          .strength((d) => 0.18 + Math.min(d.weight, 1) * 0.48)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX(initWidth / 2).strength(0.08))
      .force('y', d3.forceY(initHeight / 2).strength(0.08))
      .force('collision', d3.forceCollide<SimNode>().radius((d) => 14 + d.centrality * 10));

    simulationRef.current = simulation;

    const link = svg
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', (d) => edgeColor(d.relation))
      .attr('stroke-opacity', 0.58)
      .attr('stroke-width', (d) => 0.8 + Math.min(d.weight, 1.6) * 2.2);

    const node = svg
      .append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('r', (d) => 5 + d.centrality * 12)
      .attr('fill', (d) => nodeColor(d))
      .attr('stroke', 'var(--bg)')
      .attr('stroke-width', 1.5)
      .style('cursor', 'grab')
      .call(
        d3
          .drag<SVGCircleElement, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) {
              simulation.alphaTarget(0.3).restart();
            }
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) {
              simulation.alphaTarget(0);
            }
            d.fx = null;
            d.fy = null;
          })
      );

    node.append('title').text((d) => `${d.kind}: ${d.label}`);

    const labels = svg
      .append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(nodes.filter((n) => n.kind !== 'sentence'))
      .enter()
      .append('text')
      .text((d) => trimLabel(d.label))
      .attr('font-size', 10)
      .attr('fill', '#E6EDF3')
      .attr('pointer-events', 'none');

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (typeof d.source === 'string' ? 0 : d.source.x ?? 0))
        .attr('y1', (d) => (typeof d.source === 'string' ? 0 : d.source.y ?? 0))
        .attr('x2', (d) => (typeof d.target === 'string' ? 0 : d.target.x ?? 0))
        .attr('y2', (d) => (typeof d.target === 'string' ? 0 : d.target.y ?? 0));

      node.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0);

      labels
        .attr('x', (d) => (d.x ?? 0) + 8)
        .attr('y', (d) => (d.y ?? 0) - 8)
        .attr('opacity', (d) => Math.min(1, 0.35 + d.centrality));
    });

    return () => {
      simulation.stop();
    };
    // `size` is intentionally excluded: dimensions are read directly from the SVG
    // element via getBoundingClientRect() to avoid stale closure issues.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  return (
    <section className="panel panel-center" ref={containerRef}>
      <div className="panel-header">Assumption Graph</div>
      {!graph && (
        <div className="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="5" r="3"></circle>
            <line x1="12" y1="22" x2="12" y2="8"></line>
            <path d="M5 12H2a10 10 0 0 0 20 0h-3"></path>
          </svg>
          <p>Ready for analysis</p>
          <span>Select a preset or input text to populate the discourse graph.</span>
        </div>
      )}
      <svg ref={mergedRef} className="graph-canvas" />
    </section>
  );
});

function nodeColor(node: SimNode): string {
  if (node.kind === 'language_game') {
    return 'var(--color-cyan)';
  }
  if (node.kind === 'polarity') {
    return node.id.endsWith('_pos') ? 'var(--color-red)' : 'var(--color-green)';
  }
  if (node.kind === 'assumption') {
    if (node.label.startsWith('ontological')) {
      return 'var(--color-blue)';
    }
    if (node.label.startsWith('normative')) {
      return 'var(--color-amber)';
    }
    return 'var(--accent-dim)';
  }
  return 'var(--text-muted)';
}

function edgeColor(relation: string): string {
  if (relation === 'contradicts') {
    return 'var(--color-red)';
  }
  if (relation === 'depends_on') {
    return 'var(--color-amber)';
  }
  if (relation === 'derived_from') {
    return 'var(--color-blue)';
  }
  return 'var(--border-focus)';
}

function trimLabel(label: string): string {
  if (label.length <= 32) {
    return label;
  }
  return `${label.slice(0, 29)}...`;
}
