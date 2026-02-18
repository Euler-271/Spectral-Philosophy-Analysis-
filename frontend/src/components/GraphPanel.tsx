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

    // Re-setup standard definitions
    const defs = svg.append('defs');
    const glow = defs.append('filter').attr('id', 'node-glow').attr('height', '220%').attr('width', '220%');
    glow.append('feGaussianBlur').attr('stdDeviation', '3.2').attr('result', 'coloredBlur');
    const merge = glow.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

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
      .attr('stroke', '#0F1117')
      .attr('stroke-width', 1.2)
      .style('filter', (d) => (d.centrality > 0.22 ? 'url(#node-glow)' : 'none'))
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
      {!graph ? <div className="placeholder">Run analysis to generate discourse graph.</div> : null}
      <svg ref={mergedRef} className="graph-canvas" />
    </section>
  );
});

function nodeColor(node: SimNode): string {
  if (node.kind === 'language_game') {
    return '#00D4FF';
  }
  if (node.kind === 'polarity') {
    return node.id.endsWith('_pos') ? '#FF5D5D' : '#4ADE80';
  }
  if (node.kind === 'assumption') {
    if (node.label.startsWith('ontological')) {
      return '#3B82F6';
    }
    if (node.label.startsWith('normative')) {
      return '#F59E0B';
    }
    return '#A8B3CF';
  }
  return '#8B949E';
}

function edgeColor(relation: string): string {
  if (relation === 'contradicts') {
    return '#FF7B72';
  }
  if (relation === 'depends_on') {
    return '#F59E0B';
  }
  if (relation === 'derived_from') {
    return '#58A6FF';
  }
  return '#6EE7FF';
}

function trimLabel(label: string): string {
  if (label.length <= 24) {
    return label;
  }
  return `${label.slice(0, 21)}...`;
}
