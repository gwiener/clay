#!/usr/bin/env python3
"""
Clay Layout Engine - Comprehensive Examples

This script demonstrates the clay layout engine with various graph types.
"""

from pathlib import Path

from clay import Node, layout_graph, render_graph_matplotlib, render_graph_svg

# Create output directories
FLOWS_DIR = Path('output/flows')
USAGE_DIR = Path('output/usage')

FLOWS_DIR.mkdir(parents=True, exist_ok=True)
USAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SOFTWARE DIAGRAMS → output/flows/
# ============================================================================

def example_architecture() -> None:
    """Software architecture diagram"""
    print("Architecture Diagram")
    print("-" * 70)

    nodes = {
        'user': Node('User'),
        'web': Node('Web Server'),
        'api': Node('API Gateway'),
        'auth': Node('Auth Service'),
        'db': Node('Database'),
        'cache': Node('Cache'),
    }
    edges = [
        ('user', 'web'),
        ('web', 'api'),
        ('api', 'auth'),
        ('api', 'db'),
        ('api', 'cache'),
    ]

    positions = layout_graph(nodes, edges, target_bbox=(700, 500))
    render_graph_matplotlib(nodes, edges, positions, str(FLOWS_DIR / 'architecture.png'))
    render_graph_svg(nodes, edges, positions, str(FLOWS_DIR / 'architecture.svg'))
    print()


def example_state_machine() -> None:
    """State machine diagram"""
    print("State Machine Diagram")
    print("-" * 70)

    nodes = {
        'idle': Node('Idle'),
        'processing': Node('Processing'),
        'waiting': Node('Waiting'),
        'complete': Node('Complete'),
        'error': Node('Error'),
    }
    edges = [
        ('idle', 'processing'),
        ('processing', 'waiting'),
        ('waiting', 'processing'),  # Cycle!
        ('processing', 'complete'),
        ('processing', 'error'),
        ('error', 'idle'),
    ]

    positions = layout_graph(nodes, edges, target_bbox=(600, 500))
    render_graph_matplotlib(nodes, edges, positions, str(FLOWS_DIR / 'state_machine.png'))
    render_graph_svg(nodes, edges, positions, str(FLOWS_DIR / 'state_machine.svg'))
    print()


def example_flowchart() -> None:
    """Flowchart diagram"""
    print("Flowchart Diagram")
    print("-" * 70)

    nodes = {
        'start': Node('Start'),
        'input': Node('Get Input'),
        'valid': Node('Is Valid?'),
        'process': Node('Process'),
        'error': Node('Show Error'),
        'output': Node('Display Result'),
        'end': Node('End'),
    }
    edges = [
        ('start', 'input'),
        ('input', 'valid'),
        ('valid', 'process'),
        ('valid', 'error'),
        ('process', 'output'),
        ('error', 'input'),
        ('output', 'end'),
    ]

    positions = layout_graph(nodes, edges, target_bbox=(700, 600))
    render_graph_matplotlib(nodes, edges, positions, str(FLOWS_DIR / 'flowchart.png'))
    render_graph_svg(nodes, edges, positions, str(FLOWS_DIR / 'flowchart.svg'))
    print()


# ============================================================================
# GRAPH PATTERNS → output/usage/
# ============================================================================

def example_chain() -> None:
    """Simple chain: A→B→C→D"""
    print("Simple Chain (A→B→C→D)")
    print("-" * 70)

    nodes = {
        'A': Node('Start'),
        'B': Node('Process'),
        'C': Node('Decision'),
        'D': Node('End'),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]

    positions = layout_graph(nodes, edges, target_bbox=(600, 200))
    render_graph_matplotlib(nodes, edges, positions, str(USAGE_DIR / 'chain.png'))
    render_graph_svg(nodes, edges, positions, str(USAGE_DIR / 'chain.svg'))
    print()


def example_yshape() -> None:
    """Y-shape: A→B splits to C and D"""
    print("Y-Shape (A→B splits to C and D)")
    print("-" * 70)

    nodes = {
        'A': Node('Input'),
        'B': Node('Process'),
        'C': Node('Output 1'),
        'D': Node('Output 2'),
    }
    edges = [('A', 'B'), ('B', 'C'), ('B', 'D')]

    positions = layout_graph(nodes, edges, target_bbox=(500, 400))
    render_graph_matplotlib(nodes, edges, positions, str(USAGE_DIR / 'yshape.png'))
    render_graph_svg(nodes, edges, positions, str(USAGE_DIR / 'yshape.svg'))
    print()


def example_cycle() -> None:
    """Cycle: A→B→C→A"""
    print("Cycle (A→B→C→A)")
    print("-" * 70)

    nodes = {
        'A': Node('State A'),
        'B': Node('State B'),
        'C': Node('State C'),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]

    positions = layout_graph(nodes, edges, target_bbox=(400, 400))
    render_graph_matplotlib(nodes, edges, positions, str(USAGE_DIR / 'cycle.png'))
    render_graph_svg(nodes, edges, positions, str(USAGE_DIR / 'cycle.svg'))
    print()


def example_workflow() -> None:
    """Complex workflow with error handling"""
    print("Complex Workflow")
    print("-" * 70)

    nodes = {
        'start': Node('Start', width=100, height=50),
        'fetch': Node('Fetch Data', width=120, height=40),
        'validate': Node('Validate', width=100, height=40),
        'process': Node('Process', width=100, height=40),
        'store': Node('Store', width=90, height=40),
        'notify': Node('Notify', width=90, height=40),
        'error': Node('Error Handler', width=130, height=40),
        'end': Node('End', width=100, height=50),
    }
    edges = [
        ('start', 'fetch'),
        ('fetch', 'validate'),
        ('validate', 'process'),
        ('validate', 'error'),
        ('process', 'store'),
        ('store', 'notify'),
        ('notify', 'end'),
        ('error', 'notify'),
    ]

    positions = layout_graph(nodes, edges, target_bbox=(800, 600))
    render_graph_matplotlib(nodes, edges, positions, str(USAGE_DIR / 'workflow.png'))
    render_graph_svg(nodes, edges, positions, str(USAGE_DIR / 'workflow.svg'))
    print()


def example_diamond() -> None:
    """Diamond: A splits to B,C which merge at D"""
    print("Diamond (A splits to B,C which merge at D)")
    print("-" * 70)

    nodes = {
        'A': Node('Start'),
        'B': Node('Branch 1'),
        'C': Node('Branch 2'),
        'D': Node('Merge'),
        'E': Node('End'),
    }
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('D', 'E'),
    ]

    positions = layout_graph(nodes, edges, target_bbox=(600, 500))
    render_graph_matplotlib(nodes, edges, positions, str(USAGE_DIR / 'diamond.png'))
    render_graph_svg(nodes, edges, positions, str(USAGE_DIR / 'diamond.svg'))
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Clay Layout Engine - Examples")
    print("=" * 70)
    print()

    # Software diagrams → flows
    example_architecture()
    example_state_machine()
    example_flowchart()

    # Graph patterns → usage
    example_chain()
    example_yshape()
    example_cycle()
    example_workflow()
    example_diamond()

    print("=" * 70)
    print("All examples completed!")
    print()
    print("Generated files:")
    print("  output/flows/:")
    print("    - architecture.png/svg")
    print("    - state_machine.png/svg")
    print("    - flowchart.png/svg")
    print()
    print("  output/usage/:")
    print("    - chain.png/svg")
    print("    - yshape.png/svg")
    print("    - cycle.png/svg")
    print("    - workflow.png/svg")
    print("    - diamond.png/svg")
    print("=" * 70)
