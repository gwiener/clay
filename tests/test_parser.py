"""
Tests for the Clay DSL parser.

Tests parsing of all DSL statement types:
- Simple nodes (auto-generated labels)
- Nodes with custom labels
- Nodes with properties
- Simple edges
- Chained edges
- Graph settings (@bbox, @verbose, @weight)
"""

from pathlib import Path

from clay.parser import parse_clay_text


def test_comprehensive_parsing():
    """Test parsing all DSL statement types from test_input.clay."""
    # Read test input file
    test_file = Path(__file__).parent / "test_input.clay"
    clay_text = test_file.read_text()

    # Parse the DSL
    parsed = parse_clay_text(clay_text)

    # Test nodes - should have 5 nodes
    assert len(parsed.nodes) == 5, f"Expected 5 nodes, got {len(parsed.nodes)}"
    assert set(parsed.nodes.keys()) == {
        'simple_node', 'labeled_node', 'sized_node', 'fancy_node', 'end_node'
    }

    # Test simple node (auto-generated label)
    custom_label, properties = parsed.nodes['simple_node']
    assert custom_label is None, "simple_node should have no custom label"
    assert properties == {}, "simple_node should have no properties"

    # Test node with custom label
    custom_label, properties = parsed.nodes['labeled_node']
    assert custom_label == "Custom Label", "labeled_node should have custom label"
    assert properties == {}, "labeled_node should have no properties"

    # Test node with properties
    custom_label, properties = parsed.nodes['sized_node']
    assert custom_label == "Sized Node", "sized_node should have custom label"
    assert properties == {'width': 120, 'height': 60}, \
        f"sized_node properties mismatch: {properties}"

    # Test node with multiple properties
    custom_label, properties = parsed.nodes['fancy_node']
    assert custom_label == "Fancy", "fancy_node should have custom label"
    assert properties == {'width': 150, 'height': 50, 'fontsize': 14}, \
        f"fancy_node properties mismatch: {properties}"

    # Test node declared after being referenced in edge
    custom_label, properties = parsed.nodes['end_node']
    assert custom_label == "End", "end_node should have custom label"
    assert properties == {}, "end_node should have no properties"

    # Test edges - should have 4 edges total
    assert len(parsed.edges) == 4, f"Expected 4 edges, got {len(parsed.edges)}"

    expected_edges = [
        ('simple_node', 'labeled_node'),
        ('labeled_node', 'sized_node'),
        ('sized_node', 'fancy_node'),
        ('fancy_node', 'end_node'),
    ]

    # Check all expected edges are present
    for edge in expected_edges:
        assert edge in parsed.edges, f"Expected edge {edge} not found"

    # Verify chained edge was parsed correctly
    assert ('sized_node', 'fancy_node') in parsed.edges, \
        "First part of chain not found"
    assert ('fancy_node', 'end_node') in parsed.edges, \
        "Second part of chain not found"

    # Test settings
    assert len(parsed.settings) == 3, f"Expected 3 settings, got {len(parsed.settings)}"

    # Test @bbox setting
    assert 'bbox' in parsed.settings, "@bbox setting not found"
    assert parsed.settings['bbox'] == ['1000', '800'], \
        f"@bbox values mismatch: {parsed.settings['bbox']}"

    # Test @verbose setting
    assert 'verbose' in parsed.settings, "@verbose setting not found"
    assert parsed.settings['verbose'] == ['false'], \
        f"@verbose value mismatch: {parsed.settings['verbose']}"

    # Test @weight settings (should have 2 weight directives)
    assert 'weight' in parsed.settings, "@weight setting not found"
    # Note: When multiple @weight directives exist, only the last one is stored
    # This tests the current behavior - might need revisiting
    assert parsed.settings['weight'] == ['edge_length', '12'], \
        f"@weight values mismatch: {parsed.settings['weight']}"

    # Test metadata tracking
    assert len(parsed.node_line_numbers) == 5, \
        "Should track line numbers for all nodes"
    assert len(parsed.edge_line_numbers) == 4, \
        "Should track line numbers for all edges"

    # Verify no parsing errors occurred
    assert all(node_id in parsed.nodes for node_id in [
        'simple_node', 'labeled_node', 'sized_node', 'fancy_node', 'end_node'
    ]), "Some nodes were not parsed correctly"
