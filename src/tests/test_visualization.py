import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from processors.visualization import (build_reasoning_graph,
                                      visualize_reasoning_chain)


def test_build_reasoning_graph_basic():
    deps = [
        {"source": "A", "target": "B", "relation": "depends_on"},
        {"source": "B", "target": "C", "relation": "depends_on"}
    ]
    G, node_type_map = build_reasoning_graph([deps], ["explicit"])
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert isinstance(node_type_map, dict)

def test_visualize_reasoning_chain(tmp_path):
    deps = [
        {"source": "A", "target": "B", "relation": "depends_on"},
        {"source": "B", "target": "C", "relation": "depends_on"}
    ]
    G, node_type_map = build_reasoning_graph([deps], ["explicit"])
    save_path = tmp_path / "test_chain.png"
    visualize_reasoning_chain(G, node_type_map, title="Test Chain", save_path=str(save_path), show=False)
    assert save_path.exists() 