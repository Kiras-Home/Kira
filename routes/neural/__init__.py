"""
Kira Neural Module
Neural Network Visualization, Brain Waves und Connection Management

Module:
- network.py: Neural Network Data Generation und Visualization
- brain_waves.py: Brain Wave Simulation und Pattern Analysis
- connections.py: Node Connections, Synergies und Network Topology
- visualization.py: Chart Data, Visual Helpers und Rendering Support
"""

from .network import (
    generate_neural_network_data,
    calculate_network_topology,
    analyze_network_efficiency,
    simulate_neural_activity
)

from .brain_waves import (
    generate_brain_wave_data,
    simulate_brain_wave_patterns,
    analyze_wave_frequencies,
    calculate_wave_coherence
)

from .connections import (
    analyze_node_connections,
    calculate_connection_strength,
    identify_neural_clusters,
    optimize_network_connections
)

from .visualization import (
    prepare_neural_chart_data,
    generate_network_visualization,
    create_brain_wave_charts,
    render_connection_maps
)

__all__ = [
    # Network
    'generate_neural_network_data',
    'calculate_network_topology',
    'analyze_network_efficiency',
    'simulate_neural_activity',
    
    # Brain Waves
    'generate_brain_wave_data',
    'simulate_brain_wave_patterns',
    'analyze_wave_frequencies',
    'calculate_wave_coherence',
    
    # Connections
    'analyze_node_connections',
    'calculate_connection_strength',
    'identify_neural_clusters',
    'optimize_network_connections',
    
    # Visualization
    'prepare_neural_chart_data',
    'generate_network_visualization',
    'create_brain_wave_charts',
    'render_connection_maps'
]