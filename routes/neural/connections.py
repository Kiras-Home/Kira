"""
Neural Visualization Module
Chart Data Preparation, Visual Helpers und Rendering Support
"""

import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def prepare_neural_chart_data(neural_data: Dict,
                            chart_types: List[str] = None,
                            visualization_params: Dict = None) -> Dict[str, Any]:
    """
    Bereitet Neural Chart Data für Visualization vor
    
    Extrahiert aus kira_routes.py.backup Chart Data Preparation Logic
    """
    try:
        if chart_types is None:
            chart_types = ['network_graph', 'brain_waves', 'activity_timeline', 'frequency_spectrum']
        
        if visualization_params is None:
            visualization_params = {
                'max_nodes': 100,
                'max_edges': 200,
                'time_resolution': 100,
                'color_scheme': 'neural_default'
            }
        
        chart_data = {}
        
        # Network Graph Chart Data
        if 'network_graph' in chart_types:
            chart_data['network_graph'] = _prepare_network_graph_data(
                neural_data, visualization_params
            )
        
        # Brain Waves Chart Data
        if 'brain_waves' in chart_types:
            chart_data['brain_waves'] = _prepare_brain_waves_chart_data(
                neural_data, visualization_params
            )
        
        # Activity Timeline Chart Data
        if 'activity_timeline' in chart_types:
            chart_data['activity_timeline'] = _prepare_activity_timeline_data(
                neural_data, visualization_params
            )
        
        # Frequency Spectrum Chart Data
        if 'frequency_spectrum' in chart_types:
            chart_data['frequency_spectrum'] = _prepare_frequency_spectrum_data(
                neural_data, visualization_params
            )
        
        # Connection Strength Heatmap Data
        if 'connection_heatmap' in chart_types:
            chart_data['connection_heatmap'] = _prepare_connection_heatmap_data(
                neural_data, visualization_params
            )
        
        # Node Activity Bubble Chart Data
        if 'node_activity_bubbles' in chart_types:
            chart_data['node_activity_bubbles'] = _prepare_node_activity_bubble_data(
                neural_data, visualization_params
            )
        
        # Chart metadata
        chart_data['chart_metadata'] = {
            'preparation_timestamp': datetime.now().isoformat(),
            'chart_types_prepared': chart_types,
            'visualization_params': visualization_params,
            'data_quality_score': _assess_chart_data_quality(chart_data),
            'rendering_recommendations': _generate_rendering_recommendations(chart_data)
        }
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Neural chart data preparation failed: {e}")
        return {
            'error': str(e),
            'fallback_chart_data': _generate_fallback_chart_data(chart_types)
        }

def generate_network_visualization(network_structure: Dict,
                                 visualization_type: str = 'force_directed',
                                 layout_params: Dict = None) -> Dict[str, Any]:
    """
    Generiert Network Visualization Data
    
    Basiert auf kira_routes.py.backup Network Visualization Logic
    """
    try:
        if not network_structure:
            return {'available': False, 'reason': 'no_network_structure'}
        
        if layout_params is None:
            layout_params = {
                'width': 1000,
                'height': 800,
                'node_size_factor': 1.0,
                'edge_thickness_factor': 1.0,
                'spacing': 50
            }
        
        nodes = network_structure.get('nodes', [])
        edges = network_structure.get('edges', [])
        
        # Generate visualization based on type
        if visualization_type == 'force_directed':
            visualization_data = _generate_force_directed_layout(nodes, edges, layout_params)
        elif visualization_type == 'hierarchical':
            visualization_data = _generate_hierarchical_layout(nodes, edges, layout_params)
        elif visualization_type == 'circular':
            visualization_data = _generate_circular_layout(nodes, edges, layout_params)
        elif visualization_type == 'cluster_based':
            visualization_data = _generate_cluster_based_layout(nodes, edges, layout_params)
        else:
            visualization_data = _generate_force_directed_layout(nodes, edges, layout_params)
        
        # Add visualization enhancements
        enhanced_visualization = _enhance_visualization_data(
            visualization_data, network_structure, layout_params
        )
        
        # Generate interaction data
        interaction_data = _generate_visualization_interaction_data(enhanced_visualization)
        
        # Visualization metadata
        visualization_result = {
            'visualization_type': visualization_type,
            'visualization_data': enhanced_visualization,
            'interaction_data': interaction_data,
            'layout_params': layout_params,
            'rendering_instructions': _generate_rendering_instructions(enhanced_visualization, visualization_type),
            'performance_optimization': _generate_performance_optimization_suggestions(enhanced_visualization)
        }
        
        return visualization_result
        
    except Exception as e:
        logger.error(f"Network visualization generation failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_visualization': _generate_fallback_network_visualization()
        }

def create_brain_wave_charts(brain_wave_data: Dict,
                           chart_configurations: List[Dict] = None) -> Dict[str, Any]:
    """
    Erstellt Brain Wave Charts
    
    Extrahiert aus kira_routes.py.backup Brain Wave Chart Creation Logic
    """
    try:
        if not brain_wave_data:
            return {'available': False, 'reason': 'no_brain_wave_data'}
        
        if chart_configurations is None:
            chart_configurations = [
                {
                    'chart_type': 'time_series',
                    'wave_types': ['alpha', 'beta', 'theta', 'delta', 'gamma'],
                    'time_range': 'full',
                    'style': 'overlaid'
                },
                {
                    'chart_type': 'frequency_spectrum',
                    'wave_types': ['all'],
                    'frequency_range': (0.5, 100),
                    'style': 'power_spectrum'
                },
                {
                    'chart_type': 'coherence_matrix',
                    'wave_types': ['alpha', 'beta', 'theta'],
                    'style': 'heatmap'
                }
            ]
        
        brain_wave_charts = {}
        
        for config in chart_configurations:
            chart_type = config.get('chart_type', 'time_series')
            chart_id = f"{chart_type}_{len(brain_wave_charts)}"
            
            if chart_type == 'time_series':
                brain_wave_charts[chart_id] = _create_time_series_chart(brain_wave_data, config)
            elif chart_type == 'frequency_spectrum':
                brain_wave_charts[chart_id] = _create_frequency_spectrum_chart(brain_wave_data, config)
            elif chart_type == 'coherence_matrix':
                brain_wave_charts[chart_id] = _create_coherence_matrix_chart(brain_wave_data, config)
            elif chart_type == 'phase_plot':
                brain_wave_charts[chart_id] = _create_phase_plot_chart(brain_wave_data, config)
            elif chart_type == 'wavelet_transform':
                brain_wave_charts[chart_id] = _create_wavelet_transform_chart(brain_wave_data, config)
        
        # Charts metadata
        charts_result = {
            'brain_wave_charts': brain_wave_charts,
            'chart_count': len(brain_wave_charts),
            'chart_configurations': chart_configurations,
            'charts_metadata': {
                'creation_timestamp': datetime.now().isoformat(),
                'data_source_quality': _assess_brain_wave_data_quality(brain_wave_data),
                'rendering_requirements': _determine_rendering_requirements(brain_wave_charts)
            }
        }
        
        return charts_result
        
    except Exception as e:
        logger.error(f"Brain wave chart creation failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_charts': _generate_fallback_brain_wave_charts()
        }

def render_connection_maps(connection_data: Dict,
                         map_type: str = 'strength_based',
                         rendering_params: Dict = None) -> Dict[str, Any]:
    """
    Rendert Connection Maps
    
    Basiert auf kira_routes.py.backup Connection Map Rendering Logic
    """
    try:
        if not connection_data:
            return {'available': False, 'reason': 'no_connection_data'}
        
        if rendering_params is None:
            rendering_params = {
                'map_size': (1000, 800),
                'connection_threshold': 0.3,
                'color_intensity_mapping': True,
                'show_labels': True,
                'animation_enabled': False
            }
        
        # Generate connection map based on type
        if map_type == 'strength_based':
            connection_map = _render_strength_based_map(connection_data, rendering_params)
        elif map_type == 'cluster_based':
            connection_map = _render_cluster_based_map(connection_data, rendering_params)
        elif map_type == 'activity_based':
            connection_map = _render_activity_based_map(connection_data, rendering_params)
        elif map_type == 'hierarchy_based':
            connection_map = _render_hierarchy_based_map(connection_data, rendering_params)
        else:
            connection_map = _render_strength_based_map(connection_data, rendering_params)
        
        # Add interactive elements
        interactive_elements = _generate_interactive_elements(connection_map, connection_data)
        
        # Map enhancements
        enhanced_map = _enhance_connection_map(connection_map, interactive_elements, rendering_params)
        
        # Rendering result
        rendering_result = {
            'map_type': map_type,
            'connection_map': enhanced_map,
            'interactive_elements': interactive_elements,
            'rendering_params': rendering_params,
            'map_statistics': _calculate_map_statistics(enhanced_map),
            'rendering_metadata': {
                'rendering_timestamp': datetime.now().isoformat(),
                'map_complexity': _assess_map_complexity(enhanced_map),
                'performance_metrics': _calculate_rendering_performance_metrics(enhanced_map)
            }
        }
        
        return rendering_result
        
    except Exception as e:
        logger.error(f"Connection map rendering failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_map': _generate_fallback_connection_map()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _prepare_network_graph_data(neural_data: Dict, params: Dict) -> Dict[str, Any]:
    """Bereitet Network Graph Data vor"""
    try:
        network_structure = neural_data.get('network_structure', {})
        nodes = network_structure.get('nodes', [])
        edges = network_structure.get('edges', [])
        
        max_nodes = params.get('max_nodes', 100)
        max_edges = params.get('max_edges', 200)
        
        # Limit nodes and edges for performance
        limited_nodes = nodes[:max_nodes]
        limited_edges = edges[:max_edges]
        
        # Prepare graph data
        graph_data = {
            'nodes': [
                {
                    'id': node.get('id', f"node_{i}"),
                    'label': node.get('type', 'Node'),
                    'x': node.get('position', {}).get('x', random.uniform(0, 1000)),
                    'y': node.get('position', {}).get('y', random.uniform(0, 800)),
                    'size': _calculate_node_size(node),
                    'color': _determine_node_color(node, params.get('color_scheme', 'neural_default'))
                }
                for i, node in enumerate(limited_nodes)
            ],
            'edges': [
                {
                    'id': edge.get('id', f"edge_{i}"),
                    'source': edge.get('source', ''),
                    'target': edge.get('target', ''),
                    'weight': edge.get('properties', {}).get('weight', 0.5),
                    'color': _determine_edge_color(edge, params.get('color_scheme', 'neural_default')),
                    'thickness': _calculate_edge_thickness(edge)
                }
                for i, edge in enumerate(limited_edges)
            ]
        }
        
        return graph_data
        
    except Exception as e:
        logger.debug(f"Network graph data preparation failed: {e}")
        return {'nodes': [], 'edges': []}

def _prepare_brain_waves_chart_data(neural_data: Dict, params: Dict) -> Dict[str, Any]:
    """Bereitet Brain Waves Chart Data vor"""
    try:
        brain_wave_data = neural_data.get('wave_patterns', {})
        time_resolution = params.get('time_resolution', 100)
        
        chart_data = {
            'datasets': [],
            'labels': [],
            'time_axis': []
        }
        
        # Process each wave type
        for wave_type, wave_info in brain_wave_data.items():
            if isinstance(wave_info, dict) and 'samples' in wave_info:
                samples = wave_info['samples']
                sample_rate = wave_info.get('sample_rate', 100)
                
                # Downsample for visualization if needed
                if len(samples) > time_resolution:
                    step = len(samples) // time_resolution
                    samples = samples[::step]
                
                # Create time axis
                duration = wave_info.get('duration', 60)
                time_points = [i * duration / len(samples) for i in range(len(samples))]
                
                dataset = {
                    'label': wave_type.replace('_', ' ').title(),
                    'data': samples,
                    'time_points': time_points,
                    'color': _get_wave_color(wave_type),
                    'line_style': _get_wave_line_style(wave_type)
                }
                
                chart_data['datasets'].append(dataset)
        
        # Set common time axis
        if chart_data['datasets']:
            chart_data['time_axis'] = chart_data['datasets'][0]['time_points']
        
        return chart_data
        
    except Exception as e:
        logger.debug(f"Brain waves chart data preparation failed: {e}")
        return {'datasets': [], 'labels': [], 'time_axis': []}

def _generate_force_directed_layout(nodes: List, edges: List, params: Dict) -> Dict[str, Any]:
    """Generiert Force-Directed Layout"""
    try:
        width = params.get('width', 1000)
        height = params.get('height', 800)
        
        # Initialize random positions if not provided
        positioned_nodes = []
        for i, node in enumerate(nodes):
            position = node.get('position', {})
            positioned_node = {
                'id': node.get('id', f"node_{i}"),
                'x': position.get('x', random.uniform(100, width - 100)),
                'y': position.get('y', random.uniform(100, height - 100)),
                'type': node.get('type', 'unknown'),
                'properties': node.get('properties', {}),
                'force_data': {
                    'mass': _calculate_node_mass(node),
                    'charge': _calculate_node_charge(node)
                }
            }
            positioned_nodes.append(positioned_node)
        
        # Process edges for force calculation
        processed_edges = []
        for edge in edges:
            processed_edge = {
                'source': edge.get('source', ''),
                'target': edge.get('target', ''),
                'strength': edge.get('properties', {}).get('weight', 0.5),
                'ideal_length': _calculate_ideal_edge_length(edge),
                'spring_constant': _calculate_spring_constant(edge)
            }
            processed_edges.append(processed_edge)
        
        # Force-directed layout simulation (simplified)
        layout_data = {
            'nodes': positioned_nodes,
            'edges': processed_edges,
            'layout_type': 'force_directed',
            'simulation_params': {
                'iterations': 100,
                'cooling_factor': 0.99,
                'repulsion_strength': 1000,
                'attraction_strength': 0.1
            }
        }
        
        return layout_data
        
    except Exception as e:
        logger.debug(f"Force-directed layout generation failed: {e}")
        return {'nodes': [], 'edges': [], 'layout_type': 'force_directed'}

def _create_time_series_chart(brain_wave_data: Dict, config: Dict) -> Dict[str, Any]:
    """Erstellt Time Series Chart"""
    try:
        wave_patterns = brain_wave_data.get('wave_patterns', {})
        wave_types = config.get('wave_types', ['alpha'])
        style = config.get('style', 'overlaid')
        
        chart_data = {
            'chart_type': 'time_series',
            'chart_style': style,
            'datasets': [],
            'axes': {
                'x': {'label': 'Time (seconds)', 'type': 'continuous'},
                'y': {'label': 'Amplitude', 'type': 'continuous'}
            }
        }
        
        for wave_type in wave_types:
            if wave_type in wave_patterns:
                wave_info = wave_patterns[wave_type]
                if isinstance(wave_info, dict) and 'samples' in wave_info:
                    samples = wave_info['samples']
                    duration = wave_info.get('duration', 60)
                    
                    # Create time points
                    time_points = [i * duration / len(samples) for i in range(len(samples))]
                    
                    dataset = {
                        'label': wave_type.replace('_', ' ').title(),
                        'data': list(zip(time_points, samples)),
                        'color': _get_wave_color(wave_type),
                        'line_width': 2,
                        'opacity': 0.8 if style == 'overlaid' else 1.0
                    }
                    
                    chart_data['datasets'].append(dataset)
        
        return chart_data
        
    except Exception as e:
        logger.debug(f"Time series chart creation failed: {e}")
        return {'chart_type': 'time_series', 'datasets': []}

def _calculate_node_size(node: Dict) -> float:
    """Berechnet Node Size für Visualization"""
    try:
        base_size = 10
        
        # Size based on node properties
        properties = node.get('properties', {})
        processing_capacity = properties.get('processing_capacity', 0.5)
        memory_capacity = properties.get('memory_capacity', 50)
        
        # Calculate size factor
        size_factor = 1.0 + (processing_capacity * 0.5) + (min(memory_capacity, 100) / 100 * 0.3)
        
        return base_size * size_factor
        
    except Exception as e:
        logger.debug(f"Node size calculation failed: {e}")
        return 10

def _determine_node_color(node: Dict, color_scheme: str) -> str:
    """Bestimmt Node Color für Visualization"""
    try:
        node_type = node.get('type', 'unknown')
        
        color_schemes = {
            'neural_default': {
                'memory_node': '#FF6B6B',
                'processing_node': '#4ECDC4',
                'sensory_node': '#45B7D1',
                'motor_node': '#96CEB4',
                'emotional_node': '#FFEAA7',
                'cognitive_node': '#DDA0DD',
                'association_node': '#FFB347',
                'output_node': '#87CEEB',
                'unknown': '#CCCCCC'
            },
            'activity_based': {
                'high_activity': '#FF4444',
                'medium_activity': '#FFAA44',
                'low_activity': '#44AAFF',
                'inactive': '#CCCCCC'
            }
        }
        
        scheme = color_schemes.get(color_scheme, color_schemes['neural_default'])
        return scheme.get(node_type, scheme.get('unknown', '#CCCCCC'))
        
    except Exception as e:
        logger.debug(f"Node color determination failed: {e}")
        return '#CCCCCC'

def _get_wave_color(wave_type: str) -> str:
    """Holt Wave Color für Visualization"""
    wave_colors = {
        'delta_waves': '#8B4513',
        'theta_waves': '#4B0082',
        'alpha_waves': '#006400',
        'beta_waves': '#FF4500',
        'gamma_waves': '#FF1493'
    }
    return wave_colors.get(wave_type, '#000000')

def _generate_fallback_chart_data(chart_types: List[str]) -> Dict[str, Any]:
    """Generiert Fallback Chart Data"""
    fallback_data = {
        'fallback_mode': True,
        'chart_types_requested': chart_types
    }
    
    for chart_type in chart_types:
        if chart_type == 'network_graph':
            fallback_data['network_graph'] = {
                'nodes': [{'id': f"node_{i}", 'x': i*50, 'y': i*30} for i in range(5)],
                'edges': []
            }
        elif chart_type == 'brain_waves':
            fallback_data['brain_waves'] = {
                'datasets': [{
                    'label': 'Alpha Wave',
                    'data': [math.sin(i/10) for i in range(100)],
                    'color': '#006400'
                }]
            }
    
    return fallback_data

def _calculate_wave_synchronization(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Berechnet Wave Synchronization zwischen verschiedenen Brain Wave Types"""
    try:
        synchronization_metrics = {}
        wave_types = list(brain_wave_patterns.keys())
        
        # Calculate synchronization between all wave pairs
        for i, wave_type1 in enumerate(wave_types):
            for wave_type2 in wave_types[i+1:]:
                wave_data1 = brain_wave_patterns[wave_type1]
                wave_data2 = brain_wave_patterns[wave_type2]
                
                if isinstance(wave_data1, dict) and isinstance(wave_data2, dict):
                    samples1 = wave_data1.get('samples', [])
                    samples2 = wave_data2.get('samples', [])
                    
                    if samples1 and samples2:
                        sync_key = f"{wave_type1}_{wave_type2}_sync"
                        
                        # Calculate various synchronization metrics
                        phase_sync = _calculate_phase_synchronization(samples1, samples2)
                        amplitude_sync = _calculate_amplitude_synchronization(samples1, samples2)
                        frequency_sync = _calculate_frequency_synchronization(samples1, samples2)
                        
                        synchronization_metrics[sync_key] = {
                            'phase_synchronization': phase_sync,
                            'amplitude_synchronization': amplitude_sync,
                            'frequency_synchronization': frequency_sync,
                            'overall_synchronization': (phase_sync + amplitude_sync + frequency_sync) / 3
                        }
        
        # Calculate global synchronization metrics
        if synchronization_metrics:
            sync_values = [metrics['overall_synchronization'] for metrics in synchronization_metrics.values()]
            global_sync = {
                'mean_synchronization': statistics.mean(sync_values),
                'synchronization_stability': 1.0 - statistics.stdev(sync_values) if len(sync_values) > 1 else 1.0,
                'peak_synchronization': max(sync_values),
                'synchronization_consistency': min(sync_values) / max(sync_values) if max(sync_values) > 0 else 0.0
            }
        else:
            global_sync = {
                'mean_synchronization': 0.5,
                'synchronization_stability': 0.5,
                'peak_synchronization': 0.5,
                'synchronization_consistency': 0.5
            }
        
        return {
            'pairwise_synchronization': synchronization_metrics,
            'global_synchronization': global_sync,
            'synchronization_patterns': _identify_synchronization_patterns(synchronization_metrics),
            'neural_coherence_index': _calculate_neural_coherence_index(synchronization_metrics)
        }
        
    except Exception as e:
        logger.debug(f"Wave synchronization calculation failed: {e}")
        return {
            'error': str(e),
            'fallback_synchronization': {
                'global_synchronization': {'mean_synchronization': 0.6},
                'neural_coherence_index': 0.6
            }
        }

def _calculate_phase_synchronization(samples1: List[float], samples2: List[float]) -> float:
    """Berechnet Phase Synchronization zwischen zwei Wave Patterns"""
    try:
        if not samples1 or not samples2 or len(samples1) != len(samples2):
            return 0.5
        
        # Calculate instantaneous phases using Hilbert transform approximation
        phases1 = _estimate_instantaneous_phase(samples1)
        phases2 = _estimate_instantaneous_phase(samples2)
        
        # Calculate phase differences
        phase_diffs = []
        for p1, p2 in zip(phases1, phases2):
            diff = abs(p1 - p2)
            # Normalize to [0, π]
            diff = min(diff, 2 * math.pi - diff)
            phase_diffs.append(diff)
        
        # Phase synchronization index (0 = no sync, 1 = perfect sync)
        mean_phase_diff = statistics.mean(phase_diffs)
        phase_sync = 1.0 - (mean_phase_diff / math.pi)
        
        return max(0.0, min(1.0, phase_sync))
        
    except Exception as e:
        logger.debug(f"Phase synchronization calculation failed: {e}")
        return 0.5

def _calculate_amplitude_synchronization(samples1: List[float], samples2: List[float]) -> float:
    """Berechnet Amplitude Synchronization zwischen zwei Wave Patterns"""
    try:
        if not samples1 or not samples2 or len(samples1) != len(samples2):
            return 0.5
        
        # Calculate amplitude envelopes
        envelope1 = _calculate_amplitude_envelope(samples1)
        envelope2 = _calculate_amplitude_envelope(samples2)
        
        # Calculate correlation between envelopes
        correlation = _calculate_correlation(envelope1, envelope2)
        
        # Convert correlation to synchronization (0-1 scale)
        amplitude_sync = (correlation + 1.0) / 2.0
        
        return max(0.0, min(1.0, amplitude_sync))
        
    except Exception as e:
        logger.debug(f"Amplitude synchronization calculation failed: {e}")
        return 0.5

def _calculate_frequency_synchronization(samples1: List[float], samples2: List[float]) -> float:
    """Berechnet Frequency Synchronization zwischen zwei Wave Patterns"""
    try:
        if not samples1 or not samples2:
            return 0.5
        
        # Calculate instantaneous frequencies
        freq1 = _estimate_instantaneous_frequency(samples1)
        freq2 = _estimate_instantaneous_frequency(samples2)
        
        if not freq1 or not freq2:
            return 0.5
        
        # Calculate frequency synchronization
        freq_diffs = []
        for f1, f2 in zip(freq1, freq2):
            if f1 > 0 and f2 > 0:
                ratio = min(f1/f2, f2/f1)  # Ratio between 0 and 1
                freq_diffs.append(ratio)
        
        if freq_diffs:
            frequency_sync = statistics.mean(freq_diffs)
        else:
            frequency_sync = 0.5
        
        return max(0.0, min(1.0, frequency_sync))
        
    except Exception as e:
        logger.debug(f"Frequency synchronization calculation failed: {e}")
        return 0.5

def _estimate_instantaneous_phase(samples: List[float]) -> List[float]:
    """Schätzt Instantaneous Phase (vereinfachte Hilbert Transform)"""
    try:
        phases = []
        for i in range(len(samples)):
            # Simplified phase estimation using derivative
            if i == 0:
                phase = 0.0
            else:
                # Use arctan of sample and its derivative
                derivative = samples[i] - samples[i-1]
                phase = math.atan2(derivative, samples[i]) if samples[i] != 0 else 0.0
            
            phases.append(phase)
        
        return phases
        
    except Exception as e:
        logger.debug(f"Instantaneous phase estimation failed: {e}")
        return [0.0] * len(samples)

def _calculate_amplitude_envelope(samples: List[float]) -> List[float]:
    """Berechnet Amplitude Envelope"""
    try:
        envelope = []
        window_size = max(5, len(samples) // 100)  # Adaptive window size
        
        for i in range(len(samples)):
            # Calculate local maximum in window
            start = max(0, i - window_size // 2)
            end = min(len(samples), i + window_size // 2 + 1)
            
            local_samples = samples[start:end]
            local_max = max(abs(s) for s in local_samples) if local_samples else 0.0
            envelope.append(local_max)
        
        return envelope
        
    except Exception as e:
        logger.debug(f"Amplitude envelope calculation failed: {e}")
        return [abs(s) for s in samples]

def _estimate_instantaneous_frequency(samples: List[float]) -> List[float]:
    """Schätzt Instantaneous Frequency"""
    try:
        frequencies = []
        
        # Calculate zero crossings for frequency estimation
        for i in range(len(samples) - 1):
            # Simple frequency estimation based on zero crossings
            if len(samples) > i + 10:  # Need enough samples
                window = samples[i:i+10]
                zero_crossings = 0
                
                for j in range(1, len(window)):
                    if (window[j-1] >= 0 and window[j] < 0) or (window[j-1] < 0 and window[j] >= 0):
                        zero_crossings += 1
                
                # Estimate frequency (assuming 100 Hz sample rate)
                freq = zero_crossings * 100 / (2 * 10) if zero_crossings > 0 else 0.0
                frequencies.append(freq)
            else:
                frequencies.append(0.0)
        
        # Add last frequency
        if frequencies:
            frequencies.append(frequencies[-1])
        else:
            frequencies = [0.0] * len(samples)
        
        return frequencies
        
    except Exception as e:
        logger.debug(f"Instantaneous frequency estimation failed: {e}")
        return [0.0] * len(samples)

def _calculate_correlation(data1: List[float], data2: List[float]) -> float:
    """Berechnet Correlation zwischen zwei Datensätzen"""
    try:
        if not data1 or not data2 or len(data1) != len(data2):
            return 0.0
        
        n = len(data1)
        if n < 2:
            return 0.0
        
        # Calculate means
        mean1 = sum(data1) / n
        mean2 = sum(data2) / n
        
        # Calculate correlation coefficient
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))
        
        sum_sq1 = sum((x - mean1) ** 2 for x in data1)
        sum_sq2 = sum((x - mean2) ** 2 for x in data2)
        
        denominator = (sum_sq1 * sum_sq2) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return max(-1.0, min(1.0, correlation))
        
    except Exception as e:
        logger.debug(f"Correlation calculation failed: {e}")
        return 0.0

def _identify_synchronization_patterns(synchronization_metrics: Dict) -> Dict[str, Any]:
    """Identifiziert Synchronization Patterns"""
    try:
        if not synchronization_metrics:
            return {'pattern_type': 'none', 'pattern_strength': 0.0}
        
        # Extract synchronization values
        sync_values = []
        for metrics in synchronization_metrics.values():
            if isinstance(metrics, dict):
                sync_values.append(metrics.get('overall_synchronization', 0.0))
        
        if not sync_values:
            return {'pattern_type': 'none', 'pattern_strength': 0.0}
        
        mean_sync = statistics.mean(sync_values)
        sync_variance = statistics.variance(sync_values) if len(sync_values) > 1 else 0.0
        
        # Determine pattern type
        if mean_sync > 0.8:
            pattern_type = 'high_coherence'
        elif mean_sync > 0.6:
            pattern_type = 'moderate_coherence'
        elif sync_variance < 0.05:
            pattern_type = 'stable_low_coherence'
        else:
            pattern_type = 'variable_coherence'
        
        return {
            'pattern_type': pattern_type,
            'pattern_strength': mean_sync,
            'pattern_stability': 1.0 - sync_variance,
            'synchronization_distribution': _classify_sync_distribution(sync_values)
        }
        
    except Exception as e:
        logger.debug(f"Synchronization patterns identification failed: {e}")
        return {'pattern_type': 'unknown', 'pattern_strength': 0.5}

def _calculate_neural_coherence_index(synchronization_metrics: Dict) -> float:
    """Berechnet Neural Coherence Index"""
    try:
        if not synchronization_metrics:
            return 0.5
        
        # Collect all synchronization values
        all_sync_values = []
        for metrics in synchronization_metrics.values():
            if isinstance(metrics, dict):
                phase_sync = metrics.get('phase_synchronization', 0.0)
                amp_sync = metrics.get('amplitude_synchronization', 0.0)
                freq_sync = metrics.get('frequency_synchronization', 0.0)
                all_sync_values.extend([phase_sync, amp_sync, freq_sync])
        
        if not all_sync_values:
            return 0.5
        
        # Neural coherence is weighted average with stability factor
        mean_coherence = statistics.mean(all_sync_values)
        coherence_stability = 1.0 - statistics.stdev(all_sync_values) if len(all_sync_values) > 1 else 1.0
        
        # Weight coherence by stability
        neural_coherence_index = mean_coherence * (0.7 + 0.3 * coherence_stability)
        
        return max(0.0, min(1.0, neural_coherence_index))
        
    except Exception as e:
        logger.debug(f"Neural coherence index calculation failed: {e}")
        return 0.6

def _classify_sync_distribution(sync_values: List[float]) -> str:
    """Klassifiziert Synchronization Distribution"""
    try:
        if not sync_values:
            return 'empty'
        
        if len(sync_values) == 1:
            return 'single'
        
        mean_val = statistics.mean(sync_values)
        median_val = statistics.median(sync_values)
        
        # Check distribution shape
        if abs(mean_val - median_val) < 0.1:
            return 'balanced'
        elif mean_val > median_val:
            return 'right_skewed'
        else:
            return 'left_skewed'
            
    except Exception as e:
        logger.debug(f"Sync distribution classification failed: {e}")
        return 'unknown'

def _analyze_wave_patterns(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Analysiert Wave Patterns"""
    try:
        pattern_analysis = {}
        
        for wave_type, wave_data in brain_wave_patterns.items():
            if isinstance(wave_data, dict) and 'samples' in wave_data:
                samples = wave_data['samples']
                
                if samples:
                    # Pattern characteristics
                    pattern_analysis[wave_type] = {
                        'amplitude_statistics': _calculate_amplitude_statistics(samples),
                        'frequency_stability': _assess_frequency_stability(samples),
                        'wave_regularity': _assess_wave_regularity(samples),
                        'burst_patterns': _detect_burst_patterns(samples),
                        'rhythm_consistency': _assess_rhythm_consistency(samples)
                    }
        
        # Overall pattern assessment
        overall_patterns = {
            'pattern_complexity': _assess_overall_pattern_complexity(pattern_analysis),
            'pattern_coherence': _assess_pattern_coherence(pattern_analysis),
            'dominant_pattern_type': _identify_dominant_pattern_type(pattern_analysis),
            'pattern_health_score': _calculate_pattern_health_score(pattern_analysis)
        }
        
        return {
            'wave_specific_patterns': pattern_analysis,
            'overall_patterns': overall_patterns,
            'pattern_insights': _generate_pattern_insights(pattern_analysis, overall_patterns)
        }
        
    except Exception as e:
        logger.debug(f"Wave patterns analysis failed: {e}")
        return {
            'error': str(e),
            'fallback_patterns': {'overall_patterns': {'pattern_health_score': 0.7}}
        }

def _calculate_amplitude_statistics(samples: List[float]) -> Dict[str, float]:
    """Berechnet Amplitude Statistics"""
    try:
        if not samples:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        abs_samples = [abs(s) for s in samples]
        
        return {
            'mean_amplitude': statistics.mean(abs_samples),
            'std_amplitude': statistics.stdev(abs_samples) if len(abs_samples) > 1 else 0.0,
            'max_amplitude': max(abs_samples),
            'min_amplitude': min(abs_samples),
            'amplitude_range': max(abs_samples) - min(abs_samples),
            'rms_amplitude': (sum(s**2 for s in samples) / len(samples)) ** 0.5
        }
        
    except Exception as e:
        logger.debug(f"Amplitude statistics calculation failed: {e}")
        return {'mean_amplitude': 0.0, 'std_amplitude': 0.0}

def _assess_frequency_stability(samples: List[float]) -> float:
    """Bewertet Frequency Stability"""
    try:
        if len(samples) < 20:
            return 0.5
        
        # Estimate frequency at different time windows
        window_size = len(samples) // 10
        frequencies = []
        
        for i in range(0, len(samples) - window_size, window_size):
            window = samples[i:i + window_size]
            freq = _calculate_dominant_frequency(window, 100)  # Assuming 100 Hz sample rate
            if freq > 0:
                frequencies.append(freq)
        
        if len(frequencies) < 2:
            return 0.5
        
        # Calculate stability as inverse of coefficient of variation
        mean_freq = statistics.mean(frequencies)
        std_freq = statistics.stdev(frequencies)
        
        if mean_freq == 0:
            return 0.5
        
        cv = std_freq / mean_freq
        stability = max(0.0, 1.0 - cv)
        
        return min(1.0, stability)
        
    except Exception as e:
        logger.debug(f"Frequency stability assessment failed: {e}")
        return 0.5

def _assess_wave_regularity(samples: List[float]) -> float:
    """Bewertet Wave Regularity"""
    try:
        if len(samples) < 10:
            return 0.5
        
        # Calculate regularity based on autocorrelation
        autocorr = _calculate_autocorrelation(samples)
        
        # Find peaks in autocorrelation (indicates regularity)
        peaks = _find_peaks(autocorr)
        
        if not peaks:
            return 0.3
        
        # Regularity based on peak prominence and spacing
        peak_values = [autocorr[p] for p in peaks]
        if peak_values:
            regularity = statistics.mean(peak_values)
        else:
            regularity = 0.3
        
        return max(0.0, min(1.0, regularity))
        
    except Exception as e:
        logger.debug(f"Wave regularity assessment failed: {e}")
        return 0.5

def _calculate_autocorrelation(samples: List[float]) -> List[float]:
    """Berechnet Autocorrelation"""
    try:
        if len(samples) < 2:
            return [1.0]
        
        n = len(samples)
        autocorr = []
        
        # Calculate autocorrelation for different lags
        max_lag = min(n // 4, 50)  # Limit computation
        
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                # Calculate correlation at this lag
                sum_product = 0.0
                for i in range(n - lag):
                    sum_product += samples[i] * samples[i + lag]
                
                # Normalize
                norm = sum(s**2 for s in samples[:n-lag])
                correlation = sum_product / norm if norm > 0 else 0.0
                autocorr.append(correlation)
        
        return autocorr
        
    except Exception as e:
        logger.debug(f"Autocorrelation calculation failed: {e}")
        return [1.0]

def _find_peaks(data: List[float]) -> List[int]:
    """Findet Peaks in Daten"""
    try:
        if len(data) < 3:
            return []
        
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > 0.1:  # Threshold
                peaks.append(i)
        
        return peaks
        
    except Exception as e:
        logger.debug(f"Peak finding failed: {e}")
        return []

def _detect_burst_patterns(samples: List[float]) -> Dict[str, Any]:
    """Detektiert Burst Patterns"""
    try:
        if len(samples) < 20:
            return {'burst_detected': False, 'burst_frequency': 0.0}
        
        # Calculate amplitude envelope
        envelope = _calculate_amplitude_envelope(samples)
        
        # Detect bursts as periods of high amplitude
        threshold = statistics.mean(envelope) + statistics.stdev(envelope) if len(envelope) > 1 else statistics.mean(envelope)
        
        bursts = []
        in_burst = False
        burst_start = 0
        
        for i, amp in enumerate(envelope):
            if amp > threshold and not in_burst:
                burst_start = i
                in_burst = True
            elif amp <= threshold and in_burst:
                bursts.append((burst_start, i))
                in_burst = False
        
        # Close last burst if needed
        if in_burst:
            bursts.append((burst_start, len(envelope)))
        
        # Analyze burst characteristics
        if bursts:
            burst_durations = [end - start for start, end in bursts]
            inter_burst_intervals = []
            for i in range(1, len(bursts)):
                interval = bursts[i][0] - bursts[i-1][1]
                inter_burst_intervals.append(interval)
            
            return {
                'burst_detected': True,
                'burst_count': len(bursts),
                'burst_frequency': len(bursts) / (len(samples) / 100),  # Bursts per second
                'mean_burst_duration': statistics.mean(burst_durations) if burst_durations else 0.0,
                'mean_inter_burst_interval': statistics.mean(inter_burst_intervals) if inter_burst_intervals else 0.0
            }
        else:
            return {
                'burst_detected': False,
                'burst_count': 0,
                'burst_frequency': 0.0
            }
            
    except Exception as e:
        logger.debug(f"Burst pattern detection failed: {e}")
        return {'burst_detected': False, 'burst_frequency': 0.0}

def _assess_rhythm_consistency(samples: List[float]) -> float:
    """Bewertet Rhythm Consistency"""
    try:
        if len(samples) < 20:
            return 0.5
        
        # Find zero crossings for rhythm analysis
        zero_crossings = []
        for i in range(1, len(samples)):
            if (samples[i-1] >= 0 and samples[i] < 0) or (samples[i-1] < 0 and samples[i] >= 0):
                zero_crossings.append(i)
        
        if len(zero_crossings) < 4:
            return 0.3
        
        # Calculate intervals between zero crossings
        intervals = []
        for i in range(1, len(zero_crossings)):
            intervals.append(zero_crossings[i] - zero_crossings[i-1])
        
        # Consistency based on interval variability
        if intervals:
            mean_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
            
            if mean_interval == 0:
                return 0.3
            
            cv = std_interval / mean_interval
            consistency = max(0.0, 1.0 - cv)
            return min(1.0, consistency)
        
        return 0.3
        
    except Exception as e:
        logger.debug(f"Rhythm consistency assessment failed: {e}")
        return 0.5

def _assess_overall_pattern_complexity(pattern_analysis: Dict) -> float:
    """Bewertet Overall Pattern Complexity"""
    try:
        if not pattern_analysis:
            return 0.5
        
        complexity_scores = []
        
        for wave_type, patterns in pattern_analysis.items():
            if isinstance(patterns, dict):
                # Extract complexity indicators
                burst_patterns = patterns.get('burst_patterns', {})
                amplitude_stats = patterns.get('amplitude_statistics', {})
                
                # Complexity based on multiple factors
                burst_complexity = 1.0 if burst_patterns.get('burst_detected', False) else 0.3
                amplitude_variability = amplitude_stats.get('std_amplitude', 0.0) / max(amplitude_stats.get('mean_amplitude', 1.0), 0.1)
                
                wave_complexity = (burst_complexity + min(1.0, amplitude_variability * 2)) / 2
                complexity_scores.append(wave_complexity)
        
        if complexity_scores:
            return statistics.mean(complexity_scores)
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"Overall pattern complexity assessment failed: {e}")
        return 0.5

def _assess_pattern_coherence(pattern_analysis: Dict) -> float:
    """Bewertet Pattern Coherence"""
    try:
        if not pattern_analysis:
            return 0.5
        
        coherence_scores = []
        
        for wave_type, patterns in pattern_analysis.items():
            if isinstance(patterns, dict):
                regularity = patterns.get('wave_regularity', 0.5)
                frequency_stability = patterns.get('frequency_stability', 0.5)
                rhythm_consistency = patterns.get('rhythm_consistency', 0.5)
                
                # Coherence as combination of regularity, stability, and consistency
                wave_coherence = (regularity + frequency_stability + rhythm_consistency) / 3
                coherence_scores.append(wave_coherence)
        
        if coherence_scores:
            return statistics.mean(coherence_scores)
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"Pattern coherence assessment failed: {e}")
        return 0.5

def _identify_dominant_pattern_type(pattern_analysis: Dict) -> str:
    """Identifiziert Dominant Pattern Type"""
    try:
        if not pattern_analysis:
            return 'unknown'
        
        pattern_scores = {}
        
        for wave_type, patterns in pattern_analysis.items():
            if isinstance(patterns, dict):
                # Score each wave type based on its characteristics
                amplitude_stats = patterns.get('amplitude_statistics', {})
                burst_patterns = patterns.get('burst_patterns', {})
                
                mean_amplitude = amplitude_stats.get('mean_amplitude', 0.0)
                burst_frequency = burst_patterns.get('burst_frequency', 0.0)
                
                # Simple scoring system
                score = mean_amplitude + burst_frequency
                pattern_scores[wave_type] = score
        
        if pattern_scores:
            dominant_type = max(pattern_scores.items(), key=lambda x: x[1])[0]
            return dominant_type.replace('_waves', '')
        else:
            return 'balanced'
            
    except Exception as e:
        logger.debug(f"Dominant pattern type identification failed: {e}")
        return 'unknown'

def _calculate_pattern_health_score(pattern_analysis: Dict) -> float:
    """Berechnet Pattern Health Score"""
    try:
        if not pattern_analysis:
            return 0.5
        
        health_factors = []
        
        for wave_type, patterns in pattern_analysis.items():
            if isinstance(patterns, dict):
                regularity = patterns.get('wave_regularity', 0.5)
                frequency_stability = patterns.get('frequency_stability', 0.5)
                rhythm_consistency = patterns.get('rhythm_consistency', 0.5)
                
                # Health score for this wave type
                wave_health = (regularity + frequency_stability + rhythm_consistency) / 3
                health_factors.append(wave_health)
        
        if health_factors:
            overall_health = statistics.mean(health_factors)
            return max(0.0, min(1.0, overall_health))
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"Pattern health score calculation failed: {e}")
        return 0.5

def _generate_pattern_insights(pattern_analysis: Dict, overall_patterns: Dict) -> List[str]:
    """Generiert Pattern Insights"""
    try:
        insights = []
        
        # Health insights
        health_score = overall_patterns.get('pattern_health_score', 0.5)
        if health_score > 0.8:
            insights.append("Excellent brain wave pattern health detected")
        elif health_score < 0.4:
            insights.append("Brain wave patterns may need attention")
        
        # Dominant pattern insights
        dominant_type = overall_patterns.get('dominant_pattern_type', 'unknown')
        if dominant_type == 'alpha':
            insights.append("Alpha wave dominance suggests relaxed, focused state")
        elif dominant_type == 'beta':
            insights.append("Beta wave dominance indicates active, alert state")
        elif dominant_type == 'theta':
            insights.append("Theta wave dominance suggests creative, meditative state")
        elif dominant_type == 'delta':
            insights.append("Delta wave dominance indicates deep, restorative state")
        elif dominant_type == 'gamma':
            insights.append("Gamma wave dominance suggests high cognitive activity")
        
        # Pattern complexity insights
        complexity = overall_patterns.get('pattern_complexity', 0.5)
        if complexity > 0.7:
            insights.append("High pattern complexity indicates rich neural activity")
        elif complexity < 0.3:
            insights.append("Low complexity may suggest need for cognitive stimulation")
        
        return insights[:3]  # Limit to top 3 insights
        
    except Exception as e:
        logger.debug(f"Pattern insights generation failed: {e}")
        return ["Brain wave patterns showing normal activity"]
    
def _generate_hierarchical_layout(nodes: List, edges: List, params: Dict) -> Dict[str, Any]:
    """Generiert Hierarchical Layout für Network Visualization"""
    try:
        width = params.get('width', 1000)
        height = params.get('height', 800)
        spacing = params.get('spacing', 50)
        
        # Analyze network structure for hierarchy
        hierarchy_info = _analyze_network_hierarchy(nodes, edges)
        levels = hierarchy_info.get('levels', {})
        
        if not levels:
            # Fallback to simple layered layout
            levels = _create_simple_hierarchy(nodes)
        
        # Position nodes in hierarchical layout
        positioned_nodes = []
        level_heights = {}
        
        # Calculate level positions
        max_level = max(levels.keys()) if levels else 0
        for level in range(max_level + 1):
            level_heights[level] = (height / (max_level + 1)) * level + spacing
        
        # Position nodes within levels
        for level, level_nodes in levels.items():
            node_count = len(level_nodes)
            if node_count == 0:
                continue
            
            y_position = level_heights[level]
            
            # Distribute nodes horizontally within level
            if node_count == 1:
                x_positions = [width / 2]
            else:
                x_spacing = (width - 2 * spacing) / (node_count - 1)
                x_positions = [spacing + i * x_spacing for i in range(node_count)]
            
            # Create positioned nodes
            for i, node_id in enumerate(level_nodes):
                # Find original node data
                original_node = next((n for n in nodes if n.get('id') == node_id), None)
                if original_node:
                    positioned_node = {
                        'id': node_id,
                        'x': x_positions[i],
                        'y': y_position,
                        'level': level,
                        'type': original_node.get('type', 'unknown'),
                        'properties': original_node.get('properties', {}),
                        'hierarchy_data': {
                            'level': level,
                            'level_position': i,
                            'total_in_level': node_count,
                            'parent_nodes': _find_parent_nodes(node_id, edges),
                            'child_nodes': _find_child_nodes(node_id, edges)
                        }
                    }
                    positioned_nodes.append(positioned_node)
        
        # Process edges for hierarchical layout
        processed_edges = []
        for edge in edges:
            source_node = next((n for n in positioned_nodes if n['id'] == edge.get('source')), None)
            target_node = next((n for n in positioned_nodes if n['id'] == edge.get('target')), None)
            
            if source_node and target_node:
                processed_edge = {
                    'source': edge.get('source'),
                    'target': edge.get('target'),
                    'strength': edge.get('properties', {}).get('weight', 0.5),
                    'hierarchy_data': {
                        'level_difference': abs(source_node['level'] - target_node['level']),
                        'direction': 'down' if source_node['level'] < target_node['level'] else 'up',
                        'edge_type': _classify_hierarchical_edge(source_node, target_node)
                    }
                }
                processed_edges.append(processed_edge)
        
        layout_data = {
            'nodes': positioned_nodes,
            'edges': processed_edges,
            'layout_type': 'hierarchical',
            'hierarchy_info': hierarchy_info,
            'layout_params': {
                'total_levels': max_level + 1,
                'level_spacing': height / (max_level + 1) if max_level > 0 else height,
                'node_spacing': spacing,
                'layout_direction': 'top_to_bottom'
            }
        }
        
        return layout_data
        
    except Exception as e:
        logger.debug(f"Hierarchical layout generation failed: {e}")
        return {'nodes': [], 'edges': [], 'layout_type': 'hierarchical', 'error': str(e)}

def _generate_circular_layout(nodes: List, edges: List, params: Dict) -> Dict[str, Any]:
    """Generiert Circular Layout für Network Visualization"""
    try:
        width = params.get('width', 1000)
        height = params.get('height', 800)
        
        # Calculate circle parameters
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 - 50  # Leave margin
        
        node_count = len(nodes)
        if node_count == 0:
            return {'nodes': [], 'edges': [], 'layout_type': 'circular'}
        
        # Position nodes in circle
        positioned_nodes = []
        for i, node in enumerate(nodes):
            # Calculate angle for this node
            angle = (2 * math.pi * i) / node_count
            
            # Calculate position
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            positioned_node = {
                'id': node.get('id', f"node_{i}"),
                'x': x,
                'y': y,
                'angle': angle,
                'type': node.get('type', 'unknown'),
                'properties': node.get('properties', {}),
                'circular_data': {
                    'angle_degrees': math.degrees(angle),
                    'radius': radius,
                    'position_index': i,
                    'total_nodes': node_count
                }
            }
            positioned_nodes.append(positioned_node)
        
        # Process edges for circular layout
        processed_edges = []
        for edge in edges:
            source_node = next((n for n in positioned_nodes if n['id'] == edge.get('source')), None)
            target_node = next((n for n in positioned_nodes if n['id'] == edge.get('target')), None)
            
            if source_node and target_node:
                # Calculate edge properties for circular layout
                angle_diff = abs(source_node['angle'] - target_node['angle'])
                arc_length = radius * angle_diff
                
                processed_edge = {
                    'source': edge.get('source'),
                    'target': edge.get('target'),
                    'strength': edge.get('properties', {}).get('weight', 0.5),
                    'circular_data': {
                        'angle_difference': angle_diff,
                        'arc_length': arc_length,
                        'chord_length': 2 * radius * math.sin(angle_diff / 2),
                        'edge_curvature': _calculate_circular_edge_curvature(angle_diff)
                    }
                }
                processed_edges.append(processed_edge)
        
        layout_data = {
            'nodes': positioned_nodes,
            'edges': processed_edges,
            'layout_type': 'circular',
            'layout_params': {
                'center': (center_x, center_y),
                'radius': radius,
                'total_nodes': node_count,
                'angular_spacing': (2 * math.pi) / node_count if node_count > 0 else 0
            }
        }
        
        return layout_data
        
    except Exception as e:
        logger.debug(f"Circular layout generation failed: {e}")
        return {'nodes': [], 'edges': [], 'layout_type': 'circular', 'error': str(e)}

def _generate_cluster_based_layout(nodes: List, edges: List, params: Dict) -> Dict[str, Any]:
    """Generiert Cluster-based Layout für Network Visualization"""
    try:
        width = params.get('width', 1000)
        height = params.get('height', 800)
        spacing = params.get('spacing', 50)
        
        # Analyze network for clusters
        clusters = _detect_network_clusters(nodes, edges)
        
        if not clusters:
            # Fallback to simple clustering by node type
            clusters = _create_type_based_clusters(nodes)
        
        # Calculate cluster positions
        cluster_count = len(clusters)
        if cluster_count == 0:
            return {'nodes': [], 'edges': [], 'layout_type': 'cluster_based'}
        
        # Arrange clusters in grid pattern
        grid_size = math.ceil(math.sqrt(cluster_count))
        cluster_width = width / grid_size
        cluster_height = height / grid_size
        
        positioned_nodes = []
        cluster_info = {}
        
        for cluster_idx, (cluster_id, cluster_nodes) in enumerate(clusters.items()):
            # Calculate cluster center
            cluster_row = cluster_idx // grid_size
            cluster_col = cluster_idx % grid_size
            
            cluster_center_x = cluster_col * cluster_width + cluster_width / 2
            cluster_center_y = cluster_row * cluster_height + cluster_height / 2
            
            # Position nodes within cluster
            cluster_node_count = len(cluster_nodes)
            cluster_radius = min(cluster_width, cluster_height) / 3
            
            cluster_positioned_nodes = []
            for i, node_id in enumerate(cluster_nodes):
                # Find original node data
                original_node = next((n for n in nodes if n.get('id') == node_id), None)
                if original_node:
                    if cluster_node_count == 1:
                        # Single node at cluster center
                        x, y = cluster_center_x, cluster_center_y
                    else:
                        # Multiple nodes in circular arrangement within cluster
                        angle = (2 * math.pi * i) / cluster_node_count
                        x = cluster_center_x + cluster_radius * math.cos(angle)
                        y = cluster_center_y + cluster_radius * math.sin(angle)
                    
                    positioned_node = {
                        'id': node_id,
                        'x': x,
                        'y': y,
                        'type': original_node.get('type', 'unknown'),
                        'properties': original_node.get('properties', {}),
                        'cluster_data': {
                            'cluster_id': cluster_id,
                            'cluster_center': (cluster_center_x, cluster_center_y),
                            'cluster_radius': cluster_radius,
                            'position_in_cluster': i,
                            'cluster_size': cluster_node_count
                        }
                    }
                    positioned_nodes.append(positioned_node)
                    cluster_positioned_nodes.append(positioned_node)
            
            cluster_info[cluster_id] = {
                'center': (cluster_center_x, cluster_center_y),
                'radius': cluster_radius,
                'node_count': cluster_node_count,
                'nodes': cluster_positioned_nodes
            }
        
        # Process edges for cluster layout
        processed_edges = []
        for edge in edges:
            source_node = next((n for n in positioned_nodes if n['id'] == edge.get('source')), None)
            target_node = next((n for n in positioned_nodes if n['id'] == edge.get('target')), None)
            
            if source_node and target_node:
                source_cluster = source_node['cluster_data']['cluster_id']
                target_cluster = target_node['cluster_data']['cluster_id']
                
                processed_edge = {
                    'source': edge.get('source'),
                    'target': edge.get('target'),
                    'strength': edge.get('properties', {}).get('weight', 0.5),
                    'cluster_data': {
                        'source_cluster': source_cluster,
                        'target_cluster': target_cluster,
                        'is_inter_cluster': source_cluster != target_cluster,
                        'edge_type': 'inter_cluster' if source_cluster != target_cluster else 'intra_cluster'
                    }
                }
                processed_edges.append(processed_edge)
        
        layout_data = {
            'nodes': positioned_nodes,
            'edges': processed_edges,
            'layout_type': 'cluster_based',
            'cluster_info': cluster_info,
            'layout_params': {
                'cluster_count': cluster_count,
                'grid_size': grid_size,
                'cluster_spacing': spacing
            }
        }
        
        return layout_data
        
    except Exception as e:
        logger.debug(f"Cluster-based layout generation failed: {e}")
        return {'nodes': [], 'edges': [], 'layout_type': 'cluster_based', 'error': str(e)}

# ====================================
# MISSING HELPER FUNCTIONS FOR LAYOUTS
# ====================================

def _analyze_network_hierarchy(nodes: List, edges: List) -> Dict[str, Any]:
    """Analysiert Network Hierarchy"""
    try:
        # Build adjacency information
        node_connections = {}
        for node in nodes:
            node_id = node.get('id')
            node_connections[node_id] = {'incoming': [], 'outgoing': []}
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in node_connections and target in node_connections:
                node_connections[source]['outgoing'].append(target)
                node_connections[target]['incoming'].append(source)
        
        # Determine hierarchy levels using topological sorting approach
        levels = {}
        visited = set()
        
        # Start from nodes with no incoming connections (root nodes)
        root_nodes = [node_id for node_id, connections in node_connections.items() 
                      if not connections['incoming']]
        
        if not root_nodes:
            # If no clear roots, pick nodes with highest outgoing connections
            root_nodes = sorted(node_connections.keys(), 
                              key=lambda x: len(node_connections[x]['outgoing']), 
                              reverse=True)[:3]
        
        # Assign levels using BFS
        current_level = 0
        current_nodes = root_nodes
        
        while current_nodes:
            levels[current_level] = []
            next_level_nodes = []
            
            for node_id in current_nodes:
                if node_id not in visited:
                    levels[current_level].append(node_id)
                    visited.add(node_id)
                    
                    # Add children to next level
                    children = node_connections[node_id]['outgoing']
                    for child in children:
                        if child not in visited:
                            next_level_nodes.append(child)
            
            current_level += 1
            current_nodes = list(set(next_level_nodes))  # Remove duplicates
            
            # Prevent infinite loops
            if current_level > 10:
                break
        
        # Add any remaining unvisited nodes to the last level
        unvisited = [node_id for node_id in node_connections.keys() if node_id not in visited]
        if unvisited:
            if current_level not in levels:
                levels[current_level] = []
            levels[current_level].extend(unvisited)
        
        return {
            'levels': levels,
            'max_level': max(levels.keys()) if levels else 0,
            'root_nodes': root_nodes,
            'hierarchy_depth': len(levels),
            'node_connections': node_connections
        }
        
    except Exception as e:
        logger.debug(f"Network hierarchy analysis failed: {e}")
        return {'levels': {}, 'max_level': 0}

def _create_simple_hierarchy(nodes: List) -> Dict[int, List]:
    """Erstellt einfache Hierarchy basierend auf Node Types"""
    try:
        # Group nodes by type for simple hierarchy
        type_groups = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in type_groups:
                type_groups[node_type] = []
            type_groups[node_type].append(node.get('id'))
        
        # Assign levels based on type priority
        type_priority = {
            'memory_node': 0,
            'processing_node': 1,
            'sensory_node': 2,
            'motor_node': 3,
            'emotional_node': 1,
            'cognitive_node': 1,
            'association_node': 2,
            'output_node': 3,
            'unknown': 4
        }
        
        levels = {}
        for node_type, node_ids in type_groups.items():
            level = type_priority.get(node_type, 4)
            if level not in levels:
                levels[level] = []
            levels[level].extend(node_ids)
        
        return levels
        
    except Exception as e:
        logger.debug(f"Simple hierarchy creation failed: {e}")
        return {}

def _find_parent_nodes(node_id: str, edges: List) -> List[str]:
    """Findet Parent Nodes für hierarchical Layout"""
    try:
        parents = []
        for edge in edges:
            if edge.get('target') == node_id:
                parents.append(edge.get('source'))
        return parents
        
    except Exception as e:
        logger.debug(f"Parent nodes finding failed: {e}")
        return []

def _find_child_nodes(node_id: str, edges: List) -> List[str]:
    """Findet Child Nodes für hierarchical Layout"""
    try:
        children = []
        for edge in edges:
            if edge.get('source') == node_id:
                children.append(edge.get('target'))
        return children
        
    except Exception as e:
        logger.debug(f"Child nodes finding failed: {e}")
        return []

def _classify_hierarchical_edge(source_node: Dict, target_node: Dict) -> str:
    """Klassifiziert Edge Type im hierarchical Layout"""
    try:
        source_level = source_node.get('level', 0)
        target_level = target_node.get('level', 0)
        
        if source_level < target_level:
            return 'forward'  # Top-down
        elif source_level > target_level:
            return 'backward'  # Bottom-up
        else:
            return 'lateral'  # Same level
            
    except Exception as e:
        logger.debug(f"Hierarchical edge classification failed: {e}")
        return 'unknown'

def _calculate_circular_edge_curvature(angle_diff: float) -> float:
    """Berechnet Edge Curvature für circular Layout"""
    try:
        # More curvature for edges spanning larger angles
        normalized_angle = angle_diff / math.pi  # 0 to 1
        curvature = 0.2 + (normalized_angle * 0.3)  # 0.2 to 0.5
        return min(0.5, curvature)
        
    except Exception as e:
        logger.debug(f"Circular edge curvature calculation failed: {e}")
        return 0.3

def _detect_network_clusters(nodes: List, edges: List) -> Dict[str, List]:
    """Detektiert Network Clusters"""
    try:
        # Simple clustering based on connectivity
        # Build adjacency list
        adjacency = {}
        for node in nodes:
            node_id = node.get('id')
            adjacency[node_id] = set()
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)
        
        # Find connected components using DFS
        visited = set()
        clusters = {}
        cluster_id = 0
        
        for node_id in adjacency.keys():
            if node_id not in visited:
                cluster_nodes = []
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster_nodes.append(current)
                        
                        # Add neighbors to stack
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if cluster_nodes:
                    clusters[f"cluster_{cluster_id}"] = cluster_nodes
                    cluster_id += 1
        
        return clusters
        
    except Exception as e:
        logger.debug(f"Network cluster detection failed: {e}")
        return {}

def _create_type_based_clusters(nodes: List) -> Dict[str, List]:
    """Erstellt Clusters basierend auf Node Types"""
    try:
        clusters = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            cluster_key = f"cluster_{node_type}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(node.get('id'))
        
        return clusters
        
    except Exception as e:
        logger.debug(f"Type-based cluster creation failed: {e}")
        return {}

def _calculate_node_mass(node: Dict) -> float:
    """Berechnet Node Mass für Force-Directed Layout"""
    try:
        properties = node.get('properties', {})
        base_mass = 1.0
        
        # Mass based on processing capacity and memory
        processing_capacity = properties.get('processing_capacity', 0.5)
        memory_capacity = properties.get('memory_capacity', 50)
        
        mass_factor = 1.0 + (processing_capacity * 0.5) + (min(memory_capacity, 100) / 100 * 0.3)
        return base_mass * mass_factor
        
    except Exception as e:
        logger.debug(f"Node mass calculation failed: {e}")
        return 1.0

def _calculate_node_charge(node: Dict) -> float:
    """Berechnet Node Charge für Force-Directed Layout"""
    try:
        node_type = node.get('type', 'unknown')
        
        # Different charges for different node types
        charge_map = {
            'memory_node': -30,
            'processing_node': -25,
            'sensory_node': -20,
            'motor_node': -20,
            'emotional_node': -15,
            'cognitive_node': -25,
            'association_node': -20,
            'output_node': -20,
            'unknown': -15
        }
        
        return charge_map.get(node_type, -20)
        
    except Exception as e:
        logger.debug(f"Node charge calculation failed: {e}")
        return -20

def _calculate_ideal_edge_length(edge: Dict) -> float:
    """Berechnet ideale Edge Length für Force-Directed Layout"""
    try:
        properties = edge.get('properties', {})
        weight = properties.get('weight', 0.5)
        
        # Stronger connections should be shorter
        base_length = 100
        length_factor = 1.0 - (weight * 0.5)  # 0.5 to 1.0
        
        return base_length * length_factor
        
    except Exception as e:
        logger.debug(f"Ideal edge length calculation failed: {e}")
        return 100

def _calculate_spring_constant(edge: Dict) -> float:
    """Berechnet Spring Constant für Force-Directed Layout"""
    try:
        properties = edge.get('properties', {})
        weight = properties.get('weight', 0.5)
        
        # Stronger connections should have higher spring constants
        base_spring = 0.1
        spring_factor = 1.0 + (weight * 2.0)  # 1.0 to 3.0
        
        return base_spring * spring_factor
        
    except Exception as e:
        logger.debug(f"Spring constant calculation failed: {e}")
        return 0.1

# ====================================
# MISSING ENHANCEMENT FUNCTIONS
# ====================================

def _enhance_visualization_data(visualization_data: Dict, network_structure: Dict, 
                              layout_params: Dict) -> Dict[str, Any]:
    """Erweitert Visualization Data mit zusätzlichen Features"""
    try:
        enhanced_data = visualization_data.copy()
        
        # Add visual enhancements to nodes
        if 'nodes' in enhanced_data:
            for node in enhanced_data['nodes']:
                # Add visual properties
                node['visual_properties'] = {
                    'size': _calculate_node_size(node),
                    'color': _determine_node_color(node, layout_params.get('color_scheme', 'neural_default')),
                    'opacity': _calculate_node_opacity(node),
                    'border_width': _calculate_node_border_width(node),
                    'animation_speed': _calculate_node_animation_speed(node)
                }
                
                # Add interaction properties
                node['interaction_properties'] = {
                    'clickable': True,
                    'hoverable': True,
                    'draggable': True,
                    'tooltip_info': _generate_node_tooltip_info(node)
                }
        
        # Add visual enhancements to edges
        if 'edges' in enhanced_data:
            for edge in enhanced_data['edges']:
                # Add visual properties
                edge['visual_properties'] = {
                    'thickness': _calculate_edge_thickness(edge),
                    'color': _determine_edge_color(edge, layout_params.get('color_scheme', 'neural_default')),
                    'opacity': _calculate_edge_opacity(edge),
                    'style': _determine_edge_style(edge),
                    'animation_flow': _calculate_edge_animation_flow(edge)
                }
                
                # Add interaction properties
                edge['interaction_properties'] = {
                    'clickable': True,
                    'hoverable': True,
                    'tooltip_info': _generate_edge_tooltip_info(edge)
                }
        
        # Add layout-specific enhancements
        layout_type = visualization_data.get('layout_type', 'force_directed')
        if layout_type == 'hierarchical':
            enhanced_data = _add_hierarchical_enhancements(enhanced_data)
        elif layout_type == 'circular':
            enhanced_data = _add_circular_enhancements(enhanced_data)
        elif layout_type == 'cluster_based':
            enhanced_data = _add_cluster_enhancements(enhanced_data)
        
        return enhanced_data
        
    except Exception as e:
        logger.debug(f"Visualization data enhancement failed: {e}")
        return visualization_data

def _generate_visualization_interaction_data(visualization_data: Dict) -> Dict[str, Any]:
    """Generiert Interaction Data für Visualization"""
    try:
        interaction_data = {
            'zoom_settings': {
                'min_zoom': 0.1,
                'max_zoom': 5.0,
                'initial_zoom': 1.0,
                'zoom_step': 0.1
            },
            'pan_settings': {
                'enabled': True,
                'bounds': 'auto',
                'inertia': True
            },
            'selection_settings': {
                'single_select': True,
                'multi_select': True,
                'select_on_click': True,
                'deselect_on_background_click': True
            },
            'hover_effects': {
                'node_highlight': True,
                'edge_highlight': True,
                'neighbor_highlight': True,
                'tooltip_delay': 500
            },
            'drag_settings': {
                'nodes_draggable': True,
                'maintain_connections': True,
                'snap_to_grid': False
            }
        }
        
        # Add layout-specific interactions
        layout_type = visualization_data.get('layout_type', 'force_directed')
        if layout_type == 'force_directed':
            interaction_data['physics_settings'] = {
                'simulation_enabled': True,
                'pause_on_drag': True,
                'resume_on_release': True
            }
        
        return interaction_data
        
    except Exception as e:
        logger.debug(f"Visualization interaction data generation failed: {e}")
        return {'zoom_settings': {'min_zoom': 0.1, 'max_zoom': 5.0}}

def _generate_rendering_instructions(visualization_data: Dict, visualization_type: str) -> Dict[str, Any]:
    """Generiert Rendering Instructions"""
    try:
        instructions = {
            'rendering_order': ['edges', 'nodes', 'labels'],
            'performance_hints': {
                'use_webgl': len(visualization_data.get('nodes', [])) > 100,
                'enable_culling': True,
                'batch_rendering': True,
                'level_of_detail': len(visualization_data.get('nodes', [])) > 500
            },
            'animation_settings': {
                'enable_animations': True,
                'animation_duration': 1000,
                'easing_function': 'ease-in-out'
            },
            'style_settings': {
                'anti_aliasing': True,
                'smooth_curves': True,
                'high_dpi_support': True
            }
        }
        
        # Add visualization-specific instructions
        if visualization_type == 'hierarchical':
            instructions['hierarchical_specific'] = {
                'level_separation': True,
                'edge_routing': 'orthogonal',
                'label_positioning': 'beside_nodes'
            }
        elif visualization_type == 'circular':
            instructions['circular_specific'] = {
                'curved_edges': True,
                'radial_labels': True,
                'arc_optimization': True
            }
        
        return instructions
        
    except Exception as e:
        logger.debug(f"Rendering instructions generation failed: {e}")
        return {'rendering_order': ['edges', 'nodes']}

def _generate_performance_optimization_suggestions(visualization_data: Dict) -> List[str]:
    """Generiert Performance Optimization Suggestions"""
    try:
        suggestions = []
        
        node_count = len(visualization_data.get('nodes', []))
        edge_count = len(visualization_data.get('edges', []))
        
        # Node count optimizations
        if node_count > 1000:
            suggestions.append("Consider using level-of-detail rendering for large node counts")
            suggestions.append("Enable node clustering for better performance")
        elif node_count > 500:
            suggestions.append("Enable WebGL rendering for improved performance")
        
        # Edge count optimizations
        if edge_count > 2000:
            suggestions.append("Consider edge bundling for dense networks")
            suggestions.append("Enable edge culling based on zoom level")
        elif edge_count > 1000:
            suggestions.append("Use simplified edge rendering at low zoom levels")
        
        # General optimizations
        if node_count > 100 or edge_count > 200:
            suggestions.append("Enable viewport culling to improve rendering performance")
            suggestions.append("Consider using canvas instead of SVG for large networks")
        
        # Layout-specific optimizations
        layout_type = visualization_data.get('layout_type', 'force_directed')
        if layout_type == 'force_directed' and node_count > 200:
            suggestions.append("Use spatial indexing to optimize force calculations")
        
        return suggestions[:5]  # Limit to top 5 suggestions
        
    except Exception as e:
        logger.debug(f"Performance optimization suggestions generation failed: {e}")
        return ["Enable basic performance optimizations"]

# ====================================
# MISSING CALCULATION HELPER FUNCTIONS
# ====================================

def _calculate_node_opacity(node: Dict) -> float:
    """Berechnet Node Opacity"""
    try:
        properties = node.get('properties', {})
        activity_level = properties.get('activity_level', 0.5)
        
        # More active nodes are more opaque
        base_opacity = 0.7
        opacity_factor = 0.3 * activity_level
        
        return min(1.0, base_opacity + opacity_factor)
        
    except Exception as e:
        logger.debug(f"Node opacity calculation failed: {e}")
        return 0.8

def _calculate_node_border_width(node: Dict) -> float:
    """Berechnet Node Border Width"""
    try:
        properties = node.get('properties', {})
        importance = properties.get('importance', 0.5)
        
        # More important nodes have thicker borders
        base_width = 1.0
        width_factor = importance * 2.0
        
        return base_width + width_factor
        
    except Exception as e:
        logger.debug(f"Node border width calculation failed: {e}")
        return 1.0

def _calculate_node_animation_speed(node: Dict) -> float:
    """Berechnet Node Animation Speed"""
    try:
        properties = node.get('properties', {})
        activity_level = properties.get('activity_level', 0.5)
        
        # More active nodes animate faster
        base_speed = 1.0
        speed_factor = activity_level * 0.5
        
        return base_speed + speed_factor
        
    except Exception as e:
        logger.debug(f"Node animation speed calculation failed: {e}")
        return 1.0

def _calculate_edge_thickness(edge: Dict) -> float:
    """Berechnet Edge Thickness"""
    try:
        properties = edge.get('properties', {})
        weight = properties.get('weight', 0.5)
        
        # Stronger connections are thicker
        base_thickness = 1.0
        thickness_factor = weight * 3.0
        
        return base_thickness + thickness_factor
        
    except Exception as e:
        logger.debug(f"Edge thickness calculation failed: {e}")
        return 2.0

def _determine_edge_color(edge: Dict, color_scheme: str) -> str:
    """Bestimmt Edge Color"""
    try:
        properties = edge.get('properties', {})
        edge_type = properties.get('type', 'connection')
        
        color_schemes = {
            'neural_default': {
                'excitatory': '#FF6B6B',
                'inhibitory': '#4ECDC4',
                'connection': '#CCCCCC',
                'strong': '#FF4444',
                'weak': '#DDDDDD'
            }
        }
        
        scheme = color_schemes.get(color_scheme, color_schemes['neural_default'])
        
        # Determine color based on edge strength if type not specified
        if edge_type == 'connection':
            weight = properties.get('weight', 0.5)
            if weight > 0.7:
                return scheme.get('strong', '#FF4444')
            elif weight < 0.3:
                return scheme.get('weak', '#DDDDDD')
        
        return scheme.get(edge_type, scheme.get('connection', '#CCCCCC'))
        
    except Exception as e:
        logger.debug(f"Edge color determination failed: {e}")
        return '#CCCCCC'

def _calculate_edge_opacity(edge: Dict) -> float:
    """Berechnet Edge Opacity"""
    try:
        properties = edge.get('properties', {})
        weight = properties.get('weight', 0.5)
        
        # Stronger connections are more opaque
        base_opacity = 0.5
        opacity_factor = weight * 0.4
        
        return min(1.0, base_opacity + opacity_factor)
        
    except Exception as e:
        logger.debug(f"Edge opacity calculation failed: {e}")
        return 0.7

def _determine_edge_style(edge: Dict) -> str:
    """Bestimmt Edge Style"""
    try:
        properties = edge.get('properties', {})
        edge_type = properties.get('type', 'connection')
        
        style_map = {
            'excitatory': 'solid',
            'inhibitory': 'dashed',
            'connection': 'solid',
            'weak': 'dotted'
        }
        
        return style_map.get(edge_type, 'solid')
        
    except Exception as e:
        logger.debug(f"Edge style determination failed: {e}")
        return 'solid'

def _calculate_edge_animation_flow(edge: Dict) -> float:
    """Berechnet Edge Animation Flow"""
    try:
        properties = edge.get('properties', {})
        activity = properties.get('activity', 0.5)
        
        # More active edges have faster flow
        return activity * 2.0
        
    except Exception as e:
        logger.debug(f"Edge animation flow calculation failed: {e}")
        return 1.0

def _generate_node_tooltip_info(node: Dict) -> Dict[str, str]:
    """Generiert Node Tooltip Information"""
    try:
        tooltip_info = {
            'title': f"Node: {node.get('id', 'Unknown')}",
            'type': f"Type: {node.get('type', 'Unknown')}",
        }
        
        properties = node.get('properties', {})
        if properties:
            if 'processing_capacity' in properties:
                tooltip_info['processing'] = f"Processing: {properties['processing_capacity']:.2f}"
            if 'memory_capacity' in properties:
                tooltip_info['memory'] = f"Memory: {properties['memory_capacity']}"
            if 'activity_level' in properties:
                tooltip_info['activity'] = f"Activity: {properties['activity_level']:.2f}"
        
        return tooltip_info
        
    except Exception as e:
        logger.debug(f"Node tooltip info generation failed: {e}")
        return {'title': 'Node Information'}

def _generate_edge_tooltip_info(edge: Dict) -> Dict[str, str]:
    """Generiert Edge Tooltip Information"""
    try:
        tooltip_info = {
            'title': f"Connection: {edge.get('source', 'Unknown')} → {edge.get('target', 'Unknown')}",
        }
        
        properties = edge.get('properties', {})
        if properties:
            if 'weight' in properties:
                tooltip_info['strength'] = f"Strength: {properties['weight']:.2f}"
            if 'type' in properties:
                tooltip_info['type'] = f"Type: {properties['type']}"
        
        return tooltip_info
        
    except Exception as e:
        logger.debug(f"Edge tooltip info generation failed: {e}")
        return {'title': 'Connection Information'}

__all__ = [
    'prepare_neural_chart_data',
    'generate_network_visualization',
    'create_brain_wave_charts',
    'render_connection_maps'
]