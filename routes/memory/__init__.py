"""
Kira Memory Module
Memory Operations, Analysis, Consolidation und Management

Module:
- operations.py: Memory CRUD Operations, Search, Retrieval
- analysis.py: Memory Pattern Analysis, Statistics, Insights
- consolidation.py: Memory Consolidation, LTM Transfer, Optimization
- patterns.py: Memory Pattern Detection, Learning Patterns, Behavioral Analysis
- routes.py: Memory API Routes für Web Interface
"""

from .operations import (
    get_memory_data,
    add_memory_entry,
    search_memories,
    get_memory_statistics,
    manage_memory_lifecycle
)

from .analysis import (
    analyze_memory_patterns,
    calculate_memory_efficiency,
    assess_memory_health,
    generate_memory_insights
)

from .consolidation import (
    consolidate_memories,
    transfer_to_long_term,
    optimize_memory_storage,
    manage_memory_retention
)

from .patterns import (
    detect_learning_patterns,
    analyze_behavioral_patterns,
    identify_memory_trends,
    calculate_pattern_significance
)

# Memory Routes API Functions (for web interface)
try:
    from .routes import (
        initialize_memory_system,
        process_memory_interaction,
        manage_memory_consolidation,
        handle_cross_platform_integration,
        query_memory_system,
        manage_personality_evolution,
        start_background_processing,
        stop_background_processing,
        get_memory_status
    )
    
    # Add routes to __all__
    __all__ = [
        # Operations
        'get_memory_data',
        'add_memory_entry',
        'search_memories',
        'get_memory_statistics',
        'manage_memory_lifecycle',
        
        # Analysis
        'analyze_memory_patterns',
        'calculate_memory_efficiency',
        'assess_memory_health',
        'generate_memory_insights',
        
        # Consolidation
        'consolidate_memories',
        'transfer_to_long_term',
        'optimize_memory_storage',
        'manage_memory_retention',
        
        # Patterns
        'detect_learning_patterns',
        'analyze_behavioral_patterns',
        'identify_memory_trends',
        'calculate_pattern_significance',
        
        # Memory Routes API
        'initialize_memory_system',
        'process_memory_interaction',
        'manage_memory_consolidation',
        'handle_cross_platform_integration',
        'query_memory_system',
        'manage_personality_evolution',
        'start_background_processing',
        'stop_background_processing',
        'get_memory_status'
    ]

except ImportError as e:
    print(f"⚠️  Memory Routes not available: {e}")
    
    # Fallback: Only export existing memory functions
    __all__ = [
        # Operations
        'get_memory_data',
        'add_memory_entry',
        'search_memories',
        'get_memory_statistics',
        'manage_memory_lifecycle',
        
        # Analysis
        'analyze_memory_patterns',
        'calculate_memory_efficiency',
        'assess_memory_health',
        'generate_memory_insights',
        
        # Consolidation
        'consolidate_memories',
        'transfer_to_long_term',
        'optimize_memory_storage',
        'manage_memory_retention',
        
        # Patterns
        'detect_learning_patterns',
        'analyze_behavioral_patterns',
        'identify_memory_trends',
        'calculate_pattern_significance'
    ]