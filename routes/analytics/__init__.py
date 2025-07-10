"""
Kira Analytics Module
Comprehensive Analytics System für Memory, Personality und Performance

Module:
- reports.py: Analytics Summary und Comprehensive Reports
- trends.py: Trend Analysis und Future Predictions  
- metrics.py: Key Metrics und Performance Indicators
"""

from .reports import (
    generate_analytics_summary,
    generate_comprehensive_report,
    calculate_growth_analysis,
    analyze_memory_distribution
)

from .trends import (
    analyze_personality_trends,
    predict_future_development,
    calculate_trend_predictions,
    generate_trend_visualizations
)

from .metrics import (
    calculate_key_metrics,
    get_performance_indicators,
    calculate_efficiency_metrics,
    get_real_time_metrics
)

__all__ = [
    # Reports
    'generate_analytics_summary',
    'generate_comprehensive_report', 
    'calculate_growth_analysis',
    'analyze_memory_distribution',
    
    # Trends
    'analyze_personality_trends',
    'predict_future_development',
    'calculate_trend_predictions',
    'generate_trend_visualizations',
    
    # Metrics
    'calculate_key_metrics',
    'get_performance_indicators',
    'calculate_efficiency_metrics',
    'get_real_time_metrics'
]