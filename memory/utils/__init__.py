"""
Memory Utils Package - Utility-Funktionen für das Memory System
"""

# Cross-Platform Utils
from .cross_platform_utils import (
    CrossPlatformRecognition,
    EnhancedUserMatcher,
    extract_name,
    detect_cross_platform_reference,
    analyze_user_introduction
)

# Enhanced Search Utils
from .enhanced_search_utils import MemorySearchEnhancer

__all__ = [
    # Cross-Platform
    'CrossPlatformRecognition',
    'EnhancedUserMatcher', 
    'extract_name',
    'detect_cross_platform_reference',
    'analyze_user_introduction',
    
    # Enhanced Search
    'MemorySearchEnhancer'
]