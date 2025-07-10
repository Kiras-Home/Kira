"""
Enhanced Memory System Configuration - Optimiert für Human-Like Memory System
Unterstützt STM/LTM Integration, Emotion Memory, Cross-Platform und Enhanced Features
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
from enum import Enum

# === ENHANCED MEMORY CONFIGURATION ===

class MemorySystemMode(Enum):
    """Memory System Betriebsmodi"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    RESEARCH = "research"

class LearningProfile(Enum):
    """Lernprofile für verschiedene Anwendungsfälle"""
    CONSERVATIVE = "conservative"      # Vorsichtiges Lernen, hohe Threshold
    BALANCED = "balanced"             # Ausgewogenes Lernen (Standard)
    AGGRESSIVE = "aggressive"         # Aggressives Lernen, niedrige Threshold
    ADAPTIVE = "adaptive"             # Selbst-anpassendes Lernen

# === HAUPTKONFIGURATION ===

ENHANCED_MEMORY_CONFIG = {
    
    # === SYSTEM SETTINGS ===
    'system': {
        'mode': MemorySystemMode.PRODUCTION,
        'learning_profile': LearningProfile.BALANCED,
        'auto_optimization': True,
        'debug_mode': False,
        'telemetry_enabled': True,
        'backup_enabled': True,
        'cross_platform_enabled': True,
        'emotion_processing_enabled': True
    },
    
    # === STM (SHORT TERM MEMORY) ENHANCED ===
    'stm': {
        # Core STM Settings
        'max_working_memory_items': 7,        # Miller's Magic Number 7±2
        'working_memory_decay_minutes': 15,   # Working Memory Decay
        'attention_window_items': 3,          # Items in active attention
        'rehearsal_threshold': 3,             # Rehearsals before LTM consideration
        
        # Session Management
        'max_entries_per_session': 25,       # Increased for Enhanced System
        'session_timeout_hours': 4,          # Extended timeout
        'session_cleanup_days': 14,          # Longer retention
        'auto_session_save': True,
        
        # STM Intelligence
        'importance_amplification': 1.5,     # Multiply importance for significant items
        'emotional_boost_factor': 1.3,       # Boost for emotional content
        'pattern_recognition_threshold': 0.6, # Threshold for pattern detection
        'consolidation_trigger_score': 0.7,  # When to trigger LTM consolidation
        
        # Performance
        'cache_enabled': True,
        'cache_size_mb': 50,
        'batch_processing': True,
        'parallel_processing': True
    },
    
    # === LTM (LONG TERM MEMORY) ENHANCED ===
    'ltm': {
        # Core LTM Settings
        'consolidation_threshold': 8,         # Higher threshold for quality
        'significance_threshold': 0.6,       # Minimum significance for LTM
        'retrieval_strength_threshold': 0.4, # Minimum strength for retrieval
        'forgetting_curve_factor': 0.85,     # Ebbinghaus forgetting curve
        
        # Content Management
        'max_memories_per_user': 10000,      # Significantly increased
        'max_facts_per_category': 500,       # Category-based limits
        'cleanup_interval_days': 180,        # 6 months retention
        'auto_consolidate_interval': 5,      # Every 5 interactions
        
        # Intelligence Features
        'semantic_clustering': True,         # Group related memories
        'importance_redistribution': True,   # Adjust importance over time
        'cross_reference_building': True,    # Build memory relationships
        'pattern_extraction': True,          # Extract behavioral patterns
        
        # Quality Control
        'duplicate_detection': True,         # Prevent duplicate storage
        'quality_scoring': True,             # Score memory quality
        'relevance_decay': True,             # Decrease relevance over time
        'reinforcement_learning': True       # Strengthen accessed memories
    },
    
    # === EMOTION MEMORY ENHANCED ===
    'emotion': {
        # Core Emotion Settings
        'emotion_threshold': 0.3,            # Minimum intensity to store
        'peak_emotion_threshold': 0.8,       # Threshold for peak moments
        'mood_window_minutes': 45,           # Extended mood tracking window
        'emotional_memory_boost': 2.0,       # Strong boost for emotional content
        
        # Emotional Intelligence
        'empathy_learning': True,            # Learn empathetic responses
        'mood_prediction': True,             # Predict mood changes
        'emotional_pattern_detection': True, # Detect emotional patterns
        'cross_emotional_learning': True,    # Learn across emotional contexts
        
        # Retention & Analysis
        'emotion_cleanup_days': 120,         # 4 months emotional memory
        'pattern_analysis_days': 45,         # Extended analysis period
        'min_emotions_for_pattern': 8,       # More data for reliable patterns
        'emotional_significance_boost': 1.8, # Boost emotional memories
        
        # Advanced Features
        'valence_tracking': True,            # Track positive/negative emotions
        'arousal_tracking': True,            # Track emotional arousal
        'dominance_tracking': True,          # Track emotional dominance
        'emotional_contagion': True,         # Model emotional influence
        'trauma_detection': True,            # Detect potentially traumatic content
        'therapeutic_mode': False            # Special mode for therapeutic applications
    },
    
    # === CROSS-PLATFORM ENHANCED ===
    'cross_platform': {
        # User Recognition
        'auto_user_detection': True,         # Automatic user detection
        'cross_platform_matching': True,     # Match users across platforms
        'name_extraction': True,             # Extract names from conversations
        'confidence_threshold': 0.6,         # Minimum confidence for matching
        
        # Device Integration
        'device_context_tracking': True,     # Track device contexts
        'device_preference_learning': True,  # Learn device preferences
        'context_continuity': True,          # Maintain context across devices
        'smart_handoff': True,               # Smart context handoff between devices
        
        # Platform Support
        'supported_platforms': [
            'web_interface', 'mobile_app', 'voice_assistant', 
            'smart_display', 'api_integration', 'telegram_bot'
        ],
        'platform_specific_adaptation': True, # Adapt to platform capabilities
        'unified_user_profiles': True,       # Unified profiles across platforms
        
        # Security & Privacy
        'cross_platform_encryption': True,   # Encrypt cross-platform data
        'user_consent_required': True,       # Require consent for cross-platform
        'data_minimization': True,           # Minimize cross-platform data sharing
        'anonymization_support': True        # Support for anonymous users
    },
    
    # === DATABASE ENHANCED ===
    'database': {
        # Core Database Settings
        'path': 'data/enhanced_kira_memory.db',
        'enable_wal_mode': True,             # Write-Ahead Logging for performance
        'enable_foreign_keys': True,         # Referential integrity
        'page_size': 4096,                   # Optimal page size
        'cache_size_mb': 100,                # Database cache
        
        # Backup & Recovery
        'backup_enabled': True,
        'backup_interval_hours': 12,         # More frequent backups
        'backup_retention_days': 30,         # Keep backups for 30 days
        'incremental_backup': True,          # Incremental backups
        'auto_recovery': True,               # Automatic recovery on corruption
        
        # Maintenance
        'vacuum_interval_days': 3,           # More frequent vacuum
        'analyze_interval_days': 7,          # Statistics updates
        'integrity_check_days': 14,          # Database integrity checks
        'auto_optimize': True,               # Automatic query optimization
        
        # Performance Tuning
        'connection_pooling': True,          # Connection pooling
        'prepared_statements': True,         # Use prepared statements
        'batch_operations': True,            # Batch database operations
        'async_writes': True,                # Asynchronous writes where possible
        
        # Advanced Features
        'full_text_search': True,            # Enable FTS5
        'json_support': True,                # JSON column support
        'vector_embeddings': False,          # Vector embeddings (experimental)
        'compression_enabled': True          # Compress large text fields
    },
    
    # === PERFORMANCE ENHANCED ===
    'performance': {
        # Memory Management
        'memory_cache_size_mb': 200,         # Increased memory cache
        'object_pool_size': 1000,           # Object pooling
        'lazy_loading': True,                # Lazy load heavy objects
        'smart_prefetching': True,           # Predictive prefetching
        
        # Query Optimization
        'search_limit_default': 15,          # Increased default search limit
        'context_limit_default': 10,         # More context items
        'max_search_results': 100,           # Maximum search results
        'query_timeout_seconds': 30,         # Query timeout
        
        # Batch Processing
        'batch_size': 100,                   # Larger batch sizes
        'batch_insert_threshold': 25,        # When to use batch inserts
        'async_processing': True,            # Asynchronous processing
        'queue_size': 500,                   # Processing queue size
        
        # Caching Strategy
        'multi_level_caching': True,         # Multiple cache levels
        'cache_ttl_minutes': 60,             # Cache time-to-live
        'cache_warming': True,               # Pre-warm caches
        'adaptive_cache_size': True,         # Adjust cache size dynamically
        
        # Monitoring & Optimization
        'performance_monitoring': True,      # Monitor performance metrics
        'auto_scaling': True,                # Auto-scale resources
        'bottleneck_detection': True,        # Detect performance bottlenecks
        'optimization_suggestions': True     # Suggest optimizations
    },
    
    # === LEARNING & ADAPTATION ===
    'learning': {
        # Core Learning Settings
        'adaptive_learning': True,           # Enable adaptive learning
        'learning_rate': 0.1,               # Base learning rate
        'learning_decay': 0.95,             # Learning rate decay
        'feedback_integration': True,        # Integrate user feedback
        
        # Pattern Learning
        'pattern_recognition': True,         # Enable pattern recognition
        'behavioral_modeling': True,         # Model user behavior
        'preference_learning': True,         # Learn user preferences
        'predictive_modeling': True,         # Predict user needs
        
        # Personalization
        'personality_profiling': True,       # Build personality profiles
        'communication_style_adaptation': True, # Adapt communication style
        'interest_tracking': True,           # Track user interests
        'expertise_level_detection': True,   # Detect user expertise levels
        
        # Quality Improvement
        'response_quality_learning': True,   # Learn from response quality
        'conversation_flow_optimization': True, # Optimize conversation flow
        'context_relevance_learning': True,  # Learn context relevance
        'error_correction_learning': True    # Learn from mistakes
    },
    
    # === PRIVACY & SECURITY ===
    'privacy': {
        # Data Protection
        'encryption_at_rest': True,          # Encrypt stored data
        'encryption_in_transit': True,       # Encrypt data transmission
        'data_anonymization': True,          # Anonymize personal data
        'sensitive_data_detection': True,    # Detect sensitive information
        
        # User Control
        'user_data_control': True,           # User control over their data
        'right_to_deletion': True,           # Support right to be forgotten
        'data_export': True,                 # Allow data export
        'consent_management': True,          # Manage user consent
        
        # Compliance
        'gdpr_compliance': True,             # GDPR compliance features
        'audit_logging': True,               # Audit data access
        'data_retention_policies': True,     # Enforce retention policies
        'privacy_by_design': True            # Privacy by design principles
    }
}

# === ENHANCED EMOTION TYPES ===

ENHANCED_EMOTION_TYPES = {
    # Primary Emotions (Plutchik's Wheel)
    'primary': [
        'joy', 'sadness', 'anger', 'fear', 
        'surprise', 'disgust', 'anticipation', 'trust'
    ],
    
    # Secondary Emotions
    'secondary': [
        'excitement', 'calm', 'frustrated', 'confused',
        'satisfied', 'disappointed', 'curious', 'bored',
        'grateful', 'proud', 'ashamed', 'guilty',
        'hopeful', 'anxious', 'relieved', 'nostalgic'
    ],
    
    # Social Emotions
    'social': [
        'empathy', 'sympathy', 'compassion', 'love',
        'jealousy', 'envy', 'admiration', 'contempt',
        'respect', 'pity', 'gratitude', 'forgiveness'
    ],
    
    # Cognitive Emotions
    'cognitive': [
        'wonder', 'awe', 'inspiration', 'determination',
        'confidence', 'doubt', 'certainty', 'uncertainty',
        'realization', 'understanding', 'confusion', 'clarity'
    ],
    
    # Meta-Emotions (feelings about feelings)
    'meta': [
        'emotional_overwhelm', 'emotional_numbness',
        'mixed_feelings', 'emotional_confusion',
        'emotional_awareness', 'emotional_growth'
    ]
}

# === ENHANCED IMPORTANCE LEVELS ===

ENHANCED_IMPORTANCE_LEVELS = {
    1: {
        'name': 'Trivial',
        'description': 'Completely unimportant information',
        'ltm_probability': 0.01,
        'retention_days': 1
    },
    2: {
        'name': 'Very Low',
        'description': 'Barely worth remembering',
        'ltm_probability': 0.05,
        'retention_days': 3
    },
    3: {
        'name': 'Low',
        'description': 'Minor information',
        'ltm_probability': 0.15,
        'retention_days': 7
    },
    4: {
        'name': 'Below Average',
        'description': 'Somewhat important',
        'ltm_probability': 0.30,
        'retention_days': 14
    },
    5: {
        'name': 'Baseline',
        'description': 'Standard importance level',
        'ltm_probability': 0.50,
        'retention_days': 30
    },
    6: {
        'name': 'Above Average',
        'description': 'Moderately important',
        'ltm_probability': 0.70,
        'retention_days': 60
    },
    7: {
        'name': 'Important',
        'description': 'Clearly important information',
        'ltm_probability': 0.85,
        'retention_days': 120
    },
    8: {
        'name': 'Very Important',
        'description': 'Highly significant information',
        'ltm_probability': 0.95,
        'retention_days': 365
    },
    9: {
        'name': 'Critical',
        'description': 'Mission-critical information',
        'ltm_probability': 0.99,
        'retention_days': 1095  # 3 years
    },
    10: {
        'name': 'Essential',
        'description': 'Core identity/knowledge information',
        'ltm_probability': 1.0,
        'retention_days': -1    # Never delete
    }
}

# === ENHANCED TAGS SYSTEM ===

ENHANCED_DEFAULT_TAGS = {
    'conversation': {
        'primary': ['dialogue', 'exchange', 'communication'],
        'secondary': ['question', 'answer', 'explanation', 'clarification'],
        'meta': ['conversation_start', 'conversation_end', 'topic_change']
    },
    
    'emotion': {
        'primary': ['emotional', 'feeling', 'mood'],
        'secondary': ['positive_emotion', 'negative_emotion', 'neutral_emotion'],
        'meta': ['emotion_peak', 'emotion_transition', 'mood_change']
    },
    
    'knowledge': {
        'primary': ['fact', 'information', 'knowledge'],
        'secondary': ['learning', 'teaching', 'explanation', 'definition'],
        'meta': ['knowledge_gap', 'expertise', 'uncertainty']
    },
    
    'personality': {
        'primary': ['trait', 'preference', 'behavior'],
        'secondary': ['interest', 'hobby', 'skill', 'talent'],
        'meta': ['personality_insight', 'behavior_pattern', 'growth']
    },
    
    'relationship': {
        'primary': ['social', 'interpersonal', 'relationship'],
        'secondary': ['family', 'friend', 'colleague', 'mentor'],
        'meta': ['relationship_development', 'social_dynamics']
    },
    
    'technical': {
        'primary': ['programming', 'technology', 'technical'],
        'secondary': ['code', 'development', 'solution', 'problem'],
        'meta': ['technical_expertise', 'learning_curve', 'innovation']
    },
    
    'smart_home': {
        'primary': ['device', 'automation', 'control'],
        'secondary': ['lighting', 'security', 'climate', 'entertainment'],
        'meta': ['user_preference', 'automation_rule', 'device_learning']
    }
}

# === DEVICE CONTEXTS ===

DEVICE_CONTEXTS = {
    'mobile': {
        'platforms': ['android', 'ios'],
        'capabilities': ['touch', 'voice', 'camera', 'location'],
        'limitations': ['small_screen', 'limited_input']
    },
    
    'desktop': {
        'platforms': ['windows', 'macos', 'linux'],
        'capabilities': ['keyboard', 'mouse', 'large_screen', 'multitasking'],
        'limitations': ['stationary', 'no_mobility']
    },
    
    'voice_assistant': {
        'platforms': ['alexa', 'google_assistant', 'siri'],
        'capabilities': ['voice_only', 'always_listening', 'smart_home_control'],
        'limitations': ['no_visual', 'limited_context', 'privacy_concerns']
    },
    
    'smart_display': {
        'platforms': ['echo_show', 'nest_hub', 'portal'],
        'capabilities': ['voice', 'visual', 'touch', 'video_call'],
        'limitations': ['fixed_location', 'limited_privacy']
    },
    
    'wearable': {
        'platforms': ['apple_watch', 'fitbit', 'garmin'],
        'capabilities': ['always_on', 'health_monitoring', 'notifications'],
        'limitations': ['tiny_screen', 'minimal_input', 'battery_life']
    }
}

# === LEARNING PROFILES ===

LEARNING_PROFILE_CONFIGS = {
    LearningProfile.CONSERVATIVE: {
        'consolidation_threshold': 10,
        'significance_threshold': 0.8,
        'importance_amplification': 1.2,
        'pattern_recognition_threshold': 0.8,
        'emotion_threshold': 0.5
    },
    
    LearningProfile.BALANCED: {
        'consolidation_threshold': 7,
        'significance_threshold': 0.6,
        'importance_amplification': 1.5,
        'pattern_recognition_threshold': 0.6,
        'emotion_threshold': 0.3
    },
    
    LearningProfile.AGGRESSIVE: {
        'consolidation_threshold': 4,
        'significance_threshold': 0.4,
        'importance_amplification': 2.0,
        'pattern_recognition_threshold': 0.4,
        'emotion_threshold': 0.2
    },
    
    LearningProfile.ADAPTIVE: {
        'consolidation_threshold': 'auto',  # Adapts based on user behavior
        'significance_threshold': 'auto',
        'importance_amplification': 'auto',
        'pattern_recognition_threshold': 'auto',
        'emotion_threshold': 'auto'
    }
}

# === UTILITY FUNCTIONS ===

def get_config_for_profile(profile: LearningProfile) -> Dict[str, Any]:
    """
    Holt Konfiguration für ein spezifisches Learning Profile
    
    Args:
        profile: Learning Profile Enum
        
    Returns:
        Konfiguration für das Profile
    """
    base_config = ENHANCED_MEMORY_CONFIG.copy()
    profile_config = LEARNING_PROFILE_CONFIGS.get(profile, LEARNING_PROFILE_CONFIGS[LearningProfile.BALANCED])
    
    # Apply profile-specific overrides
    for key, value in profile_config.items():
        if key in base_config['stm']:
            base_config['stm'][key] = value
        elif key in base_config['ltm']:
            base_config['ltm'][key] = value
        elif key in base_config['emotion']:
            base_config['emotion'][key] = value
    
    return base_config

def get_emotion_category(emotion: str) -> Optional[str]:
    """
    Bestimmt die Kategorie einer Emotion
    
    Args:
        emotion: Emotion string
        
    Returns:
        Kategorie der Emotion oder None
    """
    for category, emotions in ENHANCED_EMOTION_TYPES.items():
        if emotion.lower() in emotions:
            return category
    return None

def get_importance_config(level: int) -> Dict[str, Any]:
    """
    Holt Konfiguration für ein Importance Level
    
    Args:
        level: Importance Level (1-10)
        
    Returns:
        Konfiguration für das Level
    """
    return ENHANCED_IMPORTANCE_LEVELS.get(level, ENHANCED_IMPORTANCE_LEVELS[5])

def validate_config() -> List[str]:
    """
    Validiert die Konfiguration und gibt Warnungen zurück
    
    Returns:
        Liste von Validierungsfehlern/Warnungen
    """
    warnings = []
    
    # Check memory limits
    stm_config = ENHANCED_MEMORY_CONFIG['stm']
    if stm_config['max_working_memory_items'] > 9:
        warnings.append("Working memory items > 9 may cause cognitive overload")
    
    # Check database settings
    db_config = ENHANCED_MEMORY_CONFIG['database']
    if db_config['cache_size_mb'] > 500:
        warnings.append("Database cache > 500MB may use excessive memory")
    
    # Check performance settings
    perf_config = ENHANCED_MEMORY_CONFIG['performance']
    if perf_config['memory_cache_size_mb'] > 1000:
        warnings.append("Memory cache > 1GB may cause system issues")
    
    return warnings

# === BACKWARDS COMPATIBILITY ===

# Legacy config for existing code
MEMORY_CONFIG = {
    'short_term': ENHANCED_MEMORY_CONFIG['stm'],
    'long_term': ENHANCED_MEMORY_CONFIG['ltm'],
    'emotion': ENHANCED_MEMORY_CONFIG['emotion'],
    'database': ENHANCED_MEMORY_CONFIG['database'],
    'performance': ENHANCED_MEMORY_CONFIG['performance']
}

# Legacy emotion types
EMOTION_TYPES = (
    ENHANCED_EMOTION_TYPES['primary'] + 
    ENHANCED_EMOTION_TYPES['secondary'] + 
    ['neutral']
)

# Legacy importance levels (simplified)
IMPORTANCE_LEVELS = {
    level: config['name'] 
    for level, config in ENHANCED_IMPORTANCE_LEVELS.items()
}

# Legacy default tags (simplified)
DEFAULT_TAGS = {
    category: config['primary'] + config['secondary']
    for category, config in ENHANCED_DEFAULT_TAGS.items()
}

# Export all
__all__ = [
    'ENHANCED_MEMORY_CONFIG',
    'ENHANCED_EMOTION_TYPES', 
    'ENHANCED_IMPORTANCE_LEVELS',
    'ENHANCED_DEFAULT_TAGS',
    'DEVICE_CONTEXTS',
    'LEARNING_PROFILE_CONFIGS',
    'MemorySystemMode',
    'LearningProfile',
    'get_config_for_profile',
    'get_emotion_category',
    'get_importance_config',
    'validate_config',
    # Legacy exports
    'MEMORY_CONFIG',
    'EMOTION_TYPES',
    'IMPORTANCE_LEVELS', 
    'DEFAULT_TAGS'
]