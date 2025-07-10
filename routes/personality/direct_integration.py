"""
Personality Direct Integration Module
Direkte Integration mit dem Personality System für Real-time Zugriff
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def get_direct_personality_data(memory_manager=None, 
                              personality_system=None,
                              data_scope: str = 'comprehensive') -> Dict[str, Any]:
    """
    Holt Personality Data direkt aus dem Personality System
    
    Extrahiert aus kira_routes.py.backup Direct Personality Access Logic
    """
    try:
        # Versuche verschiedene Quellen für Personality System
        active_personality_system = None
        
        if personality_system:
            active_personality_system = personality_system
        elif memory_manager and hasattr(memory_manager, 'kira_personality'):
            active_personality_system = memory_manager.kira_personality
        elif memory_manager and hasattr(memory_manager, 'personality_system'):
            active_personality_system = memory_manager.personality_system
        
        if not active_personality_system:
            return _generate_fallback_personality_data()
        
        # Extract comprehensive personality data
        personality_data = {
            'system_info': _extract_system_info(active_personality_system),
            'traits': _extract_traits_data(active_personality_system, data_scope),
            'current_state': _extract_current_state(active_personality_system),
            'development_info': _extract_development_info(active_personality_system),
            'behavioral_patterns': _extract_behavioral_patterns(active_personality_system)
        }
        
        # Enhanced data extraction for comprehensive scope
        if data_scope == 'comprehensive':
            personality_data.update({
                'interaction_history': _extract_interaction_history(active_personality_system),
                'development_history': _extract_development_history(active_personality_system),
                'learning_patterns': _extract_learning_patterns(active_personality_system),
                'adaptation_metrics': _extract_adaptation_metrics(active_personality_system),
                'evolution_tracking': _extract_evolution_tracking(active_personality_system)
            })
        
        # Integration metadata
        personality_data['integration_metadata'] = {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_scope': data_scope,
            'personality_system_type': type(active_personality_system).__name__,
            'data_freshness': _assess_data_freshness(active_personality_system),
            'integration_quality': _assess_integration_quality(personality_data)
        }
        
        return personality_data
        
    except Exception as e:
        logger.error(f"Direct personality data extraction failed: {e}")
        return {
            'error': str(e),
            'fallback_data': _generate_fallback_personality_data(),
            'integration_status': 'failed'
        }

def integrate_with_personality_system(memory_manager=None,
                                    personality_system=None,
                                    integration_level: str = 'full') -> Dict[str, Any]:
    """
    Integriert mit dem Personality System
    
    Basiert auf kira_routes.py.backup Integration Logic
    """
    try:
        integration_result = {
            'integration_timestamp': datetime.now().isoformat(),
            'integration_level': integration_level,
            'integration_steps': [],
            'integration_success': False
        }
        
        # Step 1: Validate Personality System Access
        validation_result = validate_personality_integration(memory_manager, personality_system)
        integration_result['integration_steps'].append({
            'step': 'validation',
            'result': validation_result,
            'success': validation_result.get('integration_possible', False)
        })
        
        if not validation_result.get('integration_possible', False):
            integration_result['integration_error'] = 'Personality system validation failed'
            return integration_result
        
        # Step 2: Establish Connection
        connection_result = _establish_personality_connection(memory_manager, personality_system)
        integration_result['integration_steps'].append({
            'step': 'connection',
            'result': connection_result,
            'success': connection_result.get('connected', False)
        })
        
        if not connection_result.get('connected', False):
            integration_result['integration_error'] = 'Failed to establish personality system connection'
            return integration_result
        
        # Step 3: Sync Initial State
        sync_result = sync_personality_state(memory_manager, personality_system)
        integration_result['integration_steps'].append({
            'step': 'initial_sync',
            'result': sync_result,
            'success': sync_result.get('sync_successful', False)
        })
        
        # Step 4: Configure Integration Level
        if integration_level == 'full':
            config_result = _configure_full_integration(memory_manager, personality_system)
        elif integration_level == 'read_only':
            config_result = _configure_read_only_integration(memory_manager, personality_system)
        else:  # basic
            config_result = _configure_basic_integration(memory_manager, personality_system)
        
        integration_result['integration_steps'].append({
            'step': 'configuration',
            'result': config_result,
            'success': config_result.get('configured', False)
        })
        
        # Step 5: Test Integration
        test_result = _test_personality_integration(memory_manager, personality_system)
        integration_result['integration_steps'].append({
            'step': 'testing',
            'result': test_result,
            'success': test_result.get('test_passed', False)
        })
        
        # Final Integration Status
        all_steps_successful = all(step['success'] for step in integration_result['integration_steps'])
        integration_result['integration_success'] = all_steps_successful
        
        if all_steps_successful:
            integration_result['integration_capabilities'] = _determine_integration_capabilities(
                memory_manager, personality_system, integration_level
            )
            integration_result['usage_recommendations'] = _generate_integration_usage_recommendations(
                integration_result
            )
        
        return integration_result
        
    except Exception as e:
        logger.error(f"Personality system integration failed: {e}")
        return {
            'integration_success': False,
            'integration_error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def sync_personality_state(memory_manager=None, 
                         personality_system=None,
                         sync_direction: str = 'bidirectional') -> Dict[str, Any]:
    """
    Synchronisiert Personality State
    
    Extrahiert aus kira_routes.py.backup State Sync Logic
    """
    try:
        sync_result = {
            'sync_timestamp': datetime.now().isoformat(),
            'sync_direction': sync_direction,
            'sync_operations': [],
            'sync_successful': False
        }
        
        # Identify active personality system
        active_personality_system = _identify_active_personality_system(memory_manager, personality_system)
        
        if not active_personality_system:
            sync_result['sync_error'] = 'No active personality system found'
            return sync_result
        
        # Sync operations based on direction
        if sync_direction in ['bidirectional', 'from_personality']:
            # Sync FROM personality system TO routes
            from_personality_result = _sync_from_personality_system(active_personality_system)
            sync_result['sync_operations'].append({
                'operation': 'from_personality_system',
                'result': from_personality_result,
                'success': from_personality_result.get('success', False)
            })
        
        if sync_direction in ['bidirectional', 'to_personality']:
            # Sync TO personality system FROM routes
            to_personality_result = _sync_to_personality_system(active_personality_system, memory_manager)
            sync_result['sync_operations'].append({
                'operation': 'to_personality_system', 
                'result': to_personality_result,
                'success': to_personality_result.get('success', False)
            })
        
        # Verify sync consistency
        consistency_check = _verify_sync_consistency(active_personality_system, memory_manager)
        sync_result['sync_operations'].append({
            'operation': 'consistency_verification',
            'result': consistency_check,
            'success': consistency_check.get('consistent', False)
        })
        
        # Overall sync success
        sync_result['sync_successful'] = all(
            op['success'] for op in sync_result['sync_operations']
        )
        
        if sync_result['sync_successful']:
            sync_result['sync_summary'] = {
                'traits_synced': consistency_check.get('traits_synced', 0),
                'state_components_synced': consistency_check.get('state_components_synced', 0),
                'history_entries_synced': consistency_check.get('history_entries_synced', 0),
                'data_integrity_score': consistency_check.get('data_integrity_score', 0.0)
            }
        
        return sync_result
        
    except Exception as e:
        logger.error(f"Personality state sync failed: {e}")
        return {
            'sync_successful': False,
            'sync_error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def validate_personality_integration(memory_manager=None,
                                   personality_system=None) -> Dict[str, Any]:
    """
    Validiert Personality Integration Möglichkeiten
    
    Basiert auf kira_routes.py.backup Validation Logic
    """
    try:
        validation_result = {
            'validation_timestamp': datetime.now().isoformat(),
            'integration_possible': False,
            'validation_checks': [],
            'integration_recommendations': []
        }
        
        # Check 1: Memory Manager Availability
        memory_manager_check = _validate_memory_manager_availability(memory_manager)
        validation_result['validation_checks'].append({
            'check': 'memory_manager_availability',
            'result': memory_manager_check,
            'passed': memory_manager_check.get('available', False)
        })
        
        # Check 2: Personality System Availability
        personality_system_check = _validate_personality_system_availability(personality_system, memory_manager)
        validation_result['validation_checks'].append({
            'check': 'personality_system_availability',
            'result': personality_system_check,
            'passed': personality_system_check.get('available', False)
        })
        
        # Check 3: Integration Interface Compatibility
        interface_check = _validate_integration_interface_compatibility(memory_manager, personality_system)
        validation_result['validation_checks'].append({
            'check': 'interface_compatibility',
            'result': interface_check,
            'passed': interface_check.get('compatible', False)
        })
        
        # Check 4: Data Access Permissions
        permissions_check = _validate_data_access_permissions(memory_manager, personality_system)
        validation_result['validation_checks'].append({
            'check': 'data_access_permissions',
            'result': permissions_check,
            'passed': permissions_check.get('accessible', False)
        })
        
        # Check 5: System State Consistency
        consistency_check = _validate_system_state_consistency(memory_manager, personality_system)
        validation_result['validation_checks'].append({
            'check': 'system_state_consistency',
            'result': consistency_check,
            'passed': consistency_check.get('consistent', False)
        })
        
        # Overall validation result
        all_checks_passed = all(check['passed'] for check in validation_result['validation_checks'])
        critical_checks_passed = any(check['passed'] for check in validation_result['validation_checks'][:2])  # At least one system available
        
        validation_result['integration_possible'] = critical_checks_passed
        validation_result['full_integration_possible'] = all_checks_passed
        
        # Generate recommendations
        validation_result['integration_recommendations'] = _generate_integration_recommendations(
            validation_result['validation_checks']
        )
        
        # Determine best integration strategy
        validation_result['recommended_integration_strategy'] = _recommend_integration_strategy(
            validation_result
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Personality integration validation failed: {e}")
        return {
            'integration_possible': False,
            'validation_error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _extract_system_info(personality_system) -> Dict[str, Any]:
    """Extrahiert System Info aus Personality System"""
    try:
        system_info = {
            'system_type': type(personality_system).__name__,
            'system_status': 'active',
            'available_methods': _get_available_methods(personality_system),
            'system_capabilities': _assess_system_capabilities(personality_system)
        }
        
        # Try to get system-specific info
        if hasattr(personality_system, 'get_system_info'):
            system_specific_info = personality_system.get_system_info()
            system_info.update(system_specific_info)
        
        return system_info
        
    except Exception as e:
        logger.debug(f"System info extraction failed: {e}")
        return {
            'system_type': 'unknown',
            'system_status': 'limited_access',
            'error': str(e)
        }

def _extract_traits_data(personality_system, data_scope: str) -> Dict[str, Any]:
    """Extrahiert Traits Data aus Personality System"""
    try:
        traits_data = {}
        
        # Try different methods to access traits
        if hasattr(personality_system, 'traits'):
            traits_data = _convert_traits_to_dict(personality_system.traits)
        elif hasattr(personality_system, 'get_traits'):
            traits_data = personality_system.get_traits()
        elif hasattr(personality_system, 'personality_traits'):
            traits_data = _convert_traits_to_dict(personality_system.personality_traits)
        
        # Enhanced data extraction for comprehensive scope
        if data_scope == 'comprehensive' and traits_data:
            for trait_name, trait_data in traits_data.items():
                if isinstance(trait_data, dict):
                    # Add calculated fields
                    trait_data['extraction_timestamp'] = datetime.now().isoformat()
                    trait_data['data_freshness'] = _calculate_trait_data_freshness(trait_data)
        
        return traits_data
        
    except Exception as e:
        logger.debug(f"Traits data extraction failed: {e}")
        return {}

def _extract_current_state(personality_system) -> Dict[str, Any]:
    """Extrahiert Current State aus Personality System"""
    try:
        current_state = {}
        
        # Try different methods to access current state
        if hasattr(personality_system, 'current_state'):
            current_state = _convert_state_to_dict(personality_system.current_state)
        elif hasattr(personality_system, 'get_current_state'):
            current_state = personality_system.get_current_state()
        elif hasattr(personality_system, 'emotional_state'):
            current_state = _convert_state_to_dict(personality_system.emotional_state)
        
        # Add computed state metrics
        if current_state:
            current_state['state_extraction_timestamp'] = datetime.now().isoformat()
            current_state['state_coherence_score'] = _calculate_state_coherence(current_state)
        
        return current_state
        
    except Exception as e:
        logger.debug(f"Current state extraction failed: {e}")
        return {}

def _convert_traits_to_dict(traits_obj) -> Dict[str, Any]:
    """Konvertiert Traits Objekt zu Dictionary"""
    try:
        if isinstance(traits_obj, dict):
            return traits_obj
        
        traits_dict = {}
        
        # Try to iterate over traits object
        if hasattr(traits_obj, '__dict__'):
            for attr_name, attr_value in traits_obj.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    if hasattr(attr_value, '__dict__'):
                        # Nested trait object
                        traits_dict[attr_name] = attr_value.__dict__
                    else:
                        traits_dict[attr_name] = attr_value
        
        # Try alternative iteration methods
        elif hasattr(traits_obj, 'items'):
            traits_dict = dict(traits_obj.items())
        elif hasattr(traits_obj, 'keys'):
            for key in traits_obj.keys():
                traits_dict[key] = getattr(traits_obj, key, None)
        
        return traits_dict
        
    except Exception as e:
        logger.debug(f"Traits conversion to dict failed: {e}")
        return {}

def _convert_state_to_dict(state_obj) -> Dict[str, Any]:
    """Konvertiert State Objekt zu Dictionary"""
    try:
        if isinstance(state_obj, dict):
            return state_obj
        
        state_dict = {}
        
        # Standard state attributes to look for
        standard_attributes = [
            'emotional_stability', 'adaptability', 'empathy_level',
            'social_confidence', 'learning_motivation', 'creativity_level',
            'curiosity', 'openness', 'conscientiousness'
        ]
        
        # Extract standard attributes
        for attr in standard_attributes:
            if hasattr(state_obj, attr):
                value = getattr(state_obj, attr)
                if isinstance(value, (int, float, str, bool)):
                    state_dict[attr] = value
        
        # Extract any additional attributes
        if hasattr(state_obj, '__dict__'):
            for attr_name, attr_value in state_obj.__dict__.items():
                if not attr_name.startswith('_') and attr_name not in state_dict:
                    if isinstance(attr_value, (int, float, str, bool, list, dict)):
                        state_dict[attr_name] = attr_value
        
        return state_dict
        
    except Exception as e:
        logger.debug(f"State conversion to dict failed: {e}")
        return {}

def _establish_personality_connection(memory_manager, personality_system) -> Dict[str, Any]:
    """Etabliert Verbindung zum Personality System"""
    try:
        connection_result = {
            'connected': False,
            'connection_type': 'none',
            'connection_quality': 0.0
        }
        
        # Try direct personality system connection
        if personality_system:
            connection_result['connected'] = True
            connection_result['connection_type'] = 'direct_personality_system'
            connection_result['connection_quality'] = 1.0
            return connection_result
        
        # Try memory manager personality connection
        if memory_manager:
            if hasattr(memory_manager, 'kira_personality'):
                connection_result['connected'] = True
                connection_result['connection_type'] = 'memory_manager_personality'
                connection_result['connection_quality'] = 0.9
                return connection_result
            elif hasattr(memory_manager, 'personality_system'):
                connection_result['connected'] = True
                connection_result['connection_type'] = 'memory_manager_personality_system'
                connection_result['connection_quality'] = 0.8
                return connection_result
        
        return connection_result
        
    except Exception as e:
        logger.debug(f"Personality connection establishment failed: {e}")
        return {
            'connected': False,
            'connection_error': str(e)
        }

def _validate_memory_manager_availability(memory_manager) -> Dict[str, Any]:
    """Validiert Memory Manager Verfügbarkeit"""
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'memory_manager_not_provided'
            }
        
        # Check basic memory manager functionality
        has_basic_methods = all(hasattr(memory_manager, method) for method in [
            'get_memories', 'add_memory'
        ])
        
        # Check personality-related attributes
        has_personality_attributes = any(hasattr(memory_manager, attr) for attr in [
            'kira_personality', 'personality_system', 'personality'
        ])
        
        return {
            'available': True,
            'has_basic_methods': has_basic_methods,
            'has_personality_attributes': has_personality_attributes,
            'memory_manager_type': type(memory_manager).__name__,
            'integration_potential': 'high' if has_personality_attributes else 'medium' if has_basic_methods else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Memory manager validation failed: {e}")
        return {
            'available': False,
            'validation_error': str(e)
        }

def _validate_personality_system_availability(personality_system, memory_manager) -> Dict[str, Any]:
    """Validiert Personality System Verfügbarkeit"""
    try:
        availability_result = {
            'available': False,
            'sources_checked': [],
            'best_source': None
        }
        
        # Check direct personality system
        if personality_system:
            availability_result['sources_checked'].append({
                'source': 'direct_personality_system',
                'available': True,
                'quality': 'high'
            })
            availability_result['available'] = True
            availability_result['best_source'] = 'direct_personality_system'
        
        # Check memory manager personality
        if memory_manager:
            if hasattr(memory_manager, 'kira_personality') and memory_manager.kira_personality:
                availability_result['sources_checked'].append({
                    'source': 'memory_manager_kira_personality',
                    'available': True,
                    'quality': 'high'
                })
                if not availability_result['available']:
                    availability_result['available'] = True
                    availability_result['best_source'] = 'memory_manager_kira_personality'
            
            if hasattr(memory_manager, 'personality_system') and memory_manager.personality_system:
                availability_result['sources_checked'].append({
                    'source': 'memory_manager_personality_system',
                    'available': True,
                    'quality': 'medium'
                })
                if not availability_result['available']:
                    availability_result['available'] = True
                    availability_result['best_source'] = 'memory_manager_personality_system'
        
        return availability_result
        
    except Exception as e:
        logger.debug(f"Personality system validation failed: {e}")
        return {
            'available': False,
            'validation_error': str(e)
        }

def _generate_fallback_personality_data() -> Dict[str, Any]:
    """Generiert Fallback Personality Data"""
    return {
        'fallback_mode': True,
        'system_info': {
            'system_type': 'fallback',
            'system_status': 'limited'
        },
        'traits': {
            'curiosity': {'current_strength': 0.8, 'base_strength': 0.7},
            'empathy': {'current_strength': 0.7, 'base_strength': 0.6},
            'creativity': {'current_strength': 0.6, 'base_strength': 0.5}
        },
        'current_state': {
            'emotional_stability': 0.7,
            'adaptability': 0.6,
            'empathy_level': 0.8
        },
        'integration_metadata': {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_scope': 'fallback',
            'integration_status': 'fallback_mode',
            'recommendation': 'Enable personality system integration for full functionality'
        }
    }

__all__ = [
    'get_direct_personality_data',
    'integrate_with_personality_system', 
    'sync_personality_state',
    'validate_personality_integration'
]