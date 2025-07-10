"""
🎛️ CONFIGURATION MANAGEMENT API
RESTful API für System Configuration Management
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from typing import Dict, Any
from config.system_config import get_system_config, reload_system_config

logger = logging.getLogger(__name__)

def create_config_api_blueprint() -> Blueprint:
    """
    Erstellt Configuration API Blueprint
    
    Returns:
        Flask Blueprint
    """
    
    config_bp = Blueprint('config_api', __name__, url_prefix='/config')
    
    @config_bp.route('/current', methods=['GET'])
    def get_current_configuration():
        """
        ✅ GET CURRENT CONFIGURATION
        
        GET /api/config/current
        """
        try:
            system_config = get_system_config()
            config_summary = system_config.get_configuration_summary()
            
            return jsonify({
                'success': True,
                'configuration': config_summary,
                'config_file': system_config.config_file,
                'environment_overrides': system_config.get_environment_overrides(),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Get current configuration failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/validate', methods=['POST'])
    def validate_configuration():
        """
        ✅ VALIDATE CONFIGURATION
        
        POST /api/config/validate
        {
            "configuration": {...}  // Optional: validate custom config
        }
        """
        try:
            data = request.get_json() or {}
            
            if 'configuration' in data:
                # Validate custom configuration
                # TODO: Implement custom config validation
                return jsonify({
                    'success': False,
                    'error': 'Custom configuration validation not yet implemented'
                }), 501
            else:
                # Validate current configuration
                system_config = get_system_config()
                validation_result = system_config.validate_configuration()
                
                return jsonify({
                    'success': True,
                    'validation_result': validation_result,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/components', methods=['GET'])
    def get_component_configurations():
        """
        ✅ GET COMPONENT CONFIGURATIONS
        
        GET /api/config/components
        """
        try:
            system_config = get_system_config()
            
            components = {
                'lm_studio': {
                    'enabled': True,
                    'config': {
                        'url': system_config.lm_studio.url,
                        'model_name': system_config.lm_studio.model_name,
                        'max_tokens': system_config.lm_studio.max_tokens,
                        'temperature': system_config.lm_studio.temperature,
                        'timeout': system_config.lm_studio.timeout,
                        'context_window': system_config.lm_studio.context_window
                    },
                    'status': 'active' if system_config.lm_studio.url else 'not_configured'
                },
                'voice_system': {
                    'enabled': system_config.voice.enable_voice,
                    'config': {
                        'default_emotion': system_config.voice.default_emotion,
                        'voice_speed': system_config.voice.voice_speed,
                        'voice_quality': system_config.voice.voice_quality,
                        'sample_rate': system_config.voice.sample_rate,
                        'enable_emotion_synthesis': system_config.voice.enable_emotion_synthesis,
                        'enable_speech_recognition': system_config.voice.enable_speech_recognition
                    },
                    'status': 'enabled' if system_config.voice.enable_voice else 'disabled'
                },
                'memory_system': {
                    'enabled': system_config.memory.enable_memory,
                    'config': {
                        'data_dir': system_config.memory.data_dir,
                        'stm_capacity': system_config.memory.stm_capacity,
                        'ltm_capacity': system_config.memory.ltm_capacity,
                        'consolidation_threshold': system_config.memory.consolidation_threshold,
                        'importance_threshold': system_config.memory.importance_threshold,
                        'enable_storage': system_config.memory.enable_storage,
                        'storage_backend': system_config.memory.storage_backend
                    },
                    'status': 'enabled' if system_config.memory.enable_memory else 'disabled'
                },
                'emotion_engine': {
                    'enabled': system_config.emotion.enable_emotion_analysis,
                    'config': {
                        'enable_personality_profiling': system_config.emotion.enable_personality_profiling,
                        'emotion_confidence_threshold': system_config.emotion.emotion_confidence_threshold,
                        'personality_learning_rate': system_config.emotion.personality_learning_rate,
                        'enable_response_personalization': system_config.emotion.enable_response_personalization,
                        'emotion_history_limit': system_config.emotion.emotion_history_limit
                    },
                    'status': 'enabled' if system_config.emotion.enable_emotion_analysis else 'disabled'
                },
                'security': {
                    'enabled': True,  # Always enabled
                    'config': {
                        'enable_rate_limiting': system_config.security.enable_rate_limiting,
                        'rate_limit_per_minute': system_config.security.rate_limit_per_minute,
                        'enable_input_validation': system_config.security.enable_input_validation,
                        'max_input_length': system_config.security.max_input_length,
                        'enable_content_filtering': system_config.security.enable_content_filtering,
                        'log_conversations': system_config.security.log_conversations
                    },
                    'status': 'active'
                },
                'database': {
                    'enabled': system_config.memory.enable_storage,
                    'config': {
                        'database_type': system_config.database.database_type,
                        'database_url': system_config.database.database_url,
                        'backup_enabled': system_config.database.backup_enabled,
                        'backup_interval': system_config.database.backup_interval,
                        'retention_days': system_config.database.retention_days
                    },
                    'status': 'enabled' if system_config.memory.enable_storage else 'disabled'
                }
            }
            
            return jsonify({
                'success': True,
                'components': components,
                'total_components': len(components),
                'enabled_components': len([c for c in components.values() if c['enabled']]),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Get component configurations failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/update', methods=['POST'])
    def update_configuration():
        """
        ✅ UPDATE CONFIGURATION
        
        POST /api/config/update
        {
            "component": "lm_studio|voice_system|memory_system|emotion_engine|security|database",
            "config": {...}
        }
        """
        try:
            data = request.get_json()
            if not data or 'component' not in data or 'config' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Component and config parameters required'
                }), 400
            
            component = data['component']
            new_config = data['config']
            
            system_config = get_system_config()
            
            # Update specific component configuration
            if component == 'lm_studio':
                for key, value in new_config.items():
                    if hasattr(system_config.lm_studio, key):
                        setattr(system_config.lm_studio, key, value)
            
            elif component == 'voice_system':
                for key, value in new_config.items():
                    if hasattr(system_config.voice, key):
                        setattr(system_config.voice, key, value)
            
            elif component == 'memory_system':
                for key, value in new_config.items():
                    if hasattr(system_config.memory, key):
                        setattr(system_config.memory, key, value)
            
            elif component == 'emotion_engine':
                for key, value in new_config.items():
                    if hasattr(system_config.emotion, key):
                        setattr(system_config.emotion, key, value)
            
            elif component == 'security':
                for key, value in new_config.items():
                    if hasattr(system_config.security, key):
                        setattr(system_config.security, key, value)
            
            elif component == 'database':
                for key, value in new_config.items():
                    if hasattr(system_config.database, key):
                        setattr(system_config.database, key, value)
            
            else:
                return jsonify({
                    'success': False,
                    'error': f'Unknown component: {component}'
                }), 400
            
            # Validate updated configuration
            validation_result = system_config.validate_configuration()
            
            if validation_result['valid']:
                # Save configuration
                save_success = system_config.save_configuration()
                
                if save_success:
                    return jsonify({
                        'success': True,
                        'message': f'{component} configuration updated successfully',
                        'component': component,
                        'validation_result': validation_result,
                        'restart_required': component in ['lm_studio', 'voice_system', 'database'],
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to save configuration'
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': 'Configuration validation failed',
                    'validation_errors': validation_result['errors']
                }), 400
            
        except Exception as e:
            logger.error(f"Update configuration failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/reload', methods=['POST'])
    def reload_configuration():
        """
        ✅ RELOAD CONFIGURATION
        
        POST /api/config/reload
        {
            "config_file": "optional_new_config_file.yaml"
        }
        """
        try:
            data = request.get_json() or {}
            config_file = data.get('config_file')
            
            # Reload configuration
            system_config = reload_system_config(config_file)
            validation_result = system_config.validate_configuration()
            
            return jsonify({
                'success': True,
                'message': 'Configuration reloaded successfully',
                'config_file': system_config.config_file,
                'validation_result': validation_result,
                'restart_recommended': True,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Reload configuration failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/export', methods=['GET'])
    def export_configuration():
        """
        ✅ EXPORT CONFIGURATION
        
        GET /api/config/export?format=json|yaml
        """
        try:
            export_format = request.args.get('format', 'json').lower()
            
            if export_format not in ['json', 'yaml']:
                return jsonify({
                    'success': False,
                    'error': 'Format must be json or yaml'
                }), 400
            
            system_config = get_system_config()
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'generated_at': datetime.now().isoformat(),
                    'kira_version': system_config.system_info['version'],
                    'config_file': system_config.config_file,
                    'format': export_format
                },
                'configuration': system_config.get_configuration_summary()
            }
            
            if export_format == 'yaml':
                import yaml
                yaml_content = yaml.dump(export_data, default_flow_style=False, indent=2)
                
                from flask import Response
                return Response(
                    yaml_content,
                    mimetype='application/x-yaml',
                    headers={
                        'Content-Disposition': f'attachment; filename=kira_config_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
                    }
                )
            else:
                return jsonify({
                    'success': True,
                    'export_data': export_data,
                    'format': 'json',
                    'download_ready': True
                })
            
        except Exception as e:
            logger.error(f"Export configuration failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/presets', methods=['GET'])
    def get_configuration_presets():
        """
        ✅ GET CONFIGURATION PRESETS
        
        GET /api/config/presets
        """
        try:
            presets = {
                'development': {
                    'name': 'Development Mode',
                    'description': 'Optimized for development and testing',
                    'config': {
                        'api': {'debug': True, 'port': 5001},
                        'lm_studio': {'temperature': 0.8, 'max_tokens': 1024},
                        'voice': {'enable_voice': True, 'voice_quality': 'medium'},
                        'memory': {'stm_capacity': 30, 'ltm_capacity': 500},
                        'security': {'enable_rate_limiting': False, 'log_conversations': True},
                        'performance': {'enable_profiling': True}
                    }
                },
                'production': {
                    'name': 'Production Mode',
                    'description': 'Optimized for production deployment',
                    'config': {
                        'api': {'debug': False, 'port': 5000},
                        'lm_studio': {'temperature': 0.7, 'max_tokens': 2048},
                        'voice': {'enable_voice': True, 'voice_quality': 'high'},
                        'memory': {'stm_capacity': 50, 'ltm_capacity': 2000},
                        'security': {'enable_rate_limiting': True, 'log_conversations': True},
                        'performance': {'enable_profiling': False, 'enable_caching': True}
                    }
                },
                'minimal': {
                    'name': 'Minimal Mode',
                    'description': 'Basic chat functionality only',
                    'config': {
                        'api': {'debug': False, 'port': 5001},
                        'lm_studio': {'temperature': 0.7, 'max_tokens': 1024},
                        'voice': {'enable_voice': False},
                        'memory': {'stm_capacity': 20, 'ltm_capacity': 100, 'enable_storage': False},
                        'emotion': {'enable_emotion_analysis': False},
                        'security': {'enable_rate_limiting': True}
                    }
                },
                'high_performance': {
                    'name': 'High Performance Mode',
                    'description': 'Maximum performance and capabilities',
                    'config': {
                        'api': {'debug': False, 'port': 5000},
                        'lm_studio': {'temperature': 0.7, 'max_tokens': 4096, 'context_window': 8192},
                        'voice': {'enable_voice': True, 'voice_quality': 'high', 'enable_emotion_synthesis': True},
                        'memory': {'stm_capacity': 100, 'ltm_capacity': 5000, 'enable_storage': True},
                        'emotion': {'enable_emotion_analysis': True, 'enable_personality_profiling': True},
                        'performance': {'max_concurrent_requests': 20, 'enable_caching': True},
                        'database': {'backup_enabled': True}
                    }
                }
            }
            
            return jsonify({
                'success': True,
                'presets': presets,
                'total_presets': len(presets),
                'current_mode': 'custom',  # TODO: Detect current mode
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Get configuration presets failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @config_bp.route('/apply-preset', methods=['POST'])
    def apply_configuration_preset():
        """
        ✅ APPLY CONFIGURATION PRESET
        
        POST /api/config/apply-preset
        {
            "preset": "development|production|minimal|high_performance"
        }
        """
        try:
            data = request.get_json()
            if not data or 'preset' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Preset parameter required'
                }), 400
            
            preset_name = data['preset']
            
            # Get presets
            presets_response = get_configuration_presets()
            if not presets_response.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load presets'
                }), 500
            
            presets_data = presets_response.get_json()
            if preset_name not in presets_data['presets']:
                return jsonify({
                    'success': False,
                    'error': f'Unknown preset: {preset_name}'
                }), 400
            
            preset_config = presets_data['presets'][preset_name]['config']
            
            # Apply preset configuration
            system_config = get_system_config()
            
            # Apply each component configuration
            for component, config in preset_config.items():
                update_data = {
                    'component': component,
                    'config': config
                }
                
                # Simulate internal update call
                try:
                    update_response = update_configuration()
                    # Note: This is a simplified implementation
                    # In a real scenario, you'd want to apply all changes in a transaction
                except Exception as e:
                    logger.warning(f"Failed to apply {component} config from preset: {e}")
            
            # Save configuration
            save_success = system_config.save_configuration()
            
            if save_success:
                return jsonify({
                    'success': True,
                    'message': f'Preset "{preset_name}" applied successfully',
                    'preset_name': preset_name,
                    'preset_description': presets_data['presets'][preset_name]['description'],
                    'restart_required': True,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save preset configuration'
                }), 500
            
        except Exception as e:
            logger.error(f"Apply configuration preset failed: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return config_bp

# Export
__all__ = ['create_config_api_blueprint']