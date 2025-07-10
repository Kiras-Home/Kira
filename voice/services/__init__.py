"""
Voice Services Package
Enterprise-grade voice services similar to memory services
"""

from .voice_service import VoiceService, VoiceServiceConfig, VoiceRequest, VoiceResponse

# Export main classes
__all__ = [
    'VoiceService',
    'VoiceServiceConfig', 
    'VoiceRequest',
    'VoiceResponse'
]

# Convenience function for creating voice service
def create_voice_service(
    memory_service=None,
    command_processor=None,
    config_dict: dict = None
) -> VoiceService:
    """
    Create and configure voice service
    
    Args:
        memory_service: Memory service instance
        command_processor: Command processor instance  
        config_dict: Configuration dictionary
        
    Returns:
        VoiceService: Configured voice service instance
    """
    config = None
    if config_dict:
        config = VoiceServiceConfig(**config_dict)
    
    return VoiceService(
        config=config,
        memory_service=memory_service,
        command_processor=command_processor
    )