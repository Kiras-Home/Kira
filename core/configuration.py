"""
Kira Configuration Management
Handles logging setup and configuration utilities
"""

import logging
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: str = 'kira_system.log') -> logging.Logger:
    """
    Setup comprehensive logging for Kira system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file name
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(logs_dir / log_file),
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger('kira')
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
    
    return logger


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'logs',
        'data',
        'voice/output',
        'memory/data',
        'config'
    ]
    
    project_root = get_project_root()
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ All required directories ensured")