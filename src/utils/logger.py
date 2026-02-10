"""
Logging configuration for DMOS
Uses loguru for better formatting and features
"""

import sys
from loguru import logger
from pathlib import Path

def get_logger(name: str = "DMOS", level: str = "INFO", log_file: str = None):
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (will appear in log messages)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
    
    Returns:
        Configured logger instance
    """
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip"  # Compress rotated logs
        )
    
    # Bind context (logger name)
    return logger.bind(name=name)


# Convenience function for common use case
def setup_logging(component_name: str, log_dir: str = "logs", level: str = "INFO"):
    """
    Setup logging for a DMOS component
    
    Args:
        component_name: Name of the component (e.g., "coordinator", "score_agent")
        log_dir: Directory for log files
        level: Log level
    
    Returns:
        Configured logger
    """
    log_file = f"{log_dir}/{component_name}.log"
    return get_logger(name=component_name, level=level, log_file=log_file)