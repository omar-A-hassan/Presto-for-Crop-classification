#!/usr/bin/env python3
"""
Professional Logging Utility
============================

Standardized logging for the Enhanced PRESTO Crop Classification System.
Provides consistent logging with levels, timestamps, and output formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PrestoLogger:
    """Professional logger for PRESTO crop classification system"""
    
    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def section(self, title: str, level: str = "INFO"):
        """Log a section header"""
        separator = "=" * 60
        getattr(self.logger, level.lower())(separator)
        getattr(self.logger, level.lower())(f"🚀 {title.upper()}")
        getattr(self.logger, level.lower())(separator)
    
    def subsection(self, title: str, level: str = "INFO"):
        """Log a subsection header"""
        separator = "-" * 40
        getattr(self.logger, level.lower())(separator)
        getattr(self.logger, level.lower())(f"📋 {title}")
        getattr(self.logger, level.lower())(separator)
    
    def success(self, message: str):
        """Log success message"""
        self.logger.info(f"✅ {message}")
    
    def progress(self, current: int, total: int, item: str = "items"):
        """Log progress update"""
        percentage = (current / total) * 100
        self.logger.info(f"Progress: {current:,}/{total:,} {item} ({percentage:.1f}%)")


def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> PrestoLogger:
    """
    Get a standardized logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        PrestoLogger instance
    """
    return PrestoLogger(name, level, log_file)


def setup_training_logger(experiment_name: str) -> PrestoLogger:
    """
    Set up logger for training experiments
    
    Args:
        experiment_name: Name of the training experiment
        
    Returns:
        PrestoLogger configured for training
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training/{experiment_name}_{timestamp}.log"
    return get_logger("training", level="INFO", log_file=log_file)


def setup_inference_logger() -> PrestoLogger:
    """
    Set up logger for inference
    
    Returns:
        PrestoLogger configured for inference
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/inference/inference_{timestamp}.log"
    return get_logger("inference", level="INFO", log_file=log_file)