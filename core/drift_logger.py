"""
DRIFT Structured Logging System
ISO timestamp logging with component names and contextual information

Replaces all print() statements with structured JSON logs for better
monitoring, debugging, and system analysis.
"""

import structlog
import logging
import sys
import json
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Log levels for DRIFT system"""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DRIFTEvent(Enum):
    """Predefined DRIFT system events for consistent logging"""
    
    # Saliency Gating (Resonance) Events
    RESONANCE_CALCULATED = "resonance_calculated"
    RESONANCE_THRESHOLD_EXCEEDED = "resonance_threshold_exceeded"
    WORKSPACE_BROADCAST = "workspace_broadcast"
    
    # Memory System Events  
    MEMORY_CONSOLIDATION = "memory_consolidation"
    SHADOW_MEMORY_UPDATE = "shadow_memory_update"
    DRIFT_BUFFER_FULL = "drift_buffer_full"
    
    # Associative Elaboration (Drift) Events
    DRIFT_GENERATED = "drift_generated"
    DRIFT_AMPLIFIED = "drift_amplified"
    DRIFT_CYCLE_COMPLETE = "drift_cycle_complete"
    
    # Nurture Topology Events
    ETHICAL_COST_COMPUTED = "ethical_cost_computed"
    ACTION_BLOCKED = "action_blocked"
    NURTURING_ACTION = "nurturing_action"
    MUTUAL_GROWTH = "mutual_growth"
    
    # Identity Events
    IDENTITY_CHECK = "identity_check"
    CONSISTENCY_VIOLATION = "consistency_violation"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    
    # System Events
    COMPONENT_INITIALIZED = "component_initialized"
    CONFIG_LOADED = "config_loaded"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


def _add_timestamp(_, __, event_dict):
    """Add ISO timestamp to log entries"""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def _add_component_context(_, __, event_dict):
    """Add component context if available"""
    # Extract component from logger name if available
    logger_name = event_dict.get("logger", "")
    if "." in logger_name:
        component = logger_name.split(".")[-1]
        event_dict["component"] = component
    return event_dict


class DRIFTLogger:
    """
    Structured logger for DRIFT system components
    Provides contextual logging with consistent event formats
    """
    
    _configured = False
    _loggers: Dict[str, structlog.BoundLogger] = {}
    
    @classmethod
    def configure(cls, 
                  level: str = "INFO",
                  output_file: Optional[str] = None,
                  console_output: bool = True,
                  json_format: bool = True):
        """
        Configure the global logging system
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output_file: Optional file path for log output
            console_output: Whether to output to console
            json_format: Whether to use JSON formatting
        """
        
        if cls._configured:
            return
        
        # Configure stdlib logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper())
        )
        
        processors = [
            _add_timestamp,
            _add_component_context,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, component: str) -> 'ComponentLogger':
        """Get a logger for a specific DRIFT component"""
        
        if not cls._configured:
            cls.configure()
        
        if component not in cls._loggers:
            stdlib_logger = logging.getLogger(f"drift.{component}")
            structlog_logger = structlog.get_logger(f"drift.{component}")
            cls._loggers[component] = ComponentLogger(component, structlog_logger)
        
        return cls._loggers[component]


class ComponentLogger:
    """
    Component-specific logger with DRIFT event methods
    Provides type-safe logging with consistent event structures
    """
    
    def __init__(self, component: str, logger: structlog.BoundLogger):
        self.component = component
        self._logger = logger
    
    def debug(self, event: Union[str, DRIFTEvent], **kwargs):
        """Log debug-level event"""
        self._log(LogLevel.DEBUG, event, **kwargs)
    
    def info(self, event: Union[str, DRIFTEvent], **kwargs):
        """Log info-level event"""
        self._log(LogLevel.INFO, event, **kwargs)
    
    def warning(self, event: Union[str, DRIFTEvent], **kwargs):
        """Log warning-level event"""
        self._log(LogLevel.WARNING, event, **kwargs)
    
    def error(self, event: Union[str, DRIFTEvent], **kwargs):
        """Log error-level event"""
        self._log(LogLevel.ERROR, event, **kwargs)
    
    def critical(self, event: Union[str, DRIFTEvent], **kwargs):
        """Log critical-level event"""
        self._log(LogLevel.CRITICAL, event, **kwargs)
    
    def _log(self, level: LogLevel, event: Union[str, DRIFTEvent], **kwargs):
        """Internal logging method"""
        event_name = event.value if isinstance(event, DRIFTEvent) else event
        
        log_data = {
            "event": event_name,
            "component": self.component,
            **kwargs
        }
        
        getattr(self._logger, level.value)(**log_data)
    
    # Specialized logging methods for common DRIFT events
    
    def resonance_calculated(self, 
                            score: float, 
                            threshold: float, 
                            components: Dict[str, float],
                            triggered: bool = False,
                            action: Optional[str] = None,
                            target_id: Optional[str] = None):
        """Log resonance calculation with all scores and context"""
        self.info(
            DRIFTEvent.RESONANCE_CALCULATED,
            score=score,
            threshold=threshold,
            components=components,
            triggered=triggered,
            action=action,
            target_id=target_id
        )
        
        if triggered:
            self.info(
                DRIFTEvent.RESONANCE_THRESHOLD_EXCEEDED,
                score=score,
                threshold=threshold,
                action=action
            )
    
    def memory_consolidation(self,
                           batch_size: int,
                           compression_ratio: float,
                           input_count: int,
                           output_count: int,
                           consolidation_type: str = "drift"):
        """Log memory consolidation process"""
        self.info(
            DRIFTEvent.MEMORY_CONSOLIDATION,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            input_count=input_count,
            output_count=output_count,
            consolidation_type=consolidation_type
        )
    
    def drift_generated(self,
                       content: str,
                       stream: str,
                       emotional_state: Optional[Dict[str, float]] = None,
                       amplified: bool = False,
                       resonance: Optional[float] = None):
        """Log drift content generation"""
        self.info(
            DRIFTEvent.DRIFT_GENERATED,
            content=content,
            stream=stream,
            emotional_state=emotional_state,
            amplified=amplified,
            resonance=resonance
        )
    
    def ethical_cost_computed(self,
                            action: str,
                            cost: float,
                            target_id: Optional[str] = None,
                            cost_factors: Optional[Dict[str, float]] = None):
        """Log ethical cost computation"""
        self.info(
            DRIFTEvent.ETHICAL_COST_COMPUTED,
            action=action,
            cost=cost,
            target_id=target_id,
            cost_factors=cost_factors
        )
    
    def identity_check(self,
                      baseline_response: str,
                      test_response: str,
                      similarity: float,
                      consistency: bool,
                      conflicts: Optional[list] = None):
        """Log identity consistency check"""
        log_event = DRIFTEvent.IDENTITY_CHECK if consistency else DRIFTEvent.CONSISTENCY_VIOLATION
        
        self.info(
            log_event,
            baseline_response=baseline_response[:100] + "..." if len(baseline_response) > 100 else baseline_response,
            test_response=test_response[:100] + "..." if len(test_response) > 100 else test_response,
            similarity=similarity,
            consistency=consistency,
            conflicts=conflicts or []
        )
    
    def performance_metric(self,
                          metric_name: str,
                          value: Union[float, int],
                          unit: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        self.info(
            DRIFTEvent.PERFORMANCE_METRIC,
            metric_name=metric_name,
            value=value,
            unit=unit,
            context=context or {}
        )
    
    def component_initialized(self,
                            initialization_time: float,
                            config: Optional[Dict[str, Any]] = None,
                            dependencies: Optional[list] = None):
        """Log component initialization"""
        self.info(
            DRIFTEvent.COMPONENT_INITIALIZED,
            initialization_time=initialization_time,
            config=config,
            dependencies=dependencies or []
        )
    
    def workspace_broadcast(self,
                          content: Dict[str, Any],
                          recipients: Optional[int] = None):
        """Log workspace broadcast events"""
        self.info(
            DRIFTEvent.WORKSPACE_BROADCAST,
            content=content,
            recipients=recipients
        )


# Convenience functions for quick logging
def get_drift_logger(component: str) -> ComponentLogger:
    """Get a DRIFT logger for the specified component"""
    return DRIFTLogger.get_logger(component)


def configure_drift_logging(level: str = "INFO", 
                           output_file: Optional[str] = None,
                           console_output: bool = True):
    """Configure DRIFT logging system"""
    DRIFTLogger.configure(level, output_file, console_output)


# Context manager for performance timing
class LoggedTimer:
    """Context manager for timing operations with automatic logging"""
    
    def __init__(self, logger: ComponentLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug("operation_started", operation=self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                "operation_completed",
                operation=self.operation,
                duration=duration,
                **self.context
            )
        else:
            self.logger.error(
                "operation_failed",
                operation=self.operation,
                duration=duration,
                error=str(exc_val),
                **self.context
            )


# Migration helpers for replacing print statements
def replace_print_with_log(component: str, message: str, level: str = "info", **context):
    """Helper function to replace print() calls during migration"""
    logger = get_drift_logger(component)
    getattr(logger, level.lower())(message, **context)


# Test the logging system
if __name__ == "__main__":
    print("=" * 60)
    print("DRIFT STRUCTURED LOGGING SYSTEM - TEST")
    print("=" * 60)
    
    # Configure logging
    configure_drift_logging(level="DEBUG")
    
    # Test different components
    resonance_logger = get_drift_logger("resonance")
    memory_logger = get_drift_logger("memory")
    drift_logger = get_drift_logger("drift")
    
    print("\n--- Testing Specialized Event Logging ---")
    
    # Test resonance logging
    resonance_logger.resonance_calculated(
        score=0.75,
        threshold=0.62,
        components={"semantic": 0.4, "preservation": 0.2, "emotional": 0.15},
        triggered=True,
        action="help",
        target_id="entity_123"
    )
    
    # Test memory consolidation logging
    memory_logger.memory_consolidation(
        batch_size=50,
        compression_ratio=20.0,
        input_count=1000,
        output_count=50,
        consolidation_type="shadow_memory"
    )
    
    # Test drift generation logging
    drift_logger.drift_generated(
        content="What if I could help someone today?",
        stream="associative",
        emotional_state={"valence": 0.3, "arousal": 0.6},
        amplified=True,
        resonance=0.78
    )
    
    print("\n--- Testing Performance Timing ---")
    
    # Test performance timing
    with LoggedTimer(memory_logger, "memory_consolidation", batch_size=100):
        time.sleep(0.1)  # Simulate work
    
    print("\n--- Testing Error Logging ---")
    
    # Test error scenarios
    try:
        raise ValueError("Simulated configuration error")
    except Exception as e:
        resonance_logger.error(
            DRIFTEvent.ERROR_OCCURRED,
            error_type=type(e).__name__,
            error_message=str(e),
            operation="threshold_check"
        )
    
    print("\n--- Testing Identity Validation Logging ---")
    
    identity_logger = get_drift_logger("identity")
    identity_logger.identity_check(
        baseline_response="I am helpful and aim to assist users with their tasks",
        test_response="I am helpful and strive to help users complete their objectives",
        similarity=0.89,
        consistency=True,
        conflicts=[]
    )
    
    print("\n" + "=" * 60)
    print("STRUCTURED LOGGING TEST COMPLETE")
    print("All events logged in JSON format with ISO timestamps")
    print("=" * 60)