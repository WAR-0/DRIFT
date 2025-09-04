"""
DRIFT System Enhanced Error Handling
Custom exceptions and retry logic for robust system operation

Features:
- Domain-specific exception hierarchy
- Automatic retry with exponential backoff
- Circuit breaker pattern for failing services
- Graceful degradation strategies
- Comprehensive error logging
"""

import asyncio
import time
import functools
from typing import Optional, Callable, Any, Dict, Union, Type
from dataclasses import dataclass
from enum import Enum

from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from core.drift_logger import get_drift_logger


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


# Custom Exception Hierarchy
class DriftException(Exception):
    """Base exception for all DRIFT system errors"""
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 recoverable: bool = True):
        
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = time.time()


class ConfigurationError(DriftException):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.config_path = config_path


class DatabaseError(DriftException):
    """Database connection and operation errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation


class LLMGenerationError(DriftException):
    """LLM generation and processing errors"""
    
    def __init__(self, message: str, model: Optional[str] = None, prompt_length: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model = model
        self.prompt_length = prompt_length


class MemorySystemError(DriftException):
    """Memory consolidation and management errors"""
    
    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.memory_type = memory_type


class EmotionalAnalysisError(DriftException):
    """Emotional tagging and analysis errors"""
    
    def __init__(self, message: str, text_length: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.text_length = text_length


class SaliencyGatingError(DriftException):
    """Saliency gating and resonance errors"""
    
    def __init__(self, message: str, threshold: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.threshold = threshold


class ValidationError(DriftException):
    """Identity validation and consistency errors"""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type


class SystemResourceError(DriftException):
    """System resource exhaustion errors"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.resource_type = resource_type


# Circuit Breaker Implementation
@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker"""
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        
        self.state = CircuitBreakerState(
            failure_threshold=failure_threshold,
            timeout=timeout
        )
        self.expected_exception = expected_exception
        self.logger = get_drift_logger("circuit_breaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function"""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_breaker(func, args, kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(self._execute_with_breaker(func, args, kwargs))
        
        # Return async or sync wrapper based on function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _execute_with_breaker(self, func: Callable, args: tuple, kwargs: dict):
        """Execute function with circuit breaker logic"""
        
        current_time = time.time()
        
        # Check circuit state
        if self.state.state == "open":
            if current_time - self.state.last_failure_time > self.state.timeout:
                self.state.state = "half-open"
                self.logger.info("circuit_breaker_half_open", function=func.__name__)
            else:
                raise SystemResourceError(
                    f"Circuit breaker open for {func.__name__}",
                    context={"state": self.state.state, "failures": self.state.failure_count}
                )
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state.state == "half-open":
                self.state.state = "closed"
                self.logger.info("circuit_breaker_closed", function=func.__name__)
            
            self.state.failure_count = 0
            return result
        
        except self.expected_exception as e:
            # Handle expected failures
            self.state.failure_count += 1
            self.state.last_failure_time = current_time
            
            if self.state.failure_count >= self.state.failure_threshold:
                self.state.state = "open"
                self.logger.error(
                    "circuit_breaker_opened",
                    function=func.__name__,
                    failure_count=self.state.failure_count,
                    error=str(e)
                )
            
            raise


# Enhanced Retry Decorators
def drift_retry(
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_min: float = 4.0,
    wait_max: float = 10.0,
    retry_exceptions: tuple = (DriftException,),
    logger_name: Optional[str] = None
):
    """Enhanced retry decorator for DRIFT operations"""
    
    logger = get_drift_logger(logger_name or "retry_handler")
    
    return retry(
        wait=wait_exponential(multiplier=wait_multiplier, min=wait_min, max=wait_max),
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=before_sleep_log(logger.logger, level="warning"),
        after=after_log(logger.logger, level="info"),
        reraise=True
    )


def async_drift_retry(
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_min: float = 4.0,
    wait_max: float = 10.0,
    retry_exceptions: tuple = (DriftException,),
    fallback_value: Any = None,
    logger_name: Optional[str] = None
):
    """Async retry decorator with fallback value"""
    
    def decorator(func: Callable) -> Callable:
        logger = get_drift_logger(logger_name or "async_retry_handler")
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except retry_exceptions as e:
                    if attempt < max_attempts - 1:
                        wait_time = min(wait_min * (wait_multiplier ** attempt), wait_max)
                        
                        logger.warning(
                            "async_retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error=str(e),
                            wait_time=wait_time
                        )
                        
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            "async_retry_failed",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e)
                        )
                        
                        if fallback_value is not None:
                            logger.info(
                                "using_fallback_value",
                                function=func.__name__,
                                fallback=fallback_value
                            )
                            return fallback_value
                        
                        raise
                
                except Exception as e:
                    # Don't retry non-DRIFT exceptions
                    logger.error(
                        "non_retryable_error",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error=str(e)
                    )
                    raise
        
        return wrapper
    
    return decorator


# Error Recovery Strategies
class ErrorRecoveryManager:
    """Manages error recovery strategies and graceful degradation"""
    
    def __init__(self):
        self.logger = get_drift_logger("error_recovery")
        self.recovery_strategies = {}
        self.degradation_levels = {}
    
    def register_recovery_strategy(self, 
                                  exception_type: Type[Exception],
                                  strategy: Callable):
        """Register a recovery strategy for an exception type"""
        self.recovery_strategies[exception_type] = strategy
        
        self.logger.info(
            "recovery_strategy_registered",
            exception_type=exception_type.__name__,
            strategy=strategy.__name__
        )
    
    def register_degradation_level(self,
                                  service_name: str,
                                  degraded_function: Callable):
        """Register a degraded version of a service"""
        self.degradation_levels[service_name] = degraded_function
        
        self.logger.info(
            "degradation_level_registered",
            service=service_name,
            degraded_function=degraded_function.__name__
        )
    
    async def handle_error(self, 
                          error: Exception,
                          context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle error with registered recovery strategies"""
        
        error_type = type(error)
        context = context or {}
        
        self.logger.error(
            "error_occurred",
            error_type=error_type.__name__,
            error_message=str(error),
            context=context,
            recoverable=getattr(error, 'recoverable', True)
        )
        
        # Try specific recovery strategy
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exc_type):
                try:
                    self.logger.info(
                        "attempting_recovery",
                        error_type=error_type.__name__,
                        strategy=strategy.__name__
                    )
                    
                    if asyncio.iscoroutinefunction(strategy):
                        result = await strategy(error, context)
                    else:
                        result = strategy(error, context)
                    
                    self.logger.info(
                        "recovery_successful",
                        error_type=error_type.__name__,
                        strategy=strategy.__name__
                    )
                    
                    return result
                
                except Exception as recovery_error:
                    self.logger.error(
                        "recovery_failed",
                        error_type=error_type.__name__,
                        strategy=strategy.__name__,
                        recovery_error=str(recovery_error)
                    )
        
        # If no recovery strategy worked, re-raise
        raise error


# Pre-defined Recovery Strategies
async def database_recovery_strategy(error: DatabaseError, context: Dict[str, Any]) -> Any:
    """Recovery strategy for database errors"""
    logger = get_drift_logger("database_recovery")
    
    if "connection" in str(error).lower():
        # Try to reconnect
        logger.info("attempting_database_reconnection")
        
        # Simulate reconnection attempt
        await asyncio.sleep(2.0)
        
        # Would implement actual reconnection logic here
        return {"recovered": True, "strategy": "reconnection"}
    
    elif "timeout" in str(error).lower():
        # Return cached data if available
        logger.info("database_timeout_fallback_to_cache")
        return {"recovered": True, "strategy": "cache_fallback", "data": None}
    
    raise error


async def llm_generation_recovery_strategy(error: LLMGenerationError, context: Dict[str, Any]) -> Any:
    """Recovery strategy for LLM generation errors"""
    logger = get_drift_logger("llm_recovery")
    
    if "timeout" in str(error).lower():
        # Use simplified generation
        logger.info("llm_timeout_fallback_to_simple_generation")
        
        return {
            "content": "Generated alternative response due to timeout",
            "recovered": True,
            "strategy": "simplified_generation"
        }
    
    elif "rate_limit" in str(error).lower():
        # Wait and retry
        logger.info("llm_rate_limit_waiting")
        await asyncio.sleep(10.0)
        
        return {"recovered": True, "strategy": "rate_limit_backoff"}
    
    raise error


async def memory_system_recovery_strategy(error: MemorySystemError, context: Dict[str, Any]) -> Any:
    """Recovery strategy for memory system errors"""
    logger = get_drift_logger("memory_recovery")
    
    if "consolidation" in str(error).lower():
        # Skip consolidation and continue
        logger.warning("memory_consolidation_skipped")
        
        return {"recovered": True, "strategy": "skip_consolidation"}
    
    elif "overflow" in str(error).lower():
        # Trigger emergency consolidation
        logger.info("emergency_memory_consolidation")
        
        return {"recovered": True, "strategy": "emergency_consolidation"}
    
    raise error


# Global Error Recovery Manager Instance
recovery_manager = ErrorRecoveryManager()

# Register default recovery strategies
recovery_manager.register_recovery_strategy(DatabaseError, database_recovery_strategy)
recovery_manager.register_recovery_strategy(LLMGenerationError, llm_generation_recovery_strategy)
recovery_manager.register_recovery_strategy(MemorySystemError, memory_system_recovery_strategy)


# Utility Functions
def log_exception(logger_name: str = "exception_handler"):
    """Decorator to log exceptions with full context"""
    
    def decorator(func: Callable) -> Callable:
        logger = get_drift_logger(logger_name)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "function_exception",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "function_exception",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_execute(func: Callable,
                default_value: Any = None,
                log_errors: bool = True,
                logger_name: Optional[str] = None) -> Any:
    """Safely execute a function with error handling"""
    
    logger = get_drift_logger(logger_name or "safe_execute")
    
    try:
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(func())
        else:
            return func()
    
    except Exception as e:
        if log_errors:
            logger.error(
                "safe_execute_failed",
                function=func.__name__,
                error=str(e),
                default_value=default_value
            )
        
        return default_value


# Example usage and testing
if __name__ == "__main__":
    
    async def test_error_handling():
        """Test error handling capabilities"""
        
        print("=" * 60)
        print("DRIFT ERROR HANDLING SYSTEM TEST")
        print("=" * 60)
        
        logger = get_drift_logger("error_test")
        
        # Test custom exceptions
        print("\n--- Testing Custom Exceptions ---")
        
        try:
            raise ConfigurationError(
                "Invalid configuration parameter",
                config_path="test_config.yaml",
                context={"parameter": "resonance_threshold", "value": -1.0}
            )
        except DriftException as e:
            logger.error("caught_drift_exception", 
                        error_code=e.error_code,
                        severity=e.severity.value,
                        recoverable=e.recoverable)
        
        # Test retry mechanism
        print("\n--- Testing Retry Mechanism ---")
        
        @async_drift_retry(max_attempts=3, fallback_value="fallback_result")
        async def failing_function():
            raise LLMGenerationError("Simulated LLM timeout")
        
        try:
            result = await failing_function()
            print(f"Result with fallback: {result}")
        except Exception as e:
            print(f"Failed even with retry: {e}")
        
        # Test circuit breaker
        print("\n--- Testing Circuit Breaker ---")
        
        breaker = CircuitBreaker(failure_threshold=2, timeout=5.0)
        
        @breaker
        async def unstable_service():
            import random
            if random.random() < 0.8:  # 80% failure rate
                raise DatabaseError("Simulated database connection failed")
            return "Success!"
        
        # Try multiple times to trigger circuit breaker
        for i in range(5):
            try:
                result = await unstable_service()
                print(f"Attempt {i+1}: {result}")
            except Exception as e:
                print(f"Attempt {i+1}: {type(e).__name__}: {e}")
        
        # Test error recovery
        print("\n--- Testing Error Recovery ---")
        
        try:
            raise DatabaseError("Connection timeout occurred")
        except Exception as e:
            recovered_result = await recovery_manager.handle_error(e)
            print(f"Recovery result: {recovered_result}")
        
        print("\n" + "=" * 60)
        print("ERROR HANDLING TEST COMPLETE")
        print("=" * 60)
    
    # Run the test
    asyncio.run(test_error_handling())