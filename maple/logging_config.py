"""
Logging configuration for PRISM assistant.

This module sets up structured logging using structlog with comprehensive
output formatting for observability.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        config: Logging configuration
    """
    # Set log level
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Configure shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if config.format == "structured":
        # JSON output for production/parsing
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging to also use structured format
        logging.basicConfig(
            format="%(message)s",
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        # Rich console output for development
        console = Console()
        
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True)
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging with Rich handler
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )
    
    # Quiet noisy libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Bound structlog logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


class LogContext:
    """Context manager for adding temporary log context."""
    
    def __init__(self, **kwargs: Any):
        """
        Initialize with context variables.
        
        Args:
            **kwargs: Key-value pairs to add to log context
        """
        self._context = kwargs
        self._tokens = []
    
    def __enter__(self) -> "LogContext":
        """Enter context and bind variables."""
        for key, value in self._context.items():
            token = structlog.contextvars.bind_contextvars(**{key: value})
            self._tokens.append((key, token))
        return self
    
    def __exit__(self, *args) -> None:
        """Exit context and unbind variables."""
        structlog.contextvars.unbind_contextvars(
            *[key for key, _ in self._tokens]
        )


def log_tool_call(
    logger: structlog.BoundLogger,
    tool_name: str,
    params: Dict[str, Any],
    result: Any,
    latency_ms: float,
    success: bool,
    include_result: bool = True
) -> None:
    """
    Log a tool call with standardized format.
    
    Args:
        logger: Logger instance
        tool_name: Name of the tool
        params: Input parameters
        result: Tool result
        latency_ms: Execution time in milliseconds
        success: Whether execution succeeded
        include_result: Whether to include result in log
    """
    log_data = {
        "tool_name": tool_name,
        "params": params,
        "latency_ms": round(latency_ms, 2),
        "success": success
    }
    
    if include_result:
        # Truncate large results
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "...[truncated]"
        log_data["result"] = result_str
    
    if success:
        logger.info("tool_executed", **log_data)
    else:
        logger.error("tool_failed", **log_data)


def log_llm_call(
    logger: structlog.BoundLogger,
    model_id: str,
    message_count: int,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    has_tool_calls: bool
) -> None:
    """
    Log an LLM call with standardized format.
    
    Args:
        logger: Logger instance
        model_id: Model identifier
        message_count: Number of messages in request
        input_tokens: Input token count
        output_tokens: Output token count
        latency_ms: Execution time in milliseconds
        has_tool_calls: Whether response contains tool calls
    """
    logger.info(
        "llm_call_complete",
        model_id=model_id,
        message_count=message_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=round(latency_ms, 2),
        has_tool_calls=has_tool_calls
    )


def log_request(
    logger: structlog.BoundLogger,
    user_id: str,
    session_id: str,
    message: str,
    flags: Dict[str, bool]
) -> None:
    """
    Log an incoming chat request.
    
    Args:
        logger: Logger instance
        user_id: User identifier
        session_id: Session identifier
        message: User message (truncated)
        flags: Feature flags
    """
    # Truncate long messages
    message_preview = message[:100] + "..." if len(message) > 100 else message
    
    logger.info(
        "chat_request_received",
        user_id=user_id,
        session_id=session_id,
        message_preview=message_preview,
        message_length=len(message),
        memory_enabled=flags.get("memory_enabled", True),
        learning_enabled=flags.get("learning_enabled", True),
        personalization_enabled=flags.get("personalization_enabled", True)
    )


def log_response(
    logger: structlog.BoundLogger,
    user_id: str,
    turn_id: str,
    tools_used: list,
    personalization_applied: bool,
    latency_ms: float
) -> None:
    """
    Log an outgoing chat response.
    
    Args:
        logger: Logger instance
        user_id: User identifier
        turn_id: Turn identifier
        tools_used: List of tool names used
        personalization_applied: Whether personalization was applied
        latency_ms: Total request latency
    """
    logger.info(
        "chat_response_sent",
        user_id=user_id,
        turn_id=turn_id,
        tools_used=tools_used,
        tools_count=len(tools_used),
        personalization_applied=personalization_applied,
        latency_ms=round(latency_ms, 2)
    )


def log_learning_event(
    logger: structlog.BoundLogger,
    user_id: str,
    trigger: str,
    insights_count: int,
    turn_id: str = None
) -> None:
    """
    Log a learning event.
    
    Args:
        logger: Logger instance
        user_id: User identifier
        trigger: What triggered learning
        insights_count: Number of insights extracted
        turn_id: Associated turn ID
    """
    logger.info(
        "learning_complete",
        user_id=user_id,
        trigger=trigger,
        insights_count=insights_count,
        turn_id=turn_id
    )


def log_personalization(
    logger: structlog.BoundLogger,
    user_id: str,
    has_profile: bool,
    insights_count: int,
    recent_turns_count: int,
    token_count: int
) -> None:
    """
    Log personalization context building.
    
    Args:
        logger: Logger instance
        user_id: User identifier
        has_profile: Whether user has a profile
        insights_count: Number of insights included
        recent_turns_count: Number of recent turns included
        token_count: Estimated token count
    """
    logger.info(
        "personalization_context_built",
        user_id=user_id,
        has_profile=has_profile,
        insights_count=insights_count,
        recent_turns_count=recent_turns_count,
        token_count=token_count
    )
