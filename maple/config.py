"""
Configuration management for PRISM assistant.

This module provides typed configuration classes that can be loaded from
YAML files or instantiated with defaults.
"""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


# =============================================================================
# Feature Configuration
# =============================================================================


class MemoryConfig(BaseModel):
    """Configuration for the Memory component."""
    
    enabled: bool = Field(
        default=True,
        description="Enable memory storage"
    )
    storage_path: str = Field(
        default="./storage/memory",
        description="Path to memory storage directory"
    )
    retention_days: int = Field(
        default=-1,
        description="Days to retain data (-1 = indefinite)"
    )


class LearningConfig(BaseModel):
    """Configuration for the Learning component."""
    
    enabled: bool = Field(
        default=True,
        description="Enable learning agent"
    )
    async_mode: bool = Field(
        default=True,
        description="Run learning in background"
    )
    triggers: List[str] = Field(
        default=["end_of_turn", "explicit_feedback"],
        description="Events that trigger learning"
    )


class PersonalizationConfig(BaseModel):
    """Configuration for the Personalization component."""
    
    enabled: bool = Field(
        default=True,
        description="Enable personalization"
    )
    context_budget_tokens: int = Field(
        default=2000,
        description="Max tokens for personalization context"
    )


class FeaturesConfig(BaseModel):
    """Configuration for all feature flags."""
    
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    personalization: PersonalizationConfig = Field(
        default_factory=PersonalizationConfig
    )


# =============================================================================
# LLM Configuration
# =============================================================================


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""
    
    provider: str = Field(
        default="bedrock",
        description="LLM provider (bedrock)"
    )
    model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Model identifier"
    )
    region: str = Field(
        default="us-west-2",
        description="AWS region for Bedrock"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )


# =============================================================================
# Tool Configuration
# =============================================================================


class SearchToolConfig(BaseModel):
    """Configuration for the search tool."""
    
    enabled: bool = Field(default=True, description="Enable search tool")
    max_results: int = Field(
        default=5,
        description="Maximum search results to return"
    )


class FilesystemToolConfig(BaseModel):
    """Configuration for the filesystem tool."""
    
    enabled: bool = Field(default=True, description="Enable filesystem tool")
    allowed_paths: List[str] = Field(
        default=["./workspace", "./storage"],
        description="Allowed directories for file operations"
    )


class CodeExecutorConfig(BaseModel):
    """Configuration for the code executor tool."""
    
    enabled: bool = Field(default=True, description="Enable code executor")
    timeout_seconds: int = Field(
        default=30,
        description="Execution timeout in seconds"
    )


class WebFetchConfig(BaseModel):
    """Configuration for the web fetch tool."""
    
    enabled: bool = Field(default=True, description="Enable web fetch tool")
    timeout_seconds: int = Field(
        default=15,
        description="HTTP request timeout in seconds"
    )
    max_content_length: int = Field(
        default=10000,
        description="Maximum content length to return (characters)"
    )
    allowed_domains: List[str] = Field(
        default=[],
        description="Allowed domains (empty = all domains)"
    )


class ToolsConfig(BaseModel):
    """Configuration for all tools."""
    
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    filesystem: FilesystemToolConfig = Field(
        default_factory=FilesystemToolConfig
    )
    code_executor: CodeExecutorConfig = Field(
        default_factory=CodeExecutorConfig
    )
    web_fetch: WebFetchConfig = Field(
        default_factory=WebFetchConfig
    )


# =============================================================================
# Logging Configuration
# =============================================================================


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)"
    )
    format: str = Field(
        default="structured",
        description="Log format (structured or plain)"
    )
    include_tool_results: bool = Field(
        default=True,
        description="Include tool outputs in logs"
    )


# =============================================================================
# Main Configuration
# =============================================================================


class AppConfig(BaseModel):
    """Complete application configuration."""
    
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            AppConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)
    
    @classmethod
    def default(cls) -> "AppConfig":
        """
        Create a default configuration.
        
        Returns:
            AppConfig instance with default values
        """
        return cls()
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save the configuration
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False
            )
