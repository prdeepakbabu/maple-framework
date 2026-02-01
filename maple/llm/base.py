"""Base LLM interface for PRISM assistant."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..models import ToolCall


class Message(BaseModel):
    """A message in the conversation."""
    role: str = Field(description="Message role: user, assistant, system")
    content: str = Field(description="Message content")


class LLMResponse(BaseModel):
    """Response from LLM."""
    content: str = Field(default="", description="Text content")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self,
        messages: List[Message],
        system_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable,
        max_iterations: int = 10
    ) -> Tuple[str, List[ToolCall]]:
        """Generate response with tool use loop."""
        pass
