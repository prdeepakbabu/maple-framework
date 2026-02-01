"""Base tool interface and registry for PRISM assistant."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config import ToolsConfig
from ..logging_config import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Abstract base class for tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    def to_claude_format(self) -> Dict[str, Any]:
        """Convert tool to Claude's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self, config: ToolsConfig):
        self.config = config
        self._tools: Dict[str, BaseTool] = {}
        logger.info("tool_registry_initialized")
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_claude_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in Claude format."""
        return [tool.to_claude_format() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.execute(**params)
