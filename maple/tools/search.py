"""DuckDuckGo search tool for PRISM assistant."""

from typing import Any, Dict, List

from ..config import SearchToolConfig
from ..logging_config import get_logger, log_tool_call
from .base import BaseTool

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Web search using DuckDuckGo."""
    
    def __init__(self, config: SearchToolConfig):
        self.config = config
        # Use the new ddgs package
        try:
            from ddgs import DDGS
            self._ddgs = DDGS()
        except ImportError:
            # Fall back to old package name
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        logger.info("search_tool_initialized", max_results=config.max_results)
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information using DuckDuckGo."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        }
    
    async def execute(self, query: str) -> List[Dict[str, str]]:
        """Execute a web search."""
        logger.debug("search_executing", query=query)
        
        try:
            results = list(self._ddgs.text(
                query,
                max_results=self.config.max_results
            ))
            
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
            
            logger.info("search_complete", query=query, results_count=len(formatted))
            return formatted
            
        except Exception as e:
            logger.error("search_error", query=query, error=str(e))
            raise
