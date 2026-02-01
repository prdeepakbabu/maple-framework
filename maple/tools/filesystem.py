"""Filesystem tool for PRISM assistant."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from ..config import FilesystemToolConfig
from ..logging_config import get_logger
from .base import BaseTool

logger = get_logger(__name__)


class FilesystemTool(BaseTool):
    """Filesystem operations with path restrictions."""
    
    def __init__(self, config: FilesystemToolConfig):
        self.config = config
        self._allowed_paths = [Path(p).resolve() for p in config.allowed_paths]
        logger.info("filesystem_tool_initialized", allowed_paths=config.allowed_paths)
    
    @property
    def name(self) -> str:
        return "filesystem"
    
    @property
    def description(self) -> str:
        return "Read, write, and list files in allowed directories."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list"],
                "description": "The filesystem action to perform"
            },
            "path": {
                "type": "string",
                "description": "The file or directory path"
            },
            "content": {
                "type": "string",
                "description": "Content to write (for write action)"
            }
        }
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories."""
        resolved = path.resolve()
        return any(
            resolved == allowed or allowed in resolved.parents
            for allowed in self._allowed_paths
        )
    
    async def execute(
        self,
        action: str,
        path: str,
        content: str = None
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Execute a filesystem operation."""
        target = Path(path)
        
        if not self._is_path_allowed(target):
            error = f"Path not allowed: {path}"
            logger.warning("filesystem_path_denied", path=path)
            raise PermissionError(error)
        
        logger.debug("filesystem_executing", action=action, path=path)
        
        try:
            if action == "read":
                return await self._read(target)
            elif action == "write":
                return await self._write(target, content or "")
            elif action == "list":
                return await self._list(target)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            logger.error("filesystem_error", action=action, path=path, error=str(e))
            raise
    
    async def _read(self, path: Path) -> str:
        """Read file contents."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = path.read_text(encoding="utf-8")
        logger.info("filesystem_read", path=str(path), size=len(content))
        return content
    
    async def _write(self, path: Path, content: str) -> Dict[str, Any]:
        """Write content to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        logger.info("filesystem_write", path=str(path), size=len(content))
        return {"success": True, "path": str(path), "bytes_written": len(content)}
    
    async def _list(self, path: Path) -> List[str]:
        """List directory contents."""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        entries = [
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in sorted(path.iterdir())
        ]
        
        logger.info("filesystem_list", path=str(path), entries_count=len(entries))
        return entries
