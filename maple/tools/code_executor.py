"""Python code executor tool for PRISM assistant."""

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict

from ..config import CodeExecutorConfig
from ..logging_config import get_logger
from .base import BaseTool

logger = get_logger(__name__)


class CodeExecutorTool(BaseTool):
    """Execute Python code in a sandboxed environment."""
    
    def __init__(self, config: CodeExecutorConfig):
        self.config = config
        logger.info("code_executor_initialized", timeout=config.timeout_seconds)
    
    @property
    def name(self) -> str:
        return "code_executor"
    
    @property
    def description(self) -> str:
        return "Execute Python code and return the output."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            }
        }
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code with timeout."""
        logger.debug("code_executor_running", code_length=len(code))
        
        try:
            result = await asyncio.wait_for(
                self._run_code(code),
                timeout=self.config.timeout_seconds
            )
            logger.info("code_executor_complete", success=result["success"])
            return result
            
        except asyncio.TimeoutError:
            logger.warning("code_executor_timeout", timeout=self.config.timeout_seconds)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.config.timeout_seconds}s",
                "return_value": None
            }
        except Exception as e:
            logger.error("code_executor_error", error=str(e))
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_value": None
            }
    
    async def _run_code(self, code: str) -> Dict[str, Any]:
        """Run code in isolated namespace."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Restricted globals
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "type": type,
            }
        }
        local_vars = {}
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)
            
            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "return_value": local_vars.get("result")
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": traceback.format_exc(),
                "return_value": None
            }
