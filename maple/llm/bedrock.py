"""AWS Bedrock LLM implementation for PRISM assistant."""

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

from ..config import LLMConfig
from ..logging_config import get_logger, log_llm_call
from ..models import ToolCall
from .base import BaseLLM, LLMResponse, Message

logger = get_logger(__name__)


class BedrockLLM(BaseLLM):
    """AWS Bedrock implementation using Claude."""
    
    # Refresh credentials every 30 minutes to pick up external updates
    CREDENTIAL_REFRESH_INTERVAL = 30 * 60  # seconds
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._client_created_at = 0
        self._create_client()
    
    def _create_client(self):
        """Create or refresh the boto3 client to pick up new credentials."""
        # Create NEW session to force re-reading credentials file
        # (boto3.client() uses cached default session which doesn't refresh)
        session = boto3.Session()
        self._client = session.client(
            "bedrock-runtime",
            region_name=self.config.region,
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"})
        )
        self._client_created_at = time.time()
        logger.info(
            "bedrock_client_created",
            model_id=self.config.model_id,
            region=self.config.region
        )
    
    @property
    def client(self):
        """Get the boto3 client, refreshing if credentials are stale."""
        age = time.time() - self._client_created_at
        if age > self.CREDENTIAL_REFRESH_INTERVAL:
            logger.info(
                "bedrock_credentials_refresh",
                age_seconds=int(age),
                interval_seconds=self.CREDENTIAL_REFRESH_INTERVAL
            )
            self._create_client()
        return self._client
    
    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate a response using Claude via Bedrock."""
        start_time = time.time()
        
        # Format messages for Claude
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        
        # Build request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        if tools:
            body["tools"] = tools
        
        logger.debug(
            "bedrock_request",
            message_count=len(messages),
            has_system=bool(system_prompt),
            tools_count=len(tools) if tools else 0
        )
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body)
            )
            
            result = json.loads(response["body"].read())
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content and tool calls
            content = ""
            tool_calls = []
            
            for block in result.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {})
                    })
            
            usage = result.get("usage", {})
            
            log_llm_call(
                logger,
                model_id=self.config.model_id,
                message_count=len(messages),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=latency_ms,
                has_tool_calls=bool(tool_calls)
            )
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error("bedrock_error", error=str(e))
            raise
    
    async def generate_with_tools(
        self,
        messages: List[Message],
        system_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable,
        max_iterations: int = 10
    ) -> Tuple[str, List[ToolCall]]:
        """Generate response with agentic tool use loop."""
        all_tool_calls = []
        current_messages = list(messages)
        
        for iteration in range(max_iterations):
            logger.debug("tool_loop_iteration", iteration=iteration)
            
            response = await self.generate(
                messages=current_messages,
                system_prompt=system_prompt,
                tools=tools,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # If no tool calls, return the final response
            if not response.tool_calls:
                logger.debug("tool_loop_complete", iterations=iteration + 1)
                return response.content, all_tool_calls
            
            # Add assistant message with tool calls
            assistant_content = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"]
                })
            
            current_messages.append(Message(
                role="assistant",
                content=json.dumps(assistant_content)
            ))
            
            # Execute tools and collect results
            tool_results = []
            for tc in response.tool_calls:
                start_time = time.time()
                try:
                    result = await tool_executor(tc["name"], tc["input"])
                    latency_ms = (time.time() - start_time) * 1000
                    
                    tool_call = ToolCall(
                        tool_name=tc["name"],
                        input_params=tc["input"],
                        output=result,
                        latency_ms=latency_ms,
                        success=True
                    )
                    all_tool_calls.append(tool_call)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": str(result)
                    })
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    tool_call = ToolCall(
                        tool_name=tc["name"],
                        input_params=tc["input"],
                        output=None,
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
                    all_tool_calls.append(tool_call)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })
            
            # Add tool results as user message
            current_messages.append(Message(
                role="user",
                content=json.dumps(tool_results)
            ))
        
        logger.warning("tool_loop_max_iterations", max=max_iterations)
        return response.content, all_tool_calls
