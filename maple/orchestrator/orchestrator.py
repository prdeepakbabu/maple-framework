"""Main orchestrator for PRISM assistant."""

import time
import uuid
from typing import Any, Dict, List, Optional

from ..agents.learning_agent import LearningAgent
from ..agents.memory_agent import MemoryAgent
from ..agents.personalization import PersonalizationAgent
from ..config import AppConfig
from ..llm.base import Message
from ..llm.bedrock import BedrockLLM
from ..logging_config import get_logger, log_request, log_response
from ..models import (
    ChatRequest, ChatResponse, FeedbackRequest, FeedbackType,
    Session, Turn, ToolCall
)
from ..tools.base import ToolRegistry
from ..tools.code_executor import CodeExecutorTool
from ..tools.filesystem import FilesystemTool
from ..tools.search import SearchTool
from ..tools.web_fetch import WebFetchTool

logger = get_logger(__name__)

# Base system prompt
SYSTEM_PROMPT_BASE = """You are PRISM, a helpful AI assistant.

You have access to tools for searching the web, reading/writing files, and executing Python code.

Be helpful, accurate, and concise. When using tools, explain what you're doing.
"""


class Orchestrator:
    """Central coordinator for all PRISM components."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize LLM
        self.llm = BedrockLLM(config.llm)
        
        # Initialize Memory
        self.memory = MemoryAgent(config.features.memory)
        
        # Initialize Learning
        self.learning = LearningAgent(
            config.features.learning,
            self.memory,
            self.llm
        )
        
        # Initialize Personalization
        self.personalization = PersonalizationAgent(
            config.features.personalization,
            self.memory
        )
        
        # Initialize Tools
        self.tools = ToolRegistry(config.tools)
        self._register_tools()
        
        # Session cache
        self._sessions: Dict[str, Session] = {}
        
        logger.info("orchestrator_initialized")
    
    def _register_tools(self) -> None:
        """Register all enabled tools."""
        if self.config.tools.search.enabled:
            self.tools.register(SearchTool(self.config.tools.search))
        
        if self.config.tools.filesystem.enabled:
            self.tools.register(FilesystemTool(self.config.tools.filesystem))
        
        if self.config.tools.code_executor.enabled:
            self.tools.register(CodeExecutorTool(self.config.tools.code_executor))
        
        if self.config.tools.web_fetch.enabled:
            self.tools.register(WebFetchTool(
                timeout=self.config.tools.web_fetch.timeout_seconds,
                max_content_length=self.config.tools.web_fetch.max_content_length,
                allowed_domains=self.config.tools.web_fetch.allowed_domains
            ))
        
        logger.info("tools_registered", count=len(self.tools.list_tools()))
    
    async def start(self) -> None:
        """Start background services."""
        await self.learning.start()
        logger.info("orchestrator_started")
    
    async def stop(self) -> None:
        """Stop background services."""
        await self.learning.stop()
        logger.info("orchestrator_stopped")
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request."""
        start_time = time.time()
        
        # Log request
        log_request(
            logger,
            user_id=request.user_id,
            session_id=request.session_id or "new",
            message=request.message,
            flags={
                "memory_enabled": request.flags.memory_enabled,
                "learning_enabled": request.flags.learning_enabled,
                "personalization_enabled": request.flags.personalization_enabled
            }
        )
        
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        session = await self._get_or_create_session(session_id, request.user_id)
        
        # Ensure user profile exists
        await self.memory.get_or_create_profile(request.user_id)
        
        # Build personalization context
        personalization_applied = False
        system_prompt = SYSTEM_PROMPT_BASE
        
        if request.flags.personalization_enabled:
            p_context = await self.personalization.build_context(
                request.user_id,
                request.message
            )
            if p_context.system_instructions:
                system_prompt = f"{SYSTEM_PROMPT_BASE}\n\n{p_context.system_instructions}"
                personalization_applied = True
        
        # Build conversation messages
        messages = self._build_messages(session, request.message)
        
        # Generate response with tools
        response_text, tool_calls = await self.llm.generate_with_tools(
            messages=messages,
            system_prompt=system_prompt,
            tools=self.tools.get_claude_tools(),
            tool_executor=self._execute_tool
        )
        
        # Create turn
        turn_id = str(uuid.uuid4())
        turn = Turn(
            turn_id=turn_id,
            user_message=request.message,
            assistant_response=response_text,
            tools_used=[
                ToolCall(
                    tool_name=tc.tool_name,
                    input_params=tc.input_params,
                    output=tc.output,
                    latency_ms=tc.latency_ms,
                    success=tc.success,
                    error=tc.error
                )
                for tc in tool_calls
            ],
            personalization_applied=personalization_applied
        )
        
        # Store turn in memory
        if request.flags.memory_enabled:
            session.turns.append(turn)
            await self.memory.save_session(session)
        
        # Queue learning
        if request.flags.learning_enabled:
            await self.learning.queue_learning(
                user_id=request.user_id,
                turn=turn,
                trigger="end_of_turn"
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        response = ChatResponse(
            turn_id=turn_id,
            session_id=session_id,
            message=response_text,
            tools_used=[tc.tool_name for tc in tool_calls],
            personalization_applied=personalization_applied,
            latency_ms=latency_ms
        )
        
        log_response(
            logger,
            user_id=request.user_id,
            turn_id=turn_id,
            tools_used=response.tools_used,
            personalization_applied=personalization_applied,
            latency_ms=latency_ms
        )
        
        return response
    
    async def feedback(self, request: FeedbackRequest) -> None:
        """Process user feedback."""
        logger.info(
            "feedback_received",
            user_id=request.user_id,
            turn_id=request.turn_id,
            feedback=request.feedback.value,
            learning_enabled=request.learning_enabled
        )
        
        # Update turn in memory
        await self.memory.update_turn_feedback(
            session_id=request.session_id,
            turn_id=request.turn_id,
            feedback=request.feedback
        )
        
        # Get the turn for learning (only if learning is enabled)
        if request.learning_enabled:
            session = await self.memory.get_session(request.session_id)
            if session:
                for turn in session.turns:
                    if turn.turn_id == request.turn_id:
                        turn.feedback = request.feedback
                        await self.learning.queue_learning(
                            user_id=request.user_id,
                            turn=turn,
                            trigger="explicit_feedback"
                        )
                        break
        else:
            logger.info("learning_skipped_on_feedback", reason="learning_disabled")
    
    async def _get_or_create_session(
        self,
        session_id: str,
        user_id: str
    ) -> Session:
        """Get existing session or create new one."""
        session = await self.memory.get_session(session_id)
        
        if session is None:
            session = Session(
                session_id=session_id,
                user_id=user_id
            )
            await self.memory.save_session(session)
            logger.info("session_created", session_id=session_id, user_id=user_id)
        
        return session
    
    def _build_messages(
        self,
        session: Session,
        current_message: str
    ) -> List[Message]:
        """Build message history for LLM."""
        messages = []
        
        # Include recent turns from session
        for turn in session.turns[-5:]:  # Last 5 turns
            messages.append(Message(role="user", content=turn.user_message))
            messages.append(Message(role="assistant", content=turn.assistant_response))
        
        # Add current message
        messages.append(Message(role="user", content=current_message))
        
        return messages
    
    async def _execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        logger.debug("executing_tool", tool_name=name, params=params)
        return await self.tools.execute(name, params)
