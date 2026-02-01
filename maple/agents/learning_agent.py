"""Learning agent for PRISM assistant."""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional

from ..config import LearningConfig
from ..logging_config import get_logger, log_learning_event
from ..models import FeedbackType, Insight, InsightType, Turn
from .memory_agent import MemoryAgent

logger = get_logger(__name__)

# Prompt for insight extraction
LEARNING_PROMPT = """Analyze this conversation turn and extract insights about the user.

User message: {user_message}
Assistant response: {assistant_response}
User feedback: {feedback}

Extract ANY useful insights about the user, such as:
- PREFERENCE: How they want information presented (with citations, concise, detailed, code examples, etc.)
- FACT: Information about them (role, expertise, company, domain interests)
- BEHAVIOR: Communication patterns (asks follow-ups, prefers examples, etc.)

IMPORTANT: If the user explicitly states a preference or request about how they want information, ALWAYS capture it.

Return insights as JSON array (return at least 1 insight if user message contains any useful information):
[
  {{"type": "preference|fact|behavior", "content": "detailed insight text", "confidence": 0.0-1.0}}
]

Example - If user says "I want URLs for sources", return:
[{{"type": "preference", "content": "User wants citation URLs and source links when facts are presented", "confidence": 0.95}}]

Return [] only if the message is a simple greeting with no information.
"""


class LearningAgent:
    """Async learning from user interactions."""
    
    def __init__(self, config: LearningConfig, memory: MemoryAgent, llm=None):
        self.config = config
        self.memory = memory
        self.llm = llm
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        logger.info(
            "learning_agent_initialized",
            async_mode=config.async_mode,
            triggers=config.triggers
        )
    
    async def start(self) -> None:
        """Start the background learning worker."""
        if self.config.async_mode and self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("learning_worker_started")
    
    async def stop(self) -> None:
        """Stop the background worker, waiting for pending tasks to complete."""
        if self._worker_task:
            # Wait for queue to drain before stopping
            if not self._queue.empty():
                logger.info("learning_worker_draining_queue", queue_size=self._queue.qsize())
                try:
                    # Wait up to 30 seconds for queue to drain
                    await asyncio.wait_for(self._queue.join(), timeout=30.0)
                    logger.info("learning_queue_drained")
                except asyncio.TimeoutError:
                    logger.warning("learning_queue_drain_timeout", remaining=self._queue.qsize())
            
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("learning_worker_stopped")
    
    async def _worker(self) -> None:
        """Background worker that processes learning tasks."""
        logger.info("learning_worker_running_and_waiting")
        
        while True:
            try:
                logger.info("learning_worker_waiting_for_task", queue_size=self._queue.qsize())
                task = await self._queue.get()
                user_id = task["user_id"]
                turn = task["turn"]
                trigger = task["trigger"]
                
                logger.info(
                    "learning_task_processing",
                    user_id=user_id,
                    turn_id=turn.turn_id,
                    trigger=trigger,
                    user_message_preview=turn.user_message[:50]
                )
                
                insights = await self._extract_insights(turn)
                
                logger.info(
                    "learning_insights_extracted",
                    user_id=user_id,
                    insights_count=len(insights)
                )
                
                for insight in insights:
                    await self.memory.add_insight(user_id, insight)
                    logger.info(
                        "insight_saved",
                        user_id=user_id,
                        insight_type=insight.type.value,
                        insight_content=insight.content[:50]
                    )
                
                log_learning_event(
                    logger,
                    user_id=user_id,
                    trigger=trigger,
                    insights_count=len(insights),
                    turn_id=turn.turn_id
                )
                
                self._queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("learning_worker_cancelled")
                break
            except Exception as e:
                logger.error("learning_worker_error", error=str(e), error_type=type(e).__name__)
                import traceback
                logger.error("learning_worker_traceback", traceback=traceback.format_exc())
    
    async def queue_learning(
        self,
        user_id: str,
        turn: Turn,
        trigger: str
    ) -> None:
        """Queue a turn for learning."""
        logger.info(
            "learning_queue_called",
            user_id=user_id,
            turn_id=turn.turn_id,
            trigger=trigger,
            configured_triggers=self.config.triggers
        )
        
        if trigger not in self.config.triggers:
            logger.info("learning_trigger_skipped", trigger=trigger, expected=self.config.triggers)
            return
        
        if self.config.async_mode:
            await self._queue.put({
                "user_id": user_id,
                "turn": turn,
                "trigger": trigger
            })
            logger.info(
                "learning_task_queued",
                user_id=user_id,
                turn_id=turn.turn_id,
                queue_size=self._queue.qsize()
            )
        else:
            # Synchronous learning
            insights = await self._extract_insights(turn)
            for insight in insights:
                await self.memory.add_insight(user_id, insight)
            
            log_learning_event(
                logger,
                user_id=user_id,
                trigger=trigger,
                insights_count=len(insights),
                turn_id=turn.turn_id
            )
    
    async def _extract_insights(self, turn: Turn) -> List[Insight]:
        """Extract insights from a turn using LLM."""
        if not self.llm:
            logger.info("llm_not_available_using_rules")
            return await self._rule_based_extraction(turn)
        
        try:
            logger.info(
                "llm_extraction_starting",
                turn_id=turn.turn_id,
                user_message_preview=turn.user_message[:100]
            )
            
            prompt = LEARNING_PROMPT.format(
                user_message=turn.user_message,
                assistant_response=turn.assistant_response[:500],
                feedback=turn.feedback.value
            )
            
            from ..llm.base import Message
            response = await self.llm.generate(
                messages=[Message(role="user", content=prompt)],
                max_tokens=500,
                temperature=0.3
            )
            
            logger.info(
                "llm_extraction_response",
                turn_id=turn.turn_id,
                response_preview=response.content[:200] if response.content else "empty"
            )
            
            import json
            import re
            
            # Try to extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            raw_insights = json.loads(content)
            
            insights = []
            for raw in raw_insights:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    type=InsightType(raw["type"]),
                    content=raw["content"],
                    confidence=raw.get("confidence", 0.7),
                    source_turns=[turn.turn_id]
                )
                insights.append(insight)
            
            logger.info(
                "llm_extraction_complete",
                turn_id=turn.turn_id,
                insights_count=len(insights),
                insights_preview=[i.content[:50] for i in insights]
            )
            
            return insights
            
        except json.JSONDecodeError as e:
            logger.warning(
                "llm_extraction_json_error",
                error=str(e),
                response_content=response.content[:300] if response else "no response"
            )
            return await self._rule_based_extraction(turn)
        except Exception as e:
            logger.warning("llm_extraction_failed", error=str(e), error_type=type(e).__name__)
            return await self._rule_based_extraction(turn)
    
    async def _rule_based_extraction(self, turn: Turn) -> List[Insight]:
        """Simple rule-based insight extraction as fallback."""
        insights = []
        msg = turn.user_message.lower()
        
        # Detect expertise level
        if any(w in msg for w in ["beginner", "new to", "just started", "learning"]):
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.FACT,
                content="User appears to be a beginner in this topic",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        elif any(w in msg for w in ["advanced", "expert", "years of experience", "scientist", "engineer", "developer"]):
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.FACT,
                content="User appears to have advanced expertise",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        
        # Detect role/profession
        role_keywords = {
            "scientist": "User is a scientist",
            "engineer": "User is an engineer", 
            "developer": "User is a developer",
            "manager": "User is a manager",
            "researcher": "User is a researcher",
            "student": "User is a student",
        }
        for keyword, content in role_keywords.items():
            if keyword in msg:
                insights.append(Insight(
                    insight_id=str(uuid.uuid4()),
                    type=InsightType.FACT,
                    content=content,
                    confidence=0.8,
                    source_turns=[turn.turn_id]
                ))
                break
        
        # Detect preferences from feedback
        if turn.feedback == FeedbackType.NEGATIVE:
            if len(turn.assistant_response) > 1000:
                insights.append(Insight(
                    insight_id=str(uuid.uuid4()),
                    type=InsightType.PREFERENCE,
                    content="User may prefer more concise responses",
                    confidence=0.6,
                    source_turns=[turn.turn_id]
                ))
        elif turn.feedback == FeedbackType.POSITIVE:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.BEHAVIOR,
                content=f"User gave positive feedback on response style",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        
        # Detect code preference
        if "code" in msg or "example" in msg or "show me" in msg:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.PREFERENCE,
                content="User appreciates code examples",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        
        # Detect search/web preference
        if "search" in msg or "find" in msg or "look up" in msg:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.PREFERENCE,
                content="User values web search for current information",
                confidence=0.6,
                source_turns=[turn.turn_id]
            ))
        
        # Detect technical vs simple preference
        if any(w in msg for w in ["technical", "detailed", "in-depth", "explain how"]):
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.PREFERENCE,
                content="User prefers technical, detailed explanations",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        elif any(w in msg for w in ["simple", "briefly", "quick", "summary", "tldr"]):
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                type=InsightType.PREFERENCE,
                content="User prefers concise, simple explanations",
                confidence=0.7,
                source_turns=[turn.turn_id]
            ))
        
        # Log extraction result
        logger.debug(
            "rule_based_extraction_complete",
            turn_id=turn.turn_id,
            insights_extracted=len(insights)
        )
        
        return insights
