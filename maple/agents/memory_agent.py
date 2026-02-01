"""Memory agent for PRISM assistant."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import MemoryConfig
from ..logging_config import get_logger
from ..models import (
    FeedbackType, Insight, Session, Turn, UserInsights, UserProfile
)

logger = get_logger(__name__)


class MemoryAgent:
    """Handles persistent storage of user data, sessions, and insights."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.base_path = Path(config.storage_path)
        self._ensure_directories()
        logger.info("memory_agent_initialized", storage_path=config.storage_path)
    
    def _ensure_directories(self) -> None:
        """Create storage directories if needed."""
        (self.base_path / "users").mkdir(parents=True, exist_ok=True)
        (self.base_path / "episodic").mkdir(parents=True, exist_ok=True)
        (self.base_path / "semantic").mkdir(parents=True, exist_ok=True)
        logger.debug("memory_directories_created", base=str(self.base_path))
    
    # =========================================================================
    # User Profile Operations
    # =========================================================================
    
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile if exists."""
        path = self.base_path / "users" / f"{user_id}.json"
        
        if not path.exists():
            logger.debug("profile_not_found", user_id=user_id)
            return None
        
        try:
            data = json.loads(path.read_text())
            profile = UserProfile(**data)
            logger.debug("profile_loaded", user_id=user_id)
            return profile
        except Exception as e:
            logger.error("profile_load_error", user_id=user_id, error=str(e))
            return None
    
    async def save_profile(self, profile: UserProfile) -> None:
        """Save user profile."""
        profile.updated_at = datetime.utcnow()
        path = self.base_path / "users" / f"{profile.user_id}.json"
        
        try:
            path.write_text(profile.model_dump_json(indent=2))
            logger.info("profile_saved", user_id=profile.user_id)
        except Exception as e:
            logger.error("profile_save_error", user_id=profile.user_id, error=str(e))
            raise
    
    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        profile = await self.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            await self.save_profile(profile)
            logger.info("profile_created", user_id=user_id)
        return profile
    
    # =========================================================================
    # Session Operations
    # =========================================================================
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session if exists."""
        path = self.base_path / "episodic" / f"{session_id}.json"
        
        if not path.exists():
            logger.debug("session_not_found", session_id=session_id)
            return None
        
        try:
            data = json.loads(path.read_text())
            session = Session(**data)
            logger.debug("session_loaded", session_id=session_id, turns=len(session.turns))
            return session
        except Exception as e:
            logger.error("session_load_error", session_id=session_id, error=str(e))
            return None
    
    async def save_session(self, session: Session) -> None:
        """Save session."""
        path = self.base_path / "episodic" / f"{session.session_id}.json"
        
        try:
            path.write_text(session.model_dump_json(indent=2))
            logger.debug("session_saved", session_id=session.session_id, turns=len(session.turns))
        except Exception as e:
            logger.error("session_save_error", session_id=session.session_id, error=str(e))
            raise
    
    async def add_turn(self, session_id: str, turn: Turn) -> None:
        """Add turn to session."""
        session = await self.get_session(session_id)
        if session is None:
            logger.error("session_not_found_for_turn", session_id=session_id)
            raise ValueError(f"Session not found: {session_id}")
        
        session.turns.append(turn)
        await self.save_session(session)
        logger.info("turn_added", session_id=session_id, turn_id=turn.turn_id)
    
    async def update_turn_feedback(
        self,
        session_id: str,
        turn_id: str,
        feedback: FeedbackType
    ) -> None:
        """Update feedback for a specific turn."""
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        
        for turn in session.turns:
            if turn.turn_id == turn_id:
                turn.feedback = feedback
                await self.save_session(session)
                logger.info("feedback_updated", turn_id=turn_id, feedback=feedback.value)
                return
        
        raise ValueError(f"Turn not found: {turn_id}")
    
    async def get_recent_turns(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Turn]:
        """Get recent turns across all sessions for a user."""
        episodic_dir = self.base_path / "episodic"
        all_turns = []
        
        for session_file in episodic_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                session = Session(**data)
                if session.user_id == user_id:
                    all_turns.extend(session.turns)
            except Exception as e:
                logger.warning("session_parse_error", file=str(session_file), error=str(e))
        
        # Sort by timestamp and return most recent
        all_turns.sort(key=lambda t: t.timestamp, reverse=True)
        return all_turns[:limit]
    
    # =========================================================================
    # Insight Operations
    # =========================================================================
    
    async def get_insights(self, user_id: str) -> UserInsights:
        """Get insights for a user."""
        path = self.base_path / "semantic" / f"{user_id}.json"
        
        if not path.exists():
            return UserInsights(user_id=user_id, insights=[])
        
        try:
            data = json.loads(path.read_text())
            insights = UserInsights(**data)
            logger.debug("insights_loaded", user_id=user_id, count=len(insights.insights))
            return insights
        except Exception as e:
            logger.error("insights_load_error", user_id=user_id, error=str(e))
            return UserInsights(user_id=user_id, insights=[])
    
    async def save_insights(self, user_insights: UserInsights) -> None:
        """Save user insights."""
        path = self.base_path / "semantic" / f"{user_insights.user_id}.json"
        
        try:
            path.write_text(user_insights.model_dump_json(indent=2))
            logger.info("insights_saved", user_id=user_insights.user_id, count=len(user_insights.insights))
        except Exception as e:
            logger.error("insights_save_error", user_id=user_insights.user_id, error=str(e))
            raise
    
    async def add_insight(self, user_id: str, insight: Insight) -> None:
        """Add a new insight for a user."""
        user_insights = await self.get_insights(user_id)
        user_insights.insights.append(insight)
        await self.save_insights(user_insights)
        logger.info("insight_added", user_id=user_id, insight_type=insight.type.value)
    
    async def get_recent_insights(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Insight]:
        """Get most recent insights for a user."""
        user_insights = await self.get_insights(user_id)
        sorted_insights = sorted(
            user_insights.insights,
            key=lambda i: i.extracted_at,
            reverse=True
        )
        return sorted_insights[:limit]
