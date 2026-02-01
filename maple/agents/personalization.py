"""Personalization agent for PRISM assistant."""

from typing import List, Optional

from ..config import PersonalizationConfig
from ..logging_config import get_logger, log_personalization
from ..models import Insight, PersonalizationContext, Turn, UserProfile
from .memory_agent import MemoryAgent

logger = get_logger(__name__)

# Approximate tokens per character
CHARS_PER_TOKEN = 4


class PersonalizationAgent:
    """Builds personalized context for LLM requests."""
    
    def __init__(self, config: PersonalizationConfig, memory: MemoryAgent):
        self.config = config
        self.memory = memory
        logger.info(
            "personalization_agent_initialized",
            context_budget=config.context_budget_tokens
        )
    
    async def build_context(
        self,
        user_id: str,
        current_message: str
    ) -> PersonalizationContext:
        """Build personalization context for a request."""
        logger.debug("building_personalization_context", user_id=user_id)
        
        # Gather data from memory
        profile = await self.memory.get_profile(user_id)
        insights = await self.memory.get_recent_insights(user_id, limit=100)  # No limit for demo
        recent_turns = await self.memory.get_recent_turns(user_id, limit=5)
        
        # Build system instructions within token budget
        system_instructions = self._build_instructions(
            profile=profile,
            insights=insights,
            recent_turns=recent_turns,
            current_message=current_message
        )
        
        # Estimate token count
        token_count = len(system_instructions) // CHARS_PER_TOKEN
        
        context = PersonalizationContext(
            user_id=user_id,
            profile=profile,
            recent_insights=insights,
            recent_turns=recent_turns,
            system_instructions=system_instructions,
            token_count=token_count
        )
        
        # Debug: log the actual instructions being sent
        logger.info(
            "personalization_instructions_preview",
            user_id=user_id,
            instructions_preview=system_instructions[:500] if system_instructions else "empty"
        )
        
        log_personalization(
            logger,
            user_id=user_id,
            has_profile=profile is not None,
            insights_count=len(insights),
            recent_turns_count=len(recent_turns),
            token_count=token_count
        )
        
        return context
    
    def _build_instructions(
        self,
        profile: Optional[UserProfile],
        insights: List[Insight],
        recent_turns: List[Turn],
        current_message: str
    ) -> str:
        """Build system instructions from user data."""
        sections = []
        budget_chars = self.config.context_budget_tokens * CHARS_PER_TOKEN
        used_chars = 0
        
        # Section 1: User Profile
        if profile:
            profile_section = self._format_profile(profile)
            if used_chars + len(profile_section) < budget_chars:
                sections.append(profile_section)
                used_chars += len(profile_section)
        
        # Section 2: Insights
        if insights:
            insights_section = self._format_insights(insights)
            if used_chars + len(insights_section) < budget_chars:
                sections.append(insights_section)
                used_chars += len(insights_section)
        
        # Section 3: Recent Context
        if recent_turns:
            recent_section = self._format_recent_context(recent_turns)
            remaining = budget_chars - used_chars
            if len(recent_section) > remaining:
                recent_section = recent_section[:remaining] + "..."
            if recent_section:
                sections.append(recent_section)
        
        if not sections:
            return ""
        
        return "\n\n".join([
            "## User Context",
            *sections
        ])
    
    def _format_profile(self, profile: UserProfile) -> str:
        """Format user profile for context."""
        lines = ["### User Profile"]
        
        attrs = profile.attributes
        prefs = profile.preferences
        
        if attrs.role:
            lines.append(f"- Role: {attrs.role}")
        if attrs.expertise_level:
            lines.append(f"- Expertise: {attrs.expertise_level}")
        if attrs.preferred_language:
            lines.append(f"- Preferred language: {attrs.preferred_language}")
        
        if prefs.response_style:
            lines.append(f"- Preferred style: {prefs.response_style}")
        if prefs.prefers_code_examples:
            lines.append("- Appreciates code examples")
        if prefs.verbosity:
            lines.append(f"- Verbosity preference: {prefs.verbosity}")
        
        if len(lines) == 1:
            return ""
        
        return "\n".join(lines)
    
    def _format_insights(self, insights: List[Insight]) -> str:
        """Format insights for context."""
        if not insights:
            return ""
        
        # Filter high-confidence insights
        high_confidence = [i for i in insights if i.confidence >= 0.6]
        
        if not high_confidence:
            return ""
        
        # Separate by type and sort by confidence (highest first) to prioritize important facts like name
        facts = sorted([i for i in high_confidence if i.type.value == "fact"], key=lambda x: -x.confidence)
        preferences = sorted([i for i in high_confidence if i.type.value == "preference"], key=lambda x: -x.confidence)
        behaviors = sorted([i for i in high_confidence if i.type.value == "behavior"], key=lambda x: -x.confidence)
        
        lines = ["### What You Know About This User"]
        lines.append("Use these facts and preferences when responding:")
        
        if facts:
            lines.append("\n**Facts:**")
            for insight in facts:  # No limit - show all
                lines.append(f"- {insight.content}")
        
        if preferences:
            lines.append("\n**Preferences:**")
            for insight in preferences:  # No limit - show all
                lines.append(f"- {insight.content}")
        
        if behaviors:
            lines.append("\n**Communication patterns:**")
            for insight in behaviors:  # No limit - show all
                lines.append(f"- {insight.content}")
        
        return "\n".join(lines)
    
    def _format_recent_context(self, turns: List[Turn]) -> str:
        """Format recent conversation context."""
        if not turns:
            return ""
        
        lines = ["### Recent Conversation Summary"]
        
        for turn in turns[:3]:  # Most recent 3
            topic = turn.user_message[:100]
            if len(turn.user_message) > 100:
                topic += "..."
            lines.append(f"- User asked about: {topic}")
        
        return "\n".join(lines)
