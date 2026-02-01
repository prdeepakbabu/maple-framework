"""
Data models for PRISM assistant.

This module defines all Pydantic models used throughout the system for
type safety, validation, and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class FeedbackType(str, Enum):
    """Type of user feedback on a response."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


class InsightType(str, Enum):
    """Type of learned insight about a user."""
    
    PREFERENCE = "preference"  # How user likes to receive information
    FACT = "fact"              # Factual info about user (role, expertise)
    BEHAVIOR = "behavior"      # Patterns in user interactions


class ToolName(str, Enum):
    """Available tool names."""
    
    SEARCH = "search"
    FILESYSTEM = "filesystem"
    CODE_EXECUTOR = "code_executor"


# =============================================================================
# User Models
# =============================================================================


class UserAttributes(BaseModel):
    """Static user attributes that rarely change."""
    
    role: Optional[str] = Field(
        default=None,
        description="User's role or job title"
    )
    expertise_level: Optional[str] = Field(
        default=None,
        description="User's expertise level: beginner, intermediate, senior"
    )
    preferred_language: Optional[str] = Field(
        default=None,
        description="User's preferred programming language"
    )
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom user attributes"
    )


class UserPreferences(BaseModel):
    """User interaction preferences."""
    
    response_style: Optional[str] = Field(
        default=None,
        description="Preferred response style: technical, simple, balanced"
    )
    prefers_code_examples: bool = Field(
        default=False,
        description="Whether user prefers code examples in responses"
    )
    verbosity: Optional[str] = Field(
        default=None,
        description="Preferred verbosity: concise, detailed"
    )
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom user preferences"
    )


class UserProfile(BaseModel):
    """Complete user profile combining attributes and preferences."""
    
    user_id: str = Field(description="Unique user identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    attributes: UserAttributes = Field(default_factory=UserAttributes)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserState(BaseModel):
    """Dynamic user state that changes frequently."""
    
    user_id: str = Field(description="Unique user identifier")
    current_session_id: Optional[str] = Field(
        default=None,
        description="Currently active session ID"
    )
    last_active: datetime = Field(default_factory=datetime.utcnow)
    recent_topics: List[str] = Field(
        default_factory=list,
        description="Recently discussed topics"
    )
    emotional_tone: Optional[str] = Field(
        default=None,
        description="Detected emotional tone of user"
    )


# =============================================================================
# Conversation Models
# =============================================================================


class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    
    tool_name: str = Field(description="Name of the tool invoked")
    input_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters passed to the tool"
    )
    output: Any = Field(
        default=None,
        description="Tool output/result"
    )
    latency_ms: float = Field(
        default=0.0,
        description="Tool execution time in milliseconds"
    )
    success: bool = Field(
        default=True,
        description="Whether tool execution succeeded"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if tool failed"
    )


class Turn(BaseModel):
    """Single conversation turn (user message + assistant response)."""
    
    turn_id: str = Field(description="Unique turn identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_message: str = Field(description="User's input message")
    assistant_response: str = Field(description="Assistant's response")
    tools_used: List[ToolCall] = Field(
        default_factory=list,
        description="Tools invoked during this turn"
    )
    feedback: FeedbackType = Field(
        default=FeedbackType.NONE,
        description="User feedback on this turn"
    )
    personalization_applied: bool = Field(
        default=False,
        description="Whether personalization was applied"
    )
    personalization_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Personalization context used (for debugging)"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Session(BaseModel):
    """Conversation session containing multiple turns."""
    
    session_id: str = Field(description="Unique session identifier")
    user_id: str = Field(description="User who owns this session")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = Field(
        default=None,
        description="When session ended (None if active)"
    )
    turns: List[Turn] = Field(
        default_factory=list,
        description="Turns in this session"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# Learning Models
# =============================================================================


class Insight(BaseModel):
    """A learned insight about a user extracted from interactions."""
    
    insight_id: str = Field(description="Unique insight identifier")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    type: InsightType = Field(description="Type of insight")
    content: str = Field(description="Human-readable insight content")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    source_turns: List[str] = Field(
        default_factory=list,
        description="Turn IDs that contributed to this insight"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserInsights(BaseModel):
    """Collection of insights for a specific user."""
    
    user_id: str = Field(description="User these insights belong to")
    insights: List[Insight] = Field(
        default_factory=list,
        description="List of learned insights"
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class FeatureFlags(BaseModel):
    """Runtime feature toggles for individual requests."""
    
    memory_enabled: bool = Field(
        default=True,
        description="Enable memory storage for this request"
    )
    learning_enabled: bool = Field(
        default=True,
        description="Enable learning for this request"
    )
    personalization_enabled: bool = Field(
        default=True,
        description="Enable personalization for this request"
    )


class ChatRequest(BaseModel):
    """Incoming chat request from UI."""
    
    user_id: str = Field(description="User identifier")
    message: str = Field(description="User's message")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID (auto-generated if not provided)"
    )
    flags: FeatureFlags = Field(
        default_factory=FeatureFlags,
        description="Feature toggles for this request"
    )


class ChatResponse(BaseModel):
    """Outgoing chat response to UI."""
    
    turn_id: str = Field(description="Unique turn identifier")
    session_id: str = Field(description="Session identifier")
    message: str = Field(description="Assistant's response message")
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools used"
    )
    personalization_applied: bool = Field(
        default=False,
        description="Whether response was personalized"
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total request latency in milliseconds"
    )


class FeedbackRequest(BaseModel):
    """Explicit feedback signal from user."""
    
    user_id: str = Field(description="User identifier")
    session_id: str = Field(description="Session identifier")
    turn_id: str = Field(description="Turn being rated")
    feedback: FeedbackType = Field(description="Feedback type")
    learning_enabled: bool = Field(
        default=True,
        description="Whether learning is enabled for this feedback"
    )


# =============================================================================
# Personalization Models
# =============================================================================


class PersonalizationContext(BaseModel):
    """Context assembled for personalizing a response."""
    
    user_id: str = Field(description="User identifier")
    profile: Optional[UserProfile] = Field(
        default=None,
        description="User's profile if available"
    )
    recent_insights: List[Insight] = Field(
        default_factory=list,
        description="Recent insights about the user"
    )
    recent_turns: List[Turn] = Field(
        default_factory=list,
        description="Recent conversation turns"
    )
    system_instructions: str = Field(
        default="",
        description="Generated system instructions for LLM"
    )
    token_count: int = Field(
        default=0,
        description="Estimated token count of context"
    )


# =============================================================================
# LLM Models
# =============================================================================


class Message(BaseModel):
    """A message in the conversation history."""
    
    role: Literal["user", "assistant", "system"] = Field(
        description="Message role"
    )
    content: str = Field(description="Message content")


class LLMResponse(BaseModel):
    """Response from the LLM."""
    
    content: str = Field(description="Text response content")
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls requested by LLM"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics"
    )
    latency_ms: float = Field(
        default=0.0,
        description="LLM call latency in milliseconds"
    )
