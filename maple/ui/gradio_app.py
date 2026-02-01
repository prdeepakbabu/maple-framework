"""Gradio web interface for PRISM assistant."""

import asyncio
from typing import List, Optional, Tuple

import gradio as gr

from ..logging_config import get_logger
from ..models import ChatRequest, FeedbackRequest, FeedbackType, FeatureFlags
from ..orchestrator import Orchestrator

logger = get_logger(__name__)

# Professional CSS styling
CUSTOM_CSS = """
/* Import professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: #f8fafc !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.main-header h1 {
    margin: 0 0 0.5rem 0;
    font-size: 1.875rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.main-header p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* All buttons - professional styling */
button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

/* Primary send button */
.primary-btn {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.primary-btn:hover {
    background: #1e293b !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-1px);
}

/* Secondary buttons */
button.secondary {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
}

button.secondary:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
}

/* Feedback buttons */
.feedback-btn {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
}

.feedback-btn:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #334155 !important;
}

/* Input styling */
textarea, input[type="text"] {
    font-family: 'Inter', sans-serif !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    transition: all 0.15s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #94a3b8 !important;
    box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.15) !important;
    outline: none !important;
}

/* Chatbot messages */
.message {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* Checkbox styling */
.gr-checkbox {
    font-family: 'Inter', sans-serif !important;
}

/* Info section */
.info-section {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
    margin-top: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.info-section h4 {
    margin: 0 0 0.75rem 0;
    color: #1e293b;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.info-section ul {
    margin: 0;
    padding-left: 1.25rem;
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.7;
}

.info-section li {
    margin-bottom: 0.375rem;
}

.info-section strong {
    color: #334155;
    font-weight: 500;
}

/* Labels */
label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #374151 !important;
    font-size: 0.875rem !important;
}

/* Settings section header */
h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    font-size: 1rem !important;
    letter-spacing: -0.01em !important;
}

/* Group containers */
.gr-group {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
"""


class PRISMApp:
    """Gradio application wrapper."""
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self._current_session_id: Optional[str] = None
        self._current_turn_id: Optional[str] = None
        self._current_user_id: str = "default_user"
        self._current_learning_enabled: bool = True
        logger.info("gradio_app_initialized")
    
    def _format_response(self, response) -> str:
        """Format response with beautiful metadata badges."""
        text = response.message
        
        # Add metadata section if any
        metadata_parts = []
        
        if response.tools_used:
            tool_badges = " ".join([f"🔧 {tool}" for tool in response.tools_used])
            metadata_parts.append(f"**Tools:** {tool_badges}")
        
        if response.personalization_applied:
            metadata_parts.append("🎯 **Personalized**")
        
        # Add latency
        metadata_parts.append(f"⚡ {response.latency_ms:.0f}ms")
        
        if metadata_parts:
            metadata_line = " • ".join(metadata_parts)
            text += f"\n\n---\n<small>{metadata_line}</small>"
        
        return text
    
    async def chat(
        self,
        message: str,
        history: List[dict],
        user_id: str,
        learning_enabled: bool,
        personalization_enabled: bool
    ) -> Tuple[List[dict], str]:
        """Handle chat message."""
        if not message.strip():
            return history, ""
        
        # Ensure history is a list
        if history is None:
            history = []
        
        self._current_user_id = user_id or "default_user"
        self._current_learning_enabled = learning_enabled
        
        request = ChatRequest(
            user_id=self._current_user_id,
            message=message,
            session_id=self._current_session_id,
            flags=FeatureFlags(
                memory_enabled=True,
                learning_enabled=learning_enabled,
                personalization_enabled=personalization_enabled
            )
        )
        
        try:
            response = await self.orchestrator.chat(request)
            
            self._current_session_id = response.session_id
            self._current_turn_id = response.turn_id
            
            # Format response with beautiful metadata
            response_text = self._format_response(response)
            
            # Gradio Chatbot format: list of dicts with role and content
            new_history = list(history)  # Create copy
            new_history.append({"role": "user", "content": message})
            new_history.append({"role": "assistant", "content": response_text})
            
            logger.debug(
                "chat_complete",
                user_id=self._current_user_id,
                turn_id=response.turn_id,
                latency_ms=response.latency_ms
            )
            
            return new_history, ""
            
        except Exception as e:
            logger.error("chat_error", error=str(e))
            new_history = list(history)
            new_history.append({"role": "user", "content": message})
            new_history.append({"role": "assistant", "content": f"❌ **Error:** {str(e)}"})
            return new_history, ""
    
    async def positive_feedback(self) -> str:
        """Handle positive feedback."""
        return await self._send_feedback(FeedbackType.POSITIVE)
    
    async def negative_feedback(self) -> str:
        """Handle negative feedback."""
        return await self._send_feedback(FeedbackType.NEGATIVE)
    
    async def _send_feedback(self, feedback_type: FeedbackType) -> str:
        """Send feedback for current turn."""
        if not self._current_turn_id or not self._current_session_id:
            return "💡 Send a message first to rate it"
        
        try:
            request = FeedbackRequest(
                user_id=self._current_user_id,
                session_id=self._current_session_id,
                turn_id=self._current_turn_id,
                feedback=feedback_type,
                learning_enabled=self._current_learning_enabled
            )
            await self.orchestrator.feedback(request)
            
            emoji = "✅" if feedback_type == FeedbackType.POSITIVE else "📝"
            return f"{emoji} Thanks! Feedback recorded."
            
        except Exception as e:
            logger.error("feedback_error", error=str(e))
            return f"❌ Error: {str(e)}"
    
    def new_session(self) -> Tuple[List, str]:
        """Start a new session."""
        self._current_session_id = None
        self._current_turn_id = None
        logger.info("session_reset", user_id=self._current_user_id)
        return [], "🔄 New session started"


def create_app(orchestrator: Orchestrator) -> gr.Blocks:
    """Create the Gradio application."""
    app = PRISMApp(orchestrator)
    
    with gr.Blocks(title="PRISM Assistant") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔮 PRISM Assistant</h1>
            <p><strong>P</strong>ersonalized <strong>R</strong>etrieval, <strong>I</strong>ntelligence extraction, and <strong>S</strong>elective <strong>M</strong>emory</p>
        </div>
        """)
        
        with gr.Row():
            # Settings sidebar
            with gr.Column(scale=1, min_width=280):
                gr.HTML("<h3 style='margin-top:0'>⚙️ Settings</h3>")
                
                user_id = gr.Textbox(
                    label="👤 User ID",
                    value="default_user",
                    placeholder="Enter your user ID",
                    container=True
                )
                
                with gr.Group():
                    learning_toggle = gr.Checkbox(
                        label="🧠 Learning",
                        value=True,
                        info="Learn preferences from interactions"
                    )
                    
                    personalization_toggle = gr.Checkbox(
                        label="🎯 Personalization",
                        value=True,
                        info="Adapt responses to your style"
                    )
                
                new_session_btn = gr.Button(
                    "🔄 New Session",
                    variant="secondary",
                    size="sm"
                )
                
                # Info section
                gr.HTML("""
                <div class="info-section">
                    <h4>💡 How it works</h4>
                    <ul>
                        <li><strong>Learning:</strong> Extracts insights from your conversations</li>
                        <li><strong>Personalization:</strong> Adapts responses based on learned preferences</li>
                        <li><strong>Memory:</strong> Remembers context across sessions</li>
                    </ul>
                    <p style="margin-top: 0.75rem; font-size: 0.85rem; color: #9ca3af;">
                        Toggle features to see the difference!
                    </p>
                </div>
                """)
            
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    value=[]
                )
                
                with gr.Row(elem_classes="input-row"):
                    msg = gr.Textbox(
                        placeholder="Type your message here... Press Enter to send",
                        show_label=False,
                        scale=5,
                        container=False,
                        lines=1,
                        max_lines=3
                    )
                    send_btn = gr.Button(
                        "Send →",
                        variant="primary",
                        scale=1,
                        elem_classes="primary-btn"
                    )
                
                with gr.Row():
                    positive_btn = gr.Button(
                        "👍 Good response",
                        variant="secondary",
                        size="sm",
                        elem_classes="feedback-btn"
                    )
                    negative_btn = gr.Button(
                        "👎 Could be better",
                        variant="secondary",
                        size="sm",
                        elem_classes="feedback-btn"
                    )
                    feedback_status = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        container=False,
                        placeholder="Rate the last response",
                        scale=2
                    )
        
        # Event handlers (using asyncio.run for thread safety)
        def sync_chat(*args):
            return asyncio.run(app.chat(*args))
        
        def sync_positive():
            return asyncio.run(app.positive_feedback())
        
        def sync_negative():
            return asyncio.run(app.negative_feedback())
        
        # Chat submission
        msg.submit(
            fn=sync_chat,
            inputs=[msg, chatbot, user_id, learning_toggle, personalization_toggle],
            outputs=[chatbot, msg]
        )
        
        send_btn.click(
            fn=sync_chat,
            inputs=[msg, chatbot, user_id, learning_toggle, personalization_toggle],
            outputs=[chatbot, msg]
        )
        
        # Feedback buttons
        positive_btn.click(
            fn=sync_positive,
            outputs=[feedback_status]
        )
        
        negative_btn.click(
            fn=sync_negative,
            outputs=[feedback_status]
        )
        
        # New session
        new_session_btn.click(
            fn=app.new_session,
            outputs=[chatbot, feedback_status]
        )
    
    logger.info("gradio_app_created")
    return demo.queue()
