#!/usr/bin/env python3
"""PRISM Experiment CLI - Run programmatic experiments and interactive chat."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML

from src.config import AppConfig
from src.orchestrator import Orchestrator
from src.models import ChatRequest, FeedbackRequest, FeedbackType, FeatureFlags
from src.logging_config import setup_logging, get_logger

# Matrix/Terminator green theme
MATRIX_GREEN = "#00ff00"
DARK_GREEN = "#00aa00"
DIM_GREEN = "#006600"
BLACK = "#000000"

# Rich console with green theme
console = Console(force_terminal=True, color_system="truecolor")


class MatrixChatCLI:
    """Interactive chat CLI with Matrix/Terminator aesthetic."""
    
    def __init__(
        self,
        orchestrator: Orchestrator,
        user_id: str = "default_user",
        learning_enabled: bool = True,
        personalization_enabled: bool = True
    ):
        self.orchestrator = orchestrator
        self.user_id = user_id
        self.learning_enabled = learning_enabled
        self.personalization_enabled = personalization_enabled
        self.session_id: Optional[str] = None
        self.last_turn_id: Optional[str] = None
        self.history: list = []
        
        # Setup prompt with history
        history_file = Path.home() / ".prism_history"
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file))
        )
        
        # Prompt toolkit style
        self.pt_style = PTStyle.from_dict({
            'prompt': f'{MATRIX_GREEN} bold',
        })
    
    def print_header(self):
        """Print the Matrix-style header."""
        console.print()
        header = Text()
        header.append("╔══════════════════════════════════════════════════════════════════╗\n", style=f"bold {MATRIX_GREEN}")
        header.append("║  ", style=f"bold {MATRIX_GREEN}")
        header.append("🔮 PRISM ASSISTANT", style=f"bold {MATRIX_GREEN}")
        header.append(" v1.0", style=f"{DIM_GREEN}")
        header.append("                                         ║\n", style=f"bold {MATRIX_GREEN}")
        header.append("║  ", style=f"bold {MATRIX_GREEN}")
        
        # Status indicators
        mem_status = "✓" if True else "✗"
        learn_status = "✓" if self.learning_enabled else "✗"
        pers_status = "✓" if self.personalization_enabled else "✗"
        
        header.append(f"Memory: {mem_status}", style=f"{DARK_GREEN}")
        header.append("  ", style=f"{MATRIX_GREEN}")
        header.append(f"Learning: {learn_status}", style=f"{DARK_GREEN}")
        header.append("  ", style=f"{MATRIX_GREEN}")
        header.append(f"Personalization: {pers_status}", style=f"{DARK_GREEN}")
        header.append("                      ║\n", style=f"bold {MATRIX_GREEN}")
        header.append("╠══════════════════════════════════════════════════════════════════╣", style=f"bold {MATRIX_GREEN}")
        
        console.print(header)
    
    def print_footer(self):
        """Print the command footer."""
        footer = Text()
        footer.append("╠══════════════════════════════════════════════════════════════════╣\n", style=f"bold {MATRIX_GREEN}")
        footer.append("║  ", style=f"bold {MATRIX_GREEN}")
        footer.append("Commands: ", style=f"{DIM_GREEN}")
        footer.append("/help /clear /exit /learning /personalization /feedback", style=f"{DARK_GREEN}")
        footer.append("   ║\n", style=f"bold {MATRIX_GREEN}")
        footer.append("╚══════════════════════════════════════════════════════════════════╝", style=f"bold {MATRIX_GREEN}")
        console.print(footer)
    
    def print_welcome(self):
        """Print welcome message."""
        console.clear()
        self.print_header()
        
        welcome = Text()
        welcome.append("║\n", style=f"bold {MATRIX_GREEN}")
        welcome.append("║  ", style=f"bold {MATRIX_GREEN}")
        welcome.append(f"Welcome, {self.user_id}!", style=f"bold {MATRIX_GREEN}")
        welcome.append(" Type your message to begin.\n", style=f"{DARK_GREEN}")
        welcome.append("║\n", style=f"bold {MATRIX_GREEN}")
        console.print(welcome)
        
        self.print_footer()
        console.print()
    
    def print_message(self, role: str, content: str, timestamp: str = None, metadata: dict = None):
        """Print a chat message in Matrix style."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        msg = Text()
        msg.append("║\n", style=f"bold {MATRIX_GREEN}")
        msg.append("║  ", style=f"bold {MATRIX_GREEN}")
        msg.append(f"[{timestamp}] ", style=f"{DIM_GREEN}")
        
        if role == "user":
            msg.append("You > ", style=f"bold {MATRIX_GREEN}")
            msg.append(content, style=f"{MATRIX_GREEN}")
        else:
            msg.append("PRISM > ", style=f"bold {DARK_GREEN}")
            # Wrap long content
            lines = content.split('\n')
            first = True
            for line in lines:
                if first:
                    msg.append(line + "\n", style=f"{MATRIX_GREEN}")
                    first = False
                else:
                    msg.append("║  ", style=f"bold {MATRIX_GREEN}")
                    msg.append("          ", style=f"{MATRIX_GREEN}")
                    msg.append(line + "\n", style=f"{MATRIX_GREEN}")
            
            # Print metadata if available
            if metadata:
                msg.append("║\n", style=f"bold {MATRIX_GREEN}")
                msg.append("║  ", style=f"bold {MATRIX_GREEN}")
                msg.append("  ┌", style=f"{DIM_GREEN}")
                
                parts = []
                if metadata.get('tools_used'):
                    parts.append(f"🔧 {', '.join(metadata['tools_used'])}")
                if metadata.get('personalized'):
                    parts.append("🎯 Personalized")
                if metadata.get('latency_ms'):
                    parts.append(f"⚡ {metadata['latency_ms']:.0f}ms")
                
                msg.append(" │ ".join(parts), style=f"{DIM_GREEN}")
                msg.append(" ┐\n", style=f"{DIM_GREEN}")
        
        msg.append("║\n", style=f"bold {MATRIX_GREEN}")
        console.print(msg, end="")
    
    def print_help(self):
        """Print help information."""
        help_text = Text()
        help_text.append("\n", style=f"{MATRIX_GREEN}")
        help_text.append("╔═══════════════════════════════════════╗\n", style=f"bold {MATRIX_GREEN}")
        help_text.append("║  ", style=f"bold {MATRIX_GREEN}")
        help_text.append("PRISM COMMANDS", style=f"bold {MATRIX_GREEN}")
        help_text.append("                        ║\n", style=f"bold {MATRIX_GREEN}")
        help_text.append("╠═══════════════════════════════════════╣\n", style=f"bold {MATRIX_GREEN}")
        
        commands = [
            ("/help", "Show this help"),
            ("/clear", "Clear screen & new session"),
            ("/exit", "Exit the chat"),
            ("/learning on|off", "Toggle learning"),
            ("/personalization on|off", "Toggle personalization"),
            ("/feedback +|-", "Rate last response"),
            ("/insights", "Show your insights"),
            ("/user <id>", "Switch user"),
            ("/status", "Show current settings"),
        ]
        
        for cmd, desc in commands:
            help_text.append("║  ", style=f"bold {MATRIX_GREEN}")
            help_text.append(f"{cmd:<22}", style=f"bold {DARK_GREEN}")
            help_text.append(f"{desc:<17}", style=f"{DIM_GREEN}")
            help_text.append("║\n", style=f"bold {MATRIX_GREEN}")
        
        help_text.append("╚═══════════════════════════════════════╝\n", style=f"bold {MATRIX_GREEN}")
        console.print(help_text)
    
    def print_status(self):
        """Print current status."""
        status = Text()
        status.append("\n", style=f"{MATRIX_GREEN}")
        status.append("┌─────────────────────────────────────┐\n", style=f"{DARK_GREEN}")
        status.append("│ ", style=f"{DARK_GREEN}")
        status.append("Current Status", style=f"bold {MATRIX_GREEN}")
        status.append("                      │\n", style=f"{DARK_GREEN}")
        status.append("├─────────────────────────────────────┤\n", style=f"{DARK_GREEN}")
        status.append(f"│ User ID:         {self.user_id:<18} │\n", style=f"{DARK_GREEN}")
        status.append(f"│ Learning:        {'ON' if self.learning_enabled else 'OFF':<18} │\n", style=f"{DARK_GREEN}")
        status.append(f"│ Personalization: {'ON' if self.personalization_enabled else 'OFF':<18} │\n", style=f"{DARK_GREEN}")
        status.append(f"│ Session:         {(self.session_id or 'None')[:18]:<18} │\n", style=f"{DARK_GREEN}")
        status.append("└─────────────────────────────────────┘\n", style=f"{DARK_GREEN}")
        console.print(status)
    
    async def show_insights(self):
        """Show user insights."""
        insights_data = await self.orchestrator.memory.get_insights(self.user_id)
        
        if not insights_data or not insights_data.insights:
            console.print(f"\n[{DIM_GREEN}]No insights found for {self.user_id}[/]\n")
            return
        
        console.print(f"\n[bold {MATRIX_GREEN}]═══ Insights for {self.user_id} ({len(insights_data.insights)} total) ═══[/]\n")
        
        for i, insight in enumerate(insights_data.insights, 1):
            type_color = MATRIX_GREEN if insight.type.value == "preference" else DARK_GREEN
            console.print(f"[{type_color}]{i}. [{insight.type.value.upper()}] {insight.content}[/]")
            console.print(f"[{DIM_GREEN}]   Confidence: {insight.confidence:.2f}[/]\n")
    
    async def process_command(self, user_input: str) -> bool:
        """Process a command. Returns False if should exit."""
        parts = user_input.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "/exit" or cmd == "/quit":
            console.print(f"\n[bold {MATRIX_GREEN}]Goodbye! 👋[/]\n")
            return False
        
        elif cmd == "/help":
            self.print_help()
        
        elif cmd == "/clear":
            self.session_id = None
            self.last_turn_id = None
            self.history = []
            self.print_welcome()
        
        elif cmd == "/status":
            self.print_status()
        
        elif cmd == "/learning":
            if args and args[0].lower() in ["on", "off"]:
                self.learning_enabled = args[0].lower() == "on"
                console.print(f"\n[{MATRIX_GREEN}]Learning {'enabled' if self.learning_enabled else 'disabled'}[/]\n")
            else:
                console.print(f"\n[{DIM_GREEN}]Usage: /learning on|off[/]\n")
        
        elif cmd == "/personalization":
            if args and args[0].lower() in ["on", "off"]:
                self.personalization_enabled = args[0].lower() == "on"
                console.print(f"\n[{MATRIX_GREEN}]Personalization {'enabled' if self.personalization_enabled else 'disabled'}[/]\n")
            else:
                console.print(f"\n[{DIM_GREEN}]Usage: /personalization on|off[/]\n")
        
        elif cmd == "/feedback":
            if not self.last_turn_id:
                console.print(f"\n[{DIM_GREEN}]No response to rate yet[/]\n")
            elif args and args[0] in ["+", "-"]:
                feedback_type = FeedbackType.POSITIVE if args[0] == "+" else FeedbackType.NEGATIVE
                request = FeedbackRequest(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    turn_id=self.last_turn_id,
                    feedback=feedback_type,
                    learning_enabled=self.learning_enabled
                )
                await self.orchestrator.feedback(request)
                emoji = "✅" if args[0] == "+" else "📝"
                console.print(f"\n[{MATRIX_GREEN}]{emoji} Feedback recorded![/]\n")
            else:
                console.print(f"\n[{DIM_GREEN}]Usage: /feedback +|-[/]\n")
        
        elif cmd == "/insights":
            await self.show_insights()
        
        elif cmd == "/user":
            if args:
                self.user_id = args[0]
                self.session_id = None
                console.print(f"\n[{MATRIX_GREEN}]Switched to user: {self.user_id}[/]\n")
            else:
                console.print(f"\n[{DIM_GREEN}]Usage: /user <user_id>[/]\n")
        
        else:
            console.print(f"\n[{DIM_GREEN}]Unknown command. Type /help for commands.[/]\n")
        
        return True
    
    async def send_message(self, message: str):
        """Send a message and display the response."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Print user message
        self.print_message("user", message, timestamp)
        
        # Build request
        request = ChatRequest(
            user_id=self.user_id,
            message=message,
            session_id=self.session_id,
            flags=FeatureFlags(
                memory_enabled=True,
                learning_enabled=self.learning_enabled,
                personalization_enabled=self.personalization_enabled
            )
        )
        
        # Show thinking indicator
        console.print(f"║  [{DIM_GREEN}]Thinking...[/]", end="\r")
        
        try:
            response = await self.orchestrator.chat(request)
            
            self.session_id = response.session_id
            self.last_turn_id = response.turn_id
            
            # Clear thinking indicator
            console.print("║  " + " " * 20, end="\r")
            
            # Print response
            resp_timestamp = datetime.now().strftime("%H:%M:%S")
            metadata = {
                'tools_used': response.tools_used,
                'personalized': response.personalization_applied,
                'latency_ms': response.latency_ms
            }
            self.print_message("assistant", response.message, resp_timestamp, metadata)
            
        except Exception as e:
            console.print(f"\n║  [{MATRIX_GREEN}]Error: {str(e)}[/]\n")
    
    async def run(self):
        """Run the interactive chat loop."""
        self.print_welcome()
        
        while True:
            try:
                # Get input using prompt_toolkit
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.prompt_session.prompt(
                        HTML(f'<style fg="{MATRIX_GREEN}" bold="true">┃ You > </style>'),
                        style=self.pt_style
                    )
                )
                
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    should_continue = await self.process_command(user_input)
                    if not should_continue:
                        break
                else:
                    await self.send_message(user_input)
                    
            except KeyboardInterrupt:
                console.print(f"\n\n[bold {MATRIX_GREEN}]Interrupted. Goodbye! 👋[/]\n")
                break
            except EOFError:
                console.print(f"\n\n[bold {MATRIX_GREEN}]Goodbye! 👋[/]\n")
                break


# =============================================================================
# Original Experiment CLI Functions
# =============================================================================

async def run_experiment(
    input_path: str,
    output_path: str,
    config_path: str = "config.yaml",
    verbose: bool = False
) -> None:
    """Run experiment from JSONL file."""
    
    try:
        config = AppConfig.from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        config = AppConfig()
        print("Using default configuration")
    
    if verbose:
        config.logging.level = "DEBUG"
    
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    orchestrator = Orchestrator(config)
    await orchestrator.start()
    
    results = []
    
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\nProcessing {len([l for l in lines if l.strip()])} turns from {input_path}\n")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                turn_data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("json_parse_error", line=line_num, error=str(e))
                print(f"  ❌ Line {line_num}: JSON parse error - {e}")
                continue
            
            required = ['user_id', 'session_id', 'turn_id', 'message', 
                       'enable_learning', 'enable_personalization']
            missing = [f for f in required if f not in turn_data]
            if missing:
                logger.error("missing_fields", line=line_num, fields=missing)
                print(f"  ❌ Line {line_num}: Missing fields - {missing}")
                continue
            
            print(f"  [{turn_data['turn_id']}] Processing: {turn_data['message'][:50]}...")
            result = await process_turn(orchestrator, turn_data, verbose)
            results.append(result)
            
            print(f"  [{turn_data['turn_id']}] ✓ Response: {result['output']['response'][:60]}...")
            if result['learning']['insights_extracted']:
                print(f"  [{turn_data['turn_id']}]   Learned: {len(result['learning']['insights_extracted'])} insights")
            
            logger.info("turn_processed", turn_id=turn_data['turn_id'], line=line_num)
                
    finally:
        await orchestrator.stop()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result, default=str) + '\n')
    
    print(f"\n✓ Experiment complete: {len(results)} turns processed")
    print(f"✓ Results written to: {output_path}")


async def process_turn(orchestrator: Orchestrator, turn_data: dict, verbose: bool) -> dict:
    """Process a single turn and return result record."""
    timestamp = datetime.utcnow()
    
    request = ChatRequest(
        user_id=turn_data['user_id'],
        message=turn_data['message'],
        session_id=turn_data['session_id'],
        flags=FeatureFlags(
            memory_enabled=True,
            learning_enabled=turn_data['enable_learning'],
            personalization_enabled=turn_data['enable_personalization']
        )
    )
    
    p_context = None
    if turn_data['enable_personalization']:
        p_context = await orchestrator.personalization.build_context(
            turn_data['user_id'],
            turn_data['message']
        )
    
    response = await orchestrator.chat(request)
    
    insights_extracted = []
    if turn_data['enable_learning']:
        await asyncio.sleep(1.0)
        user_insights = await orchestrator.memory.get_insights(turn_data['user_id'])
        if user_insights:
            recent = sorted(user_insights.insights, key=lambda x: x.extracted_at, reverse=True)[:5]
            insights_extracted = [
                {"type": i.type.value, "content": i.content, "confidence": i.confidence}
                for i in recent
            ]
    
    feedback = turn_data.get('feedback')
    if feedback:
        feedback_type = FeedbackType.POSITIVE if feedback == "positive" else FeedbackType.NEGATIVE
        feedback_request = FeedbackRequest(
            user_id=turn_data['user_id'],
            session_id=turn_data['session_id'],
            turn_id=response.turn_id,
            feedback=feedback_type,
            learning_enabled=turn_data['enable_learning']
        )
        await orchestrator.feedback(feedback_request)
        if turn_data['enable_learning']:
            await asyncio.sleep(1.0)
    
    result = {
        "turn_id": turn_data['turn_id'],
        "user_id": turn_data['user_id'],
        "session_id": turn_data['session_id'],
        "timestamp": timestamp.isoformat(),
        "input": {
            "message": turn_data['message'],
            "enable_learning": turn_data['enable_learning'],
            "enable_personalization": turn_data['enable_personalization']
        },
        "output": {
            "response": response.message,
            "tools_used": response.tools_used,
            "latency_ms": response.latency_ms
        },
        "personalization": {
            "applied": response.personalization_applied,
            "context_token_count": p_context.token_count if p_context else 0,
            "insights_used": [i.content for i in p_context.recent_insights] if p_context else []
        },
        "learning": {
            "enabled": turn_data['enable_learning'],
            "insights_extracted": insights_extracted
        },
        "feedback": feedback
    }
    
    if verbose and p_context:
        result["personalization"]["system_prompt_preview"] = p_context.system_instructions[:500]
    
    return result


async def run_chat(
    user_id: str = "default_user",
    learning: bool = True,
    personalization: bool = True,
    config_path: str = "config.yaml"
):
    """Run interactive chat CLI."""
    try:
        config = AppConfig.from_yaml(config_path)
    except FileNotFoundError:
        config = AppConfig()
    
    # Disable structured logging for clean CLI output
    config.logging.level = "WARNING"
    setup_logging(config.logging)
    
    orchestrator = Orchestrator(config)
    await orchestrator.start()
    
    try:
        cli = MatrixChatCLI(
            orchestrator=orchestrator,
            user_id=user_id,
            learning_enabled=learning,
            personalization_enabled=personalization
        )
        await cli.run()
    finally:
        await orchestrator.stop()


async def clear_memory(user_id: str, storage_path: str = "./storage/memory") -> None:
    """Clear all memory for a user."""
    base = Path(storage_path)
    cleared_any = False
    
    user_file = base / "users" / f"{user_id}.json"
    if user_file.exists():
        user_file.unlink()
        print(f"✓ Cleared user profile: {user_id}")
        cleared_any = True
    
    episodic_dir = base / "episodic"
    if episodic_dir.exists():
        for session_file in episodic_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                if data.get("user_id") == user_id:
                    session_file.unlink()
                    print(f"✓ Cleared session: {session_file.name}")
                    cleared_any = True
            except Exception:
                pass
    
    semantic_file = base / "semantic" / f"{user_id}.json"
    if semantic_file.exists():
        semantic_file.unlink()
        print(f"✓ Cleared insights: {user_id}")
        cleared_any = True
    
    if not cleared_any:
        print(f"No data found for user: {user_id}")


async def show_insights(user_id: str, storage_path: str = "./storage/memory") -> None:
    """Show insights for a user."""
    semantic_file = Path(storage_path) / "semantic" / f"{user_id}.json"
    
    if not semantic_file.exists():
        print(f"No insights found for user: {user_id}")
        return
    
    with open(semantic_file) as f:
        data = json.load(f)
    
    insights = data.get("insights", [])
    print(f"\n=== Insights for {user_id} ({len(insights)} total) ===\n")
    
    for i, insight in enumerate(insights, 1):
        print(f"{i}. [{insight['type'].upper()}] {insight['content']}")
        print(f"   Confidence: {insight['confidence']:.2f}")
        print(f"   Extracted: {insight['extracted_at']}")
        print()


async def show_profile(user_id: str, storage_path: str = "./storage/memory") -> None:
    """Show profile for a user."""
    profile_file = Path(storage_path) / "users" / f"{user_id}.json"
    
    if not profile_file.exists():
        print(f"No profile found for user: {user_id}")
        return
    
    with open(profile_file) as f:
        data = json.load(f)
    
    print(f"\n=== Profile for {user_id} ===\n")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Experiment CLI - Interactive chat and experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat (Matrix theme)
  python cli.py chat
  python cli.py chat --user deepak --no-learning
  
  # Run experiment from JSONL file
  python cli.py run experiment.jsonl -o results.jsonl -v
  
  # Run evaluation pipeline
  python cli.py eval -o experiments/eval_run -n 10
  
  # Manage user data
  python cli.py clear-memory -u exp_user_1
  python cli.py show-insights -u exp_user_1
  python cli.py show-profile -u exp_user_1
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Chat command (NEW!)
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat (Matrix theme)")
    chat_parser.add_argument("--user", "-u", default="default_user", help="User ID")
    chat_parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    chat_parser.add_argument("--no-learning", action="store_true", help="Disable learning")
    chat_parser.add_argument("--no-personalization", action="store_true", help="Disable personalization")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment from JSONL file")
    run_parser.add_argument("input", help="Input JSONL file path")
    run_parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    run_parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    # Clear memory command
    clear_parser = subparsers.add_parser("clear-memory", help="Clear all memory for a user")
    clear_parser.add_argument("--user-id", "-u", required=True, help="User ID to clear")
    clear_parser.add_argument("--storage", default="./storage/memory", help="Storage path")
    
    # Show insights command
    insights_parser = subparsers.add_parser("show-insights", help="Show semantic insights for a user")
    insights_parser.add_argument("--user-id", "-u", required=True, help="User ID")
    insights_parser.add_argument("--storage", default="./storage/memory", help="Storage path")
    
    # Show profile command
    profile_parser = subparsers.add_parser("show-profile", help="Show user profile")
    profile_parser.add_argument("--user-id", "-u", required=True, help="User ID")
    profile_parser.add_argument("--storage", default="./storage/memory", help="Storage path")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument("--output", "-o", default="experiments/eval_run", help="Output directory")
    eval_parser.add_argument("--num-personas", "-n", type=int, default=10, help="Number of personas")
    eval_parser.add_argument("--turns", "-t", type=int, default=10, help="Turns per conversation")
    eval_parser.add_argument("--learning-turns", "-l", type=int, default=8, help="Learning turns")
    eval_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        asyncio.run(run_chat(
            user_id=args.user,
            learning=not args.no_learning,
            personalization=not args.no_personalization,
            config_path=args.config
        ))
    elif args.command == "run":
        asyncio.run(run_experiment(
            args.input, args.output, args.config, args.verbose
        ))
    elif args.command == "clear-memory":
        asyncio.run(clear_memory(args.user_id, args.storage))
    elif args.command == "show-insights":
        asyncio.run(show_insights(args.user_id, args.storage))
    elif args.command == "show-profile":
        asyncio.run(show_profile(args.user_id, args.storage))
    elif args.command == "eval":
        from evaluation.run_evaluation import run_evaluation
        run_evaluation(
            output_dir=args.output,
            num_personas=args.num_personas,
            turns_per_conversation=args.turns,
            learning_turns=args.learning_turns,
            seed=args.seed
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
