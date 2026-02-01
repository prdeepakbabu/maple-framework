#!/usr/bin/env python3
"""
PRISM: Personalized Retrieval, Intelligence extraction, and Selective Memory

Main entry point for the PRISM assistant application.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AppConfig
from src.logging_config import setup_logging, get_logger
from src.orchestrator import Orchestrator
from src.ui import create_app


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM - AI Assistant with Memory, Learning, and Personalization"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = AppConfig.from_yaml(str(config_path))
        print(f"Loaded configuration from {config_path}")
    else:
        config = AppConfig.default()
        print("Using default configuration")
    
    # Override log level if debug flag
    if args.debug:
        config.logging.level = "DEBUG"
        config.logging.format = "plain"
    
    # Setup logging
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    logger.info(
        "prism_starting",
        config_path=str(config_path),
        port=args.port,
        host=args.host
    )
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    
    # Start background services
    await orchestrator.start()
    
    # Create Gradio app
    app = create_app(orchestrator)
    
    # Setup shutdown handler
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Launch the app
        logger.info(
            "launching_web_interface",
            host=args.host,
            port=args.port,
            share=args.share
        )
        
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            prevent_thread_lock=True
        )
        
        print(f"\n🔮 PRISM Assistant running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop\n")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error("startup_error", error=str(e))
        raise
    finally:
        # Cleanup
        logger.info("shutting_down")
        await orchestrator.stop()
        logger.info("shutdown_complete")


def run() -> None:
    """Run the application."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
