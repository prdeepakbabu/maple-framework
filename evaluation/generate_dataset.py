"""Standalone dataset generation script with resume support.

Usage:
    # Generate new dataset
    python -m evaluation.generate_dataset -o experiments/my_eval -n 20 -t 10 -l 8
    
    # Resume from interrupted generation
    python -m evaluation.generate_dataset -o experiments/my_eval -n 20 -t 10 -l 8 --resume
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.bedrock import BedrockLLM
from src.config import LLMConfig
from evaluation.datasets.generator import ConversationGenerator, create_sample_personas
from evaluation.datasets.schemas import EvaluationConversation, Persona


def load_existing_conversations(dataset_path: str) -> List[EvaluationConversation]:
    """Load existing conversations from dataset file."""
    conversations = []
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    conv = EvaluationConversation.from_dict(data)
                    conversations.append(conv)
    return conversations


def get_completed_persona_ids(conversations: List[EvaluationConversation]) -> set:
    """Get set of persona IDs that have been completed."""
    return {conv.persona.persona_id for conv in conversations}


async def generate_dataset_incremental(
    output_dir: str,
    num_personas: int,
    turns_per_conversation: int,
    learning_turns: int,
    resume: bool = False,
    use_llm: bool = True,
    seed: int = 42,
) -> str:
    """Generate dataset with incremental save and resume support.
    
    Args:
        output_dir: Directory to save dataset
        num_personas: Number of personas/conversations to generate
        turns_per_conversation: Total turns per conversation
        learning_turns: Number of learning turns (rest are test turns)
        resume: Whether to resume from existing dataset
        use_llm: Whether to use LLM for generation (False = simple templates)
        seed: Random seed for persona generation
        
    Returns:
        Path to generated dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    
    # Load existing conversations if resuming
    existing_conversations = []
    completed_persona_ids = set()
    
    if resume and os.path.exists(dataset_path):
        existing_conversations = load_existing_conversations(dataset_path)
        completed_persona_ids = get_completed_persona_ids(existing_conversations)
        print(f"📂 Resuming: Found {len(existing_conversations)} existing conversations")
    
    # Generate all personas
    personas = create_sample_personas(num_personas, seed=seed)
    
    # Filter out already-completed personas
    remaining_personas = [p for p in personas if p.persona_id not in completed_persona_ids]
    
    if not remaining_personas:
        print(f"✅ All {num_personas} conversations already generated!")
        return dataset_path
    
    print(f"📊 Generating {len(remaining_personas)} conversations ({len(completed_persona_ids)} already done)")
    
    # Initialize generator
    llm = None
    if use_llm:
        try:
            llm_config = LLMConfig()
            llm = BedrockLLM(llm_config)
            print(f"🤖 Using LLM: {llm_config.model_id}")
        except Exception as e:
            print(f"⚠️ LLM init failed, using templates: {e}")
    else:
        print("📝 Using template-based generation (no LLM)")
    
    generator = ConversationGenerator(
        llm=llm,
        turns_per_conversation=turns_per_conversation,
        learning_turns=learning_turns
    )
    
    # Open file in append mode for incremental save
    mode = 'a' if resume else 'w'
    with open(dataset_path, mode) as f:
        for i, persona in enumerate(remaining_personas):
            conv_id = f"eval_{persona.persona_id}_conv"
            global_idx = len(completed_persona_ids) + i + 1
            
            try:
                print(f"  [{global_idx}/{num_personas}] Generating {persona.persona_id}...", end=" ", flush=True)
                
                if llm:
                    conv = await generator.generate_conversation(persona, conv_id)
                else:
                    conv = generator._generate_simple(persona, conv_id)
                
                # Write immediately (incremental save)
                f.write(json.dumps(conv.to_dict()) + '\n')
                f.flush()  # Ensure written to disk
                
                print(f"✓ ({len(conv.turns)} turns)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"⚠️ Saved {global_idx - 1} conversations. Re-run with --resume to continue.")
                raise
    
    total = len(completed_persona_ids) + len(remaining_personas)
    print(f"\n✅ Dataset complete: {total} conversations saved to {dataset_path}")
    
    # Print summary
    print("\n📋 Dataset Summary:")
    print(f"   - Personas: {total}")
    print(f"   - Turns per conversation: {turns_per_conversation}")
    print(f"   - Learning turns: {learning_turns}")
    print(f"   - Test turns: {turns_per_conversation - learning_turns}")
    
    return dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset with resume support"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "-n", "--num-personas",
        type=int,
        default=10,
        help="Number of personas/conversations (default: 10)"
    )
    parser.add_argument(
        "-t", "--turns",
        type=int,
        default=10,
        help="Total turns per conversation (default: 10)"
    )
    parser.add_argument(
        "-l", "--learning-turns",
        type=int,
        default=8,
        help="Number of learning turns (default: 8)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial dataset"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use template-based generation instead of LLM"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for persona generation (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.learning_turns >= args.turns:
        print(f"Error: learning_turns ({args.learning_turns}) must be < turns ({args.turns})")
        sys.exit(1)
    
    # Run generation
    asyncio.run(generate_dataset_incremental(
        output_dir=args.output,
        num_personas=args.num_personas,
        turns_per_conversation=args.turns,
        learning_turns=args.learning_turns,
        resume=args.resume,
        use_llm=not args.no_llm,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
