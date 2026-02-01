#!/usr/bin/env python3
"""
Real evaluation script using Bedrock LLM.

Runs complete evaluation pipeline with:
1. LLM-generated conversations from personas
2. Real PRISM orchestrator 
3. LLM-as-judge evaluation

Usage:
    python -m evaluation.run_real_evaluation --output experiments/real_eval
"""

import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import boto3
from botocore.config import Config

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AppConfig, LLMConfig
from src.orchestrator import Orchestrator
from src.models import ChatRequest, FeatureFlags
from src.logging_config import setup_logging, get_logger

from .datasets.generator import ConversationGenerator, create_sample_personas
from .datasets.schemas import (
    AblationConfig, 
    EvaluationDataset, 
    EvaluationConversation,
    TurnResult,
    ExperimentResult,
)
from .judge import LLMJudge
from .analyzer import ResultsAnalyzer
from .runner import save_results

logger = get_logger(__name__)


class BedrockSimpleLLM:
    """Simple Bedrock LLM for generation/evaluation (not tool-use)."""
    
    # Refresh credentials every 30 minutes to pick up external updates
    CREDENTIAL_REFRESH_INTERVAL = 30 * 60  # seconds
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-west-2"
    ):
        self.model_id = model_id
        self.region = region
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
            region_name=self.region,
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"})
        )
        self._client_created_at = time.time()
        print(f"  ✓ Bedrock client refreshed at {datetime.now().strftime('%H:%M:%S')}")
    
    @property
    def client(self):
        """Get the boto3 client, refreshing if credentials are stale."""
        age = time.time() - self._client_created_at
        if age > self.CREDENTIAL_REFRESH_INTERVAL:
            print(f"  ⟳ Refreshing Bedrock credentials (age: {int(age/60)} min)")
            self._create_client()
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate text completion."""
        import json
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            result = json.loads(response["body"].read())
            
            for block in result.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""
            
        except Exception as e:
            print(f"Bedrock error: {e}")
            raise


class RealExperimentRunner:
    """Run experiments with real PRISM orchestrator."""
    
    def __init__(self, config: AppConfig, output_path: Path = None):
        self.config = config
        self.orchestrator: Optional[Orchestrator] = None
        self.output_path = output_path
        self.completed_experiments: set = set()
    
    async def start(self):
        """Start orchestrator."""
        self.orchestrator = Orchestrator(self.config)
        await self.orchestrator.start()
    
    async def stop(self):
        """Stop orchestrator."""
        if self.orchestrator:
            await self.orchestrator.stop()
    
    def load_completed_experiments(self, results_path: Path) -> List[ExperimentResult]:
        """Load already completed experiments for resume."""
        completed = []
        if results_path.exists() and results_path.stat().st_size > 0:
            import json
            with open(results_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            result = ExperimentResult.from_dict(data)
                            completed.append(result)
                            self.completed_experiments.add(result.experiment_id)
                        except Exception as e:
                            print(f"    Warning: Could not load result: {e}")
        return completed
    
    def append_result(self, result: ExperimentResult, results_path: Path):
        """Append a single result to the output file (for incremental saves)."""
        import json
        with open(results_path, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    async def run_conversation(
        self,
        conversation: EvaluationConversation,
        config: AblationConfig,
        experiment_id: str
    ) -> Optional[ExperimentResult]:
        """Run a conversation through real orchestrator. Returns None if all turns failed."""
        
        user_id = f"eval_user_{conversation.persona.persona_id}"
        session_id = None
        
        turn_results = []
        failed_turns = 0
        
        # OPTIMIZATION: For BASELINE, skip learning turns entirely!
        # Baseline has no memory, so T1-T8 don't contribute anything.
        # This saves 80% of LLM calls for baseline config.
        turns_to_run = conversation.turns
        if config == AblationConfig.BASELINE:
            turns_to_run = [t for t in conversation.turns if t.is_test_turn]
            # Also use a fresh user_id so no memory leaks from FULL config
            user_id = f"eval_user_{conversation.persona.persona_id}_baseline"
        
        for turn in turns_to_run:
            # Determine flags based on config
            learning = config in [AblationConfig.FULL, AblationConfig.LEARN_ONLY]
            personalization = config == AblationConfig.FULL
            
            request = ChatRequest(
                user_id=user_id,
                message=turn.user_message,
                session_id=session_id,
                flags=FeatureFlags(
                    memory_enabled=True,
                    learning_enabled=learning,
                    personalization_enabled=personalization
                )
            )
            
            start_time = time.perf_counter()
            
            try:
                response = await self.orchestrator.chat(request)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                session_id = response.session_id
                
                # Get learned insights
                learned = []
                if learning:
                    try:
                        insights_data = await self.orchestrator.memory.get_insights(user_id)
                        if insights_data:
                            learned = [i.content for i in insights_data.insights]
                    except Exception:
                        pass  # Ignore insight retrieval errors
                
                # Get personalization context
                p13n_context = ""
                if personalization and response.personalization_applied:
                    p13n_context = "Personalization applied"
                
                turn_results.append(TurnResult(
                    turn_number=turn.turn_number,
                    user_message=turn.user_message,
                    assistant_response=response.message,
                    latency_ms=latency_ms,
                    learned_insights_after=learned,
                    personalization_context_used=p13n_context,
                    revealed_traits_so_far=conversation.get_revealed_traits_at_turn(turn.turn_number)
                ))
                
            except Exception as e:
                # Log and skip failed turn
                failed_turns += 1
                print(f"      ⚠ Turn {turn.turn_number} failed: {str(e)[:50]}... (skipping)")
                
                # Add placeholder result for failed turn
                turn_results.append(TurnResult(
                    turn_number=turn.turn_number,
                    user_message=turn.user_message,
                    assistant_response=f"[ERROR: {str(e)[:100]}]",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    learned_insights_after=[],
                    personalization_context_used="",
                    revealed_traits_so_far=conversation.get_revealed_traits_at_turn(turn.turn_number)
                ))
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        # Return None only if ALL turns failed
        if failed_turns == len(conversation.turns):
            return None
        
        return ExperimentResult(
            experiment_id=experiment_id,
            config=config.value,
            conversation_id=conversation.conversation_id,
            persona_id=conversation.persona.persona_id,
            turns=turn_results,
            timestamp=datetime.now(datetime.UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat()
        )
    
    async def run_ablation(
        self,
        dataset: EvaluationDataset,
        configs: List[AblationConfig],
        results_path: Path = None
    ) -> List[ExperimentResult]:
        """Run all conversations under specified configs with resume support."""
        
        # Load already completed experiments for resume
        all_results = []
        if results_path:
            all_results = self.load_completed_experiments(results_path)
            if all_results:
                print(f"  ✓ Resuming: Found {len(all_results)} completed experiments")
        
        total = len(configs) * len(dataset.conversations)
        current = len(all_results)
        skipped = 0
        
        # Clear memory ONCE before all configs (not between)
        # This ensures FULL config can use learned insights from learning phase
        memory_path = Path("./storage/memory")
        cleared = False
        for subdir in ["users", "episodic", "semantic"]:
            p = memory_path / subdir
            if p.exists():
                for f in p.glob("eval_user_*.json"):
                    f.unlink()
                    cleared = True
        if cleared:
            print(f"  ✓ Cleared previous eval memory files")
        
        for config in configs:
            print(f"\n  Running config: {config.value}")
            
            for i, conv in enumerate(dataset.conversations):
                exp_id = f"exp_{config.value}_{conv.conversation_id}"
                
                # Skip already completed experiments
                if exp_id in self.completed_experiments:
                    skipped += 1
                    continue
                
                current += 1
                
                try:
                    result = await self.run_conversation(conv, config, exp_id)
                    
                    if result:
                        all_results.append(result)
                        self.completed_experiments.add(exp_id)
                        
                        # Incremental save
                        if results_path:
                            self.append_result(result, results_path)
                        
                        # Calculate stats
                        successful_turns = [t for t in result.turns if not t.assistant_response.startswith("[ERROR")]
                        if successful_turns:
                            avg_latency = sum(t.latency_ms for t in successful_turns) / len(successful_turns)
                        else:
                            avg_latency = 0
                        
                        failed_count = len(result.turns) - len(successful_turns)
                        status = f"✓ {len(successful_turns)}/{len(result.turns)} turns"
                        if failed_count > 0:
                            status += f" ({failed_count} skipped)"
                        
                        print(f"    [{current}/{total}] {conv.persona.persona_id} - {status} - avg latency: {avg_latency:.0f}ms")
                    else:
                        print(f"    [{current}/{total}] {conv.persona.persona_id} - ✗ All turns failed (skipping)")
                    
                except Exception as e:
                    print(f"    [{current}/{total}] {conv.persona.persona_id} - ✗ FAILED: {str(e)[:50]}")
        
        if skipped > 0:
            print(f"\n  ℹ Skipped {skipped} already-completed experiments")
        
        return all_results


async def run_real_evaluation(
    output_dir: str,
    num_personas: int = 10,
    turns_per_conversation: int = 10,
    learning_turns: int = 8,
    seed: int = 42,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-west-2",
    config_path: str = "config.yaml"
):
    """Run complete real evaluation pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("PRISM REAL EVALUATION (Bedrock)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Region: {region}")
    print(f"  Personas: {num_personas}")
    print(f"  Turns: {turns_per_conversation} ({learning_turns} learning + {turns_per_conversation - learning_turns} eval)")
    print(f"  Output: {output_path}")
    print()
    
    # Initialize Bedrock LLM
    print("-"*70)
    print("Initializing Bedrock LLM...")
    print("-"*70)
    llm = BedrockSimpleLLM(model_id=model_id, region=region)
    
    # Test connection
    try:
        test = await llm.generate("Say 'hello' in one word.")
        print(f"  ✓ Bedrock connected: {test[:50]}...")
    except Exception as e:
        print(f"  ✗ Bedrock connection failed: {e}")
        print("  Falling back to mock mode...")
        llm = None
    
    # Step 1: Generate dataset with LLM (or load existing)
    print("\n" + "-"*70)
    print("Step 1/4: Loading/generating evaluation dataset...")
    print("-"*70)
    
    dataset_path = output_path / "dataset.jsonl"
    
    if dataset_path.exists():
        # Load existing dataset
        print(f"  ✓ Found existing dataset at {dataset_path}")
        dataset = EvaluationDataset.from_jsonl(str(dataset_path))
        print(f"  ✓ Loaded {len(dataset.conversations)} conversations")
    else:
        # Generate new dataset
        personas = create_sample_personas(num_personas, seed=seed)
        print(f"  Created {len(personas)} sample personas")
        
        generator = ConversationGenerator(
            llm=llm,
            turns_per_conversation=turns_per_conversation,
            learning_turns=learning_turns,
        )
        
        if llm:
            # Use LLM generation
            dataset = await generator.generate_dataset(personas, dataset_id=f"real_eval_{timestamp}")
        else:
            # Fallback to template
            dataset = generator.generate_dataset_sync(personas, dataset_id=f"real_eval_{timestamp}")
        
        dataset.to_jsonl(str(dataset_path))
        print(f"  ✓ Saved dataset to {dataset_path}")
    
    # Step 2: Run experiments with real orchestrator
    print("\n" + "-"*70)
    print("Step 2/4: Running experiments with real PRISM orchestrator...")
    print("-"*70)
    
    try:
        config = AppConfig.from_yaml(config_path)
    except FileNotFoundError:
        config = AppConfig()
    
    config.logging.level = "WARNING"
    setup_logging(config.logging)
    
    raw_path = output_path / "raw_results.jsonl"
    
    runner = RealExperimentRunner(config, output_path=output_path)
    await runner.start()
    
    try:
        configs = [AblationConfig.BASELINE, AblationConfig.FULL]
        raw_results = await runner.run_ablation(dataset, configs, results_path=raw_path)
    finally:
        await runner.stop()
    
    # Results are already incrementally saved, but ensure final state
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        save_results(raw_results, str(raw_path))
    print(f"\n  ✓ Saved {len(raw_results)} experiment results")
    
    # Step 3: Evaluate with LLM judge
    print("\n" + "-"*70)
    print("Step 3/4: Evaluating with LLM-as-judge...")
    print("-"*70)
    
    judge = LLMJudge(llm=llm)
    scored_results = await judge.evaluate_all(raw_results, dataset)
    
    scored_path = output_path / "scored_results.jsonl"
    scored_results.to_jsonl(str(scored_path))
    print(f"  ✓ Saved scored results")
    
    # Step 4: Analyze results
    print("\n" + "-"*70)
    print("Step 4/4: Analyzing results...")
    print("-"*70)
    
    analyzer = ResultsAnalyzer(scored_results, raw_results)
    analyzer.print_comparison_table()
    
    # Export
    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    analyzer.export_metrics(str(analysis_dir / "metrics.json"))
    
    md_table = analyzer.generate_markdown_table()
    with open(analysis_dir / "results.md", 'w') as f:
        f.write(md_table)
    
    latex_table = analyzer.generate_latex_table()
    with open(analysis_dir / "results.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"\n  ✓ Exported all analysis files")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nOutputs: {output_path}")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Run real PRISM evaluation with Bedrock")
    parser.add_argument("--output", "-o", default="experiments/real_eval", help="Output directory")
    parser.add_argument("--num-personas", "-n", type=int, default=10, help="Number of personas")
    parser.add_argument("--turns", "-t", type=int, default=10, help="Turns per conversation")
    parser.add_argument("--learning-turns", "-l", type=int, default=8, help="Learning turns")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--model", "-m", default="anthropic.claude-3-haiku-20240307-v1:0", help="Bedrock model ID")
    parser.add_argument("--region", "-r", default="us-west-2", help="AWS region")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    asyncio.run(run_real_evaluation(
        output_dir=args.output,
        num_personas=args.num_personas,
        turns_per_conversation=args.turns,
        learning_turns=args.learning_turns,
        seed=args.seed,
        model_id=args.model,
        region=args.region,
        config_path=args.config
    ))


if __name__ == "__main__":
    main()
