#!/usr/bin/env python3
"""
Main evaluation script for PRISM framework.

Runs the complete evaluation pipeline:
1. Generate dataset (10 personas × 10 turns)
2. Run experiments (baseline vs full)
3. Evaluate with LLM judge
4. Generate comparison table

Usage:
    python -m evaluation.run_evaluation --output experiments/eval_run
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime

from .datasets.generator import ConversationGenerator, create_sample_personas
from .datasets.schemas import AblationConfig, EvaluationDataset
from .runner import ExperimentRunner, save_results, load_results
from .judge import LLMJudge
from .analyzer import ResultsAnalyzer


def run_evaluation(
    output_dir: str,
    num_personas: int = 10,
    turns_per_conversation: int = 10,
    learning_turns: int = 8,
    seed: int = 42,
):
    """Run complete evaluation pipeline.
    
    Args:
        output_dir: Directory for all outputs
        num_personas: Number of personas to generate
        turns_per_conversation: Total turns per conversation
        learning_turns: Number of learning turns (rest are evaluation)
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("PRISM EVALUATION PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Personas: {num_personas}")
    print(f"  Turns per conversation: {turns_per_conversation}")
    print(f"  Learning turns: {learning_turns}")
    print(f"  Evaluation turns: {turns_per_conversation - learning_turns}")
    print(f"  Output: {output_path}")
    print()
    
    # Step 1: Generate dataset
    print("-"*60)
    print("Step 1/4: Generating evaluation dataset...")
    print("-"*60)
    
    personas = create_sample_personas(num_personas, seed=seed)
    print(f"  Created {len(personas)} sample personas")
    
    generator = ConversationGenerator(
        llm=None,  # Use template-based generation
        turns_per_conversation=turns_per_conversation,
        learning_turns=learning_turns,
    )
    
    dataset = generator.generate_dataset_sync(personas, dataset_id=f"eval_{timestamp}")
    
    dataset_path = output_path / "dataset.jsonl"
    dataset.to_jsonl(str(dataset_path))
    print(f"  Saved dataset to {dataset_path}")
    print(f"  Generated {len(dataset.conversations)} conversations")
    
    # Step 2: Run experiments
    print("\n" + "-"*60)
    print("Step 2/4: Running experiments (baseline vs full)...")
    print("-"*60)
    
    runner = ExperimentRunner(orchestrator_factory=None)  # Use mock orchestrator
    
    configs = [AblationConfig.BASELINE, AblationConfig.FULL]
    raw_results = runner.run_ablation_sync(dataset, configs)
    
    raw_results_path = output_path / "raw_results.jsonl"
    save_results(raw_results, str(raw_results_path))
    print(f"\n  Saved {len(raw_results)} experiment results to {raw_results_path}")
    
    # Step 3: Evaluate with LLM judge
    print("\n" + "-"*60)
    print("Step 3/4: Evaluating with LLM judge...")
    print("-"*60)
    
    judge = LLMJudge(llm=None)  # Use mock judge
    scored_results = judge.evaluate_all_sync(raw_results, dataset)
    
    scored_path = output_path / "scored_results.jsonl"
    scored_results.to_jsonl(str(scored_path))
    print(f"  Saved scored results to {scored_path}")
    
    # Step 4: Analyze and generate comparison table
    print("\n" + "-"*60)
    print("Step 4/4: Analyzing results...")
    print("-"*60)
    
    analyzer = ResultsAnalyzer(scored_results, raw_results)
    
    # Print main result
    analyzer.print_comparison_table()
    
    # Export files
    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    analyzer.export_metrics(str(analysis_dir / "metrics.json"))
    
    md_table = analyzer.generate_markdown_table()
    with open(analysis_dir / "results.md", 'w') as f:
        f.write(md_table)
    print(f"  Exported Markdown to {analysis_dir / 'results.md'}")
    
    latex_table = analyzer.generate_latex_table()
    with open(analysis_dir / "results.tex", 'w') as f:
        f.write(latex_table)
    print(f"  Exported LaTeX to {analysis_dir / 'results.tex'}")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_path}")
    print(f"  - dataset.jsonl")
    print(f"  - raw_results.jsonl")
    print(f"  - scored_results.jsonl")
    print(f"  - analysis/metrics.json")
    print(f"  - analysis/results.md")
    print(f"  - analysis/results.tex")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run PRISM evaluation pipeline"
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments/eval_run",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-personas", "-n",
        type=int,
        default=10,
        help="Number of personas to evaluate"
    )
    parser.add_argument(
        "--turns", "-t",
        type=int,
        default=10,
        help="Turns per conversation"
    )
    parser.add_argument(
        "--learning-turns", "-l",
        type=int,
        default=8,
        help="Number of learning turns"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        output_dir=args.output,
        num_personas=args.num_personas,
        turns_per_conversation=args.turns,
        learning_turns=args.learning_turns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
