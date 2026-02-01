"""Results analyzer for generating comparison table and metrics."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

from .datasets.schemas import ScoredResults, AblationConfig


@dataclass
class ConfigMetrics:
    """Metrics for one configuration."""
    
    config: str
    mean_score: float  # Now only test turns are judged
    std_score: float
    incorporation_rate: float
    violation_rate: float
    mean_latency_ms: float
    n_conversations: int
    n_turns: int  # Number of test turns evaluated


@dataclass
class ComparisonResult:
    """Final comparison table result."""
    
    metric_name: str
    without_prism: str  # "value ± std" format
    with_prism: str
    improvement: str  # "Δ value (+%)" format


class ResultsAnalyzer:
    """Analyze scored results and generate comparison table."""
    
    def __init__(self, results: ScoredResults, raw_results: List = None):
        """Initialize analyzer.
        
        Args:
            results: Scored evaluation results
            raw_results: Raw experiment results (for latency data)
        """
        self.results = results
        self.raw_results = raw_results or []
        
        # Try to build DataFrame
        try:
            self.df = results.to_dataframe()
        except ImportError:
            self.df = None
    
    def compute_config_metrics(self, learning_turns: int = None) -> Dict[str, ConfigMetrics]:
        """Compute metrics for each configuration.
        
        Args:
            learning_turns: Number of learning turns (eval turns are after this).
                           If None, tries to detect from data or uses all turns.
        """
        
        metrics = {}
        
        for config_name in ['baseline', 'full', 'learn_only', 'p13n_only']:
            config_evals = self.results.filter_by_config(config_name)
            
            if not config_evals:
                continue
            
            # Detect test turns threshold
            # If learning_turns provided, test turns start after that
            # Otherwise detect: find max turn number and assume last ~20% are test turns
            if learning_turns is not None:
                test_turn_threshold = learning_turns
            else:
                max_turn = max(
                    turn_eval.turn_number 
                    for conv_eval in config_evals 
                    for turn_eval in conv_eval.turn_evaluations
                )
                # If only a few turns, use ~50% as threshold; otherwise 80%
                test_turn_threshold = max(1, int(max_turn * 0.5)) if max_turn <= 6 else int(max_turn * 0.8)
            
            # Collect all scores
            all_scores = []
            eval_turn_scores = []  # Test turns only
            total_incorporated = 0
            total_violated = 0
            total_traits = 0
            
            for conv_eval in config_evals:
                for turn_eval in conv_eval.turn_evaluations:
                    all_scores.append(turn_eval.personalization_score)
                    
                    # Eval/test turns (after learning phase)
                    if turn_eval.turn_number > test_turn_threshold:
                        eval_turn_scores.append(turn_eval.personalization_score)
                    
                    # Trait consistency
                    tc = turn_eval.trait_consistency
                    total_incorporated += len(tc.incorporated)
                    total_violated += len(tc.violated)
                    total_traits += len(tc.incorporated) + len(tc.violated) + len(tc.neutral)
            
            # Compute latency from raw results
            latencies = []
            for raw in self.raw_results:
                if raw.config == config_name:
                    for turn in raw.turns:
                        latencies.append(turn.latency_ms)
            
            mean_latency = sum(latencies) / len(latencies) if latencies else 0
            
            import statistics
            
            # Since we now only judge test turns, all_scores ARE the eval scores
            metrics[config_name] = ConfigMetrics(
                config=config_name,
                mean_score=statistics.mean(all_scores) if all_scores else 0,
                std_score=statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                incorporation_rate=total_incorporated / total_traits if total_traits > 0 else 0,
                violation_rate=total_violated / total_traits if total_traits > 0 else 0,
                mean_latency_ms=mean_latency,
                n_conversations=len(config_evals),
                n_turns=len(all_scores)
            )
        
        return metrics
    
    def generate_comparison_table(self) -> List[ComparisonResult]:
        """Generate the main comparison table: With vs Without PRISM."""
        
        metrics = self.compute_config_metrics()
        
        baseline = metrics.get('baseline')
        full = metrics.get('full')
        
        if not baseline or not full:
            print("Warning: Missing baseline or full config data")
            return []
        
        results = []
        
        # 1. LLM-as-Judge Score (only test turns are judged now)
        score_diff = full.mean_score - baseline.mean_score
        score_pct = (score_diff / baseline.mean_score * 100) if baseline.mean_score else 0
        
        results.append(ComparisonResult(
            metric_name="LLM-as-Judge Score (1-5)",
            without_prism=f"{baseline.mean_score:.2f} ± {baseline.std_score:.2f}",
            with_prism=f"{full.mean_score:.2f} ± {full.std_score:.2f}",
            improvement=f"+{score_diff:.2f} (+{score_pct:.0f}%)"
        ))
        
        # 2. Trait Consistency (Incorporation Rate)
        inc_diff = full.incorporation_rate - baseline.incorporation_rate
        
        results.append(ComparisonResult(
            metric_name="Trait Consistency (Inc. Rate)",
            without_prism=f"{baseline.incorporation_rate:.0%}",
            with_prism=f"{full.incorporation_rate:.0%}",
            improvement=f"+{inc_diff:.0%}"
        ))
        
        # 3. Response Latency
        latency_diff = full.mean_latency_ms - baseline.mean_latency_ms
        latency_pct = (latency_diff / baseline.mean_latency_ms * 100) if baseline.mean_latency_ms else 0
        
        results.append(ComparisonResult(
            metric_name="Response Latency (ms)",
            without_prism=f"{baseline.mean_latency_ms:.0f}",
            with_prism=f"{full.mean_latency_ms:.0f}",
            improvement=f"+{latency_diff:.0f}ms (+{latency_pct:.0f}%)"
        ))
        
        return results
    
    def print_comparison_table(self):
        """Print comparison table to console."""
        
        results = self.generate_comparison_table()
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS: With vs Without PRISM (Evaluation Turns T9-T10)")
        print("="*70)
        
        # Header
        print(f"\n{'Metric':<30} {'Without PRISM':<15} {'With PRISM':<15} {'Δ Improvement':<15}")
        print("-"*70)
        
        # Rows
        for r in results:
            print(f"{r.metric_name:<30} {r.without_prism:<15} {r.with_prism:<15} {r.improvement:<15}")
        
        print("-"*70)
        print()
    
    def export_metrics(self, output_path: str):
        """Export all metrics to JSON file."""
        
        metrics = self.compute_config_metrics()
        comparison = self.generate_comparison_table()
        
        output = {
            "config_metrics": {k: asdict(v) for k, v in metrics.items()},
            "comparison_table": [asdict(c) for c in comparison],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Exported metrics to {output_path}")
    
    def generate_markdown_table(self) -> str:
        """Generate Markdown comparison table."""
        
        results = self.generate_comparison_table()
        
        lines = [
            "## Evaluation Results: With vs Without PRISM",
            "",
            "| Metric | Without PRISM | With PRISM | Δ Improvement |",
            "|--------|---------------|------------|---------------|",
        ]
        
        for r in results:
            lines.append(f"| **{r.metric_name}** | {r.without_prism} | {r.with_prism} | {r.improvement} |")
        
        return "\n".join(lines)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        
        results = self.generate_comparison_table()
        
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Evaluation Results: With vs Without PRISM}",
            r"\label{tab:results}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Metric & Without PRISM & With PRISM & Improvement \\",
            r"\midrule",
        ]
        
        for r in results:
            name = r.metric_name.replace('%', r'\%')
            without = r.without_prism.replace('±', r'$\pm$')
            with_p = r.with_prism.replace('±', r'$\pm$')
            imp = r.improvement.replace('%', r'\%').replace('+', r'+')
            lines.append(f"{name} & {without} & {with_p} & {imp} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)


def run_full_analysis(
    dataset_path: str,
    raw_results_path: str,
    scored_results_path: str,
    output_dir: str
):
    """Run full analysis pipeline and generate all outputs.
    
    Args:
        dataset_path: Path to evaluation dataset JSONL
        raw_results_path: Path to raw experiment results JSONL
        scored_results_path: Path to scored results JSONL
        output_dir: Directory for output files
    """
    from .datasets.schemas import EvaluationDataset
    from .runner import load_results
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    scored = ScoredResults.from_jsonl(scored_results_path)
    raw_results = load_results(raw_results_path) if Path(raw_results_path).exists() else []
    
    # Analyze
    print("Analyzing results...")
    analyzer = ResultsAnalyzer(scored, raw_results)
    
    # Print to console
    analyzer.print_comparison_table()
    
    # Export JSON metrics
    analyzer.export_metrics(str(output_path / "metrics.json"))
    
    # Export Markdown
    md_table = analyzer.generate_markdown_table()
    with open(output_path / "results.md", 'w') as f:
        f.write(md_table)
    print(f"Exported Markdown to {output_path / 'results.md'}")
    
    # Export LaTeX
    latex_table = analyzer.generate_latex_table()
    with open(output_path / "results.tex", 'w') as f:
        f.write(latex_table)
    print(f"Exported LaTeX to {output_path / 'results.tex'}")
    
    return analyzer
