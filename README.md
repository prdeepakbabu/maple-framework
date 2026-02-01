# MAPLE Framework

**M**emory-**A**daptive **P**ersonalized **LE**arning for Agentic AI Systems

[![Paper](https://img.shields.io/badge/Paper-AAMAS%202026-blue)](https://example.com)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/prdeepakbabu/maple-personas)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

MAPLE decomposes agent adaptation into three architecturally distinct components:

- **Memory (M)**: Storage and retrieval infrastructure for user data
- **Learning (L)**: Asynchronous intelligence extraction from interactions  
- **Personalization (P)**: Real-time adaptation of responses based on learned preferences

This repository contains the reference implementation accompanying our AAMAS 2026 paper.

## Installation

```bash
git clone https://github.com/prdeepakbabu/maple-framework.git
cd maple-framework
pip install -r requirements.txt
```

### Prerequisites

- Python 3.10+
- AWS credentials configured for Bedrock access (Claude models)

## Quick Start

### Interactive Demo (Gradio UI)

```bash
python main.py
```

Opens a web interface at `http://localhost:7860` where you can:
- Chat with MAPLE and see personalization in action
- Toggle Learning ON/OFF to compare behavior
- Toggle Personalization ON/OFF
- Provide feedback (👍/👎) to train the system

### Command Line

```bash
python cli.py --user-id demo_user
```

## Project Structure

```
maple-framework/
├── maple/                     # Core framework
│   ├── agents/
│   │   ├── learning_agent.py  # Intelligence extraction
│   │   ├── memory_agent.py    # Storage/retrieval
│   │   └── personalization.py # Context assembly
│   ├── llm/
│   │   ├── base.py           # LLM abstraction
│   │   └── bedrock.py        # AWS Bedrock implementation
│   ├── orchestrator/
│   │   └── orchestrator.py   # Request coordination
│   ├── tools/                # Tool implementations
│   └── ui/
│       └── gradio_app.py     # Web interface
│
├── evaluation/               # Evaluation framework
│   ├── run_evaluation.py     # Main evaluation script
│   ├── judge.py              # LLM-as-judge implementation
│   └── analyzer.py           # Results analysis
│
├── experiments/              # Reproduce paper results
│   └── paper_eval/
│       └── generate_figures.py
│
├── config.yaml               # Configuration
├── main.py                   # Gradio entry point
└── cli.py                    # CLI entry point
```

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  model: "anthropic.claude-3-sonnet-20240229-v1:0"
  region: "us-west-2"

memory:
  storage_path: "./storage/memory"
  
learning:
  async_mode: true
  triggers: ["end_of_turn", "explicit_feedback"]

personalization:
  context_budget_tokens: 1000
```

## Reproducing Paper Results

```bash
# Run evaluation on MAPLE-Personas benchmark
python evaluation/run_evaluation.py --experiment paper_eval

# Generate figures
python experiments/paper_eval/generate_figures.py
```

## How It Works

### The Learning Loop

1. **User interacts** → Message processed by orchestrator
2. **Personalization retrieves** → User context from memory
3. **Response generated** → LLM with personalized context
4. **Learning extracts** → Insights from interaction (async)
5. **Memory stores** → New insights for future use

### Key Prompts

See Appendix A of the paper for the exact prompts used:
- **Learning Prompt**: Extracts facts, preferences, and behaviors
- **Personalization Context**: Structures user data for injection

## Evaluation

We use the [MAPLE-Personas](https://huggingface.co/datasets/prdeepakbabu/maple-personas) benchmark:
- 150 synthetic personas with 3-5 traits each
- 10-turn conversations (8 context + 2 evaluation)
- LLM-as-judge evaluation

Results show 14.6% improvement in personalization score (Cohen's d = 0.95).

## Citation

```bibtex
@inproceedings{maple2026,
  title={MAPLE: A Sub-Agent Architecture for Memory, Learning, and Personalization in Agentic AI Systems},
  author={Anonymous},
  booktitle={Proceedings of AAMAS},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
