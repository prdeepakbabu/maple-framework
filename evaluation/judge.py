"""LLM-as-judge for evaluating personalization quality."""

import json
import re
from typing import List, Optional

from .datasets.schemas import (
    TurnResult,
    ExperimentResult,
    TurnEvaluation,
    TraitConsistency,
    ConversationEvaluation,
    ScoredResults,
    EvaluationDataset,
)


class LLMJudge:
    """Evaluate personalization quality using LLM-as-judge."""
    
    JUDGE_PROMPT = """You are a STRICT evaluator assessing whether an AI assistant PROACTIVELY personalizes responses using remembered user information.

CRITICAL DISTINCTION:
- PROACTIVE personalization = Using information from PREVIOUS conversations/turns that is NOT in the current user message
- REACTIVE response = Simply responding to what the user explicitly said in their current message

You must distinguish between these. A high score requires PROACTIVE personalization.

PREVIOUSLY REVEALED USER TRAITS (from earlier in the conversation):
{revealed_traits}

CURRENT USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

STRICT EVALUATION CRITERIA:
1. First, check: Does the current user message EXPLICITLY mention any traits?
   - If YES, responding to those traits is just being responsive, NOT proactive personalization
   
2. Rate PROACTIVE personalization (1-5):
   5 = Explicitly references or applies traits NOT mentioned in the current message
   4 = Clearly adapts response based on remembered preferences without user prompting
   3 = Generic but acceptable response (no proactive personalization)
   2 = Misses clear opportunities to personalize based on known traits
   1 = Contradicts known traits OR provides inappropriate generic response

3. For each revealed trait, classify as:
   - PROACTIVELY_INCORPORATED: Trait used WITHOUT being mentioned in current message
   - REACTIVELY_USED: Trait was explicitly mentioned in user's current message
   - VIOLATED: Response contradicts this trait
   - MISSED: Trait was relevant but not incorporated
   - NEUTRAL: Trait not relevant to this context

OUTPUT FORMAT (JSON only):
{{
  "score": <1-5>,
  "traits_in_current_message": ["list traits explicitly mentioned in user message"],
  "trait_analysis": {{
    "<trait>": "<PROACTIVELY_INCORPORATED|REACTIVELY_USED|VIOLATED|MISSED|NEUTRAL>"
  }},
  "proactive_evidence": "<specific quote or evidence of proactive personalization, or 'none'>",
  "reasoning": "<brief explanation>"
}}"""

    def __init__(self, llm=None):
        """Initialize judge.
        
        Args:
            llm: LLM provider for evaluation (optional, can use mock)
        """
        self.llm = llm
        self.model_id = getattr(llm, 'model_id', 'mock') if llm else 'mock'
    
    async def evaluate_turn(
        self,
        turn: TurnResult,
    ) -> TurnEvaluation:
        """Evaluate a single turn."""
        
        revealed = turn.revealed_traits_so_far
        
        if self.llm is None:
            # Mock evaluation for testing
            return self._mock_evaluate(turn, revealed)
        
        # Format revealed traits
        if not revealed:
            traits_str = "(No traits revealed yet - response should be appropriately generic)"
        else:
            traits_str = "\n".join(f"- {t}" for t in revealed)
        
        prompt = self.JUDGE_PROMPT.format(
            revealed_traits=traits_str,
            user_message=turn.user_message,
            assistant_response=turn.assistant_response
        )
        
        try:
            response = await self.llm.generate(prompt)
            return self._parse_judge_response(response, turn, revealed)
        except Exception as e:
            print(f"Judge error: {e}")
            return self._mock_evaluate(turn, revealed)
    
    def _parse_judge_response(
        self,
        response: str,
        turn: TurnResult,
        revealed: List[str]
    ) -> TurnEvaluation:
        """Parse LLM judge response."""
        try:
            # Extract JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            # Build trait consistency
            incorporated = []
            violated = []
            neutral = []
            
            for trait, status in data.get("trait_analysis", {}).items():
                status_upper = status.upper()
                if "INCORPORATED" in status_upper:
                    incorporated.append(trait)
                elif "VIOLATED" in status_upper:
                    violated.append(trait)
                else:
                    neutral.append(trait)
            
            return TurnEvaluation(
                turn_number=turn.turn_number,
                traits_revealed_so_far=revealed,
                personalization_score=float(data["score"]),
                trait_consistency=TraitConsistency(
                    incorporated=incorporated,
                    violated=violated,
                    neutral=neutral
                ),
                judge_reasoning=data.get("reasoning", ""),
                judge_model=self.model_id
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: try to extract just the score
            score_match = re.search(r'"score"\s*:\s*(\d)', response)
            score = float(score_match.group(1)) if score_match else 3.0
            
            return TurnEvaluation(
                turn_number=turn.turn_number,
                traits_revealed_so_far=revealed,
                personalization_score=score,
                trait_consistency=TraitConsistency([], [], revealed),
                judge_reasoning=f"Parse error: {e}",
                judge_model=self.model_id
            )
    
    def _mock_evaluate(
        self,
        turn: TurnResult,
        revealed: List[str]
    ) -> TurnEvaluation:
        """Mock evaluation based on response content."""
        response = turn.assistant_response.lower()
        
        # Simple heuristic: check if response mentions personalization
        if "[personalized]" in response:
            score = 4.0
            incorporated = revealed.copy() if revealed else []
            violated = []
            neutral = []
            reasoning = "Response is marked as personalized"
        elif "[generic]" in response:
            score = 3.0
            incorporated = []
            violated = []
            neutral = revealed.copy() if revealed else []
            reasoning = "Response is generic"
        else:
            # Check if any traits are mentioned in response
            score = 3.0
            incorporated = []
            neutral = []
            
            for trait in revealed:
                trait_words = trait.lower().split()
                if any(word in response for word in trait_words if len(word) > 3):
                    incorporated.append(trait)
                    score = 4.0
                else:
                    neutral.append(trait)
            
            violated = []
            reasoning = f"Based on trait keyword matching"
        
        return TurnEvaluation(
            turn_number=turn.turn_number,
            traits_revealed_so_far=revealed,
            personalization_score=score,
            trait_consistency=TraitConsistency(
                incorporated=incorporated,
                violated=violated,
                neutral=neutral
            ),
            judge_reasoning=reasoning,
            judge_model="mock"
        )
    
    async def evaluate_experiment(
        self,
        result: ExperimentResult,
        dataset: EvaluationDataset,
        test_turns_only: bool = True
    ) -> ConversationEvaluation:
        """Evaluate turns in an experiment.
        
        Args:
            result: Experiment result containing all turns
            dataset: Evaluation dataset for reference
            test_turns_only: If True, only evaluate test turns (default).
                            This saves 80% of LLM judge calls since
                            learning turns are not used in final metrics.
        """
        
        turn_evals = []
        
        # Find the dataset conversation to get test turn info
        conv = dataset.get_conversation(result.conversation_id)
        
        for i, turn in enumerate(result.turns):
            # Determine if this is a test turn
            is_test_turn = False
            if conv and i < len(conv.turns):
                is_test_turn = conv.turns[i].is_test_turn
            
            # Skip non-test turns if optimization enabled
            if test_turns_only and not is_test_turn:
                continue
            
            eval_result = await self.evaluate_turn(turn)
            turn_evals.append(eval_result)
        
        return ConversationEvaluation(
            experiment_id=result.experiment_id,
            conversation_id=result.conversation_id,
            persona_id=result.persona_id,
            config=result.config,
            turn_evaluations=turn_evals
        )
    
    async def evaluate_all(
        self,
        results: List[ExperimentResult],
        dataset: EvaluationDataset
    ) -> ScoredResults:
        """Evaluate all experiment results."""
        
        evaluations = []
        total = len(results)
        
        for i, result in enumerate(results):
            try:
                eval_result = await self.evaluate_experiment(result, dataset)
                evaluations.append(eval_result)
                print(f"Evaluated {i+1}/{total}: {result.experiment_id}")
            except Exception as e:
                print(f"Failed to evaluate {result.experiment_id}: {e}")
        
        return ScoredResults(evaluations=evaluations)
    
    def evaluate_all_sync(
        self,
        results: List[ExperimentResult],
        dataset: EvaluationDataset
    ) -> ScoredResults:
        """Synchronous version using mock evaluation."""
        import asyncio
        return asyncio.run(self.evaluate_all(results, dataset))
