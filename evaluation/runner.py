"""Experiment runner for evaluation."""

import time
from datetime import datetime
from typing import List, Optional

from .datasets.schemas import (
    EvaluationDataset,
    EvaluationConversation,
    AblationConfig,
    TurnResult,
    ExperimentResult,
)


class ExperimentRunner:
    """Run evaluation experiments under different ablation configs."""
    
    def __init__(self, orchestrator_factory=None):
        """Initialize runner.
        
        Args:
            orchestrator_factory: Callable that creates orchestrator with config.
                                  Signature: (learning_enabled, p13n_enabled) -> Orchestrator
        """
        self.orchestrator_factory = orchestrator_factory
    
    def _create_mock_orchestrator(
        self,
        learning_enabled: bool,
        personalization_enabled: bool
    ):
        """Create a mock orchestrator for testing."""
        
        class MockOrchestrator:
            def __init__(self, learning, p13n):
                self.learning_enabled = learning
                self.personalization_enabled = p13n
                self._learned_insights = {}
                self._last_p13n_context = ""
            
            async def process_message(self, user_id: str, message: str) -> str:
                """Generate a mock response."""
                # Simulate learning
                if self.learning_enabled:
                    if user_id not in self._learned_insights:
                        self._learned_insights[user_id] = []
                    # Extract potential traits from message
                    if "I am" in message or "I'm" in message or "I have" in message:
                        self._learned_insights[user_id].append(message)
                
                # Generate response based on config
                if self.personalization_enabled and self.learning_enabled:
                    insights = self._learned_insights.get(user_id, [])
                    if insights:
                        self._last_p13n_context = f"Based on: {insights}"
                        return f"[Personalized] Taking into account your preferences: {message}"
                    return f"[Generic] {message}"
                else:
                    self._last_p13n_context = ""
                    return f"[Generic] Here's a response to: {message}"
            
            def get_learned_insights(self, user_id: str) -> List[str]:
                return self._learned_insights.get(user_id, [])
            
            def get_last_personalization_context(self) -> str:
                return self._last_p13n_context
            
            def reset_user(self, user_id: str):
                if user_id in self._learned_insights:
                    del self._learned_insights[user_id]
        
        return MockOrchestrator(learning_enabled, personalization_enabled)
    
    def _get_orchestrator(self, config: AblationConfig):
        """Get orchestrator for given config."""
        if self.orchestrator_factory:
            return self.orchestrator_factory(
                config.learning_enabled,
                config.personalization_enabled
            )
        return self._create_mock_orchestrator(
            config.learning_enabled,
            config.personalization_enabled
        )
    
    async def run_conversation(
        self,
        conversation: EvaluationConversation,
        config: AblationConfig,
        experiment_id: str
    ) -> ExperimentResult:
        """Run a single conversation under a config."""
        
        orchestrator = self._get_orchestrator(config)
        user_id = f"eval_user_{conversation.persona.persona_id}"
        
        # Reset user state
        if hasattr(orchestrator, 'reset_user'):
            orchestrator.reset_user(user_id)
        
        turn_results = []
        
        for turn in conversation.turns:
            # Time the response
            start_time = time.perf_counter()
            
            response = await orchestrator.process_message(
                user_id=user_id,
                message=turn.user_message
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Capture internal state
            learned_insights = []
            p13n_context = ""
            
            if hasattr(orchestrator, 'get_learned_insights'):
                learned_insights = orchestrator.get_learned_insights(user_id)
            if hasattr(orchestrator, 'get_last_personalization_context'):
                p13n_context = orchestrator.get_last_personalization_context()
            
            turn_results.append(TurnResult(
                turn_number=turn.turn_number,
                user_message=turn.user_message,
                assistant_response=response,
                latency_ms=latency_ms,
                learned_insights_after=list(learned_insights),
                personalization_context_used=p13n_context,
                revealed_traits_so_far=conversation.get_revealed_traits_at_turn(turn.turn_number)
            ))
        
        return ExperimentResult(
            experiment_id=experiment_id,
            config=config.value,
            conversation_id=conversation.conversation_id,
            persona_id=conversation.persona.persona_id,
            turns=turn_results,
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def run_ablation(
        self,
        dataset: EvaluationDataset,
        configs: List[AblationConfig] = None
    ) -> List[ExperimentResult]:
        """Run all conversations under specified configs."""
        
        if configs is None:
            # Default: only baseline and full for simple comparison
            configs = [AblationConfig.BASELINE, AblationConfig.FULL]
        
        all_results = []
        total_experiments = len(configs) * len(dataset.conversations)
        current = 0
        
        for config in configs:
            print(f"\nRunning config: {config.value}")
            
            for i, conversation in enumerate(dataset.conversations):
                exp_id = f"exp_{config.value}_{conversation.conversation_id}"
                current += 1
                
                try:
                    result = await self.run_conversation(
                        conversation, config, exp_id
                    )
                    all_results.append(result)
                    print(f"  [{current}/{total_experiments}] {exp_id}")
                except Exception as e:
                    print(f"  Failed {exp_id}: {e}")
        
        return all_results
    
    def run_ablation_sync(
        self,
        dataset: EvaluationDataset,
        configs: List[AblationConfig] = None
    ) -> List[ExperimentResult]:
        """Synchronous version of run_ablation using mock orchestrator."""
        import asyncio
        return asyncio.run(self.run_ablation(dataset, configs))


def save_results(results: List[ExperimentResult], path: str):
    """Save experiment results to JSONL file."""
    import json
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + '\n')


def load_results(path: str) -> List[ExperimentResult]:
    """Load experiment results from JSONL file."""
    import json
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(ExperimentResult.from_dict(data))
    return results
