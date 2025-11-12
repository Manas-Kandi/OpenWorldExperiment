"""
Open-Ended Learning Machine Learning Package

This package implements a comprehensive machine learning framework inspired by DeepMind's XLand
research for training generally capable agents through open-ended play.

Core Components:
- Neural Networks: Goal-Attentive Agent (GOAT) architecture with attention mechanisms
- RL Core: Deep reinforcement learning algorithms (PPO/A3C) with GAE
- Task Generation: Dynamic task generation with adaptive difficulty
- Population Training: Population-Based Training (PBT) framework
- Curriculum Learning: Adaptive curriculum scheduling and self-paced learning
- Evaluation System: Comprehensive evaluation with held-out tasks
- Behavior Analysis: Agent behavior analysis and visualization
- Multi-Agent Coordination: Team formation and role assignment
- Training Orchestrator: Main training loop coordination

Example Usage:
    from ml import TrainingOrchestrator, create_default_config
    
    config = create_default_config()
    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.train()
"""

from .neural_networks import GoalAttentiveAgent
from .rl_core import PPOAgent, A2CAgent, RLConfig
from .task_generation import DynamicTaskGenerator, TaskInstance, GameMode, TaskComplexity
from .population_training import PopulationBasedTrainer, PBTConfig, AgentState
from .curriculum_learning import AdaptiveCurriculumScheduler, CurriculumConfig, CurriculumType
from .evaluation_system import EvaluationSystem, EvaluationConfig, GeneralizationEvaluator
from .behavior_analysis import BehaviorAnalyzer, BehaviorVisualizer, BehaviorReporter
from .multi_agent_coordination import MultiAgentCoordinator, MultiAgentConfig, MultiAgentRole
from .training_orchestrator import (
    TrainingOrchestrator,
    TrainingConfig,
    create_default_config,
    create_research_config,
    create_quick_test_config,
    ExperimentRunner,
)
from .llm_bridge import (
    LLMGoalCoach,
    HybridLLMAgent,
    MiniQuestToolbox,
    SimpleMiniQuestController,
    GameToolbox,
    ToolSpec,
    ToolCall,
    ToolActionResult,
    LLMPlan,
)

__version__ = "1.0.0"
__author__ = "Open-Ended Learning Team"

# Export main classes for easy access
__all__ = [
    # Core classes
    'GoalAttentiveAgent',
    'PPOAgent',
    'A2CAgent',
    'DynamicTaskGenerator',
    'PopulationBasedTrainer',
    'AdaptiveCurriculumScheduler',
    'EvaluationSystem',
    'BehaviorAnalyzer',
    'BehaviorVisualizer',
    'BehaviorReporter',
    'MultiAgentCoordinator',
    'TrainingOrchestrator',
    'LLMGoalCoach',
    'HybridLLMAgent',
    'MiniQuestToolbox',
    'SimpleMiniQuestController',
    'GameToolbox',
    'ToolSpec',
    'ToolCall',
    'ToolActionResult',
    'LLMPlan',
    
    # Configuration classes
    'RLConfig',
    'PBTConfig',
    'CurriculumConfig',
    'EvaluationConfig',
    'MultiAgentConfig',
    'TrainingConfig',
    
    # Enums and data types
    'GameMode',
    'TaskComplexity',
    'CurriculumType',
    'MultiAgentRole',
    'TaskInstance',
    'AgentState',
    
    # Utility functions
    'create_default_config',
    'create_research_config',
    'create_quick_test_config',
    'ExperimentRunner'
]

def get_version():
    """Get package version."""
    return __version__

def get_component_info():
    """Get information about all components."""
    return {
        'neural_networks': {
            'main_class': 'GoalAttentiveAgent',
            'features': ['attention_mechanisms', 'goal_attentive_design', 'recurrent_state_processing'],
            'paper_reference': 'GOAL-ATTENTIVE AGENT (GOAT)'
        },
        'rl_core': {
            'main_classes': ['PPOAgent', 'A2CAgent'],
            'features': ['proximal_policy_optimization', 'gae_advantage_estimation', 'experience_replay'],
            'paper_reference': 'PPO, GAE'
        },
        'task_generation': {
            'main_class': 'DynamicTaskGenerator',
            'features': ['procedural_generation', 'adaptive_difficulty', 'curriculum_integration'],
            'paper_reference': 'XLand Task Generation'
        },
        'population_training': {
            'main_class': 'PopulationBasedTrainer',
            'features': ['population_based_training', 'hyperparameter_evolution', 'exploitation_exploration'],
            'paper_reference': 'PBT (Population Based Training)'
        },
        'curriculum_learning': {
            'main_class': 'AdaptiveCurriculumScheduler',
            'features': ['adaptive_curriculum', 'competency_tracking', 'self_paced_learning'],
            'paper_reference': 'Curriculum Learning for RL'
        },
        'evaluation_system': {
            'main_class': 'EvaluationSystem',
            'features': ['held_out_evaluation', 'zero_shot_generalization', 'few_shot_adaptation'],
            'paper_reference': 'Generalization Metrics in RL'
        },
        'behavior_analysis': {
            'main_classes': ['BehaviorAnalyzer', 'BehaviorVisualizer'],
            'features': ['behavior_clustering', 'emergent_behavior_detection', 'attention_analysis'],
            'paper_reference': 'Behavior Analysis in Multi-Agent Systems'
        },
        'multi_agent_coordination': {
            'main_class': 'MultiAgentCoordinator',
            'features': ['team_formation', 'role_assignment', 'coordination_learning'],
            'paper_reference': 'Multi-Agent Coordination Strategies'
        },
        'training_orchestrator': {
            'main_class': 'TrainingOrchestrator',
            'features': ['end_to_end_training', 'experiment_management', 'checkpointing'],
            'paper_reference': 'Training Orchestration for Large-Scale RL'
        }
    }

def quick_start_example():
    """
    Quick start example for the Open-Ended Learning framework.
    
    This example shows how to set up and run a basic training session.
    """
    print("Open-Ended Learning Framework - Quick Start Example")
    print("=" * 60)
    
    # Import required components
    from .training_orchestrator import create_quick_test_config
    
    # Create configuration for quick testing
    config = create_quick_test_config()
    config.total_training_steps = 1000  # Very short for demo
    config.output_dir = "quick_start_demo"
    
    print(f"Configuration: {config.total_training_steps} training steps")
    print(f"Output directory: {config.output_dir}")
    print(f"Population size: {config.pbt_config.population_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create and run orchestrator
    print("\nInitializing training orchestrator...")
    orchestrator = TrainingOrchestrator(config)
    
    print("Starting training...")
    results = orchestrator.train()
    
    print(f"\nTraining completed!")
    print(f"Best performance achieved: {results['best_performance']:.3f}")
    print(f"Total training time: {results['total_time']:.1f} seconds")
    
    # Print component information
    component_info = get_component_info()
    print(f"\nComponents used: {len(component_info)}")
    for component, info in component_info.items():
        print(f"  - {component}: {info['main_class']}")
    
    return results

if __name__ == "__main__":
    quick_start_example()
