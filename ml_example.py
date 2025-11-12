#!/usr/bin/env python3
"""
Open-Ended Learning Framework - Comprehensive Example

This script demonstrates the complete machine learning training system
inspired by DeepMind's XLand research for training generally capable agents.

Features demonstrated:
1. Goal-Attentive Agent (GOAT) neural network architecture
2. Deep reinforcement learning with PPO
3. Dynamic task generation with adaptive difficulty
4. Population-Based Training (PBT)
5. Curriculum learning with competency tracking
6. Comprehensive evaluation with held-out tasks
7. Behavior analysis and visualization
8. Multi-agent coordination and team formation
9. End-to-end training orchestration

Usage:
    python ml_example.py --mode quick_test    # Quick demonstration
    python ml_example.py --mode research      # Full research configuration
    python ml_example.py --mode ablation      # Run ablation study
    python ml_example.py --mode llm_demo      # Demonstrate LLM tool control
"""

import argparse
import logging
import time
import torch
import numpy as np
from pathlib import Path

# Import the ML framework
from ml import (
    TrainingOrchestrator, TrainingConfig,
    create_default_config, create_research_config, create_quick_test_config,
    ExperimentRunner, get_component_info, get_version
)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_example.log')
        ]
    )

def print_system_info():
    """Print system and framework information."""
    print("=" * 80)
    print("OPEN-ENDED LEARNING FRAMEWORK")
    print("=" * 80)
    print(f"Version: {get_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"NumPy Version: {np.__version__}")
    print("=" * 80)

def print_component_overview():
    """Print overview of all framework components."""
    print("\nFRAMEWORK COMPONENTS:")
    print("-" * 40)
    
    component_info = get_component_info()
    for component, info in component_info.items():
        print(f"\n{component.upper()}:")
        if 'main_class' in info:
            print(f"  Main Class: {info['main_class']}")
        elif 'main_classes' in info:
            classes = ', '.join(info['main_classes'])
            print(f"  Main Classes: {classes}")
        print(f"  Features: {', '.join(info['features'])}")
        print(f"  Reference: {info['paper_reference']}")

def run_quick_test():
    """Run a quick test of the framework."""
    print("\n" + "=" * 60)
    print("RUNNING QUICK TEST")
    print("=" * 60)
    
    # Create quick test configuration
    config = create_quick_test_config()
    config.total_training_steps = 500
    config.evaluation_interval = 250
    config.checkpoint_interval = 250
    config.output_dir = "quick_test_output"
    config.log_interval = 50
    
    print(f"Configuration:")
    print(f"  Training steps: {config.total_training_steps}")
    print(f"  Population size: {config.pbt_config.population_size}")
    print(f"  Evaluation interval: {config.evaluation_interval}")
    print(f"  Output directory: {config.output_dir}")
    
    # Create and run orchestrator
    print("\nInitializing training orchestrator...")
    start_time = time.time()
    
    try:
        orchestrator = TrainingOrchestrator(config)
        
        print("Starting training...")
        results = orchestrator.train()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nQuick test completed successfully!")
        print(f"Training time: {elapsed_time:.1f} seconds")
        print(f"Best performance: {results['best_performance']:.3f}")
        print(f"Final evaluation score: {results['final_evaluation']['population_metrics']['population_mean_score']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Quick test failed: {str(e)}")
        return None

def run_research_experiment():
    """Run a full research experiment."""
    print("\n" + "=" * 60)
    print("RUNNING RESEARCH EXPERIMENT")
    print("=" * 60)
    
    # Create research configuration
    config = create_research_config()
    config.total_training_steps = 50000  # Reduced for demo
    config.evaluation_interval = 5000
    config.checkpoint_interval = 10000
    config.output_dir = "research_experiment_output"
    config.log_interval = 1000
    
    print(f"Research Configuration:")
    print(f"  Training steps: {config.total_training_steps}")
    print(f"  Population size: {config.pbt_config.population_size}")
    print(f"  Curriculum type: {config.curriculum_config.curriculum_type.value}")
    print(f"  Held-out tasks: {config.evaluation_config.num_held_out_tasks}")
    
    # Create and run orchestrator
    print("\nInitializing research orchestrator...")
    start_time = time.time()
    
    try:
        orchestrator = TrainingOrchestrator(config)
        
        print("Starting research training...")
        results = orchestrator.train()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nResearch experiment completed!")
        print(f"Training time: {elapsed_time:.1f} seconds")
        print(f"Best performance: {results['best_performance']:.3f}")
        
        # Print detailed results
        final_eval = results['final_evaluation']['population_metrics']
        print(f"Final population metrics:")
        print(f"  Mean score: {final_eval['population_mean_score']:.3f}")
        print(f"  Zero-shot performance: {final_eval['population_mean_zero_shot']:.3f}")
        print(f"  Generalization gap: {final_eval['generalization_gap']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Research experiment failed: {str(e)}")
        return None

def run_ablation_study():
    """Run ablation study to analyze component contributions."""
    print("\n" + "=" * 60)
    print("RUNNING ABLATION STUDY")
    print("=" * 60)
    
    # Create base configuration
    base_config = create_quick_test_config()
    base_config.total_training_steps = 1000
    base_config.output_dir = "ablation_study"
    
    # Define ablation parameters
    ablation_params = {
        'population_size': [2, 4, 8],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'curriculum_type': ['linear', 'adaptive']
    }
    
    print(f"Ablation parameters:")
    for param, values in ablation_params.items():
        print(f"  {param}: {values}")
    
    # Create experiment runner
    experiment_runner = ExperimentRunner(base_config.output_dir)
    
    print("\nRunning ablation experiments...")
    start_time = time.time()
    
    try:
        results = experiment_runner.run_ablation_study(base_config, ablation_params)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nAblation study completed!")
        print(f"Total time: {elapsed_time:.1f} seconds")
        
        # Analyze results
        print("\nAblation Results Summary:")
        for param, param_results in results.items():
            print(f"\n{param}:")
            for value, result in param_results.items():
                if 'error' not in result:
                    best_perf = result.get('best_performance', 0)
                    print(f"  {value}: Best performance = {best_perf:.3f}")
                else:
                    print(f"  {value}: Failed - {result['error']}")
        
        return results
        
    except Exception as e:
        print(f"Ablation study failed: {str(e)}")
        return None

def run_llm_demo(goal_text: str, stream: bool = False):
    """Demonstrate LLM-powered tool control."""
    print("\n" + "=" * 60)
    print("RUNNING LLM TOOL DEMO")
    print("=" * 60)
    
    from ml import LLMGoalCoach, MiniQuestToolbox, SimpleMiniQuestController
    
    controller = SimpleMiniQuestController()
    toolbox = MiniQuestToolbox(controller)
    coach = LLMGoalCoach(toolbox)
    
    state_summary = controller.summarize_state()
    print(f"Goal: {goal_text}")
    print(f"Initial state: {state_summary}")
    
    plan = coach.propose_plan(goal_text, state_summary, stream=stream)
    
    print(f"\nLLM Thought: {plan.thought}")
    if not plan.actions:
        print("LLM produced no actions.")
        return
    
    for idx, action in enumerate(plan.actions, 1):
        try:
            result = toolbox.execute(action)
            print(f"Step {idx}: tool={action.tool}, obs={result.observation}")
        except Exception as exc:
            print(f"Tool execution failed: {exc}")
            break

def demonstrate_components():
    """Demonstrate individual framework components."""
    print("\n" + "=" * 60)
    print("COMPONENT DEMONSTRATIONS")
    print("=" * 60)
    
    # Import individual components
    from ml import (
        GoalAttentiveAgent, DynamicTaskGenerator, 
        PopulationBasedTrainer, AdaptiveCurriculumScheduler,
        EvaluationSystem, BehaviorAnalyzer, MultiAgentCoordinator
    )
    
    print("\n1. Goal-Attentive Agent (GOAT)")
    print("-" * 30)
    
    # Create GOAT network
    goat = GoalAttentiveAgent(
        grid_size=10,
        hidden_dim=128,
        num_attention_heads=4
    )
    
    print(f"Created GOAT network with {sum(p.numel() for p in goat.parameters())} parameters")
    
    # Test forward pass
    grid_state = torch.randn(1, 10, 10, 15)  # Batch of 1, 10x10 grid, 15 features
    goal_text = "Collect the red cube"
    
    output = goat(grid_state, goal_text)
    print(f"Forward pass successful:")
    print(f"  Policy logits shape: {output['policy_logits'].shape}")
    print(f"  Value shape: {output['value'].shape}")
    print(f"  Attention weights shape: {output['attention_weights'].shape}")
    
    print("\n2. Dynamic Task Generation")
    print("-" * 30)
    
    task_generator = DynamicTaskGenerator(seed=42)
    
    # Generate different types of tasks
    tasks = []
    for mode in ['single', 'cooperative', 'competitive']:
        task = task_generator.generate_task(game_mode=mode)
        tasks.append(task)
        print(f"Generated {mode} task: {task.goal_description}")
        print(f"  Complexity: {task.complexity.name}")
        print(f"  Expected steps: {task.expected_solution_length}")
    
    print("\n3. Curriculum Learning")
    print("-" * 30)
    
    from ml.curriculum_learning import CurriculumConfig, CurriculumType
    
    curriculum_config = CurriculumConfig(
        curriculum_type=CurriculumType.ADAPTIVE,
        evaluation_frequency=100
    )
    
    curriculum_scheduler = AdaptiveCurriculumScheduler(
        config=curriculum_config,
        task_generator=task_generator
    )
    
    # Simulate curriculum updates
    for step in [1000, 5000, 10000]:
        performance = {'success_rate': 0.3 + step / 10000, 'efficiency': 0.4 + step / 15000}
        curriculum_scheduler.update_curriculum(step, performance, [])
        
        stage_info = curriculum_scheduler.get_current_stage_info()
        print(f"Step {step}: Stage = {stage_info.get('stage_name', 'Unknown')}")
        print(f"  Progress: {stage_info.get('progress_percentage', 0):.1f}%")
    
    print("\n4. Behavior Analysis")
    print("-" * 30)
    
    behavior_analyzer = BehaviorAnalyzer(save_trajectories=True)
    
    # Simulate episode data
    episode_data = {
        'episode_id': 1,
        'task_id': 'demo_task',
        'goal_type': 'collect',
        'actions': [0, 1, 2, 3, 4, 5] * 5,  # Movement and interaction actions
        'positions': [(i % 10, i // 10) for i in range(30)],
        'rewards': [0.1, -0.1, 0.2, 0.0, 1.0, -0.1] * 5,
        'goal_description': 'Collect the red cube',
        'success': True
    }
    
    behavior_analyzer.record_episode('demo_agent', episode_data)
    
    # Analyze behavior
    movement_analysis = behavior_analyzer.analyze_movement_patterns('demo_agent')
    goal_strategies = behavior_analyzer.analyze_goal_strategies('demo_agent')
    
    print(f"Movement analysis:")
    for metric, value in movement_analysis.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"Goal strategies: {list(goal_strategies.keys())}")
    
    print("\n5. Multi-Agent Coordination")
    print("-" * 30)
    
    from ml.multi_agent_coordination import MultiAgentConfig, MultiAgentRole
    from ml.population_training import AgentState
    from ml.rl_core import RLConfig, PPOAgent
    
    # Create mock population
    population = []
    for i in range(4):
        network = GoalAttentiveAgent(hidden_dim=64)
        agent = PPOAgent(network, RLConfig(), 'cpu')
        agent_state = AgentState(
            agent_id=f'agent_{i}',
            agent=agent,
            config=RLConfig()
        )
        population.append(agent_state)
    
    multi_config = MultiAgentConfig(max_team_size=3)
    coordinator = MultiAgentCoordinator(multi_config, population)
    
    # Form teams
    teams = coordinator.form_teams(current_step=0, task_mode='cooperative')
    
    print(f"Formed {len(teams)} teams:")
    for team in teams:
        print(f"  Team {team.team_id}: {len(team.members)} members")
        for agent_id, role in team.role_assignments.items():
            print(f"    {agent_id}: {role.value}")
    
    print("\nComponent demonstrations completed!")

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(
        description='Open-Ended Learning Framework Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_example.py --mode quick_test     # Quick demonstration
  python ml_example.py --mode research       # Full research experiment
  python ml_example.py --mode ablation       # Run ablation study
  python ml_example.py --mode components     # Demonstrate components
  python ml_example.py --mode llm_demo       # Run LLM planning/tooling demo
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick_test', 'research', 'ablation', 'components', 'llm_demo'],
        default='quick_test',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ml_example_output',
        help='Base output directory'
    )
    parser.add_argument(
        '--goal',
        type=str,
        default='Collect the purple cube without touching walls.',
        help='Goal description for llm_demo mode'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print system information
    print_system_info()
    print_component_overview()
    
    # Run based on mode
    if args.mode == 'quick_test':
        results = run_quick_test()
    elif args.mode == 'research':
        results = run_research_experiment()
    elif args.mode == 'ablation':
        results = run_ablation_study()
    elif args.mode == 'components':
        demonstrate_components()
        results = None
    elif args.mode == 'llm_demo':
        run_llm_demo(args.goal)
        results = None
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXAMPLE EXECUTION SUMMARY")
    print("=" * 80)
    
    if results and 'best_performance' in results:
        print(f"Best performance achieved: {results['best_performance']:.3f}")
        print(f"Output saved to: {args.output_dir}")
    
    print("Framework components successfully demonstrated!")
    print("Check the output directories for detailed results and visualizations.")
    
    # Next steps
    print("\nNEXT STEPS:")
    print("1. Examine the generated training logs and checkpoints")
    print("2. Review the behavior analysis reports and visualizations")
    print("3. Analyze the evaluation results and generalization metrics")
    print("4. Experiment with different configurations and parameters")
    print("5. Extend the framework for your specific research needs")
    
    print("\nFor more information, see the documentation in the ml/ directory.")

if __name__ == "__main__":
    main()
