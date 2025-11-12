import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from .neural_networks import GoalAttentiveAgent
from .rl_core import PPOAgent, RLConfig
from .task_generation import DynamicTaskGenerator, GameMode
from .population_training import PopulationBasedTrainer, PBTConfig, AgentState
from .curriculum_learning import AdaptiveCurriculumScheduler, CurriculumConfig
from .evaluation_system import EvaluationSystem, EvaluationConfig
from .behavior_analysis import BehaviorAnalyzer, BehaviorVisualizer, BehaviorReporter

@dataclass
class TrainingConfig:
    """Main configuration for the entire training system."""
    # Training parameters
    total_training_steps: int = 1000000
    evaluation_interval: int = 10000
    checkpoint_interval: int = 50000
    log_interval: int = 1000
    
    # System components
    rl_config: RLConfig = None
    pbt_config: PBTConfig = None
    curriculum_config: CurriculumConfig = None
    evaluation_config: EvaluationConfig = None
    
    # Output settings
    output_dir: str = "training_output"
    save_checkpoints: bool = True
    save_behavior_data: bool = True
    generate_visualizations: bool = True
    
    def __post_init__(self):
        if self.rl_config is None:
            self.rl_config = RLConfig()
        if self.pbt_config is None:
            self.pbt_config = PBTConfig()
        if self.curriculum_config is None:
            self.curriculum_config = CurriculumConfig()
        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig()

class TrainingOrchestrator:
    """
    Main orchestrator that coordinates all components of the open-ended learning system.
    Integrates PBT, curriculum learning, evaluation, and behavior analysis.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.training_start_time = None
        self.current_step = 0
        self.best_performance = 0.0
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all training components."""
        self.logger.info("Initializing training components...")
        
        # Task generator
        self.task_generator = DynamicTaskGenerator(seed=42)
        
        # Population-based trainer
        def network_factory():
            return GoalAttentiveAgent(
                grid_size=10,
                hidden_dim=256,
                num_attention_heads=8
            ).to(self.device)
        
        self.pbt_trainer = PopulationBasedTrainer(
            pbt_config=self.config.pbt_config,
            rl_config=self.config.rl_config,
            task_generator=self.task_generator,
            network_factory=network_factory,
            device=self.device
        )
        
        # Curriculum scheduler
        self.curriculum_scheduler = AdaptiveCurriculumScheduler(
            config=self.config.curriculum_config,
            task_generator=self.task_generator
        )
        
        # Evaluation system
        self.evaluation_system = EvaluationSystem(self.config.evaluation_config)
        
        # Behavior analyzer
        self.behavior_analyzer = BehaviorAnalyzer(
            save_trajectories=self.config.save_behavior_data,
            analysis_window=1000
        )
        
        # Behavior visualizer and reporter
        self.behavior_visualizer = BehaviorVisualizer()
        self.behavior_reporter = BehaviorReporter(
            output_dir=f"{self.config.output_dir}/behavior_reports"
        )
        
        self.logger.info("All components initialized successfully")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop that orchestrates all components.
        
        Returns:
            Training results and statistics
        """
        self.logger.info(f"Starting training for {self.config.total_training_steps} steps")
        self.training_start_time = time.time()
        
        try:
            # Main training loop
            while self.current_step < self.config.total_training_steps:
                # Generate training batch
                training_tasks = self._generate_training_batch()
                
                # Train population
                training_metrics = self._train_population_step(training_tasks)
                
                # Update curriculum
                self._update_curriculum(training_metrics)
                
                # Collect behavior data
                self._collect_behavior_data(training_metrics)
                
                # Periodic evaluation
                if self.current_step % self.config.evaluation_interval == 0:
                    evaluation_results = self._perform_evaluation()
                    self._update_best_performance(evaluation_results)
                
                # Checkpoint saving
                if self.current_step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Logging
                if self.current_step % self.config.log_interval == 0:
                    self._log_progress(training_metrics)
                
                self.current_step += 1
            
            # Final evaluation and reporting
            final_results = self._finalize_training()
            
            self.logger.info("Training completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
    
    def _generate_training_batch(self) -> List[Any]:
        """Generate batch of training tasks."""
        batch_size = self.config.pbt_config.population_size
        
        # Generate diverse tasks based on current curriculum
        tasks = []
        for i in range(batch_size):
            # Vary game mode for diversity
            if i % 3 == 0:
                mode = GameMode.SINGLE_PLAYER
            elif i % 3 == 1:
                mode = GameMode.COOPERATIVE
            else:
                mode = GameMode.COMPETITIVE
            
            task = self.task_generator.generate_task(game_mode=mode)
            tasks.append(task)
        
        return tasks
    
    def _train_population_step(self, training_tasks: List[Any]) -> Dict[str, float]:
        """Train population for one step."""
        # This would interface with the actual PBT training
        # For now, simulate training metrics
        training_metrics = {
            'population_avg_reward': np.random.uniform(0.3, 0.8),
            'population_success_rate': np.random.uniform(0.4, 0.9),
            'curriculum_difficulty': np.random.uniform(0.2, 0.8),
            'behavior_diversity': np.random.uniform(0.5, 0.9)
        }
        
        return training_metrics
    
    def _update_curriculum(self, training_metrics: Dict[str, float]):
        """Update curriculum based on training progress."""
        # Get current performance metrics
        performance_metrics = {
            'success_rate': training_metrics['population_success_rate'],
            'efficiency': training_metrics['population_avg_reward'],
            'learning_progress': training_metrics['curriculum_difficulty']
        }
        
        # Update curriculum
        self.curriculum_scheduler.update_curriculum(
            current_step=self.current_step,
            agent_performance=performance_metrics,
            agent_states=self.pbt_trainer.population
        )
    
    def _collect_behavior_data(self, training_metrics: Dict[str, float]):
        """Collect and analyze behavior data."""
        if not self.config.save_behavior_data:
            return
        
        # Simulate episode data collection
        for agent_state in self.pbt_trainer.population[:3]:  # Sample a few agents
            episode_data = {
                'episode_id': self.current_step,
                'task_id': f'task_{self.current_step}',
                'goal_type': 'collect',  # Simplified
                'states': [],  # Would contain actual states
                'actions': [np.random.randint(0, 7) for _ in range(20)],
                'rewards': [np.random.uniform(-1, 1) for _ in range(20)],
                'positions': [(np.random.randint(0, 10), np.random.randint(0, 10)) for _ in range(20)],
                'goal_description': 'Collect the red cube',
                'success': np.random.random() > 0.5
            }
            
            self.behavior_analyzer.record_episode(
                agent_state.agent_id,
                episode_data
            )
    
    def _perform_evaluation(self) -> Dict[str, Any]:
        """Perform comprehensive evaluation."""
        self.logger.info(f"Performing evaluation at step {self.current_step}")
        
        # Get agents for evaluation
        agents = [agent_state.agent for agent_state in self.pbt_trainer.population]
        
        # Generate training tasks for transfer evaluation
        training_tasks = self.task_generator.generate_task_batch(20)
        
        # Perform comprehensive evaluation
        evaluation_results = self.evaluation_system.comprehensive_evaluation(
            agents=agents,
            training_tasks=training_tasks
        )
        
        # Log evaluation results
        population_metrics = evaluation_results['population_metrics']
        self.logger.info(
            f"Evaluation results - Mean score: {population_metrics['population_mean_score']:.3f}, "
            f"Zero-shot: {population_metrics['population_mean_zero_shot']:.3f}"
        )
        
        return evaluation_results
    
    def _update_best_performance(self, evaluation_results: Dict[str, Any]):
        """Update best performance tracking."""
        current_performance = evaluation_results['population_metrics']['population_mean_score']
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.logger.info(f"New best performance: {self.best_performance:.3f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        checkpoint_path = f"{self.config.output_dir}/checkpoints/checkpoint_step_{self.current_step}.json"
        
        checkpoint_data = {
            'step': self.current_step,
            'best_performance': self.best_performance,
            'training_time': time.time() - self.training_start_time,
            'curriculum_stage': self.curriculum_scheduler.current_stage_index,
            'population_size': len(self.pbt_trainer.population),
            'task_generator_state': self.task_generator.get_complexity_stats()
        }
        
        # Save checkpoint
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Save model weights for best agents
        if self.best_performance > 0.7:  # Save good models
            self._save_best_models()
        
        self.logger.info(f"Checkpoint saved at step {self.current_step}")
    
    def _save_best_models(self):
        """Save model weights for best performing agents."""
        # Sort agents by performance
        sorted_agents = sorted(
            self.pbt_trainer.population,
            key=lambda x: np.mean(list(x.performance_history)) if x.performance_history else 0,
            reverse=True
        )
        
        # Save top 3 agents
        for i, agent_state in enumerate(sorted_agents[:3]):
            model_path = f"{self.config.output_dir}/models/best_agent_{i}_step_{self.current_step}.pt"
            
            # Save model state
            torch.save({
                'network_state_dict': agent_state.agent.network.state_dict(),
                'config': agent_state.config.__dict__,
                'performance': np.mean(list(agent_state.performance_history)) if agent_state.performance_history else 0,
                'step': self.current_step
            }, model_path)
    
    def _log_progress(self, training_metrics: Dict[str, float]):
        """Log training progress."""
        elapsed_time = time.time() - self.training_start_time
        steps_per_second = self.current_step / elapsed_time if elapsed_time > 0 else 0
        
        # Get curriculum info
        curriculum_info = self.curriculum_scheduler.get_current_stage_info()
        
        self.logger.info(
            f"Step {self.current_step}/{self.config.total_training_steps} "
            f"({steps_per_second:.1f} steps/s) - "
            f"Reward: {training_metrics['population_avg_reward']:.3f}, "
            f"Success: {training_metrics['population_success_rate']:.3f}, "
            f"Stage: {curriculum_info.get('stage_name', 'Unknown')}, "
            f"Best: {self.best_performance:.3f}"
        )
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and generate final reports."""
        self.logger.info("Finalizing training...")
        
        # Final evaluation
        final_evaluation = self._perform_evaluation()
        
        # Generate behavior analysis reports
        behavior_reports = {}
        if self.config.save_behavior_data:
            for agent_state in self.pbt_trainer.population:
                behavior_report = self.behavior_analyzer.generate_behavior_report(
                    agent_state.agent_id
                )
                
                report_path = self.behavior_reporter.generate_comprehensive_report(
                    behavior_report,
                    include_visualizations=self.config.generate_visualizations
                )
                
                behavior_reports[agent_state.agent_id] = report_path
        
        # Generate final training summary
        training_summary = {
            'training_config': self.config.__dict__,
            'total_steps': self.current_step,
            'total_time': time.time() - self.training_start_time,
            'best_performance': self.best_performance,
            'final_evaluation': final_evaluation,
            'behavior_reports': behavior_reports,
            'curriculum_progress': self.curriculum_scheduler.generate_stage_report(),
            'task_generation_stats': self.task_generator.get_complexity_stats()
        }
        
        # Save final summary
        summary_path = f"{self.config.output_dir}/training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to {summary_path}")
        
        return training_summary

class ExperimentRunner:
    """Utility class for running different experimental configurations."""
    
    def __init__(self, base_output_dir: str = "experiments"):
        self.base_output_dir = base_output_dir
        self.logger = logging.getLogger(__name__)
    
    def run_ablation_study(self, 
                          base_config: TrainingConfig,
                          ablation_params: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Run ablation study by varying specific parameters.
        
        Args:
            base_config: Base training configuration
            ablation_params: Dictionary of parameters to vary and their values
            
        Returns:
            Results of all ablation experiments
        """
        results = {}
        
        for param_name, param_values in ablation_params.items():
            self.logger.info(f"Running ablation study for parameter: {param_name}")
            
            param_results = {}
            for value in param_values:
                # Create config for this experiment
                config = self._create_experiment_config(
                    base_config, param_name, value
                )
                
                # Run experiment
                experiment_name = f"{param_name}_{value}"
                output_dir = f"{self.base_output_dir}/{experiment_name}"
                config.output_dir = output_dir
                
                try:
                    orchestrator = TrainingOrchestrator(config)
                    result = orchestrator.train()
                    param_results[str(value)] = result
                    
                    self.logger.info(f"Completed experiment: {experiment_name}")
                    
                except Exception as e:
                    self.logger.error(f"Experiment {experiment_name} failed: {str(e)}")
                    param_results[str(value)] = {'error': str(e)}
            
            results[param_name] = param_results
        
        # Save ablation study results
        results_path = f"{self.base_output_dir}/ablation_study_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def run_comparison_study(self, 
                            configs: List[Tuple[str, TrainingConfig]]) -> Dict[str, Any]:
        """
        Run comparison study between different configurations.
        
        Args:
            configs: List of (name, config) tuples
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, config in configs:
            self.logger.info(f"Running comparison experiment: {name}")
            
            output_dir = f"{self.base_output_dir}/comparison_{name}"
            config.output_dir = output_dir
            
            try:
                orchestrator = TrainingOrchestrator(config)
                result = orchestrator.train()
                results[name] = result
                
                self.logger.info(f"Completed comparison experiment: {name}")
                
            except Exception as e:
                self.logger.error(f"Comparison experiment {name} failed: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Save comparison results
        results_path = f"{self.base_output_dir}/comparison_study_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _create_experiment_config(self, 
                                 base_config: TrainingConfig,
                                 param_name: str,
                                 param_value: Any) -> TrainingConfig:
        """Create experiment configuration by modifying specific parameter."""
        import copy
        
        config = copy.deepcopy(base_config)
        
        # Modify the specific parameter
        if hasattr(config, param_name):
            setattr(config, param_name, param_value)
        elif hasattr(config.rl_config, param_name):
            setattr(config.rl_config, param_name, param_value)
        elif hasattr(config.pbt_config, param_name):
            setattr(config.pbt_config, param_name, param_value)
        elif hasattr(config.curriculum_config, param_name):
            setattr(config.curriculum_config, param_name, param_value)
        elif hasattr(config.evaluation_config, param_name):
            setattr(config.evaluation_config, param_name, param_value)
        
        return config

# Utility functions for creating standard configurations
def create_default_config() -> TrainingConfig:
    """Create default training configuration."""
    return TrainingConfig(
        total_training_steps=100000,
        evaluation_interval=10000,
        checkpoint_interval=5000,
        log_interval=1000
    )

def create_research_config() -> TrainingConfig:
    """Create configuration for research experiments."""
    return TrainingConfig(
        total_training_steps=1000000,
        evaluation_interval=20000,
        checkpoint_interval=10000,
        log_interval=1000,
        rl_config=RLConfig(
            learning_rate=3e-4,
            clip_epsilon=0.2,
            entropy_coef=0.01
        ),
        pbt_config=PBTConfig(
            population_size=16,
            exploit_interval=2000,
            explore_interval=4000
        ),
        curriculum_config=CurriculumConfig(
            curriculum_type=CurriculumType.ADAPTIVE,
            evaluation_frequency=2000
        )
    )

def create_quick_test_config() -> TrainingConfig:
    """Create configuration for quick testing."""
    return TrainingConfig(
        total_training_steps=10000,
        evaluation_interval=2000,
        checkpoint_interval=5000,
        log_interval=500,
        pbt_config=PBTConfig(population_size=4),
        evaluation_config=EvaluationConfig(num_held_out_tasks=20)
    )

# Main execution function
def main():
    """Main execution function for running training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Open-Ended Learning Training')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'research', 'quick_test'],
                       help='Configuration preset')
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='Output directory')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Run specific experiment')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config == 'default':
        config = create_default_config()
    elif args.config == 'research':
        config = create_research_config()
    elif args.config == 'quick_test':
        config = create_quick_test_config()
    
    config.output_dir = args.output_dir
    
    # Run training or experiment
    if args.experiment == 'ablation':
        # Run ablation study
        experiment_runner = ExperimentRunner(f"{args.output_dir}/experiments")
        
        ablation_params = {
            'population_size': [4, 8, 16],
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'curriculum_type': ['linear', 'adaptive']
        }
        
        results = experiment_runner.run_ablation_study(config, ablation_params)
        print(f"Ablation study completed. Results saved to {args.output_dir}/experiments")
        
    else:
        # Run standard training
        orchestrator = TrainingOrchestrator(config)
        results = orchestrator.train()
        
        print(f"Training completed successfully!")
        print(f"Best performance: {results['best_performance']:.3f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
