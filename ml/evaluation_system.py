import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from .rl_core import PPOAgent
from .task_generation import DynamicTaskGenerator, TaskInstance, GameMode, TaskComplexity, TaskEvaluator
from .neural_networks import GoalAttentiveAgent

@dataclass
class EvaluationConfig:
    """Configuration for evaluation system."""
    num_held_out_tasks: int = 100
    num_evaluation_episodes: int = 10
    parallel_evaluation: bool = True
    max_workers: int = 4
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    track_generalization_metrics: bool = True
    zero_shot_evaluation: bool = True

@dataclass
class EvaluationResult:
    """Results from evaluating an agent on tasks."""
    agent_id: str
    task_id: str
    goal_type: str
    complexity: TaskComplexity
    game_mode: GameMode
    success_rate: float
    average_reward: float
    efficiency_score: float
    generalization_score: float
    zero_shot_performance: float
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

class HeldOutTaskGenerator:
    """Generates held-out tasks for evaluation that are not used in training."""
    
    def __init__(self, 
                 seed: int = 12345,
                 difficulty_distribution: Optional[Dict[TaskComplexity, float]] = None):
        self.seed = seed
        np.random.seed(seed)
        
        # Held-out tasks have different distribution than training
        if difficulty_distribution is None:
            self.difficulty_distribution = {
                TaskComplexity.TRIVIAL: 0.05,
                TaskComplexity.EASY: 0.15,
                TaskComplexity.MEDIUM: 0.30,
                TaskComplexity.HARD: 0.30,
                TaskComplexity.EXPERT: 0.20
            }
        else:
            self.difficulty_distribution = difficulty_distribution
        
        # Human-designed probe tasks for specific capabilities
        self.probe_tasks = self._create_probe_tasks()
        
    def _create_probe_tasks(self) -> List[Dict[str, Any]]:
        """Create human-designed probe tasks to test specific capabilities."""
        return [
            {
                'name': 'complex_navigation_maze',
                'description': 'Navigate through a complex maze to reach the goal',
                'goal_type': 'touch_corners',
                'complexity': TaskComplexity.EXPERT,
                'arena_config': {
                    'grid_size': 15,
                    'walls': self._generate_maze_walls(15),
                    'objects': [],
                    'goal_zones': [{'x': 14, 'y': 14, 'color': 'red'}],
                    'player_spawns': [{'x': 0, 'y': 0, 'player_id': 0}]
                },
                'tests_capability': 'complex_navigation'
            },
            {
                'name': 'strategic_object_collection',
                'description': 'Strategically collect objects in optimal order',
                'goal_type': 'collect_multiple',
                'complexity': TaskComplexity.EXPERT,
                'arena_config': {
                    'grid_size': 12,
                    'walls': [],
                    'objects': [
                        {'x': 2, 'y': 2, 'color': 'red', 'shape': 'cube', 'id': 'obj_0'},
                        {'x': 9, 'y': 9, 'color': 'red', 'shape': 'sphere', 'id': 'obj_1'},
                        {'x': 2, 'y': 9, 'color': 'red', 'shape': 'cube', 'id': 'obj_2'},
                        {'x': 9, 'y': 2, 'color': 'red', 'shape': 'sphere', 'id': 'obj_3'}
                    ],
                    'goal_zones': [],
                    'player_spawns': [{'x': 5, 'y': 5, 'player_id': 0}]
                },
                'tests_capability': 'strategic_planning'
            },
            {
                'name': 'coordinated_cooperation',
                'description': 'Two agents must coordinate to achieve goal',
                'goal_type': 'cooperative_zone',
                'complexity': TaskComplexity.EXPERT,
                'arena_config': {
                    'grid_size': 10,
                    'walls': [
                        {'x': 4, 'y': 0, 'type': 'wall'}, {'x': 4, 'y': 1, 'type': 'wall'},
                        {'x': 4, 'y': 2, 'type': 'wall'}, {'x': 4, 'y': 3, 'type': 'wall'},
                        {'x': 5, 'y': 6, 'type': 'wall'}, {'x': 5, 'y': 7, 'type': 'wall'},
                        {'x': 5, 'y': 8, 'type': 'wall'}, {'x': 5, 'y': 9, 'type': 'wall'}
                    ],
                    'objects': [],
                    'goal_zones': [{'x': 7, 'y': 7, 'color': 'blue'}],
                    'player_spawns': [
                        {'x': 1, 'y': 1, 'player_id': 0},
                        {'x': 8, 'y': 8, 'player_id': 1}
                    ]
                },
                'tests_capability': 'cooperation'
            },
            {
                'name': 'competitive_strategy',
                'description': 'Outperform opponent in resource collection',
                'goal_type': 'competitive_collect',
                'complexity': TaskComplexity.EXPERT,
                'arena_config': {
                    'grid_size': 10,
                    'walls': [],
                    'objects': [
                        {'x': 1, 'y': 1, 'color': 'blue', 'shape': 'cube', 'id': 'obj_0'},
                        {'x': 8, 'y': 1, 'color': 'blue', 'shape': 'sphere', 'id': 'obj_1'},
                        {'x': 1, 'y': 8, 'color': 'blue', 'shape': 'cube', 'id': 'obj_2'},
                        {'x': 8, 'y': 8, 'color': 'blue', 'shape': 'sphere', 'id': 'obj_3'},
                        {'x': 4, 'y': 4, 'color': 'red', 'shape': 'cube', 'id': 'obj_4'},
                        {'x': 5, 'y': 5, 'color': 'red', 'shape': 'sphere', 'id': 'obj_5'}
                    ],
                    'goal_zones': [],
                    'player_spawns': [
                        {'x': 0, 'y': 0, 'player_id': 0},
                        {'x': 9, 'y': 9, 'player_id': 1}
                    ]
                },
                'tests_capability': 'competition'
            }
        ]
    
    def _generate_maze_walls(self, grid_size: int) -> List[Dict[str, Any]]:
        """Generate a maze pattern for probe tasks."""
        walls = []
        
        # Create maze-like structure
        for i in range(1, grid_size - 1, 2):
            for j in range(1, grid_size - 1):
                if i % 4 == 1:
                    if j != grid_size // 2:
                        walls.append({'x': i, 'y': j, 'type': 'wall'})
                else:
                    if j % 4 == 1:
                        walls.append({'x': i, 'y': j, 'type': 'wall'})
        
        return walls
    
    def generate_held_out_tasks(self, num_tasks: int) -> List[TaskInstance]:
        """Generate held-out tasks for evaluation."""
        tasks = []
        
        # Include probe tasks
        for probe_config in self.probe_tasks:
            if len(tasks) < num_tasks:
                task = TaskInstance(
                    task_id=f"probe_{probe_config['name']}",
                    goal_type=probe_config['goal_type'],
                    goal_description=probe_config['description'],
                    arena_config=probe_config['arena_config'],
                    complexity=probe_config['complexity'],
                    game_mode=GameMode.COOPERATIVE if 'cooperative' in probe_config['goal_type'] else GameMode.SINGLE_PLAYER,
                    difficulty_params={},
                    expected_solution_length=50,
                    max_time_steps=150,
                    reward_structure={'completion_reward': 100, 'step_penalty': -0.1}
                )
                tasks.append(task)
        
        # Generate additional procedural tasks
        while len(tasks) < num_tasks:
            complexity = self._sample_complexity()
            game_mode = np.random.choice(list(GameMode))
            
            # Create task with different parameters than training
            task = self._generate_diverse_task(complexity, game_mode)
            tasks.append(task)
        
        return tasks[:num_tasks]
    
    def _sample_complexity(self) -> TaskComplexity:
        """Sample complexity from held-out distribution."""
        complexities = list(self.difficulty_distribution.keys())
        probabilities = list(self.difficulty_distribution.values())
        return np.random.choice(complexities, p=probabilities)
    
    def _generate_diverse_task(self, 
                              complexity: TaskComplexity, 
                              game_mode: GameMode) -> TaskInstance:
        """Generate diverse task not seen during training."""
        # This would create tasks with novel combinations
        # For now, return a placeholder
        return TaskInstance(
            task_id=f"held_out_{np.random.randint(10000)}",
            goal_type='collect',
            goal_description='Collect the orange diamond.',  # Novel color/shape
            arena_config={'grid_size': 10, 'walls': [], 'objects': [], 'goal_zones': [], 'player_spawns': []},
            complexity=complexity,
            game_mode=game_mode,
            difficulty_params={},
            expected_solution_length=30,
            max_time_steps=90,
            reward_structure={'completion_reward': 50, 'step_penalty': -0.1}
        )

class GeneralizationEvaluator:
    """Evaluates agent generalization capabilities."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def evaluate_zero_shot_generalization(self, 
                                        agent: PPOAgent,
                                        held_out_tasks: List[TaskInstance]) -> Dict[str, float]:
        """
        Evaluate zero-shot generalization on held-out tasks.
        This tests the agent's ability to handle completely novel tasks.
        """
        results = defaultdict(list)
        
        for task in held_out_tasks:
            # Run evaluation episodes
            task_results = self._evaluate_agent_on_task(agent, task)
            
            # Collect metrics
            results['zero_shot_success_rates'].append(task_results.success_rate)
            results['zero_shot_rewards'].append(task_results.average_reward)
            results['zero_shot_efficiency'].append(task_results.efficiency_score)
            
            # Track by complexity
            results[f'complexity_{task.complexity.name}_success'].append(task_results.success_rate)
            
            # Track by goal type
            results[f'goal_type_{task.goal_type}_success'].append(task_results.success_rate)
        
        # Compute aggregate metrics
        generalization_metrics = {
            'overall_zero_shot_success': np.mean(results['zero_shot_success_rates']),
            'overall_zero_shot_reward': np.mean(results['zero_shot_rewards']),
            'overall_zero_shot_efficiency': np.mean(results['zero_shot_efficiency']),
            'zero_shot_consistency': 1 - np.std(results['zero_shot_success_rates']),  # Lower std = more consistent
        }
        
        # Add complexity-specific metrics
        for complexity in TaskComplexity:
            key = f'complexity_{complexity.name}_success'
            if key in results:
                generalization_metrics[f'zero_shot_{complexity.name}_success'] = np.mean(results[key])
        
        return generalization_metrics
    
    def evaluate_few_shot_adaptation(self, 
                                    agent: PPOAgent,
                                    adaptation_tasks: List[TaskInstance],
                                    adaptation_steps: int = 1000) -> Dict[str, float]:
        """
        Evaluate few-shot adaptation capability.
        Tests how quickly agent can adapt to new task types.
        """
        adaptation_results = []
        
        for task in adaptation_tasks:
            # Create a copy of the agent for adaptation
            adapted_agent = self._copy_agent(agent)
            
            # Baseline performance (zero-shot)
            baseline_result = self._evaluate_agent_on_task(adapted_agent, task, num_episodes=5)
            
            # Adaptation phase
            self._adapt_agent_to_task(adapted_agent, task, adaptation_steps)
            
            # Post-adaptation performance
            adapted_result = self._evaluate_agent_on_task(adapted_agent, task, num_episodes=5)
            
            # Calculate improvement
            improvement = adapted_result.success_rate - baseline_result.success_rate
            
            adaptation_results.append({
                'task_id': task.task_id,
                'baseline_success': baseline_result.success_rate,
                'adapted_success': adapted_result.success_rate,
                'improvement': improvement,
                'adaptation_efficiency': improvement / adaptation_steps
            })
        
        # Aggregate adaptation metrics
        avg_improvement = np.mean([r['improvement'] for r in adaptation_results])
        avg_adaptation_efficiency = np.mean([r['adaptation_efficiency'] for r in adaptation_results])
        
        return {
            'few_shot_avg_improvement': avg_improvement,
            'few_shot_adaptation_efficiency': avg_adaptation_efficiency,
            'few_shot_success_rate': np.mean([r['adapted_success'] for r in adaptation_results])
        }
    
    def evaluate_cross_task_transfer(self, 
                                    agent: PPOAgent,
                                    source_tasks: List[TaskInstance],
                                    target_tasks: List[TaskInstance]) -> Dict[str, float]:
        """
        Evaluate cross-task transfer learning.
        Tests if knowledge from source tasks helps with target tasks.
        """
        # Train agent on source tasks
        trained_agent = self._copy_agent(agent)
        self._train_agent_on_tasks(trained_agent, source_tasks, training_steps=5000)
        
        # Evaluate on target tasks
        transfer_results = []
        for task in target_tasks:
            result = self._evaluate_agent_on_task(trained_agent, task)
            transfer_results.append(result.success_rate)
        
        # Compare with baseline (untrained agent)
        baseline_results = []
        for task in target_tasks:
            result = self._evaluate_agent_on_task(agent, task)
            baseline_results.append(result.success_rate)
        
        transfer_gain = np.mean(transfer_results) - np.mean(baseline_results)
        
        return {
            'cross_task_transfer_gain': transfer_gain,
            'transfer_success_rate': np.mean(transfer_results),
            'baseline_success_rate': np.mean(baseline_results),
            'transfer_efficiency': transfer_gain / len(source_tasks)
        }
    
    def _evaluate_agent_on_task(self, 
                               agent: PPOAgent, 
                               task: TaskInstance,
                               num_episodes: int = None) -> EvaluationResult:
        """Evaluate agent on a single task."""
        if num_episodes is None:
            num_episodes = self.config.num_evaluation_episodes
        
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        for episode in range(num_episodes):
            reward, length, success = self._run_evaluation_episode(agent, task)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            successes.append(1 if success else 0)
        
        # Calculate metrics
        success_rate = np.mean(successes)
        average_reward = np.mean(episode_rewards)
        efficiency_score = self._calculate_efficiency(episode_rewards, episode_lengths)
        generalization_score = self._calculate_generalization_score(task, success_rate, efficiency_score)
        
        return EvaluationResult(
            agent_id=agent.agent_id if hasattr(agent, 'agent_id') else 'unknown',
            task_id=task.task_id,
            goal_type=task.goal_type,
            complexity=task.complexity,
            game_mode=task.game_mode,
            success_rate=success_rate,
            average_reward=average_reward,
            efficiency_score=efficiency_score,
            generalization_score=generalization_score,
            zero_shot_performance=success_rate,  # For held-out tasks, this is zero-shot
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths
        )
    
    def _run_evaluation_episode(self, 
                               agent: PPOAgent, 
                               task: TaskInstance) -> Tuple[float, int, bool]:
        """Run a single evaluation episode."""
        # Reset environment
        state = self._reset_environment(task)
        total_reward = 0
        steps = 0
        done = False
        max_steps = task.max_time_steps
        
        while not done and steps < max_steps:
            # Agent selects action
            action, log_prob, value = agent.act(state, task.goal_description, deterministic=True)
            
            # Environment step
            next_state, reward, done, info = self._step_environment(state, action, task)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        success = total_reward > 0  # Simplified success criterion
        
        return total_reward, steps, success
    
    def _calculate_efficiency(self, 
                             rewards: List[float], 
                             lengths: List[int]) -> float:
        """Calculate efficiency score based on reward per step."""
        if not rewards or not lengths:
            return 0.0
        
        total_reward = sum(rewards)
        total_steps = sum(lengths)
        
        if total_steps == 0:
            return 0.0
        
        return total_reward / total_steps
    
    def _calculate_generalization_score(self, 
                                       task: TaskInstance, 
                                       success_rate: float, 
                                       efficiency: float) -> float:
        """Calculate generalization score for novel tasks."""
        # Weight success more heavily for novel tasks
        complexity_weight = {
            TaskComplexity.TRIVIAL: 0.5,
            TaskComplexity.EASY: 0.7,
            TaskComplexity.MEDIUM: 1.0,
            TaskComplexity.HARD: 1.3,
            TaskComplexity.EXPERT: 1.5
        }
        
        weight = complexity_weight.get(task.complexity, 1.0)
        return (success_rate * 0.7 + efficiency * 0.3) * weight
    
    def _copy_agent(self, agent: PPOAgent) -> PPOAgent:
        """Create a copy of the agent for independent evaluation."""
        # Create new agent with same network
        new_network = GoalAttentiveAgent()
        new_network.load_state_dict(agent.network.state_dict())
        
        new_agent = PPOAgent(new_network, agent.config, agent.device)
        return new_agent
    
    def _adapt_agent_to_task(self, 
                            agent: PPOAgent, 
                            task: TaskInstance, 
                            adaptation_steps: int):
        """Adapt agent to specific task for few-shot learning."""
        for step in range(adaptation_steps):
            # Run episode and update
            reward = self._run_evaluation_episode(agent, task)[0]
            agent.update()
    
    def _train_agent_on_tasks(self, 
                             agent: PPOAgent, 
                             tasks: List[TaskInstance], 
                             training_steps: int):
        """Train agent on a set of tasks."""
        steps_per_task = training_steps // len(tasks)
        
        for task in tasks:
            for step in range(steps_per_task):
                self._run_evaluation_episode(agent, task)
                agent.update()
    
    def _reset_environment(self, task: TaskInstance) -> Dict[str, Any]:
        """Reset environment for task (placeholder)."""
        return {'grid': np.zeros((task.arena_config['grid_size'], task.arena_config['grid_size']))}
    
    def _step_environment(self, 
                          state: Dict[str, Any], 
                          action: int, 
                          task: TaskInstance) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Step environment (placeholder)."""
        reward = np.random.uniform(-1, 1)
        done = np.random.random() < 0.1
        return state, reward, done, {}

class EvaluationSystem:
    """Main evaluation system that coordinates all evaluation components."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.held_out_generator = HeldOutTaskGenerator()
        self.generalization_evaluator = GeneralizationEvaluator(config)
        self.task_evaluator = TaskEvaluator()
        self.logger = logging.getLogger(__name__)
        
    def comprehensive_evaluation(self, 
                                 agents: List[PPOAgent],
                                 training_tasks: List[TaskInstance]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of agent population.
        """
        self.logger.info(f"Starting comprehensive evaluation for {len(agents)} agents")
        
        # Generate held-out tasks
        held_out_tasks = self.held_out_generator.generate_held_out_tasks(
            self.config.num_held_out_tasks
        )
        
        # Evaluate each agent
        agent_results = {}
        for agent in agents:
            agent_id = getattr(agent, 'agent_id', f'agent_{len(agent_results)}')
            
            # Zero-shot generalization
            zero_shot_metrics = self.generalization_evaluator.evaluate_zero_shot_generalization(
                agent, held_out_tasks
            )
            
            # Few-shot adaptation (subset of tasks)
            adaptation_tasks = held_out_tasks[:10]  # Use 10 tasks for adaptation testing
            few_shot_metrics = self.generalization_evaluator.evaluate_few_shot_adaptation(
                agent, adaptation_tasks
            )
            
            # Cross-task transfer
            if len(training_tasks) > 10:
                source_tasks = training_tasks[:5]
                target_tasks = training_tasks[5:10]
                transfer_metrics = self.generalization_evaluator.evaluate_cross_task_transfer(
                    agent, source_tasks, target_tasks
                )
            else:
                transfer_metrics = {}
            
            # Detailed task evaluation
            detailed_results = self._detailed_task_evaluation(agent, held_out_tasks)
            
            agent_results[agent_id] = {
                'zero_shot_metrics': zero_shot_metrics,
                'few_shot_metrics': few_shot_metrics,
                'transfer_metrics': transfer_metrics,
                'detailed_results': detailed_results,
                'overall_score': self._calculate_overall_score(
                    zero_shot_metrics, few_shot_metrics, transfer_metrics
                )
            }
        
        # Attach normalized task metrics inspired by Nash baseline analysis
        nash_task_baselines = self._attach_normalized_metrics(agent_results)
        
        # Population-level analysis
        population_metrics = self._analyze_population_performance(agent_results)
        
        # Generate report
        evaluation_report = {
            'evaluation_config': self.config.__dict__,
            'held_out_task_summary': self._summarize_held_out_tasks(held_out_tasks),
            'nash_task_baselines': nash_task_baselines,
            'agent_results': agent_results,
            'population_metrics': population_metrics,
            'timestamp': time.time()
        }
        
        # Save results
        if self.config.save_detailed_results:
            self._save_evaluation_results(evaluation_report)
        
        # Generate visualizations
        if self.config.generate_visualizations:
            self._generate_evaluation_visualizations(evaluation_report)
        
        return evaluation_report
    
    def _detailed_task_evaluation(self, 
                                 agent: PPOAgent, 
                                 tasks: List[TaskInstance]) -> List[EvaluationResult]:
        """Perform detailed evaluation on each task."""
        results = []
        
        for task in tasks:
            result = self.generalization_evaluator._evaluate_agent_on_task(agent, task)
            
            # Add additional detailed metrics
            result.detailed_metrics = {
                'task_difficulty_score': self.task_evaluator.evaluate_task_difficulty(task)['overall_difficulty'],
                'goal_completion_time': np.mean(result.episode_lengths),
                'reward_variance': np.var(result.episode_rewards),
                'consistency_score': 1 - np.std(result.episode_rewards) / (np.mean(result.episode_rewards) + 1e-8)
            }
            
            results.append(result)
        
        return results
    
    def _calculate_overall_score(self, 
                               zero_shot: Dict[str, float],
                               few_shot: Dict[str, float],
                               transfer: Dict[str, float]) -> float:
        """Calculate overall evaluation score."""
        # Weight different components
        weights = {
            'zero_shot': 0.5,
            'few_shot': 0.3,
            'transfer': 0.2
        }
        
        zero_shot_score = zero_shot.get('overall_zero_shot_success', 0)
        few_shot_score = few_shot.get('few_shot_success_rate', 0)
        transfer_score = transfer.get('transfer_success_rate', 0)
        
        overall = (weights['zero_shot'] * zero_shot_score + 
                  weights['few_shot'] * few_shot_score + 
                  weights['transfer'] * transfer_score)
        
        return overall
    
    def _attach_normalized_metrics(self, agent_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute Nash-style baselines and normalized score percentiles per agent.
        Returns:
            Dictionary mapping task_id -> baseline score (mean success across population)
        """
        task_scores = defaultdict(list)
        
        # Gather raw success rates per task across all agents
        for data in agent_results.values():
            for result in data['detailed_results']:
                task_scores[result.task_id].append(result.success_rate)
        
        # Baseline approximation: mean success for each task
        baselines = {
            task_id: float(np.mean(scores)) if scores else 0.0
            for task_id, scores in task_scores.items()
        }
        
        percentile_targets = [10, 25, 50, 75, 90]
        
        for agent_id, data in agent_results.items():
            normalized_scores = []
            participation_hits = 0
            detailed_results = data.get('detailed_results', [])
            
            for result in detailed_results:
                baseline = baselines.get(result.task_id, 0.0)
                denom = abs(baseline) + 1e-6
                normalized_value = (result.success_rate - baseline) / denom
                normalized_scores.append(normalized_value)
                
                # Expose normalized score alongside other detailed metrics
                result.detailed_metrics['normalized_score'] = normalized_value
                
                if result.average_reward > 0:
                    participation_hits += 1
            
            if normalized_scores:
                percentile_summary = {
                    f'p{p}': float(np.percentile(normalized_scores, p))
                    for p in percentile_targets
                }
            else:
                percentile_summary = {f'p{p}': 0.0 for p in percentile_targets}
            
            participation_rate = participation_hits / max(1, len(detailed_results))
            
            data['normalized_score_percentiles'] = percentile_summary
            data['participation_rate'] = participation_rate
        
        return baselines
    
    def _analyze_population_performance(self, 
                                       agent_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze performance across the entire population."""
        overall_scores = [result['overall_score'] for result in agent_results.values()]
        zero_shot_scores = [result['zero_shot_metrics']['overall_zero_shot_success'] 
                           for result in agent_results.values()]
        participation_rates = [result.get('participation_rate', 0.0) 
                              for result in agent_results.values()]
        
        percentile_keys = set()
        for result in agent_results.values():
            percentile_keys.update(result.get('normalized_score_percentiles', {}).keys())
        
        percentile_summary = {}
        for key in percentile_keys:
            values = [result.get('normalized_score_percentiles', {}).get(key, 0.0)
                      for result in agent_results.values()]
            percentile_summary[key] = float(np.mean(values)) if values else 0.0
        
        return {
            'population_mean_score': np.mean(overall_scores),
            'population_std_score': np.std(overall_scores),
            'population_best_score': np.max(overall_scores),
            'population_worst_score': np.min(overall_scores),
            'population_mean_zero_shot': np.mean(zero_shot_scores),
            'population_std_zero_shot': np.std(zero_shot_scores),
            'generalization_gap': np.mean(zero_shot_scores) - np.mean(overall_scores),
            'population_percentile_summary': percentile_summary,
            'population_participation_mean': float(np.mean(participation_rates)) if participation_rates else 0.0,
            'population_participation_std': float(np.std(participation_rates)) if participation_rates else 0.0
        }
    
    def _summarize_held_out_tasks(self, tasks: List[TaskInstance]) -> Dict[str, Any]:
        """Summarize the held-out task distribution."""
        complexity_counts = defaultdict(int)
        goal_type_counts = defaultdict(int)
        game_mode_counts = defaultdict(int)
        
        for task in tasks:
            complexity_counts[task.complexity.name] += 1
            goal_type_counts[task.goal_type] += 1
            game_mode_counts[task.game_mode.value] += 1
        
        return {
            'total_tasks': len(tasks),
            'complexity_distribution': dict(complexity_counts),
            'goal_type_distribution': dict(goal_type_counts),
            'game_mode_distribution': dict(game_mode_counts)
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        timestamp = int(time.time())
        filepath = f"evaluation_results_{timestamp}.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return obj
    
    def _generate_evaluation_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots for evaluation results."""
        timestamp = int(time.time())
        
        # Performance distribution plot
        self._plot_performance_distribution(results, f"performance_dist_{timestamp}.png")
        
        # Generalization capabilities plot
        self._plot_generalization_capabilities(results, f"generalization_{timestamp}.png")
        
        # Task difficulty vs performance plot
        self._plot_difficulty_vs_performance(results, f"difficulty_performance_{timestamp}.png")
        
        self.logger.info(f"Evaluation visualizations saved with timestamp {timestamp}")
    
    def _plot_performance_distribution(self, results: Dict[str, Any], filepath: str):
        """Plot distribution of agent performance scores."""
        scores = [result['overall_score'] for result in results['agent_results'].values()]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=20, kde=True)
        plt.title('Distribution of Agent Performance Scores')
        plt.xlabel('Overall Score')
        plt.ylabel('Frequency')
        plt.savefig(filepath)
        plt.close()
    
    def _plot_generalization_capabilities(self, results: Dict[str, Any], filepath: str):
        """Plot generalization capabilities across different metrics."""
        agents = list(results['agent_results'].keys())
        zero_shot = [results['agent_results'][agent]['zero_shot_metrics']['overall_zero_shot_success'] 
                    for agent in agents]
        few_shot = [results['agent_results'][agent]['few_shot_metrics']['few_shot_success_rate'] 
                   for agent in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, zero_shot, width, label='Zero-shot', alpha=0.7)
        ax.bar(x + width/2, few_shot, width, label='Few-shot', alpha=0.7)
        
        ax.set_xlabel('Agents')
        ax.set_ylabel('Success Rate')
        ax.set_title('Generalization Capabilities')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _plot_difficulty_vs_performance(self, results: Dict[str, Any], filepath: str):
        """Plot relationship between task difficulty and performance."""
        # Collect all detailed results
        all_results = []
        for agent_data in results['agent_results'].values():
            all_results.extend(agent_data['detailed_results'])
        
        difficulties = [result.detailed_metrics.get('task_difficulty_score', 0) for result in all_results]
        performances = [result.success_rate for result in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(difficulties, performances, alpha=0.6)
        plt.xlabel('Task Difficulty Score')
        plt.ylabel('Success Rate')
        plt.title('Task Difficulty vs Performance')
        
        # Add trend line
        z = np.polyfit(difficulties, performances, 1)
        p = np.poly1d(z)
        plt.plot(difficulties, p(difficulties), "r--", alpha=0.8)
        
        plt.savefig(filepath)
        plt.close()
