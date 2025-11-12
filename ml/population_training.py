import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import copy
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .rl_core import PPOAgent, RLConfig
from .task_generation import DynamicTaskGenerator, TaskInstance, GameMode, TaskComplexity

@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""
    population_size: int = 8
    tournament_size: int = 3
    exploit_interval: int = 1000  # Steps between exploitation/exploitation
    explore_interval: int = 2000  # Steps between exploration
    mutation_scale: float = 0.2
    elite_fraction: float = 0.2
    performance_window: int = 100
    min_performance_threshold: float = 0.1
    hyperparameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'learning_rate': (1e-5, 1e-3),
        'clip_epsilon': (0.1, 0.3),
        'entropy_coef': (0.001, 0.1),
        'gae_lambda': (0.9, 0.99),
        'value_loss_coef': (0.3, 0.7)
    })

@dataclass
class AgentState:
    """State of an individual agent in the population."""
    agent_id: str
    agent: PPOAgent
    config: RLConfig
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    current_performance: float = 0.0
    generation: int = 0
    total_steps: int = 0
    last_exploit_step: int = 0
    last_explore_step: int = 0
    is_elite: bool = False

class PopulationBasedTrainer:
    """
    Population-Based Training framework that maintains a population of agents
    with different hyperparameters, periodically exploiting and exploring
    to optimize performance across diverse tasks.
    """
    
    def __init__(self, 
                 pbt_config: PBTConfig,
                 rl_config: RLConfig,
                 task_generator: DynamicTaskGenerator,
                 network_factory: Callable[[], torch.nn.Module],
                 device: str = 'cpu'):
        
        self.pbt_config = pbt_config
        self.rl_config = rl_config
        self.task_generator = task_generator
        self.network_factory = network_factory
        self.device = device
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Training statistics
        self.global_step = 0
        self.generation = 0
        self.performance_history = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_population(self) -> List[AgentState]:
        """Initialize population with diverse hyperparameters."""
        population = []
        
        for i in range(self.pbt_config.population_size):
            # Create diverse initial configurations
            config = self._sample_hyperparameters()
            
            # Create agent
            network = self.network_factory()
            agent = PPOAgent(network, config, self.device)
            
            # Create agent state
            agent_state = AgentState(
                agent_id=f"agent_{i:03d}",
                agent=agent,
                config=config,
                generation=0
            )
            
            population.append(agent_state)
        
        return population
    
    def _sample_hyperparameters(self) -> RLConfig:
        """Sample hyperparameters within bounds."""
        config = copy.deepcopy(self.rl_config)
        
        for param, (min_val, max_val) in self.pbt_config.hyperparameter_bounds.items():
            if hasattr(config, param):
                # Log-uniform sampling for learning rate
                if param == 'learning_rate':
                    log_min = np.log(min_val)
                    log_max = np.log(max_val)
                    value = np.exp(random.uniform(log_min, log_max))
                else:
                    value = random.uniform(min_val, max_val)
                
                setattr(config, param, value)
        
        return config
    
    def train(self, 
              total_steps: int,
              eval_interval: int = 1000,
              save_interval: int = 10000) -> Dict[str, Any]:
        """
        Main training loop for population-based training.
        
        Args:
            total_steps: Total number of training steps
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoint saves
            
        Returns:
            Training statistics and final population state
        """
        self.logger.info(f"Starting PBT with {self.pbt_config.population_size} agents for {total_steps} steps")
        
        start_time = time.time()
        
        while self.global_step < total_steps:
            # Generate tasks for current step
            tasks = self._generate_training_batch()
            
            # Train each agent on tasks
            for agent_state in self.population:
                self._train_agent_step(agent_state, tasks)
            
            # Periodic exploitation
            if self.global_step % self.pbt_config.exploit_interval == 0:
                self._exploitation_phase()
            
            # Periodic exploration
            if self.global_step % self.pbt_config.explore_interval == 0:
                self._exploration_phase()
            
            # Evaluation
            if self.global_step % eval_interval == 0:
                self._evaluate_population()
                self._log_progress()
            
            # Checkpoint saving
            if self.global_step % save_interval == 0:
                self._save_checkpoint()
            
            self.global_step += 1
        
        # Final evaluation
        self._evaluate_population()
        
        training_time = time.time() - start_time
        
        return {
            'total_steps': self.global_step,
            'training_time': training_time,
            'final_population': self._get_population_stats(),
            'performance_history': self.performance_history
        }
    
    def _train_agent_step(self, 
                         agent_state: AgentState, 
                         tasks: List[TaskInstance]):
        """Train a single agent for one step."""
        agent = agent_state.agent
        total_reward = 0
        
        for task in tasks:
            # Run episode on task
            episode_reward = self._run_episode(agent, task)
            total_reward += episode_reward
            
            # Update agent
            update_metrics = agent.update()
            
            # Track performance
            agent_state.total_steps += 1
        
        # Update performance
        avg_reward = total_reward / len(tasks)
        agent_state.current_performance = avg_reward
        agent_state.performance_history.append(avg_reward)
    
    def _run_episode(self, 
                    agent: PPOAgent, 
                    task: TaskInstance,
                    max_steps: Optional[int] = None) -> float:
        """Run a single episode and return total reward."""
        if max_steps is None:
            max_steps = task.max_time_steps
        
        # Initialize environment (simplified)
        state = self._reset_environment(task)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Select action
            action, log_prob, value = agent.act(state, task.goal_description)
            
            # Take action in environment
            next_state, reward, done, info = self._step_environment(state, action, task)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done, log_prob, value)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        return total_reward
    
    def _generate_training_batch(self) -> List[TaskInstance]:
        """Generate a batch of diverse training tasks."""
        batch_size = self.pbt_config.population_size
        
        # Generate tasks with varying complexity and modes
        tasks = []
        for i in range(batch_size):
            # Alternate between game modes for diversity
            if i % 3 == 0:
                mode = GameMode.SINGLE_PLAYER
            elif i % 3 == 1:
                mode = GameMode.COOPERATIVE
            else:
                mode = GameMode.COMPETITIVE
            
            task = self.task_generator.generate_task(mode)
            tasks.append(task)
        
        return tasks
    
    def _exploitation_phase(self):
        """Periodic exploitation: copy weights from better performers."""
        self.logger.info("Starting exploitation phase")
        
        # Sort population by performance
        sorted_population = sorted(
            self.population, 
            key=lambda x: np.mean(list(x.performance_history)) if x.performance_history else 0,
            reverse=True
        )
        
        # Identify elite agents
        num_elite = int(self.pbt_config.elite_fraction * len(sorted_population))
        elite_agents = sorted_population[:num_elite]
        
        for agent_state in sorted_population:
            if agent_state not in elite_agents:
                # Select better performer to copy from
                candidate = self._tournament_selection(agent_state, elite_agents)
                
                if candidate and self._should_exploit(agent_state, candidate):
                    # Copy weights
                    agent_state.agent.load_state_dict(candidate.agent.network.state_dict())
                    agent_state.last_exploit_step = self.global_step
                    agent_state.generation += 1
                    
                    self.logger.debug(f"Agent {agent_state.agent_id} exploited from {candidate.agent_id}")
        
        # Mark elite agents
        for agent_state in self.population:
            agent_state.is_elite = agent_state in elite_agents
    
    def _exploration_phase(self):
        """Periodic exploration: mutate hyperparameters."""
        self.logger.info("Starting exploration phase")
        
        for agent_state in self.population:
            if not agent_state.is_elite or random.random() < 0.1:  # 10% chance for elites to explore
                # Mutate hyperparameters
                old_config = agent_state.config
                new_config = self._mutate_hyperparameters(old_config)
                
                # Update agent configuration
                agent_state.config = new_config
                agent_state.last_explore_step = self.global_step
                
                # Reinitialize optimizer with new learning rate
                agent_state.agent.optimizer.param_groups[0]['lr'] = new_config.learning_rate
                
                self.logger.debug(f"Agent {agent_state.agent_id} explored new hyperparameters")
    
    def _tournament_selection(self, 
                              candidate: AgentState, 
                              elite_pool: List[AgentState]) -> Optional[AgentState]:
        """Select better performer using tournament selection."""
        tournament_size = min(self.pbt_config.tournament_size, len(elite_pool))
        tournament = random.sample(elite_pool, tournament_size)
        
        # Select best performer from tournament
        best = max(tournament, key=lambda x: np.mean(list(x.performance_history)) if x.performance_history else 0)
        
        # Return if better than candidate
        candidate_perf = np.mean(list(candidate.performance_history)) if candidate.performance_history else 0
        best_perf = np.mean(list(best.performance_history)) if best.performance_history else 0
        
        return best if best_perf > candidate_perf else None
    
    def _should_exploit(self, 
                       candidate: AgentState, 
                       source: AgentState) -> bool:
        """Determine if exploitation should occur based on performance gap."""
        candidate_perf = np.mean(list(candidate.performance_history)) if candidate.performance_history else 0
        source_perf = np.mean(list(source.performance_history)) if source.performance_history else 0
        
        performance_gap = source_perf - candidate_perf
        threshold = self.pbt_config.min_performance_threshold
        
        return performance_gap > threshold
    
    def _mutate_hyperparameters(self, config: RLConfig) -> RLConfig:
        """Mutate hyperparameters for exploration."""
        new_config = copy.deepcopy(config)
        
        for param, (min_val, max_val) in self.pbt_config.hyperparameter_bounds.items():
            if hasattr(new_config, param):
                current_value = getattr(new_config, param)
                
                # Apply multiplicative mutation
                mutation_factor = 1 + random.gauss(0, self.pbt_config.mutation_scale)
                new_value = current_value * mutation_factor
                
                # Clip to bounds
                new_value = np.clip(new_value, min_val, max_val)
                
                setattr(new_config, param, new_value)
        
        return new_config
    
    def _evaluate_population(self):
        """Evaluate all agents on held-out tasks."""
        # Generate evaluation tasks
        eval_tasks = self.task_generator.generate_task_batch(
            batch_size=len(self.population),
            game_mode=GameMode.SINGLE_PLAYER
        )
        
        # Evaluate each agent
        eval_results = []
        for i, agent_state in enumerate(self.population):
            task = eval_tasks[i]
            performance = self._evaluate_agent(agent_state.agent, task)
            eval_results.append({
                'agent_id': agent_state.agent_id,
                'performance': performance,
                'generation': agent_state.generation,
                'config': agent_state.config.__dict__
            })
        
        # Record performance history
        avg_performance = np.mean([r['performance'] for r in eval_results])
        self.performance_history.append({
            'global_step': self.global_step,
            'avg_performance': avg_performance,
            'best_performance': max([r['performance'] for r in eval_results]),
            'worst_performance': min([r['performance'] for r in eval_results]),
            'generation': self.generation
        })
        
        # Update task generator based on population performance
        population_metrics = {
            'success_rate': avg_performance,
            'efficiency': avg_performance  # Simplified
        }
        self.task_generator.update_difficulty_based_on_performance(population_metrics)
    
    def _evaluate_agent(self, 
                       agent: PPOAgent, 
                       task: TaskInstance,
                       num_episodes: int = 5) -> float:
        """Evaluate agent on held-out task."""
        total_rewards = []
        
        for _ in range(num_episodes):
            reward = self._run_episode(agent, task)
            total_rewards.append(reward)
        
        return np.mean(total_rewards)
    
    def _log_progress(self):
        """Log training progress."""
        if not self.performance_history:
            return
        
        latest = self.performance_history[-1]
        
        self.logger.info(
            f"Step {self.global_step}: "
            f"Avg Performance: {latest['avg_performance']:.3f}, "
            f"Best: {latest['best_performance']:.3f}, "
            f"Generation: {self.generation}"
        )
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'generation': self.generation,
            'population': [],
            'performance_history': self.performance_history,
            'task_generator_state': self.task_generator.get_complexity_stats()
        }
        
        for agent_state in self.population:
            agent_checkpoint = {
                'agent_id': agent_state.agent_id,
                'config': agent_state.config.__dict__,
                'performance_history': list(agent_state.performance_history),
                'generation': agent_state.generation,
                'total_steps': agent_state.total_steps
            }
            checkpoint['population'].append(agent_checkpoint)
        
        # Save to file
        filepath = f"pbt_checkpoint_step_{self.global_step}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def _get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about current population."""
        performances = [np.mean(list(agent.performance_history)) if agent.performance_history else 0 
                       for agent in self.population]
        
        return {
            'population_size': len(self.population),
            'avg_performance': np.mean(performances),
            'best_performance': np.max(performances),
            'worst_performance': np.min(performances),
            'performance_std': np.std(performances),
            'avg_generation': np.mean([agent.generation for agent in self.population]),
            'total_steps': sum([agent.total_steps for agent in self.population])
        }
    
    def _reset_environment(self, task: TaskInstance) -> Dict[str, Any]:
        """Reset environment for new episode (simplified)."""
        # This would interface with the actual game environment
        return {
            'grid': np.zeros((task.arena_config['grid_size'], task.arena_config['grid_size'])),
            'player_pos': task.arena_config['player_spawns'][0],
            'objects': task.arena_config['objects'].copy(),
            'goal_zones': task.arena_config['goal_zones'].copy()
        }
    
    def _step_environment(self, 
                          state: Dict[str, Any], 
                          action: int, 
                          task: TaskInstance) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Step environment with action (simplified)."""
        # This would interface with the actual game environment
        # For now, return placeholder values
        reward = random.uniform(-1, 1)
        done = random.random() < 0.1  # 10% chance of episode ending
        
        return state, reward, done, {}

class CurriculumScheduler:
    """Curriculum learning scheduler that coordinates with PBT."""
    
    def __init__(self, 
                 task_generator: DynamicTaskGenerator,
                 pbt_trainer: PopulationBasedTrainer):
        self.task_generator = task_generator
        self.pbt_trainer = pbt_trainer
        self.curriculum_stages = self._define_curriculum_stages()
        self.current_stage = 0
        
    def _define_curriculum_stages(self) -> List[Dict[str, Any]]:
        """Define curriculum stages with increasing difficulty."""
        return [
            {
                'name': 'basic_collection',
                'duration': 5000,
                'complexity_focus': [TaskComplexity.TRIVIAL, TaskComplexity.EASY],
                'goal_types': ['collect'],
                'target_performance': 0.6
            },
            {
                'name': 'intermediate_tasks',
                'duration': 10000,
                'complexity_focus': [TaskComplexity.EASY, TaskComplexity.MEDIUM],
                'goal_types': ['collect', 'bring_to_zone', 'avoid_walls'],
                'target_performance': 0.7
            },
            {
                'name': 'advanced_challenges',
                'duration': 15000,
                'complexity_focus': [TaskComplexity.MEDIUM, TaskComplexity.HARD],
                'goal_types': ['collect_multiple', 'touch_corners', 'cooperative_collect'],
                'target_performance': 0.8
            },
            {
                'name': 'expert_level',
                'duration': 20000,
                'complexity_focus': [TaskComplexity.HARD, TaskComplexity.EXPERT],
                'goal_types': ['competitive_collect', 'competitive_race', 'cooperative_zone'],
                'target_performance': 0.9
            }
        ]
    
    def update_curriculum(self, current_step: int, current_performance: float):
        """Update curriculum based on training progress."""
        if self.current_stage >= len(self.curriculum_stages):
            return
        
        current_stage_config = self.curriculum_stages[self.current_stage]
        
        # Check if ready to advance
        if (current_step >= current_stage_config['duration'] and 
            current_performance >= current_stage_config['target_performance']):
            
            self.current_stage += 1
            self._apply_curriculum_stage()
            
            logging.info(f"Advanced to curriculum stage: {self.curriculum_stages[self.current_stage]['name']}")
    
    def _apply_curriculum_stage(self):
        """Apply current curriculum stage to task generator."""
        if self.current_stage >= len(self.curriculum_stages):
            return
        
        stage_config = self.curriculum_stages[self.current_stage]
        
        # Update task generator complexity distribution
        new_distribution = {}
        for complexity in TaskComplexity:
            if complexity in stage_config['complexity_focus']:
                new_distribution[complexity] = 1.0 / len(stage_config['complexity_focus'])
            else:
                new_distribution[complexity] = 0.0
        
        self.task_generator.complexity_distribution = new_distribution

class MultiAgentCoordinator:
    """Coordinates multi-agent training scenarios."""
    
    def __init__(self, population: List[AgentState]):
        self.population = population
        self.cooperation_history = []
        self.competition_history = []
    
    def organize_cooperative_session(self, num_pairs: int = 2) -> List[Tuple[AgentState, AgentState]]:
        """Organize agents into cooperative pairs."""
        # Sort by performance for balanced pairing
        sorted_agents = sorted(self.population, 
                             key=lambda x: np.mean(list(x.performance_history)) if x.performance_history else 0)
        
        pairs = []
        for i in range(min(num_pairs, len(sorted_agents) // 2)):
            agent1 = sorted_agents[i]
            agent2 = sorted_agents[len(sorted_agents) - 1 - i]
            pairs.append((agent1, agent2))
        
        return pairs
    
    def organize_competitive_tournament(self) -> List[Tuple[AgentState, AgentState]]:
        """Organize competitive tournament bracket."""
        # Random pairing for competition
        shuffled = random.sample(self.population, len(self.population))
        pairs = []
        
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        
        return pairs
    
    def evaluate_cooperation(self, 
                           pair: Tuple[AgentState, AgentState], 
                           task: TaskInstance) -> Dict[str, float]:
        """Evaluate cooperative performance of agent pair."""
        # This would run actual cooperative episode
        # For now, return placeholder metrics
        return {
            'team_score': random.uniform(0, 1),
            'coordination_score': random.uniform(0, 1),
            'efficiency_score': random.uniform(0, 1)
        }
    
    def evaluate_competition(self, 
                           pair: Tuple[AgentState, AgentState], 
                           task: TaskInstance) -> Dict[str, float]:
        """Evaluate competitive performance of agent pair."""
        # This would run actual competitive episode
        # For now, return placeholder metrics
        return {
            'player1_score': random.uniform(0, 1),
            'player2_score': random.uniform(0, 1),
            'game_intensity': random.uniform(0, 1)
        }
