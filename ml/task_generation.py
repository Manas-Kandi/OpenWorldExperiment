import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class TaskComplexity(Enum):
    """Complexity levels for generated tasks."""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5

class GameMode(Enum):
    """Different game modes for task generation."""
    SINGLE_PLAYER = "single"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"

@dataclass
class TaskInstance:
    """Represents a single generated task."""
    task_id: str
    goal_type: str
    goal_description: str
    arena_config: Dict[str, Any]
    complexity: TaskComplexity
    game_mode: GameMode
    difficulty_params: Dict[str, Any]
    expected_solution_length: int
    max_time_steps: int
    reward_structure: Dict[str, float]

class DynamicTaskGenerator:
    """
    Dynamic task generation system that creates tasks with adaptive difficulty
    based on agent performance. Implements curriculum learning through
    automatic difficulty adjustment.
    """
    
    def __init__(self, 
                 seed: int = 42,
                 complexity_distribution: Optional[Dict[TaskComplexity, float]] = None):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Complexity distribution (can be updated based on performance)
        if complexity_distribution is None:
            self.complexity_distribution = {
                TaskComplexity.TRIVIAL: 0.1,
                TaskComplexity.EASY: 0.3,
                TaskComplexity.MEDIUM: 0.4,
                TaskComplexity.HARD: 0.15,
                TaskComplexity.EXPERT: 0.05
            }
        else:
            self.complexity_distribution = complexity_distribution
        
        # Goal templates with placeholders
        self.goal_templates = {
            'collect': {
                'template': 'Collect the {color} {shape}.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'num_objects': 3, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'num_objects': 5, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'num_objects': 8, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'num_objects': 12, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'num_objects': 20, 'num_walls': 25}
                }
            },
            'bring_to_zone': {
                'template': 'Bring the {color} {shape} to the {zone_color} goal zone.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'num_objects': 2, 'num_zones': 1, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'num_objects': 4, 'num_zones': 2, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'num_objects': 6, 'num_zones': 3, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'num_objects': 10, 'num_zones': 4, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'num_objects': 15, 'num_zones': 5, 'num_walls': 25}
                }
            },
            'avoid_walls': {
                'template': 'Avoid the {color} wall for {moves} moves.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'moves_required': 5, 'num_walls': 1},
                    TaskComplexity.EASY: {'grid_size': 8, 'moves_required': 10, 'num_walls': 3},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'moves_required': 15, 'num_walls': 5},
                    TaskComplexity.HARD: {'grid_size': 12, 'moves_required': 25, 'num_walls': 8},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'moves_required': 40, 'num_walls': 12}
                }
            },
            'touch_corners': {
                'template': 'Touch all corners before time runs out.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'time_limit': 30, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'time_limit': 45, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'time_limit': 60, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'time_limit': 90, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'time_limit': 120, 'num_walls': 25}
                }
            },
            'collect_multiple': {
                'template': 'Collect all {color} objects.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'target_count': 2, 'num_objects': 4, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'target_count': 3, 'num_objects': 8, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'target_count': 5, 'num_objects': 12, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'target_count': 8, 'num_objects': 18, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'target_count': 12, 'num_objects': 25, 'num_walls': 25}
                }
            },
            'cooperative_collect': {
                'template': 'Work together to collect {count} objects.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'target_count': 3, 'num_objects': 6, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'target_count': 5, 'num_objects': 10, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'target_count': 8, 'num_objects': 15, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'target_count': 12, 'num_objects': 20, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'target_count': 18, 'num_objects': 30, 'num_walls': 25}
                }
            },
            'competitive_collect': {
                'template': 'Collect more objects than your opponent!',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'win_threshold': 3, 'num_objects': 8, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'win_threshold': 5, 'num_objects': 15, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'win_threshold': 8, 'num_objects': 20, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'win_threshold': 12, 'num_objects': 30, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'win_threshold': 18, 'num_objects': 40, 'num_walls': 25}
                }
            },
            'cooperative_zone': {
                'template': 'Both players reach the {color} goal zone.',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'num_zones': 1, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'num_zones': 2, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'num_zones': 3, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'num_zones': 4, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'num_zones': 5, 'num_walls': 25}
                }
            },
            'competitive_race': {
                'template': 'Be the first to touch all corners!',
                'complexity_map': {
                    TaskComplexity.TRIVIAL: {'grid_size': 5, 'time_limit': 30, 'num_walls': 2},
                    TaskComplexity.EASY: {'grid_size': 8, 'time_limit': 45, 'num_walls': 4},
                    TaskComplexity.MEDIUM: {'grid_size': 10, 'time_limit': 60, 'num_walls': 8},
                    TaskComplexity.HARD: {'grid_size': 12, 'time_limit': 90, 'num_walls': 15},
                    TaskComplexity.EXPERT: {'grid_size': 15, 'time_limit': 120, 'num_walls': 25}
                }
            }
        }
        
        # Performance tracking for adaptation
        self.performance_history = []
        self.task_generation_count = 0
        
    def generate_task(self, 
                      game_mode: GameMode = GameMode.SINGLE_PLAYER,
                      target_complexity: Optional[TaskComplexity] = None) -> TaskInstance:
        """
        Generate a single task with specified or adaptive complexity.
        
        Args:
            game_mode: Mode of play (single, cooperative, competitive)
            target_complexity: Specific complexity level or None for adaptive
            
        Returns:
            Generated task instance
        """
        # Select complexity
        if target_complexity is None:
            complexity = self._sample_complexity()
        else:
            complexity = target_complexity
        
        # Select goal type based on game mode
        goal_type = self._select_goal_type(game_mode)
        
        # Generate task parameters
        task_params = self._generate_task_parameters(goal_type, complexity)
        
        # Create arena configuration
        arena_config = self._generate_arena_config(goal_type, complexity, task_params)
        
        # Generate goal description
        goal_description = self._generate_goal_description(goal_type, task_params)
        
        # Calculate expected difficulty metrics
        expected_solution_length = self._estimate_solution_length(goal_type, complexity)
        max_time_steps = expected_solution_length * 3  # Allow 3x expected time
        reward_structure = self._generate_reward_structure(goal_type, complexity)
        
        # Create task instance
        task = TaskInstance(
            task_id=f"task_{self.task_generation_count:06d}",
            goal_type=goal_type,
            goal_description=goal_description,
            arena_config=arena_config,
            complexity=complexity,
            game_mode=game_mode,
            difficulty_params=task_params,
            expected_solution_length=expected_solution_length,
            max_time_steps=max_time_steps,
            reward_structure=reward_structure
        )
        
        self.task_generation_count += 1
        return task
    
    def generate_task_batch(self, 
                           batch_size: int,
                           game_mode: GameMode = GameMode.SINGLE_PLAYER) -> List[TaskInstance]:
        """Generate a batch of tasks with diverse complexities."""
        tasks = []
        
        for i in range(batch_size):
            # Vary complexity within batch for diversity
            if i < batch_size // 3:
                complexity = TaskComplexity.EASY
            elif i < 2 * batch_size // 3:
                complexity = TaskComplexity.MEDIUM
            else:
                complexity = TaskComplexity.HARD
            
            task = self.generate_task(game_mode, complexity)
            tasks.append(task)
        
        return tasks
    
    def update_difficulty_based_on_performance(self, 
                                               performance_metrics: Dict[str, float]):
        """
        Update task generation difficulty based on agent performance.
        Implements adaptive curriculum learning.
        
        Args:
            performance_metrics: Dictionary with performance metrics
        """
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        if len(self.performance_history) < 10:
            return  # Not enough data
        
        # Calculate recent performance trends
        recent_performance = self.performance_history[-10:]
        avg_success_rate = np.mean([p.get('success_rate', 0) for p in recent_performance])
        avg_efficiency = np.mean([p.get('efficiency', 0) for p in recent_performance])
        
        # Adjust complexity distribution
        if avg_success_rate > 0.8 and avg_efficiency > 0.7:
            # Increase difficulty
            self._increase_difficulty()
        elif avg_success_rate < 0.3 or avg_efficiency < 0.3:
            # Decrease difficulty
            self._decrease_difficulty()
        # Otherwise maintain current difficulty
    
    def _sample_complexity(self) -> TaskComplexity:
        """Sample complexity based on current distribution."""
        complexities = list(self.complexity_distribution.keys())
        probabilities = list(self.complexity_distribution.values())
        return np.random.choice(complexities, p=probabilities)
    
    def _select_goal_type(self, game_mode: GameMode) -> str:
        """Select goal type appropriate for game mode."""
        if game_mode == GameMode.COOPERATIVE:
            cooperative_goals = ['cooperative_collect', 'cooperative_zone', 'collect_multiple']
            return random.choice(cooperative_goals)
        elif game_mode == GameMode.COMPETITIVE:
            competitive_goals = ['competitive_collect', 'competitive_race', 'collect_multiple']
            return random.choice(competitive_goals)
        else:
            single_player_goals = ['collect', 'bring_to_zone', 'avoid_walls', 'touch_corners']
            return random.choice(single_player_goals)
    
    def _generate_task_parameters(self, 
                                 goal_type: str, 
                                 complexity: TaskComplexity) -> Dict[str, Any]:
        """Generate parameters for specific goal type and complexity."""
        template = self.goal_templates[goal_type]
        base_params = template['complexity_map'][complexity]
        
        # Add random variations
        params = base_params.copy()
        
        # Colors and shapes
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        shapes = ['cube', 'sphere']
        
        if '{color}' in template['template']:
            params['color'] = random.choice(colors)
        if '{shape}' in template['template']:
            params['shape'] = random.choice(shapes)
        if '{zone_color}' in template['template']:
            zone_colors = [c for c in colors if c != params.get('color', 'red')]
            params['zone_color'] = random.choice(zone_colors)
        if '{moves}' in template['template']:
            params['moves'] = base_params['moves_required']
        if '{count}' in template['template']:
            params['count'] = base_params.get('target_count', 5)
        
        return params
    
    def _generate_arena_config(self, 
                              goal_type: str, 
                              complexity: TaskComplexity,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate arena configuration based on task parameters."""
        grid_size = params['grid_size']
        
        # Generate walls
        walls = []
        for _ in range(params['num_walls']):
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            walls.append({'x': x, 'y': y, 'type': 'wall'})
        
        # Generate objects
        objects = []
        num_objects = params.get('num_objects', 5)
        
        for i in range(num_objects):
            while True:
                x = random.randint(0, grid_size - 1)
                y = random.randint(0, grid_size - 1)
                
                # Check if position is free
                if not any(w['x'] == x and w['y'] == y for w in walls):
                    color = params.get('color', random.choice(['red', 'blue', 'green', 'yellow', 'purple']))
                    shape = params.get('shape', random.choice(['cube', 'sphere']))
                    objects.append({
                        'x': x, 'y': y, 
                        'color': color, 
                        'shape': shape,
                        'id': f'obj_{i}'
                    })
                    break
        
        # Generate goal zones
        goal_zones = []
        num_zones = params.get('num_zones', 1)
        
        for i in range(num_zones):
            while True:
                x = random.randint(0, grid_size - 1)
                y = random.randint(0, grid_size - 1)
                
                # Check if position is free
                if not any(w['x'] == x and w['y'] == y for w in walls) and \
                   not any(o['x'] == x and o['y'] == y for o in objects):
                    zone_color = params.get('zone_color', random.choice(['red', 'blue', 'green', 'yellow', 'purple']))
                    goal_zones.append({
                        'x': x, 'y': y,
                        'color': zone_color,
                        'type': 'goal_zone',
                        'id': f'zone_{i}'
                    })
                    break
        
        # Player spawn positions
        player_spawns = []
        num_players = 2 if goal_type in ['cooperative_collect', 'competitive_collect', 
                                        'cooperative_zone', 'competitive_race'] else 1
        
        for i in range(num_players):
            while True:
                x = random.randint(0, grid_size - 1)
                y = random.randint(0, grid_size - 1)
                
                # Check if position is free
                if not any(w['x'] == x and w['y'] == y for w in walls) and \
                   not any(o['x'] == x and o['y'] == y for o in objects) and \
                   not any(z['x'] == x and z['y'] == y for z in goal_zones) and \
                   not any(s['x'] == x and s['y'] == y for s in player_spawns):
                    player_spawns.append({'x': x, 'y': y, 'player_id': i})
                    break
        
        return {
            'grid_size': grid_size,
            'walls': walls,
            'objects': objects,
            'goal_zones': goal_zones,
            'player_spawns': player_spawns
        }
    
    def _generate_goal_description(self, 
                                  goal_type: str, 
                                  params: Dict[str, Any]) -> str:
        """Generate final goal description with parameters filled in."""
        template = self.goal_templates[goal_type]['template']
        
        # Replace placeholders
        description = template
        for key, value in params.items():
            placeholder = '{' + key + '}'
            if placeholder in description:
                description = description.replace(placeholder, str(value))
        
        return description
    
    def _estimate_solution_length(self, 
                                 goal_type: str, 
                                 complexity: TaskComplexity) -> int:
        """Estimate expected number of steps to solve the task."""
        base_lengths = {
            TaskComplexity.TRIVIAL: 5,
            TaskComplexity.EASY: 10,
            TaskComplexity.MEDIUM: 20,
            TaskComplexity.HARD: 35,
            TaskComplexity.EXPERT: 50
        }
        
        # Adjust based on goal type
        multipliers = {
            'collect': 1.0,
            'bring_to_zone': 1.5,
            'avoid_walls': 1.2,
            'touch_corners': 1.8,
            'collect_multiple': 2.0,
            'cooperative_collect': 1.3,
            'competitive_collect': 1.4,
            'cooperative_zone': 1.6,
            'competitive_race': 1.7
        }
        
        base_length = base_lengths[complexity]
        multiplier = multipliers.get(goal_type, 1.0)
        
        return int(base_length * multiplier)
    
    def _generate_reward_structure(self, 
                                  goal_type: str, 
                                  complexity: TaskComplexity) -> Dict[str, float]:
        """Generate reward structure for the task."""
        base_rewards = {
            TaskComplexity.TRIVIAL: 10,
            TaskComplexity.EASY: 25,
            TaskComplexity.MEDIUM: 50,
            TaskComplexity.HARD: 100,
            TaskComplexity.EXPERT: 200
        }
        
        base_reward = base_rewards[complexity]
        
        return {
            'completion_reward': base_reward,
            'step_penalty': -0.1,
            'time_penalty': -0.01,
            'efficiency_bonus': base_reward * 0.3,
            'failure_penalty': -base_reward * 0.2
        }
    
    def _increase_difficulty(self):
        """Shift complexity distribution towards harder tasks."""
        shift = 0.05
        self.complexity_distribution[TaskComplexity.TRIVIAL] = max(0, self.complexity_distribution[TaskComplexity.TRIVIAL] - shift)
        self.complexity_distribution[TaskComplexity.EASY] = max(0, self.complexity_distribution[TaskComplexity.EASY] - shift/2)
        self.complexity_distribution[TaskComplexity.MEDIUM] = max(0, self.complexity_distribution[TaskComplexity.MEDIUM] - shift/4)
        self.complexity_distribution[TaskComplexity.HARD] = min(1, self.complexity_distribution[TaskComplexity.HARD] + shift/2)
        self.complexity_distribution[TaskComplexity.EXPERT] = min(1, self.complexity_distribution[TaskComplexity.EXPERT] + shift/4)
        
        # Renormalize
        total = sum(self.complexity_distribution.values())
        for key in self.complexity_distribution:
            self.complexity_distribution[key] /= total
    
    def _decrease_difficulty(self):
        """Shift complexity distribution towards easier tasks."""
        shift = 0.05
        self.complexity_distribution[TaskComplexity.TRIVIAL] = min(1, self.complexity_distribution[TaskComplexity.TRIVIAL] + shift/2)
        self.complexity_distribution[TaskComplexity.EASY] = min(1, self.complexity_distribution[TaskComplexity.EASY] + shift/4)
        self.complexity_distribution[TaskComplexity.MEDIUM] = max(0, self.complexity_distribution[TaskComplexity.MEDIUM] - shift/4)
        self.complexity_distribution[TaskComplexity.HARD] = max(0, self.complexity_distribution[TaskComplexity.HARD] - shift/2)
        self.complexity_distribution[TaskComplexity.EXPERT] = max(0, self.complexity_distribution[TaskComplexity.EXPERT] - shift)
        
        # Renormalize
        total = sum(self.complexity_distribution.values())
        for key in self.complexity_distribution:
            self.complexity_distribution[key] /= total
    
    def get_complexity_stats(self) -> Dict[str, float]:
        """Get current complexity distribution statistics."""
        return self.complexity_distribution.copy()
    
    def export_task_dataset(self, 
                           num_tasks: int = 1000,
                           filepath: str = 'tasks.json') -> List[Dict[str, Any]]:
        """Export a dataset of generated tasks for external use."""
        tasks = []
        
        for _ in range(num_tasks):
            task = self.generate_task()
            task_dict = {
                'task_id': task.task_id,
                'goal_type': task.goal_type,
                'goal_description': task.goal_description,
                'arena_config': task.arena_config,
                'complexity': task.complexity.value,
                'game_mode': task.game_mode.value,
                'difficulty_params': task.difficulty_params,
                'expected_solution_length': task.expected_solution_length,
                'max_time_steps': task.max_time_steps,
                'reward_structure': task.reward_structure
            }
            tasks.append(task_dict)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        return tasks

class TaskEvaluator:
    """Evaluates task difficulty and agent performance."""
    
    def __init__(self):
        self.evaluation_metrics = []
    
    def evaluate_task_difficulty(self, task: TaskInstance) -> Dict[str, float]:
        """Evaluate various difficulty metrics for a task."""
        arena = task.arena_config
        grid_size = arena['grid_size']
        
        # Spatial complexity
        spatial_complexity = len(arena['walls']) / (grid_size * grid_size)
        
        # Object complexity
        object_complexity = len(arena['objects']) / (grid_size * grid_size)
        
        # Goal complexity
        goal_complexity = {
            'collect': 1.0,
            'bring_to_zone': 1.5,
            'avoid_walls': 1.2,
            'touch_corners': 1.8,
            'collect_multiple': 2.0,
            'cooperative_collect': 1.3,
            'competitive_collect': 1.4,
            'cooperative_zone': 1.6,
            'competitive_race': 1.7
        }.get(task.goal_type, 1.0)
        
        # Overall difficulty score
        overall_difficulty = (spatial_complexity + object_complexity) * goal_complexity
        
        return {
            'spatial_complexity': spatial_complexity,
            'object_complexity': object_complexity,
            'goal_complexity': goal_complexity,
            'overall_difficulty': overall_difficulty
        }
    
    def evaluate_agent_performance(self, 
                                  task: TaskInstance,
                                  agent_trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate agent performance on a completed task."""
        if not agent_trajectory:
            return {'success_rate': 0.0, 'efficiency': 0.0, 'exploration_score': 0.0}
        
        # Success metrics
        final_reward = agent_trajectory[-1].get('cumulative_reward', 0)
        success_rate = 1.0 if final_reward > 0 else 0.0
        
        # Efficiency metrics
        steps_taken = len(agent_trajectory)
        expected_steps = task.expected_solution_length
        efficiency = max(0, 1 - (steps_taken - expected_steps) / expected_steps)
        
        # Exploration metrics
        visited_positions = set()
        for step in agent_trajectory:
            pos = step.get('position', (0, 0))
            visited_positions.add(pos)
        
        grid_size = task.arena_config['grid_size']
        exploration_score = len(visited_positions) / (grid_size * grid_size)
        
        return {
            'success_rate': success_rate,
            'efficiency': efficiency,
            'exploration_score': exploration_score,
            'steps_taken': steps_taken,
            'final_reward': final_reward
        }
