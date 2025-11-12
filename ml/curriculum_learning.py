import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
from enum import Enum
import json

from .task_generation import DynamicTaskGenerator, TaskInstance, GameMode, TaskComplexity
from .population_training import AgentState

class CurriculumType(Enum):
    """Types of curriculum learning strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    SELF_PACED = "self_paced"
    COMPETENCY_BASED = "competency_based"

@dataclass
class CurriculumStage:
    """Single stage in curriculum learning."""
    stage_id: str
    name: str
    description: str
    complexity_requirements: Dict[TaskComplexity, float]
    goal_type_requirements: Dict[str, float]
    performance_threshold: float
    min_steps_before_advance: int
    max_steps_in_stage: int
    adaptive_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning system."""
    curriculum_type: CurriculumType = CurriculumType.ADAPTIVE
    evaluation_frequency: int = 1000
    performance_window: int = 100
    advancement_threshold: float = 0.8
    regression_threshold: float = 0.3
    min_stage_duration: int = 5000
    max_stage_duration: int = 50000
    difficulty_smoothing: float = 0.1
    competency_decay_rate: float = 0.95

class CompetencyTracker:
    """Tracks agent competency across different skills and task types."""
    
    def __init__(self, 
                 skill_types: List[str],
                 decay_rate: float = 0.95):
        self.skill_types = skill_types
        self.decay_rate = decay_rate
        self.competencies = {skill: deque(maxlen=100) for skill in skill_types}
        self.skill_weights = {skill: 1.0 for skill in skill_types}
        
    def update_competency(self, 
                         skill_type: str, 
                         performance: float):
        """Update competency for a specific skill."""
        if skill_type in self.competencies:
            self.competencies[skill_type].append(performance)
    
    def get_current_competency(self, skill_type: str) -> float:
        """Get current competency level for a skill."""
        if skill_type not in self.competencies or not self.competencies[skill_type]:
            return 0.0
        
        # Exponential moving average
        recent_performances = list(self.competencies[skill_type])
        if len(recent_performances) == 1:
            return recent_performances[0]
        
        # Weight recent performances more heavily
        weights = np.array([self.decay_rate ** i for i in range(len(recent_performances)-1, -1, -1)])
        weights = weights / weights.sum()
        
        return np.average(recent_performances, weights=weights)
    
    def get_overall_competency(self) -> float:
        """Get weighted overall competency score."""
        competencies = [self.get_current_competency(skill) for skill in self.skill_types]
        weights = [self.skill_weights[skill] for skill in self.skill_types]
        
        if not competencies:
            return 0.0
        
        return np.average(competencies, weights=weights)
    
    def identify_weak_skills(self, threshold: float = 0.5) -> List[str]:
        """Identify skills that need improvement."""
        weak_skills = []
        for skill in self.skill_types:
            if self.get_current_competency(skill) < threshold:
                weak_skills.append(skill)
        return weak_skills
    
    def adjust_skill_weights(self, focus_skills: List[str]):
        """Adjust weights to focus on specific skills."""
        for skill in focus_skills:
            if skill in self.skill_weights:
                self.skill_weights[skill] *= 1.5
        
        # Normalize weights
        total_weight = sum(self.skill_weights.values())
        for skill in self.skill_weights:
            self.skill_weights[skill] /= total_weight

class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum scheduler that adjusts difficulty based on
    agent performance and competency across different skills.
    """
    
    def __init__(self, 
                 config: CurriculumConfig,
                 task_generator: DynamicTaskGenerator):
        self.config = config
        self.task_generator = task_generator
        self.logger = logging.getLogger(__name__)
        
        # Initialize curriculum stages
        self.stages = self._create_curriculum_stages()
        self.current_stage_index = 0
        self.stage_start_step = 0
        self.steps_in_current_stage = 0
        
        # Competency tracking
        self.skill_types = [
            'navigation', 'object_manipulation', 'goal_parsing', 
            'spatial_reasoning', 'temporal_planning', 'cooperation', 'competition'
        ]
        self.competency_tracker = CompetencyTracker(self.skill_types)
        
        # Performance tracking
        self.performance_history = deque(maxlen=config.performance_window)
        self.stage_transitions = []
        
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create predefined curriculum stages."""
        stages = [
            CurriculumStage(
                stage_id="basic_navigation",
                name="Basic Navigation and Object Collection",
                description="Learn fundamental movement and object interaction",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.6,
                    TaskComplexity.EASY: 0.4,
                    TaskComplexity.MEDIUM: 0.0,
                    TaskComplexity.HARD: 0.0,
                    TaskComplexity.EXPERT: 0.0
                },
                goal_type_requirements={
                    'collect': 0.8,
                    'bring_to_zone': 0.2,
                    'avoid_walls': 0.0,
                    'touch_corners': 0.0
                },
                performance_threshold=0.7,
                min_steps_before_advance=3000,
                max_steps_in_stage=15000,
                adaptive_params={'focus_skills': ['navigation', 'object_manipulation']}
            ),
            
            CurriculumStage(
                stage_id="intermediate_reasoning",
                name="Intermediate Spatial Reasoning",
                description="Develop spatial awareness and multi-step planning",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.2,
                    TaskComplexity.EASY: 0.4,
                    TaskComplexity.MEDIUM: 0.4,
                    TaskComplexity.HARD: 0.0,
                    TaskComplexity.EXPERT: 0.0
                },
                goal_type_requirements={
                    'collect': 0.3,
                    'bring_to_zone': 0.3,
                    'avoid_walls': 0.2,
                    'touch_corners': 0.2
                },
                performance_threshold=0.75,
                min_steps_before_advance=5000,
                max_steps_in_stage=20000,
                adaptive_params={'focus_skills': ['spatial_reasoning', 'temporal_planning']}
            ),
            
            CurriculumStage(
                stage_id="advanced_challenges",
                name="Advanced Multi-Objective Tasks",
                description="Handle complex goals with multiple constraints",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.0,
                    TaskComplexity.EASY: 0.2,
                    TaskComplexity.MEDIUM: 0.5,
                    TaskComplexity.HARD: 0.3,
                    TaskComplexity.EXPERT: 0.0
                },
                goal_type_requirements={
                    'collect': 0.2,
                    'bring_to_zone': 0.2,
                    'avoid_walls': 0.2,
                    'touch_corners': 0.2,
                    'collect_multiple': 0.2
                },
                performance_threshold=0.8,
                min_steps_before_advance=8000,
                max_steps_in_stage=30000,
                adaptive_params={'focus_skills': ['goal_parsing', 'temporal_planning']}
            ),
            
            CurriculumStage(
                stage_id="cooperative_skills",
                name="Cooperative Multi-Agent Tasks",
                description="Learn to work with other agents",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.0,
                    TaskComplexity.EASY: 0.1,
                    TaskComplexity.MEDIUM: 0.4,
                    TaskComplexity.HARD: 0.4,
                    TaskComplexity.EXPERT: 0.1
                },
                goal_type_requirements={
                    'cooperative_collect': 0.4,
                    'cooperative_zone': 0.3,
                    'collect_multiple': 0.2,
                    'bring_to_zone': 0.1
                },
                performance_threshold=0.8,
                min_steps_before_advance=10000,
                max_steps_in_stage=40000,
                adaptive_params={'focus_skills': ['cooperation', 'goal_parsing']}
            ),
            
            CurriculumStage(
                stage_id="competitive_mastery",
                name="Competitive and Strategic Play",
                description="Master competitive scenarios and strategic thinking",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.0,
                    TaskComplexity.EASY: 0.0,
                    TaskComplexity.MEDIUM: 0.2,
                    TaskComplexity.HARD: 0.5,
                    TaskComplexity.EXPERT: 0.3
                },
                goal_type_requirements={
                    'competitive_collect': 0.4,
                    'competitive_race': 0.3,
                    'cooperative_collect': 0.2,
                    'collect_multiple': 0.1
                },
                performance_threshold=0.85,
                min_steps_before_advance=15000,
                max_steps_in_stage=50000,
                adaptive_params={'focus_skills': ['competition', 'strategic_planning']}
            ),
            
            CurriculumStage(
                stage_id="expert_generalization",
                name="Expert Generalization",
                description="Achieve mastery across all task types and complexities",
                complexity_requirements={
                    TaskComplexity.TRIVIAL: 0.1,
                    TaskComplexity.EASY: 0.1,
                    TaskComplexity.MEDIUM: 0.2,
                    TaskComplexity.HARD: 0.3,
                    TaskComplexity.EXPERT: 0.3
                },
                goal_type_requirements={
                    'collect': 0.1,
                    'bring_to_zone': 0.1,
                    'avoid_walls': 0.1,
                    'touch_corners': 0.1,
                    'collect_multiple': 0.1,
                    'cooperative_collect': 0.15,
                    'cooperative_zone': 0.15,
                    'competitive_collect': 0.15,
                    'competitive_race': 0.15
                },
                performance_threshold=0.9,
                min_steps_before_advance=20000,
                max_steps_in_stage=100000,
                adaptive_params={'focus_skills': ['all_skills']}
            )
        ]
        
        return stages
    
    def update_curriculum(self, 
                         current_step: int,
                         agent_performance: Dict[str, float],
                         agent_states: List[AgentState]):
        """
        Update curriculum based on current performance and step count.
        
        Args:
            current_step: Current training step
            agent_performance: Performance metrics for agents
            agent_states: Current states of agents in population
        """
        self.steps_in_current_stage = current_step - self.stage_start_step
        
        # Update competency tracker
        self._update_competencies(agent_performance)
        
        # Check for stage advancement
        if self._should_advance_stage(agent_performance):
            self._advance_to_next_stage()
        
        # Check for stage regression (if performing poorly)
        elif self._should_regress_stage(agent_performance):
            self._regress_to_previous_stage()
        
        # Apply current stage settings
        self._apply_current_stage_settings()
        
        # Adaptive difficulty adjustment within stage
        if self.config.curriculum_type == CurriculumType.ADAPTIVE:
            self._adaptive_difficulty_adjustment(agent_performance)
    
    def _update_competencies(self, performance_metrics: Dict[str, float]):
        """Update competency tracker with current performance."""
        # Map performance metrics to skills
        skill_mapping = {
            'navigation_success': 'navigation',
            'object_collection_rate': 'object_manipulation',
            'goal_parsing_accuracy': 'goal_parsing',
            'spatial_reasoning_score': 'spatial_reasoning',
            'planning_efficiency': 'temporal_planning',
            'cooperation_success': 'cooperation',
            'competition_win_rate': 'competition'
        }
        
        for metric, skill in skill_mapping.items():
            if metric in performance_metrics:
                self.competency_tracker.update_competency(skill, performance_metrics[metric])
    
    def _should_advance_stage(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if curriculum should advance to next stage."""
        if self.current_stage_index >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_stage = self.stages[self.current_stage_index]
        
        # Check minimum duration requirement
        if self.steps_in_current_stage < current_stage.min_steps_before_advance:
            return False
        
        # Check performance threshold
        overall_performance = np.mean(list(performance_metrics.values()))
        if overall_performance < current_stage.performance_threshold:
            return False
        
        # Check competency requirements
        focus_skills = current_stage.adaptive_params.get('focus_skills', [])
        if focus_skills != ['all_skills']:
            for skill in focus_skills:
                if skill in self.skill_types:
                    competency = self.competency_tracker.get_current_competency(skill)
                    if competency < current_stage.performance_threshold:
                        return False
        
        # Check maximum duration
        if self.steps_in_current_stage >= current_stage.max_steps_in_stage:
            return True
        
        return True
    
    def _should_regress_stage(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if curriculum should regress to previous stage."""
        if self.current_stage_index == 0:
            return False  # Cannot regress from first stage
        
        overall_performance = np.mean(list(performance_metrics.values()))
        
        # Regress if performance is consistently poor
        if overall_performance < self.config.regression_threshold:
            # Check if this is persistent poor performance
            if len(self.performance_history) >= self.config.performance_window:
                recent_avg = np.mean(list(self.performance_history))
                if recent_avg < self.config.regression_threshold:
                    return True
        
        return False
    
    def _advance_to_next_stage(self):
        """Advance curriculum to next stage."""
        old_stage = self.stages[self.current_stage_index]
        self.current_stage_index += 1
        new_stage = self.stages[self.current_stage_index]
        
        self.stage_start_step = self.steps_in_current_stage + self.stage_start_step
        self.steps_in_current_stage = 0
        
        # Record transition
        transition = {
            'from_stage': old_stage.stage_id,
            'to_stage': new_stage.stage_id,
            'step': self.stage_start_step,
            'type': 'advancement'
        }
        self.stage_transitions.append(transition)
        
        self.logger.info(f"Curriculum advanced: {old_stage.name} -> {new_stage.name}")
    
    def _regress_to_previous_stage(self):
        """Regress curriculum to previous stage."""
        if self.current_stage_index == 0:
            return
        
        old_stage = self.stages[self.current_stage_index]
        self.current_stage_index -= 1
        new_stage = self.stages[self.current_stage_index]
        
        self.stage_start_step = self.steps_in_current_stage + self.stage_start_step
        self.steps_in_current_stage = 0
        
        # Record transition
        transition = {
            'from_stage': old_stage.stage_id,
            'to_stage': new_stage.stage_id,
            'step': self.stage_start_step,
            'type': 'regression'
        }
        self.stage_transitions.append(transition)
        
        self.logger.warning(f"Curriculum regressed: {old_stage.name} -> {new_stage.name}")
    
    def _apply_current_stage_settings(self):
        """Apply current curriculum stage settings to task generator."""
        if self.current_stage_index >= len(self.stages):
            return
        
        current_stage = self.stages[self.current_stage_index]
        
        # Update complexity distribution
        self.task_generator.complexity_distribution = current_stage.complexity_requirements
        
        # Adjust skill focus in competency tracker
        focus_skills = current_stage.adaptive_params.get('focus_skills', [])
        if focus_skills and focus_skills != ['all_skills']:
            self.competency_tracker.adjust_skill_weights(focus_skills)
    
    def _adaptive_difficulty_adjustment(self, performance_metrics: Dict[str, float]):
        """Fine-tune difficulty within current stage based on performance."""
        current_stage = self.stages[self.current_stage_index]
        overall_performance = np.mean(list(performance_metrics.values()))
        
        # Calculate adjustment factor
        target_performance = current_stage.performance_threshold
        performance_gap = overall_performance - target_performance
        
        # Adjust complexity distribution smoothly
        if abs(performance_gap) > 0.1:  # Only adjust if gap is significant
            adjustment = performance_gap * self.config.difficulty_smoothing
            
            current_distribution = self.task_generator.complexity_distribution.copy()
            
            # Shift distribution based on performance
            if performance_gap > 0:  # Performing well, increase difficulty
                # Shift weight towards harder complexities
                complexities = list(TaskComplexity)
                for i, complexity in enumerate(complexities[:-1]):
                    if current_distribution[complexity] > 0.05:
                        transfer = min(current_distribution[complexity] * adjustment, 
                                     current_distribution[complexity] * 0.5)
                        current_distribution[complexity] -= transfer
                        current_distribution[complexities[i + 1]] += transfer
            else:  # Performing poorly, decrease difficulty
                # Shift weight towards easier complexities
                complexities = list(TaskComplexity)
                for i, complexity in enumerate(complexities[1:], 1):
                    if current_distribution[complexity] > 0.05:
                        transfer = min(current_distribution[complexity] * abs(adjustment), 
                                     current_distribution[complexity] * 0.5)
                        current_distribution[complexity] -= transfer
                        current_distribution[complexities[i - 1]] += transfer
            
            # Ensure distribution sums to 1
            total = sum(current_distribution.values())
            for complexity in current_distribution:
                current_distribution[complexity] /= total
            
            self.task_generator.complexity_distribution = current_distribution
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current curriculum stage."""
        if self.current_stage_index >= len(self.stages):
            return {}
        
        current_stage = self.stages[self.current_stage_index]
        
        return {
            'stage_id': current_stage.stage_id,
            'stage_name': current_stage.name,
            'stage_description': current_stage.description,
            'steps_in_stage': self.steps_in_current_stage,
            'progress_percentage': min(100, (self.steps_in_current_stage / 
                                            current_stage.max_steps_in_stage) * 100),
            'performance_threshold': current_stage.performance_threshold,
            'focus_skills': current_stage.adaptive_params.get('focus_skills', []),
            'competency_scores': {
                skill: self.competency_tracker.get_current_competency(skill)
                for skill in self.skill_types
            },
            'overall_competency': self.competency_tracker.get_overall_competency()
        }
    
    def generate_stage_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of curriculum progress."""
        report = {
            'current_stage': self.get_current_stage_info(),
            'stage_transitions': self.stage_transitions,
            'total_stages': len(self.stages),
            'completed_stages': self.current_stage_index,
            'weak_skills': self.competency_tracker.identify_weak_skills(),
            'curriculum_type': self.config.curriculum_type.value
        }
        
        # Add stage-specific statistics
        stage_stats = {}
        for stage in self.stages:
            transitions_to_stage = [t for t in self.stage_transitions if t['to_stage'] == stage.stage_id]
            transitions_from_stage = [t for t in self.stage_transitions if t['from_stage'] == stage.stage_id]
            
            stage_stats[stage.stage_id] = {
                'name': stage.name,
                'times_reached': len(transitions_to_stage),
                'times_completed': len(transitions_from_stage),
                'average_duration': 0  # Would need timing data
            }
        
        report['stage_statistics'] = stage_stats
        
        return report

class SelfPacedLearningScheduler:
    """Self-paced learning scheduler that lets agents control difficulty progression."""
    
    def __init__(self, 
                 task_generator: DynamicTaskGenerator,
                 initial_difficulty: float = 0.1,
                 pacing_function: str = "linear"):
        self.task_generator = task_generator
        self.initial_difficulty = initial_difficulty
        self.current_difficulty = initial_difficulty
        self.pacing_function = pacing_function
        
        # Learning pace tracking
        self.success_history = deque(maxlen=100)
        self.difficulty_adjustments = []
        
    def update_difficulty(self, 
                         success_rate: float,
                         learning_efficiency: float):
        """Update difficulty based on agent's learning pace."""
        self.success_history.append(success_rate)
        
        if len(self.success_history) < 10:
            return  # Need more data
        
        # Calculate learning pace
        recent_success = np.mean(list(self.success_history)[-10:])
        
        # Adjust difficulty based on pacing function
        if self.pacing_function == "linear":
            new_difficulty = self._linear_pacing(recent_success, learning_efficiency)
        elif self.pacing_function == "exponential":
            new_difficulty = self._exponential_pacing(recent_success, learning_efficiency)
        elif self.pacing_function == "sigmoid":
            new_difficulty = self._sigmoid_pacing(recent_success, learning_efficiency)
        else:
            new_difficulty = self._linear_pacing(recent_success, learning_efficiency)
        
        # Smooth difficulty changes
        self.current_difficulty = 0.8 * self.current_difficulty + 0.2 * new_difficulty
        
        # Apply to task generator
        self._apply_difficulty_to_generator()
        
        # Record adjustment
        self.difficulty_adjustments.append({
            'step': len(self.difficulty_adjustments),
            'old_difficulty': self.current_difficulty,
            'new_difficulty': new_difficulty,
            'success_rate': recent_success,
            'learning_efficiency': learning_efficiency
        })
    
    def _linear_pacing(self, success_rate: float, efficiency: float) -> float:
        """Linear pacing function."""
        if success_rate > 0.8 and efficiency > 0.7:
            return min(1.0, self.current_difficulty + 0.05)
        elif success_rate < 0.4 or efficiency < 0.3:
            return max(0.1, self.current_difficulty - 0.05)
        else:
            return self.current_difficulty
    
    def _exponential_pacing(self, success_rate: float, efficiency: float) -> float:
        """Exponential pacing function."""
        combined_score = (success_rate + efficiency) / 2
        
        if combined_score > 0.8:
            return min(1.0, self.current_difficulty * 1.1)
        elif combined_score < 0.4:
            return max(0.1, self.current_difficulty * 0.9)
        else:
            return self.current_difficulty
    
    def _sigmoid_pacing(self, success_rate: float, efficiency: float) -> float:
        """Sigmoid pacing function for smooth transitions."""
        combined_score = (success_rate + efficiency) / 2
        
        # Map to sigmoid curve
        target = 1 / (1 + np.exp(-10 * (combined_score - 0.5)))
        
        # Smooth transition
        return 0.9 * self.current_difficulty + 0.1 * target
    
    def _apply_difficulty_to_generator(self):
        """Apply current difficulty to task generator."""
        # Map difficulty to complexity distribution
        difficulty = self.current_difficulty
        
        if difficulty < 0.2:
            # Very easy
            distribution = {
                TaskComplexity.TRIVIAL: 0.5,
                TaskComplexity.EASY: 0.5,
                TaskComplexity.MEDIUM: 0.0,
                TaskComplexity.HARD: 0.0,
                TaskComplexity.EXPERT: 0.0
            }
        elif difficulty < 0.4:
            # Easy
            distribution = {
                TaskComplexity.TRIVIAL: 0.2,
                TaskComplexity.EASY: 0.6,
                TaskComplexity.MEDIUM: 0.2,
                TaskComplexity.HARD: 0.0,
                TaskComplexity.EXPERT: 0.0
            }
        elif difficulty < 0.6:
            # Medium
            distribution = {
                TaskComplexity.TRIVIAL: 0.1,
                TaskComplexity.EASY: 0.3,
                TaskComplexity.MEDIUM: 0.4,
                TaskComplexity.HARD: 0.2,
                TaskComplexity.EXPERT: 0.0
            }
        elif difficulty < 0.8:
            # Hard
            distribution = {
                TaskComplexity.TRIVIAL: 0.0,
                TaskComplexity.EASY: 0.1,
                TaskComplexity.MEDIUM: 0.3,
                TaskComplexity.HARD: 0.4,
                TaskComplexity.EXPERT: 0.2
            }
        else:
            # Expert
            distribution = {
                TaskComplexity.TRIVIAL: 0.0,
                TaskComplexity.EASY: 0.0,
                TaskComplexity.MEDIUM: 0.2,
                TaskComplexity.HARD: 0.4,
                TaskComplexity.EXPERT: 0.4
            }
        
        self.task_generator.complexity_distribution = distribution
