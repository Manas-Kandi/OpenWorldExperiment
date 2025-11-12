import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import time
from enum import Enum
import json

from .rl_core import PPOAgent, RLConfig
from .task_generation import TaskInstance, GameMode, TaskComplexity
from .population_training import AgentState

class MultiAgentRole(Enum):
    """Roles that agents can play in multi-agent scenarios."""
    LEADER = "leader"
    FOLLOWER = "follower"
    COMPETITOR = "competitor"
    COLLABORATOR = "collaborator"
    EXPLORER = "explorer"
    SPECIALIST = "specialist"

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent training coordination."""
    max_team_size: int = 4
    min_team_size: int = 2
    role_assignment_strategy: str = "adaptive"  # adaptive, random, performance_based
    team_formation_frequency: int = 5000  # Steps between team reformation
    communication_protocol: str = "implicit"  # implicit, explicit, none
    coordination_reward_weight: float = 0.3
    individual_reward_weight: float = 0.7
    diversity_encouragement: float = 0.1
    skill_complementarity_weight: float = 0.2

@dataclass
class Team:
    """Represents a team of agents for cooperative/competitive tasks."""
    team_id: str
    members: List[AgentState]
    role_assignments: Dict[str, MultiAgentRole]
    team_performance: float = 0.0
    coordination_score: float = 0.0
    formation_time: int = 0
    task_history: List[str] = field(default_factory=list)

class MultiAgentCoordinator:
    """
    Coordinates multi-agent training scenarios, team formation,
    and role assignment for optimal learning dynamics.
    """
    
    def __init__(self, 
                 config: MultiAgentConfig,
                 population: List[AgentState]):
        self.config = config
        self.population = population
        self.logger = logging.getLogger(__name__)
        
        # Team management
        self.teams = []
        self.agent_team_mapping = {}  # agent_id -> team_id
        self.role_history = defaultdict(list)
        
        # Coordination metrics
        self.coordination_history = deque(maxlen=1000)
        self.team_performance_history = defaultdict(deque)
        
        # Skill tracking for role assignment
        self.agent_skills = self._initialize_skill_tracking()
        
        # Communication system
        self.communication_channels = {} if config.communication_protocol != "none" else None
        
    def _initialize_skill_tracking(self) -> Dict[str, Dict[str, float]]:
        """Initialize skill tracking for all agents."""
        skills = {}
        for agent_state in self.population:
            skills[agent_state.agent_id] = {
                'navigation': 0.5,
                'object_manipulation': 0.5,
                'coordination': 0.5,
                'competition': 0.5,
                'exploration': 0.5,
                'leadership': 0.5,
                'specialization': 0.5
            }
        return skills
    
    def form_teams(self, 
                   current_step: int,
                   task_mode: GameMode) -> List[Team]:
        """
        Form teams of agents based on current strategy and task requirements.
        
        Args:
            current_step: Current training step
            task_mode: Mode of tasks (cooperative, competitive)
            
        Returns:
            List of formed teams
        """
        if task_mode == GameMode.SINGLE_PLAYER:
            return []  # No teams needed for single player
        
        # Clear previous team assignments
        self.teams.clear()
        self.agent_team_mapping.clear()
        
        if task_mode == GameMode.COOPERATIVE:
            teams = self._form_cooperative_teams()
        elif task_mode == GameMode.COMPETITIVE:
            teams = self._form_competitive_teams()
        else:
            teams = []
        
        # Assign roles within teams
        for team in teams:
            self._assign_team_roles(team)
        
        self.teams = teams
        self._log_team_formation(current_step, task_mode)
        
        return teams
    
    def _form_cooperative_teams(self) -> List[Team]:
        """Form teams for cooperative tasks."""
        teams = []
        available_agents = self.population.copy()
        
        # Sort agents by coordination skill for better team formation
        available_agents.sort(
            key=lambda x: self.agent_skills[x.agent_id]['coordination'],
            reverse=True
        )
        
        team_id = 0
        while len(available_agents) >= self.config.min_team_size:
            team_size = min(
                self.config.max_team_size,
                np.random.randint(self.config.min_team_size, self.config.max_team_size + 1)
            )
            
            # Select team members with complementary skills
            team_members = self._select_complementary_agents(
                available_agents[:team_size * 2], team_size
            )
            
            # Remove selected agents from available pool
            for agent in team_members:
                available_agents.remove(agent)
            
            # Create team
            team = Team(
                team_id=f"cooperative_team_{team_id}",
                members=team_members,
                role_assignments={},
                formation_time=time.time()
            )
            
            teams.append(team)
            team_id += 1
        
        return teams
    
    def _form_competitive_teams(self) -> List[Team]:
        """Form teams for competitive tasks."""
        teams = []
        available_agents = self.population.copy()
        
        # Sort agents by competition skill
        available_agents.sort(
            key=lambda x: self.agent_skills[x.agent_id]['competition'],
            reverse=True
        )
        
        # Create pairs for competition
        team_id = 0
        while len(available_agents) >= 2:
            # Select agents with similar skill levels for balanced competition
            agent1 = available_agents.pop(0)
            
            # Find closest match in skill level
            best_match_idx = self._find_competitive_match(agent1, available_agents)
            if best_match_idx is not None:
                agent2 = available_agents.pop(best_match_idx)
                
                team = Team(
                    team_id=f"competitive_team_{team_id}",
                    members=[agent1, agent2],
                    role_assignments={},
                    formation_time=time.time()
                )
                
                teams.append(team)
                team_id += 1
        
        return teams
    
    def _select_complementary_agents(self, 
                                    candidates: List[AgentState], 
                                    team_size: int) -> List[AgentState]:
        """Select agents with complementary skills for a team."""
        if len(candidates) <= team_size:
            return candidates
        
        # Calculate skill diversity scores for different combinations
        best_team = None
        best_diversity = 0
        
        # Try different combinations (simplified for performance)
        for _ in range(100):  # Sample 100 random combinations
            selected = np.random.choice(candidates, team_size, replace=False).tolist()
            
            # Calculate skill diversity
            skill_vectors = []
            for agent in selected:
                skills = self.agent_skills[agent.agent_id]
                skill_vector = [skills['navigation'], skills['object_manipulation'], 
                              skills['coordination'], skills['exploration']]
                skill_vectors.append(skill_vector)
            
            skill_matrix = np.array(skill_vectors)
            diversity = np.std(skill_matrix, axis=0).mean()  # Average standard deviation
            
            if diversity > best_diversity:
                best_diversity = diversity
                best_team = selected
        
        return best_team or candidates[:team_size]
    
    def _find_competitive_match(self, 
                               agent: AgentState, 
                               candidates: List[AgentState]) -> Optional[int]:
        """Find best competitive match for an agent."""
        agent_skills = self.agent_skills[agent.agent_id]
        
        best_idx = None
        min_diff = float('inf')
        
        for i, candidate in enumerate(candidates):
            candidate_skills = self.agent_skills[candidate.agent_id]
            
            # Calculate skill difference
            skill_diff = abs(agent_skills['competition'] - candidate_skills['competition'])
            overall_diff = np.sqrt(sum([
                (agent_skills[k] - candidate_skills[k])**2 
                for k in agent_skills.keys()
            ]))
            
            if overall_diff < min_diff:
                min_diff = overall_diff
                best_idx = i
        
        return best_idx
    
    def _assign_team_roles(self, team: Team):
        """Assign roles to team members based on their skills."""
        if self.config.role_assignment_strategy == "random":
            self._assign_random_roles(team)
        elif self.config.role_assignment_strategy == "performance_based":
            self._assign_performance_based_roles(team)
        else:  # adaptive
            self._assign_adaptive_roles(team)
    
    def _assign_random_roles(self, team: Team):
        """Assign roles randomly to team members."""
        available_roles = [MultiAgentRole.LEADER, MultiAgentRole.FOLLOWER, 
                          MultiAgentRole.COLLABORATOR, MultiAgentRole.EXPLORER]
        
        for member in team.members:
            role = np.random.choice(available_roles)
            team.role_assignments[member.agent_id] = role
            self.role_history[member.agent_id].append(role)
    
    def _assign_performance_based_roles(self, team: Team):
        """Assign roles based on agent performance metrics."""
        # Sort members by overall performance
        sorted_members = sorted(
            team.members,
            key=lambda x: np.mean(list(x.performance_history)) if x.performance_history else 0,
            reverse=True
        )
        
        # Assign roles based on performance ranking
        if len(sorted_members) >= 2:
            team.role_assignments[sorted_members[0].agent_id] = MultiAgentRole.LEADER
            team.role_assignments[sorted_members[1].agent_id] = MultiAgentRole.FOLLOWER
            
            for i, member in enumerate(sorted_members[2:]):
                if i % 2 == 0:
                    team.role_assignments[member.agent_id] = MultiAgentRole.COLLABORATOR
                else:
                    team.role_assignments[member.agent_id] = MultiAgentRole.EXPLORER
        
        # Update role history
        for member in team.members:
            self.role_history[member.agent_id].append(team.role_assignments[member.agent_id])
    
    def _assign_adaptive_roles(self, team: Team):
        """Assign roles adaptively based on skills and team composition."""
        # Calculate role suitability scores for each agent
        role_scores = {}
        
        for member in team.members:
            skills = self.agent_skills[member.agent_id]
            role_scores[member.agent_id] = {
                MultiAgentRole.LEADER: skills['leadership'] * 0.7 + skills['coordination'] * 0.3,
                MultiAgentRole.FOLLOWER: skills['coordination'] * 0.8 + skills['navigation'] * 0.2,
                MultiAgentRole.COLLABORATOR: skills['coordination'] * 0.6 + skills['object_manipulation'] * 0.4,
                MultiAgentRole.EXPLORER: skills['exploration'] * 0.7 + skills['navigation'] * 0.3,
                MultiAgentRole.COMPETITOR: skills['competition'] * 0.8 + skills['leadership'] * 0.2,
                MultiAgentRole.SPECIALIST: skills['specialization'] * 0.6 + skills['object_manipulation'] * 0.4
            }
        
        # Assign roles to maximize team utility
        assigned_roles = set()
        
        # First, assign leader (highest suitability)
        leader_candidates = [(agent_id, scores[MultiAgentRole.LEADER]) 
                            for agent_id, scores in role_scores.items()]
        leader_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if leader_candidates:
            team.role_assignments[leader_candidates[0][0]] = MultiAgentRole.LEADER
            assigned_roles.add(MultiAgentRole.LEADER)
        
        # Assign remaining roles
        for member in team.members:
            if member.agent_id not in team.role_assignments:
                # Find best remaining role
                best_role = None
                best_score = -1
                
                for role, score in role_scores[member.agent_id].items():
                    if role not in assigned_roles and score > best_score:
                        best_score = score
                        best_role = role
                
                if best_role:
                    team.role_assignments[member.agent_id] = best_role
                    assigned_roles.add(best_role)
                else:
                    # Fallback to collaborator
                    team.role_assignments[member.agent_id] = MultiAgentRole.COLLABORATOR
        
        # Update role history
        for member in team.members:
            self.role_history[member.agent_id].append(team.role_assignments[member.agent_id])
    
    def coordinate_team_execution(self, 
                                 team: Team, 
                                 task: TaskInstance) -> Dict[str, Any]:
        """
        Coordinate team execution of a multi-agent task.
        
        Args:
            team: Team to execute the task
            task: Multi-agent task
            
        Returns:
            Execution results and coordination metrics
        """
        if not team.members:
            return {}
        
        # Initialize communication if needed
        if self.config.communication_protocol == "explicit":
            self._initialize_communication(team)
        
        # Execute team task
        execution_results = self._execute_team_task(team, task)
        
        # Calculate coordination metrics
        coordination_metrics = self._calculate_coordination_metrics(team, execution_results)
        
        # Update team performance
        team.team_performance = execution_results.get('team_score', 0)
        team.coordination_score = coordination_metrics.get('coordination_efficiency', 0)
        team.task_history.append(task.task_id)
        
        # Update individual agent skills based on performance
        self._update_agent_skills(team, execution_results, coordination_metrics)
        
        return {
            'execution_results': execution_results,
            'coordination_metrics': coordination_metrics,
            'team_performance': team.team_performance
        }
    
    def _initialize_communication(self, team: Team):
        """Initialize communication channels for explicit communication."""
        team_id = team.team_id
        
        if team_id not in self.communication_channels:
            self.communication_channels[team_id] = {
                'messages': deque(maxlen=100),
                'broadcasts': {},
                'agent_channels': {member.agent_id: deque(maxlen=50) 
                                 for member in team.members}
            }
    
    def _execute_team_task(self, 
                          team: Team, 
                          task: TaskInstance) -> Dict[str, Any]:
        """
        Execute team task (simplified simulation).
        In practice, this would coordinate actual multi-agent environment execution.
        """
        # Simulate team execution based on agent skills and roles
        team_skill_score = 0
        individual_scores = {}
        
        for member in team.members:
            agent_skills = self.agent_skills[member.agent_id]
            role = team.role_assignments.get(member.agent_id, MultiAgentRole.COLLABORATOR)
            
            # Calculate individual contribution based on role and skills
            if role == MultiAgentRole.LEADER:
                contribution = agent_skills['leadership'] * 0.6 + agent_skills['coordination'] * 0.4
            elif role == MultiAgentRole.FOLLOWER:
                contribution = agent_skills['coordination'] * 0.7 + agent_skills['navigation'] * 0.3
            elif role == MultiAgentRole.COLLABORATOR:
                contribution = agent_skills['coordination'] * 0.5 + agent_skills['object_manipulation'] * 0.5
            elif role == MultiAgentRole.EXPLORER:
                contribution = agent_skills['exploration'] * 0.7 + agent_skills['navigation'] * 0.3
            elif role == MultiAgentRole.COMPETITOR:
                contribution = agent_skills['competition'] * 0.8 + agent_skills['leadership'] * 0.2
            else:
                contribution = agent_skills['specialization'] * 0.6 + agent_skills['object_manipulation'] * 0.4
            
            # Add some randomness
            contribution += np.random.normal(0, 0.1)
            contribution = np.clip(contribution, 0, 1)
            
            individual_scores[member.agent_id] = contribution
            team_skill_score += contribution
        
        # Calculate team score with coordination bonus
        coordination_bonus = self._calculate_coordination_bonus(team)
        team_score = (team_skill_score / len(team.members)) * (1 + coordination_bonus)
        
        # Calculate individual rewards
        individual_rewards = {}
        for member in team.members:
            base_reward = individual_scores[member.agent_id]
            coordination_reward = coordination_bonus * self.config.coordination_reward_weight
            individual_rewards[member.agent_id] = (
                base_reward * self.config.individual_reward_weight + 
                coordination_reward
            )
        
        return {
            'team_score': team_score,
            'individual_scores': individual_scores,
            'individual_rewards': individual_rewards,
            'task_completed': team_score > 0.6,
            'execution_time': np.random.uniform(10, 100),
            'coordination_events': int(np.random.uniform(5, 20))
        }
    
    def _calculate_coordination_bonus(self, team: Team) -> float:
        """Calculate coordination bonus based on role complementarity."""
        role_counts = defaultdict(int)
        for role in team.role_assignments.values():
            role_counts[role] += 1
        
        # Bonus for having diverse roles
        diversity_bonus = len(role_counts) / len(team.members)
        
        # Bonus for having leader when team size > 2
        leader_bonus = 0.2 if role_counts.get(MultiAgentRole.LEADER, 0) > 0 and len(team.members) > 2 else 0
        
        # Bonus for role balance
        balance_bonus = 1 - (np.std(list(role_counts.values())) / len(team.members))
        
        total_bonus = (diversity_bonus * 0.5 + leader_bonus * 0.3 + balance_bonus * 0.2)
        
        return np.clip(total_bonus, 0, 0.5)  # Cap at 50% bonus
    
    def _calculate_coordination_metrics(self, 
                                       team: Team, 
                                       execution_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coordination quality metrics."""
        team_score = execution_results.get('team_score', 0)
        individual_scores = execution_results.get('individual_scores', {})
        
        # Coordination efficiency (how well team performs relative to individuals)
        avg_individual_score = np.mean(list(individual_scores.values())) if individual_scores else 0
        coordination_efficiency = team_score / (avg_individual_score + 1e-8)
        
        # Role utilization (how well roles match actual contributions)
        role_utilization = self._calculate_role_utilization(team, individual_scores)
        
        # Communication effectiveness (if applicable)
        communication_effectiveness = 0.5  # Placeholder
        
        # Synchronization score (how well agents work together)
        synchronization_score = execution_results.get('coordination_events', 0) / 20.0  # Normalize
        synchronization_score = np.clip(synchronization_score, 0, 1)
        
        return {
            'coordination_efficiency': np.clip(coordination_efficiency, 0, 2),
            'role_utilization': role_utilization,
            'communication_effectiveness': communication_effectiveness,
            'synchronization_score': synchronization_score,
            'overall_coordination': np.mean([
                coordination_efficiency, role_utilization, 
                communication_effectiveness, synchronization_score
            ])
        }
    
    def _calculate_role_utilization(self, 
                                   team: Team, 
                                   individual_scores: Dict[str, float]) -> float:
        """Calculate how well assigned roles match actual performance."""
        if not individual_scores:
            return 0.5
        
        role_performance_match = 0
        
        for member in team.members:
            agent_id = member.agent_id
            assigned_role = team.role_assignments.get(agent_id, MultiAgentRole.COLLABORATOR)
            actual_performance = individual_scores.get(agent_id, 0)
            
            # Expected performance based on role
            skills = self.agent_skills[agent_id]
            
            if assigned_role == MultiAgentRole.LEADER:
                expected = skills['leadership']
            elif assigned_role == MultiAgentRole.FOLLOWER:
                expected = skills['coordination']
            elif assigned_role == MultiAgentRole.COLLABORATOR:
                expected = (skills['coordination'] + skills['object_manipulation']) / 2
            elif assigned_role == MultiAgentRole.EXPLORER:
                expected = skills['exploration']
            elif assigned_role == MultiAgentRole.COMPETITOR:
                expected = skills['competition']
            else:
                expected = skills['specialization']
            
            # Calculate match
            match = 1 - abs(actual_performance - expected)
            role_performance_match += match
        
        return role_performance_match / len(team.members)
    
    def _update_agent_skills(self, 
                            team: Team, 
                            execution_results: Dict[str, Any],
                            coordination_metrics: Dict[str, float]):
        """Update agent skill estimates based on team performance."""
        team_score = execution_results.get('team_score', 0)
        individual_scores = execution_results.get('individual_scores', {})
        coordination_quality = coordination_metrics.get('overall_coordination', 0)
        
        for member in team.members:
            agent_id = member.agent_id
            role = team.role_assignments.get(agent_id, MultiAgentRole.COLLABORATOR)
            individual_score = individual_scores.get(agent_id, 0)
            
            # Update skills based on performance and role
            skills = self.agent_skills[agent_id]
            
            # Base skill update
            learning_rate = 0.1
            skills['navigation'] += learning_rate * (individual_score - skills['navigation'])
            skills['object_manipulation'] += learning_rate * (individual_score - skills['object_manipulation'])
            
            # Role-specific skill updates
            if role == MultiAgentRole.LEADER:
                skills['leadership'] += learning_rate * (team_score - skills['leadership'])
                skills['coordination'] += learning_rate * (coordination_quality - skills['coordination'])
            elif role == MultiAgentRole.FOLLOWER:
                skills['coordination'] += learning_rate * (coordination_quality - skills['coordination'])
            elif role == MultiAgentRole.COLLABORATOR:
                skills['coordination'] += learning_rate * (coordination_quality - skills['coordination'])
            elif role == MultiAgentRole.EXPLORER:
                skills['exploration'] += learning_rate * (individual_score - skills['exploration'])
            elif role == MultiAgentRole.COMPETITOR:
                skills['competition'] += learning_rate * (team_score - skills['competition'])
            
            # Ensure skills stay in [0, 1] range
            for skill in skills:
                skills[skill] = np.clip(skills[skill], 0, 1)
    
    def _log_team_formation(self, current_step: int, task_mode: GameMode):
        """Log team formation information."""
        self.logger.info(
            f"Step {current_step}: Formed {len(self.teams)} teams for {task_mode.value} mode"
        )
        
        for team in self.teams:
            role_str = ", ".join([f"{agent_id}:{role.value}" 
                                 for agent_id, role in team.role_assignments.items()])
            self.logger.debug(f"Team {team.team_id}: {role_str}")
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics."""
        if not self.teams:
            return {}
        
        # Team statistics
        team_performances = [team.team_performance for team in self.teams]
        coordination_scores = [team.coordination_score for team in self.teams]
        
        # Role distribution
        role_counts = defaultdict(int)
        for team in self.teams:
            for role in team.role_assignments.values():
                role_counts[role] += 1
        
        # Skill distribution
        skill_averages = defaultdict(list)
        for agent_id, skills in self.agent_skills.items():
            for skill, value in skills.items():
                skill_averages[skill].append(value)
        
        skill_stats = {skill: {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        } for skill, values in skill_averages.items()}
        
        return {
            'num_teams': len(self.teams),
            'avg_team_size': np.mean([len(team.members) for team in self.teams]),
            'team_performance_stats': {
                'mean': np.mean(team_performances) if team_performances else 0,
                'std': np.std(team_performances) if team_performances else 0,
                'min': np.min(team_performances) if team_performances else 0,
                'max': np.max(team_performances) if team_performances else 0
            },
            'coordination_stats': {
                'mean': np.mean(coordination_scores) if coordination_scores else 0,
                'std': np.std(coordination_scores) if coordination_scores else 0
            },
            'role_distribution': {role.value: count for role, count in role_counts.items()},
            'skill_statistics': skill_stats,
            'coordination_history_length': len(self.coordination_history)
        }
    
    def save_coordination_state(self, filepath: str):
        """Save current coordination state to file."""
        state = {
            'teams': [
                {
                    'team_id': team.team_id,
                    'members': [member.agent_id for member in team.members],
                    'role_assignments': {agent_id: role.value for agent_id, role in team.role_assignments.items()},
                    'team_performance': team.team_performance,
                    'coordination_score': team.coordination_score,
                    'task_history': team.task_history
                }
                for team in self.teams
            ],
            'agent_skills': self.agent_skills,
            'coordination_statistics': self.get_coordination_statistics(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Coordination state saved to {filepath}")

class MultiAgentEvaluator:
    """Evaluates multi-agent coordination and team performance."""
    
    def __init__(self, coordinator: MultiAgentCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
    
    def evaluate_team_performance(self, 
                                 teams: List[Team],
                                 evaluation_tasks: List[TaskInstance]) -> Dict[str, Any]:
        """Evaluate team performance on held-out tasks."""
        team_results = {}
        
        for team in teams:
            team_scores = []
            coordination_scores = []
            
            for task in evaluation_tasks:
                # Execute task with team
                result = self.coordinator.coordinate_team_execution(team, task)
                
                team_scores.append(result['team_performance'])
                coordination_scores.append(result['coordination_metrics']['overall_coordination'])
            
            team_results[team.team_id] = {
                'avg_team_score': np.mean(team_scores),
                'avg_coordination_score': np.mean(coordination_scores),
                'score_consistency': 1 - np.std(team_scores),
                'coordination_consistency': 1 - np.std(coordination_scores),
                'num_tasks': len(evaluation_tasks)
            }
        
        return team_results
    
    def evaluate_role_effectiveness(self) -> Dict[str, Any]:
        """Evaluate effectiveness of role assignments."""
        role_performance = defaultdict(list)
        
        for team in self.coordinator.teams:
            for member in team.members:
                agent_id = member.agent_id
                role = team.role_assignments.get(agent_id)
                
                if role:
                    # Get agent's recent performance
                    if member.performance_history:
                        recent_performance = np.mean(list(member.performance_history)[-10:])
                        role_performance[role.value].append(recent_performance)
        
        # Calculate role effectiveness statistics
        role_stats = {}
        for role, performances in role_performance.items():
            if performances:
                role_stats[role] = {
                    'mean_performance': np.mean(performances),
                    'performance_std': np.std(performances),
                    'num_agents': len(performances)
                }
        
        return role_stats
    
    def analyze_coordination_patterns(self) -> Dict[str, Any]:
        """Analyze coordination patterns across teams."""
        if not self.coordinator.teams:
            return {}
        
        patterns = {
            'team_composition_analysis': self._analyze_team_composition(),
            'role_synergy_analysis': self._analyze_role_synergy(),
            'skill_development_analysis': self._analyze_skill_development()
        }
        
        return patterns
    
    def _analyze_team_composition(self) -> Dict[str, Any]:
        """Analyze team composition patterns."""
        team_sizes = [len(team.members) for team in self.coordinator.teams]
        role_diversity = [len(set(team.role_assignments.values())) 
                         for team in self.coordinator.teams]
        
        return {
            'avg_team_size': np.mean(team_sizes),
            'team_size_distribution': {size: team_sizes.count(size) for size in set(team_sizes)},
            'avg_role_diversity': np.mean(role_diversity),
            'role_diversity_distribution': {div: role_diversity.count(div) for div in set(role_diversity)}
        }
    
    def _analyze_role_synergy(self) -> Dict[str, Any]:
        """Analyze synergy between different roles."""
        role_combinations = defaultdict(list)
        
        for team in self.coordinator.teams:
            roles = sorted([role.value for role in team.role_assignments.values()])
            role_combo = "_".join(roles)
            
            role_combinations[role_combo].append(team.team_performance)
        
        # Calculate synergy scores
        synergy_analysis = {}
        for combo, performances in role_combinations.items():
            if len(performances) >= 2:
                synergy_analysis[combo] = {
                    'avg_performance': np.mean(performances),
                    'performance_consistency': 1 - np.std(performances),
                    'num_teams': len(performances)
                }
        
        return synergy_analysis
    
    def _analyze_skill_development(self) -> Dict[str, Any]:
        """Analyze skill development trends."""
        skill_trends = {}
        
        for skill in self.coordinator.agent_skills[next(iter(self.coordinator.agent_skills))].keys():
            skill_values = [skills[skill] for skills in self.coordinator.agent_skills.values()]
            skill_trends[skill] = {
                'current_average': np.mean(skill_values),
                'skill_variance': np.var(skill_values),
                'skill_range': (np.min(skill_values), np.max(skill_values)),
                'development_needed': np.mean(skill_values) < 0.7
            }
        
        return skill_trends
