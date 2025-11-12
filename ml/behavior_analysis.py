import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

from .neural_networks import GoalAttentiveAgent
from .rl_core import PPOAgent
from .task_generation import TaskInstance, TaskComplexity

@dataclass
class BehaviorMetrics:
    """Metrics for analyzing agent behavior."""
    action_distribution: Dict[str, float]
    movement_patterns: List[Tuple[int, int]]
    goal_completion_strategies: List[str]
    exploration_efficiency: float
    learning_curves: Dict[str, List[float]]
    attention_patterns: Dict[str, np.ndarray]
    emergent_behaviors: List[str]

class BehaviorAnalyzer:
    """Analyzes agent behavior patterns and emergent capabilities."""
    
    def __init__(self, 
                 save_trajectories: bool = True,
                 analysis_window: int = 1000):
        self.save_trajectories = save_trajectories
        self.analysis_window = analysis_window
        self.logger = logging.getLogger(__name__)
        
        # Behavior tracking
        self.trajectory_buffer = deque(maxlen=analysis_window)
        self.action_history = defaultdict(list)
        self.position_history = defaultdict(list)
        self.goal_history = defaultdict(list)
        self.attention_weights = defaultdict(list)
        
        # Emergent behavior detection
        self.behavior_patterns = {}
        self.cluster_models = {}
        
    def record_episode(self, 
                      agent_id: str,
                      episode_data: Dict[str, Any]):
        """Record episode data for behavior analysis."""
        trajectory = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'episode_id': episode_data.get('episode_id', len(self.trajectory_buffer)),
            'task_id': episode_data.get('task_id', 'unknown'),
            'goal_type': episode_data.get('goal_type', 'unknown'),
            'states': episode_data.get('states', []),
            'actions': episode_data.get('actions', []),
            'rewards': episode_data.get('rewards', []),
            'positions': episode_data.get('positions', []),
            'goal_description': episode_data.get('goal_description', ''),
            'attention_weights': episode_data.get('attention_weights', []),
            'success': episode_data.get('success', False),
            'episode_length': len(episode_data.get('actions', [])),
            'total_reward': sum(episode_data.get('rewards', []))
        }
        
        self.trajectory_buffer.append(trajectory)
        
        # Update specific tracking
        self.action_history[agent_id].extend(episode_data.get('actions', []))
        self.position_history[agent_id].extend(episode_data.get('positions', []))
        self.goal_history[agent_id].append(episode_data.get('goal_type', 'unknown'))
        
        if 'attention_weights' in episode_data:
            self.attention_weights[agent_id].extend(episode_data['attention_weights'])
    
    def analyze_movement_patterns(self, 
                                 agent_id: str) -> Dict[str, Any]:
        """Analyze movement patterns and navigation strategies."""
        if agent_id not in self.position_history:
            return {}
        
        positions = self.position_history[agent_id]
        if len(positions) < 10:
            return {}
        
        # Calculate movement statistics
        movements = np.diff(positions, axis=0)
        movement_distances = np.sqrt(np.sum(movements**2, axis=1))
        
        # Movement efficiency
        total_distance = np.sum(movement_distances)
        net_displacement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        efficiency = net_displacement / (total_distance + 1e-8)
        
        # Direction preferences
        directions = np.arctan2(movements[:, 1], movements[:, 0])
        direction_bins = np.histogram(directions, bins=8, range=(-np.pi, np.pi))[0]
        direction_preferences = direction_bins / np.sum(direction_bins)
        
        # Area coverage
        unique_positions = len(set(map(tuple, positions)))
        max_positions = len(positions)
        coverage_ratio = unique_positions / max_positions
        
        # Path complexity (turn frequency)
        turn_angles = np.diff(directions)
        turn_frequency = np.sum(np.abs(turn_angles) > np.pi/4) / len(turn_angles)
        
        return {
            'movement_efficiency': efficiency,
            'direction_preferences': direction_preferences.tolist(),
            'area_coverage': coverage_ratio,
            'path_complexity': turn_frequency,
            'avg_step_size': np.mean(movement_distances),
            'movement_variance': np.var(movement_distances)
        }
    
    def analyze_goal_strategies(self, 
                               agent_id: str) -> Dict[str, Any]:
        """Analyze strategies used for different goal types."""
        if agent_id not in self.goal_history:
            return {}
        
        goal_types = self.goal_history[agent_id]
        episodes_by_goal = defaultdict(list)
        
        # Group episodes by goal type
        for i, trajectory in enumerate(self.trajectory_buffer):
            if trajectory['agent_id'] == agent_id:
                episodes_by_goal[trajectory['goal_type']].append(trajectory)
        
        strategy_analysis = {}
        
        for goal_type, episodes in episodes_by_goal.items():
            if len(episodes) < 3:
                continue
            
            # Success rate by goal type
            success_rate = np.mean([ep['success'] for ep in episodes])
            
            # Average completion time
            avg_length = np.mean([ep['episode_length'] for ep in episodes])
            
            # Action patterns for this goal type
            all_actions = []
            for ep in episodes:
                all_actions.extend(ep['actions'])
            
            action_distribution = self._calculate_action_distribution(all_actions)
            
            # Identify common strategies
            strategies = self._identify_goal_strategies(goal_type, episodes)
            
            strategy_analysis[goal_type] = {
                'success_rate': success_rate,
                'avg_completion_time': avg_length,
                'action_distribution': action_distribution,
                'identified_strategies': strategies,
                'episode_count': len(episodes)
            }
        
        return strategy_analysis
    
    def analyze_attention_patterns(self, 
                                  agent_id: str) -> Dict[str, Any]:
        """Analyze attention patterns across different goals and states."""
        if agent_id not in self.attention_weights:
            return {}
        
        attention_data = self.attention_weights[agent_id]
        if not attention_data:
            return {}
        
        # Aggregate attention weights
        attention_matrix = np.array(attention_data)
        
        # Attention consistency
        attention_consistency = 1 - np.mean(np.std(attention_matrix, axis=0))
        
        # Attention focus (how concentrated attention is)
        attention_entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-8), axis=1)
        avg_attention_entropy = np.mean(attention_entropy)
        
        # Goal-specific attention patterns
        goal_attention = defaultdict(list)
        for i, trajectory in enumerate(self.trajectory_buffer):
            if trajectory['agent_id'] == agent_id and i < len(attention_data):
                goal_attention[trajectory['goal_type']].append(attention_data[i])
        
        goal_attention_analysis = {}
        for goal_type, weights in goal_attention.items():
            if weights:
                avg_weights = np.mean(weights, axis=0)
                goal_attention_analysis[goal_type] = {
                    'average_attention': avg_weights.tolist(),
                    'attention_consistency': 1 - np.mean(np.std(weights, axis=0)),
                    'attention_entropy': -np.sum(avg_weights * np.log(avg_weights + 1e-8))
                }
        
        return {
            'overall_consistency': attention_consistency,
            'attention_entropy': avg_attention_entropy,
            'goal_specific_patterns': goal_attention_analysis
        }
    
    def detect_emergent_behaviors(self, 
                                 agent_id: str) -> List[str]:
        """Detect emergent behaviors in agent's play."""
        if len(self.trajectory_buffer) < 50:
            return []
        
        emergent_behaviors = []
        
        # Analyze recent trajectories
        recent_trajectories = [t for t in self.trajectory_buffer 
                              if t['agent_id'] == agent_id][-20:]
        
        # Tool use detection
        tool_use_patterns = self._detect_tool_use(recent_trajectories)
        emergent_behaviors.extend(tool_use_patterns)
        
        # Strategic planning detection
        strategic_patterns = self._detect_strategic_planning(recent_trajectories)
        emergent_behaviors.extend(strategic_patterns)
        
        # Social behavior detection (for multi-agent)
        social_patterns = self._detect_social_behaviors(recent_trajectories)
        emergent_behaviors.extend(social_patterns)
        
        # Experimentation behavior
        experimentation = self._detect_experimentation(recent_trajectories)
        emergent_behaviors.extend(experimentation)
        
        return emergent_behaviors
    
    def cluster_behavior_patterns(self, 
                                 agent_id: str,
                                 n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster behavior patterns to identify different strategies."""
        if agent_id not in self.position_history:
            return {}
        
        # Extract feature vectors for clustering
        feature_vectors = self._extract_behavior_features(agent_id)
        
        if len(feature_vectors) < n_clusters:
            return {}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_vectors)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = feature_vectors[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': np.sum(cluster_mask),
                'centroid': kmeans.cluster_centers_[cluster_id].tolist(),
                'characteristics': self._describe_cluster(cluster_features)
            }
        
        return {
            'cluster_analysis': cluster_analysis,
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': n_clusters
        }
    
    def generate_behavior_report(self, 
                                agent_id: str) -> Dict[str, Any]:
        """Generate comprehensive behavior analysis report."""
        report = {
            'agent_id': agent_id,
            'analysis_timestamp': time.time(),
            'data_points': len(self.trajectory_buffer),
            'movement_analysis': self.analyze_movement_patterns(agent_id),
            'goal_strategy_analysis': self.analyze_goal_strategies(agent_id),
            'attention_analysis': self.analyze_attention_patterns(agent_id),
            'emergent_behaviors': self.detect_emergent_behaviors(agent_id),
            'behavior_clusters': self.cluster_behavior_patterns(agent_id)
        }
        
        return report
    
    def _calculate_action_distribution(self, actions: List[int]) -> Dict[str, float]:
        """Calculate distribution of action types."""
        action_names = ['up', 'down', 'left', 'right', 'pickup', 'drop', 'interact']
        action_counts = defaultdict(int)
        
        for action in actions:
            if 0 <= action < len(action_names):
                action_counts[action_names[action]] += 1
        
        total_actions = len(actions)
        if total_actions == 0:
            return {}
        
        return {name: count / total_actions for name, count in action_counts.items()}
    
    def _identify_goal_strategies(self, 
                                 goal_type: str, 
                                 episodes: List[Dict]) -> List[str]:
        """Identify common strategies for specific goal types."""
        strategies = []
        
        if goal_type == 'collect':
            # Analyze collection strategies
            strategies.append(self._analyze_collection_strategy(episodes))
        elif goal_type == 'bring_to_zone':
            # Analyze transport strategies
            strategies.append(self._analyze_transport_strategy(episodes))
        elif goal_type == 'avoid_walls':
            # Analyze avoidance strategies
            strategies.append(self._analyze_avoidance_strategy(episodes))
        elif goal_type == 'touch_corners':
            # Analyze exploration strategies
            strategies.append(self._analyze_exploration_strategy(episodes))
        
        return [s for s in strategies if s]  # Filter out empty strings
    
    def _analyze_collection_strategy(self, episodes: List[Dict]) -> str:
        """Analyze object collection strategies."""
        # Check if agent systematically searches
        avg_lengths = [ep['episode_length'] for ep in episodes]
        success_rates = [ep['success'] for ep in episodes]
        
        if np.mean(success_rates) > 0.8 and np.mean(avg_lengths) < 20:
            return "efficient_direct_collection"
        elif np.mean(success_rates) > 0.6:
            return "systematic_search_collection"
        else:
            return "random_collection"
    
    def _analyze_transport_strategy(self, episodes: List[Dict]) -> str:
        """Analyze object transport strategies."""
        # Check pickup/drop patterns
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep['actions'])
        
        pickup_count = all_actions.count(4)  # Assuming pickup is action 4
        drop_count = all_actions.count(5)    # Assuming drop is action 5
        
        if pickup_count > 0 and drop_count > 0:
            return "coordinated_transport"
        else:
            return "inefficient_transport"
    
    def _analyze_avoidance_strategy(self, episodes: List[Dict]) -> str:
        """Analyze wall avoidance strategies."""
        # Check movement patterns
        avg_lengths = [ep['episode_length'] for ep in episodes]
        
        if np.mean(avg_lengths) > 15:
            return "cautious_avoidance"
        else:
            return "direct_path_avoidance"
    
    def _analyze_exploration_strategy(self, episodes: List[Dict]) -> str:
        """Analyze corner touching strategies."""
        # Check if agent systematically visits corners
        strategies = []
        for ep in episodes:
            positions = ep.get('positions', [])
            unique_corners = len(set([tuple(p) for p in positions 
                                    if p in [(0, 0), (0, 9), (9, 0), (9, 9)]]))
            if unique_corners >= 3:
                strategies.append("systematic_corner_exploration")
        
        if strategies:
            return np.random.choice(strategies)
        else:
            return "random_exploration"
    
    def _detect_tool_use(self, episodes: List[Dict]) -> List[str]:
        """Detect tool use behaviors."""
        tool_behaviors = []
        
        # Look for object manipulation patterns
        for ep in episodes:
            actions = ep.get('actions', [])
            # Check for complex action sequences
            for i in range(len(actions) - 2):
                if (actions[i] == 4 and actions[i+1] in [0, 1, 2, 3] and actions[i+2] == 5):
                    tool_behaviors.append("object_transport")
                    break
        
        return list(set(tool_behaviors))
    
    def _detect_strategic_planning(self, episodes: List[Dict]) -> List[str]:
        """Detect strategic planning behaviors."""
        strategic_behaviors = []
        
        # Look for efficient path planning
        for ep in episodes:
            if ep['success'] and ep['episode_length'] < 15:
                strategic_behaviors.append("efficient_planning")
        
        return list(set(strategic_behaviors))
    
    def _detect_social_behaviors(self, episodes: List[Dict]) -> List[str]:
        """Detect social behaviors in multi-agent scenarios."""
        social_behaviors = []
        
        # This would analyze multi-agent interactions
        # For now, return placeholder
        return social_behaviors
    
    def _detect_experimentation(self, episodes: List[Dict]) -> List[str]:
        """Detect experimentation behaviors."""
        experimentation_behaviors = []
        
        # Look for diverse action patterns
        for ep in episodes:
            actions = ep.get('actions', [])
            unique_actions = len(set(actions))
            if unique_actions >= 5:  # Uses many different actions
                experimentation_behaviors.append("action_exploration")
        
        return list(set(experimentation_behaviors))
    
    def _extract_behavior_features(self, agent_id: str) -> np.ndarray:
        """Extract feature vectors for behavior clustering."""
        features = []
        
        for trajectory in self.trajectory_buffer:
            if trajectory['agent_id'] != agent_id:
                continue
            
            # Extract features from trajectory
            episode_features = [
                trajectory['episode_length'],
                trajectory['total_reward'],
                len(set(trajectory['actions'])),  # Action diversity
                len(set(map(tuple, trajectory['positions']))),  # Position diversity
                trajectory['success']  # Success indicator
            ]
            
            features.append(episode_features)
        
        return np.array(features) if features else np.array([])
    
    def _describe_cluster(self, cluster_features: np.ndarray) -> Dict[str, float]:
        """Describe characteristics of a behavior cluster."""
        if len(cluster_features) == 0:
            return {}
        
        return {
            'avg_episode_length': np.mean(cluster_features[:, 0]),
            'avg_reward': np.mean(cluster_features[:, 1]),
            'avg_action_diversity': np.mean(cluster_features[:, 2]),
            'avg_position_diversity': np.mean(cluster_features[:, 3]),
            'success_rate': np.mean(cluster_features[:, 4])
        }

class BehaviorVisualizer:
    """Visualizes agent behavior patterns and analysis results."""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.color_palette = sns.color_palette("husl", 10)
    
    def plot_movement_heatmap(self, 
                             positions: List[Tuple[int, int]], 
                             grid_size: int = 10,
                             title: str = "Agent Movement Heatmap") -> plt.Figure:
        """Plot heatmap of agent movement positions."""
        if not positions:
            return None
        
        # Create position frequency matrix
        position_matrix = np.zeros((grid_size, grid_size))
        for x, y in positions:
            if 0 <= x < grid_size and 0 <= y < grid_size:
                position_matrix[y, x] += 1
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(position_matrix, annot=True, fmt='g', cmap='YlOrRd', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        return fig
    
    def plot_action_distribution(self, 
                                action_counts: Dict[str, float],
                                title: str = "Action Distribution") -> plt.Figure:
        """Plot distribution of actions taken by agent."""
        if not action_counts:
            return None
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(actions, counts, color=self.color_palette[:len(actions)])
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:.2f}', ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xlabel('Actions')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)
        
        return fig
    
    def plot_learning_curves(self, 
                            learning_data: Dict[str, List[float]],
                            title: str = "Learning Curves") -> plt.Figure:
        """Plot learning curves over time."""
        if not learning_data:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric, values in learning_data.items():
            if values:
                x = range(len(values))
                ax.plot(x, values, label=metric, linewidth=2, 
                       color=self.color_palette[len(learning_data) % len(self.color_palette)])
        
        ax.set_title(title)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_attention_patterns(self, 
                               attention_data: Dict[str, np.ndarray],
                               title: str = "Attention Patterns") -> plt.Figure:
        """Plot attention patterns across different goals."""
        if not attention_data:
            return None
        
        n_goals = len(attention_data)
        if n_goals == 0:
            return None
        
        fig, axes = plt.subplots(1, n_goals, figsize=(5*n_goals, 4))
        if n_goals == 1:
            axes = [axes]
        
        for i, (goal_type, attention_weights) in enumerate(attention_data.items()):
            if isinstance(attention_weights, list) and attention_weights:
                attention_weights = np.array(attention_weights)
            
            if len(attention_weights.shape) == 1:
                attention_weights = attention_weights.reshape(-1, 1)
            
            sns.heatmap(attention_weights, ax=axes[i], cmap='Blues', 
                       annot=True, fmt='.2f')
            axes[i].set_title(f'{goal_type.replace("_", " ").title()}')
            axes[i].set_xlabel('Attention Dimension')
            axes[i].set_ylabel('Time Step')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_behavior_clusters(self, 
                              cluster_data: Dict[str, Any],
                              title: str = "Behavior Clusters") -> plt.Figure:
        """Plot visualization of behavior clusters."""
        if 'cluster_labels' not in cluster_data:
            return None
        
        cluster_labels = np.array(cluster_data['cluster_labels'])
        n_clusters = cluster_data['n_clusters']
        
        # Create 2D visualization using t-SNE
        # This would need actual feature data, for now create dummy visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate dummy data for visualization
        np.random.seed(42)
        features = np.random.randn(len(cluster_labels), 2)
        
        # Plot clusters
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_points = features[mask]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      label=f'Cluster {cluster_id}', 
                      color=self.color_palette[cluster_id % len(self.color_palette)],
                      alpha=0.7, s=50)
        
        ax.set_title(title)
        ax.set_xlabel('Feature Dimension 1')
        ax.set_ylabel('Feature Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_emergent_behaviors(self, 
                               behavior_data: Dict[str, Any],
                               title: str = "Emergent Behaviors Over Time") -> plt.Figure:
        """Plot emergence of different behaviors over training."""
        emergent_behaviors = behavior_data.get('emergent_behaviors', [])
        if not emergent_behaviors:
            return None
        
        # Count behavior occurrences
        behavior_counts = defaultdict(int)
        for behavior in emergent_behaviors:
            behavior_counts[behavior] += 1
        
        if not behavior_counts:
            return None
        
        behaviors = list(behavior_counts.keys())
        counts = list(behavior_counts.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(behaviors, counts, color=self.color_palette[:len(behaviors)])
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   str(count), ha='left', va='center')
        
        ax.set_title(title)
        ax.set_xlabel('Occurrence Count')
        ax.set_ylabel('Emergent Behaviors')
        
        return fig
    
    def create_comprehensive_dashboard(self, 
                                      analysis_report: Dict[str, Any],
                                      save_path: str = "behavior_dashboard.png") -> plt.Figure:
        """Create comprehensive dashboard of behavior analysis."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Movement analysis
        if 'movement_analysis' in analysis_report:
            ax1 = fig.add_subplot(gs[0, 0])
            movement_data = analysis_report['movement_analysis']
            
            metrics = ['movement_efficiency', 'area_coverage', 'path_complexity']
            values = [movement_data.get(m, 0) for m in metrics]
            
            bars = ax1.bar(metrics, values, color=self.color_palette[:3])
            ax1.set_title('Movement Metrics')
            ax1.tick_params(axis='x', rotation=45)
        
        # Goal strategies
        if 'goal_strategy_analysis' in analysis_report:
            ax2 = fig.add_subplot(gs[0, 1])
            strategy_data = analysis_report['goal_strategy_analysis']
            
            goal_types = list(strategy_data.keys())
            success_rates = [strategy_data[gt]['success_rate'] for gt in goal_types]
            
            ax2.bar(goal_types, success_rates, color=self.color_palette[:len(goal_types)])
            ax2.set_title('Success Rate by Goal Type')
            ax2.tick_params(axis='x', rotation=45)
        
        # Action distribution
        if 'goal_strategy_analysis' in analysis_report:
            ax3 = fig.add_subplot(gs[0, 2])
            # Use first goal type's action distribution as example
            if strategy_data:
                first_goal = list(strategy_data.values())[0]
                action_dist = first_goal.get('action_distribution', {})
                
                actions = list(action_dist.keys())
                probs = list(action_dist.values())
                
                ax3.pie(probs, labels=actions, autopct='%1.1f%%')
                ax3.set_title('Action Distribution')
        
        # Attention patterns
        if 'attention_analysis' in analysis_report:
            ax4 = fig.add_subplot(gs[1, :2])
            attention_data = analysis_report['attention_analysis']
            
            if 'goal_specific_patterns' in attention_data:
                goals = list(attention_data['goal_specific_patterns'].keys())
                consistencies = [attention_data['goal_specific_patterns'][g]['attention_consistency'] 
                               for g in goals]
                
                ax4.bar(goals, consistencies, color=self.color_palette[:len(goals)])
                ax4.set_title('Attention Consistency by Goal Type')
                ax4.tick_params(axis='x', rotation=45)
        
        # Emergent behaviors
        if 'emergent_behaviors' in analysis_report:
            ax5 = fig.add_subplot(gs[1, 2])
            behaviors = analysis_report['emergent_behaviors']
            
            if behaviors:
                behavior_counts = defaultdict(int)
                for behavior in behaviors:
                    behavior_counts[behavior] += 1
                
                ax5.barh(list(behavior_counts.keys()), list(behavior_counts.values()))
                ax5.set_title('Emergent Behaviors')
        
        # Cluster analysis
        if 'behavior_clusters' in analysis_report:
            ax6 = fig.add_subplot(gs[2, :])
            cluster_data = analysis_report['behavior_clusters']
            
            if 'cluster_analysis' in cluster_data:
                clusters = list(cluster_data['cluster_analysis'].keys())
                sizes = [cluster_data['cluster_analysis'][c]['size'] for c in clusters]
                
                ax6.pie(sizes, labels=clusters, autopct='%1.1f%%')
                ax6.set_title('Behavior Cluster Distribution')
        
        fig.suptitle(f'Behavior Analysis Dashboard - Agent {analysis_report.get("agent_id", "Unknown")}', 
                    fontsize=16)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig

class BehaviorReporter:
    """Generates detailed reports of agent behavior analysis."""
    
    def __init__(self, output_dir: str = "behavior_reports"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, 
                                    analysis_report: Dict[str, Any],
                                    include_visualizations: bool = True) -> str:
        """Generate comprehensive behavior analysis report."""
        timestamp = int(time.time())
        report_path = f"{self.output_dir}/behavior_report_{timestamp}.json"
        
        # Prepare report data
        report_data = {
            'metadata': {
                'report_timestamp': time.time(),
                'agent_id': analysis_report.get('agent_id', 'unknown'),
                'analysis_data_points': analysis_report.get('data_points', 0)
            },
            'analysis_results': analysis_report,
            'summary': self._generate_summary(analysis_report),
            'recommendations': self._generate_recommendations(analysis_report)
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate visualizations
        if include_visualizations:
            visualizer = BehaviorVisualizer()
            dashboard_path = f"{self.output_dir}/behavior_dashboard_{timestamp}.png"
            visualizer.create_comprehensive_dashboard(analysis_report, dashboard_path)
        
        self.logger.info(f"Behavior analysis report saved to {report_path}")
        return report_path
    
    def _generate_summary(self, analysis_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of key findings."""
        summary = {
            'overall_assessment': 'moderate',  # Would be calculated
            'key_strengths': [],
            'areas_for_improvement': [],
            'notable_behaviors': []
        }
        
        # Analyze movement patterns
        movement_analysis = analysis_report.get('movement_analysis', {})
        if movement_analysis.get('movement_efficiency', 0) > 0.7:
            summary['key_strengths'].append('efficient_navigation')
        elif movement_analysis.get('movement_efficiency', 0) < 0.3:
            summary['areas_for_improvement'].append('navigation_efficiency')
        
        # Analyze goal strategies
        strategy_analysis = analysis_report.get('goal_strategy_analysis', {})
        for goal_type, data in strategy_analysis.items():
            if data.get('success_rate', 0) > 0.8:
                summary['key_strengths'].append(f'expertise_in_{goal_type}')
            elif data.get('success_rate', 0) < 0.4:
                summary['areas_for_improvement'].append(f'improvement_needed_{goal_type}')
        
        # Note emergent behaviors
        emergent_behaviors = analysis_report.get('emergent_behaviors', [])
        if emergent_behaviors:
            summary['notable_behaviors'] = emergent_behaviors
        
        return summary
    
    def _generate_recommendations(self, analysis_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        movement_analysis = analysis_report.get('movement_analysis', {})
        strategy_analysis = analysis_report.get('goal_strategy_analysis', {})
        
        # Movement recommendations
        if movement_analysis.get('movement_efficiency', 0) < 0.5:
            recommendations.append("Focus training on pathfinding and navigation efficiency")
        
        if movement_analysis.get('area_coverage', 0) < 0.3:
            recommendations.append("Encourage more exploration of the environment")
        
        # Strategy recommendations
        low_performance_goals = []
        for goal_type, data in strategy_analysis.items():
            if data.get('success_rate', 0) < 0.5:
                low_performance_goals.append(goal_type)
        
        if low_performance_goals:
            recommendations.append(f"Provide additional training on: {', '.join(low_performance_goals)}")
        
        # Attention recommendations
        attention_analysis = analysis_report.get('attention_analysis', {})
        if attention_analysis.get('attention_entropy', 1) > 0.8:
            recommendations.append("Agent shows unfocused attention - consider attention regularization")
        
        # General recommendations
        emergent_behaviors = analysis_report.get('emergent_behaviors', [])
        if not emergent_behaviors:
            recommendations.append("Consider curriculum adjustments to encourage emergent behaviors")
        
        return recommendations
