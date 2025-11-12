import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class GoalAttentiveAgent(nn.Module):
    """
    Goal-Attentive Agent (GOAT) neural network architecture.
    Implements attention mechanism over internal recurrent state to guide
    agent's attention with estimates of subgoals unique to each game.
    """
    
    def __init__(self, 
                 grid_size: int = 10,
                 num_object_types: int = 10,
                 num_colors: int = 5,
                 goal_embedding_dim: int = 64,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8,
                 num_layers: int = 3):
        super(GoalAttentiveAgent, self).__init__()
        
        self.grid_size = grid_size
        self.num_object_types = num_object_types
        self.num_colors = num_colors
        self.goal_embedding_dim = goal_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        
        # Input encoders
        self.grid_encoder = GridEncoder(grid_size, num_object_types, num_colors, hidden_dim)
        self.goal_encoder = GoalEncoder(goal_embedding_dim, hidden_dim)
        
        # Recurrent state processing with attention
        self.recurrent_encoder = nn.LSTM(
            input_size=hidden_dim * 2,  # grid + goal
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Goal-attention mechanism
        self.goal_attention = GoalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            goal_dim=goal_embedding_dim
        )
        
        # Subgoal estimation network
        self.subgoal_network = SubgoalNetwork(
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            num_subgoals=4
        )
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            hidden_dim=hidden_dim,
            num_actions=7,  # up, down, left, right, pickup, drop, interact
            grid_size=grid_size
        )
        
        self.value_head = ValueHead(hidden_dim=hidden_dim)
        
        # Auxiliary tasks for better representation learning
        self.next_state_predictor = NextStatePredictor(hidden_dim, grid_size, num_object_types, num_colors)
        self.reward_predictor = RewardPredictor(hidden_dim)
        
    def forward(self, 
                grid_state: torch.Tensor,
                goal_text: str,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GOAT network.
        
        Args:
            grid_state: Tensor of shape (batch_size, grid_size, grid_size, features)
            goal_text: Text description of the current goal
            hidden_state: Previous LSTM hidden state
            
        Returns:
            Dictionary containing policy, value, attention weights, and auxiliary outputs
        """
        batch_size = grid_state.shape[0]
        
        # Encode inputs
        grid_features = self.grid_encoder(grid_state)  # (batch_size, hidden_dim)
        goal_features = self.goal_encoder(goal_text)    # (batch_size, hidden_dim)
        
        # Combine features
        combined_features = torch.cat([grid_features, goal_features], dim=-1)
        combined_features = combined_features.unsqueeze(1)  # (batch_size, 1, hidden_dim*2)
        
        # Process through recurrent network
        lstm_out, new_hidden = self.recurrent_encoder(combined_features, hidden_state)
        recurrent_features = lstm_out.squeeze(1)  # (batch_size, hidden_dim)
        
        # Apply goal attention
        attended_features, attention_weights = self.goal_attention(
            recurrent_features, goal_features, grid_features
        )
        
        # Estimate subgoals
        subgoal_estimates = self.subgoal_network(attended_features)
        
        # Generate policy and value
        policy_logits = self.policy_head(attended_features, subgoal_estimates)
        value = self.value_head(attended_features)
        
        # Auxiliary predictions
        next_state_pred = self.next_state_predictor(attended_features, grid_state)
        reward_pred = self.reward_predictor(attended_features)
        
        return {
            'policy_logits': policy_logits,
            'value': value,
            'attention_weights': attention_weights,
            'subgoal_estimates': subgoal_estimates,
            'next_state_prediction': next_state_pred,
            'reward_prediction': reward_pred,
            'hidden_state': new_hidden,
            'recurrent_features': recurrent_features
        }

class GridEncoder(nn.Module):
    """Encodes the grid state into a compact representation."""
    
    def __init__(self, grid_size: int, num_object_types: int, num_colors: int, hidden_dim: int):
        super(GridEncoder, self).__init__()
        
        self.grid_size = grid_size
        
        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_object_types * num_colors + 3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Positional encoding for spatial awareness
        self.positional_encoding = PositionalEncoding2D(grid_size, 256)
        
        # Global pooling and final encoding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, grid_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_state: (batch_size, grid_size, grid_size, features)
        Returns:
            encoded_features: (batch_size, hidden_dim)
        """
        # Rearrange for convolution: (batch_size, channels, height, width)
        x = grid_state.permute(0, 3, 1, 2)
        
        # Extract spatial features
        conv_features = self.conv_layers(x)
        
        # Add positional encoding
        conv_features = conv_features + self.positional_encoding(conv_features)
        
        # Global pooling
        pooled = self.global_pool(conv_features).squeeze(-1).squeeze(-1)
        
        # Final encoding
        encoded = self.encoder(pooled)
        
        return encoded

class GoalEncoder(nn.Module):
    """Encodes text goals into vector representations."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(GoalEncoder, self).__init__()
        
        # Simple text embedding (in practice, would use pretrained language models)
        self.vocab_size = 1000  # Simplified vocabulary
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # Goal type classification
        self.goal_types = ['collect', 'bring_to_zone', 'avoid_walls', 'touch_corners', 
                          'collect_multiple', 'clear_zone', 'cooperative_collect', 
                          'competitive_collect', 'cooperative_zone', 'competitive_race']
        self.goal_type_embedding = nn.Embedding(len(self.goal_types), embedding_dim)
        
        # Text processing
        self.text_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Goal attention over text tokens
        self.goal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final encoding
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, goal_text: str) -> torch.Tensor:
        """
        Args:
            goal_text: Text description of goal
        Returns:
            encoded_goal: (batch_size, hidden_dim)
        """
        # In practice, would tokenize and embed actual text
        # For now, use simplified encoding based on goal type
        
        # Identify goal type (simplified)
        goal_type_idx = self._identify_goal_type(goal_text)
        goal_type_tensor = torch.tensor([[goal_type_idx]], dtype=torch.long)
        
        # Embed goal type
        goal_embedded = self.goal_type_embedding(goal_type_tensor)
        
        # Process through LSTM (sequence length 1 for simplicity)
        lstm_out, _ = self.text_encoder(goal_embedded)
        
        # Self-attention
        attended, _ = self.goal_attention(lstm_out, lstm_out, lstm_out)
        
        # Final encoding
        encoded = self.encoder(attended.squeeze(1))
        
        return encoded
    
    def _identify_goal_type(self, goal_text: str) -> int:
        """Simple goal type identification - would use NLP in practice."""
        for i, goal_type in enumerate(self.goal_types):
            if goal_type in goal_text.lower():
                return i
        return 0  # Default to first type

class GoalAttention(nn.Module):
    """Attention mechanism that focuses on goal-relevant features."""
    
    def __init__(self, hidden_dim: int, num_heads: int, goal_dim: int):
        super(GoalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for goal-guided feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Goal query projection
        self.goal_query = nn.Linear(goal_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, 
                recurrent_features: torch.Tensor,
                goal_features: torch.Tensor,
                grid_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            recurrent_features: (batch_size, hidden_dim)
            goal_features: (batch_size, hidden_dim)
            grid_features: (batch_size, hidden_dim)
        Returns:
            attended_features: (batch_size, hidden_dim)
            attention_weights: (batch_size, 1, 3)
        """
        batch_size = recurrent_features.shape[0]
        
        # Create query from goal features
        goal_query = self.goal_query(goal_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Create key-value pairs from all features
        features = torch.stack([recurrent_features, grid_features, goal_features], dim=1)
        # (batch_size, 3, hidden_dim)
        
        # Apply attention
        attended, attention_weights = self.attention(
            goal_query, features, features
        )
        
        # Squeeze and project
        attended_features = self.output_proj(attended.squeeze(1))
        
        return attended_features, attention_weights

class SubgoalNetwork(nn.Module):
    """Estimates subgoals based on current state and goal."""
    
    def __init__(self, hidden_dim: int, grid_size: int, num_subgoals: int):
        super(SubgoalNetwork, self).__init__()
        
        self.num_subgoals = num_subgoals
        self.grid_size = grid_size
        
        # Subgoal estimation
        self.subgoal_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_subgoals * 3),  # x, y, type for each subgoal
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, hidden_dim)
        Returns:
            subgoals: (batch_size, num_subgoals, 3) - x, y, type
        """
        subgoal_raw = self.subgoal_estimator(features)
        subgoals = subgoal_raw.view(-1, self.num_subgoals, 3)
        
        # Scale coordinates to grid size
        subgoals[:, :, :2] = subgoals[:, :, :2] * self.grid_size
        
        return subgoals

class PolicyHead(nn.Module):
    """Policy head with spatial and action components."""
    
    def __init__(self, hidden_dim: int, num_actions: int, grid_size: int):
        super(PolicyHead, self).__init__()
        
        self.num_actions = num_actions
        self.grid_size = grid_size
        
        # Movement policy (spatial)
        self.movement_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # up, down, left, right
            nn.LogSoftmax(dim=-1)
        )
        
        # Action policy (non-spatial)
        self.action_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # pickup, drop, interact
            nn.LogSoftmax(dim=-1)
        )
        
        # Subgoal-guided attention
        self.subgoal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, features: torch.Tensor, subgoals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, hidden_dim)
            subgoals: (batch_size, num_subgoals, 3)
        Returns:
            policy_logits: (batch_size, num_actions)
        """
        # Use subgoals to guide attention
        subgoal_features = subgoals.view(subgoals.shape[0], subgoals.shape[1], -1)
        subgoal_features = subgoal_features[:, :, :features.shape[-1]]  # Match dimensions
        
        # Pad or truncate to match feature dimension
        if subgoal_features.shape[-1] < features.shape[-1]:
            padding = torch.zeros(subgoal_features.shape[0], subgoal_features.shape[1], 
                                 features.shape[-1] - subgoal_features.shape[-1])
            subgoal_features = torch.cat([subgoal_features, padding], dim=-1)
        else:
            subgoal_features = subgoal_features[:, :, :features.shape[-1]]
        
        # Apply attention
        features_expanded = features.unsqueeze(1)
        attended, _ = self.subgoal_attention(features_expanded, subgoal_features, subgoal_features)
        attended_features = attended.squeeze(1)
        
        # Generate movement and action policies
        movement_logits = self.movement_policy(attended_features)
        action_logits = self.action_policy(attended_features)
        
        # Combine policies
        policy_logits = torch.cat([movement_logits, action_logits], dim=-1)
        
        return policy_logits

class ValueHead(nn.Module):
    """Value function estimator."""
    
    def __init__(self, hidden_dim: int):
        super(ValueHead, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, hidden_dim)
        Returns:
            value: (batch_size, 1)
        """
        return self.value_net(features)

class NextStatePredictor(nn.Module):
    """Auxiliary task: predict next state."""
    
    def __init__(self, hidden_dim: int, grid_size: int, num_object_types: int, num_colors: int):
        super(NextStatePredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size * grid_size * (num_object_types * num_colors + 3))
        )
        
    def forward(self, features: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Predict next state given current features and state."""
        prediction = self.predictor(features)
        return prediction

class RewardPredictor(nn.Module):
    """Auxiliary task: predict immediate reward."""
    
    def __init__(self, hidden_dim: int):
        super(RewardPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Reward between -1 and 1
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict immediate reward."""
        return self.predictor(features)

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial awareness."""
    
    def __init__(self, grid_size: int, d_model: int):
        super(PositionalEncoding2D, self).__init__()
        
        self.grid_size = grid_size
        self.d_model = d_model
        
        # Create positional encodings
        pe = torch.zeros(grid_size, grid_size, d_model)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(0, d_model, 2):
                    div_term = 10000.0 ** (k / d_model)
                    pe[i, j, k] = np.sin(i / div_term + j / div_term)
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = np.cos(i / div_term + j / div_term)
        
        self.register_buffer('pe', pe.permute(2, 0, 1).unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            positional_encoding: (batch_size, channels, height, width)
        """
        return self.pe[:, :, :x.shape[2], :x.shape[3]]
