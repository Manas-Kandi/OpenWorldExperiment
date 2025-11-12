import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

@dataclass
class RLConfig:
    """Configuration for reinforcement learning algorithms."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    buffer_size: int = 2048
    target_kl: float = 0.01
    advantage_normalization: bool = True

class PPOAgent:
    """
    Proximal Policy Optimization agent for training in Mini-Quest Arena.
    Implements clipped surrogate objective and GAE advantage estimation.
    """
    
    def __init__(self, 
                 network: nn.Module,
                 config: RLConfig,
                 device: str = 'cpu'):
        self.network = network.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Experience buffer
        self.buffer = PPOBuffer(config.buffer_size, device)
        
        # Training statistics
        self.update_count = 0
        self.total_steps = 0
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=1000000
        )
        
    def act(self, 
            state: Dict[str, torch.Tensor],
            goal_text: str,
            deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action given current state and goal.
        
        Args:
            state: Current environment state
            goal_text: Text description of current goal
            deterministic: Whether to sample or take argmax
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """
        self.network.eval()
        
        with torch.no_grad():
            # Prepare state tensor
            grid_state = self._prepare_state(state)
            grid_state = grid_state.unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.network(grid_state, goal_text)
            
            # Get action distribution
            policy_logits = output['policy_logits']
            value = output['value']
            
            # Sample action
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
                log_prob = F.log_softmax(policy_logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
            else:
                action_dist = torch.distributions.Categorical(logits=policy_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, 
                         state: Dict[str, torch.Tensor],
                         action: int,
                         reward: float,
                         next_state: Dict[str, torch.Tensor],
                         done: bool,
                         log_prob: float,
                         value: float):
        """Store experience in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done, log_prob, value)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config.mini_batch_size:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns()
        
        # Update policy
        metrics = {}
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle indices
            indices = np.random.permutation(len(self.buffer))
            
            for start in range(0, len(self.buffer), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_data = self.buffer.get_batch(batch_indices, advantages, returns)
                
                # Perform update step
                step_metrics = self._update_step(batch_data)
                
                total_policy_loss += step_metrics['policy_loss']
                total_value_loss += step_metrics['value_loss']
                total_entropy_loss += step_metrics['entropy_loss']
                total_kl += step_metrics['kl_divergence']
        
        # Average metrics
        num_updates = self.config.ppo_epochs * (len(self.buffer) // self.config.mini_batch_size)
        metrics.update({
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'kl_divergence': total_kl / num_updates,
            'mean_advantage': torch.mean(advantages).item(),
            'mean_return': torch.mean(returns).item()
        })
        
        # Clear buffer
        self.buffer.clear()
        self.update_count += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return metrics
    
    def _compute_advantages_and_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and returns."""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        next_values = self.buffer.next_values
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # GAE computation
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Normalize advantages
        if self.config.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single PPO update step."""
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']
        goal_texts = batch_data['goal_texts']
        
        self.network.train()
        
        # Forward pass
        outputs = []
        for i, goal_text in enumerate(goal_texts):
            state_tensor = states[i].unsqueeze(0)
            output = self.network(state_tensor, goal_text)
            outputs.append(output)
        
        # Concatenate outputs
        policy_logits = torch.cat([out['policy_logits'] for out in outputs], dim=0)
        values = torch.cat([out['value'] for out in outputs], dim=0)
        
        # Compute new log probabilities
        action_dist = torch.distributions.Categorical(logits=policy_logits)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute policy loss (clipped surrogate objective)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Compute total loss
        loss = (policy_loss + 
                self.config.value_loss_coef * value_loss - 
                self.config.entropy_coef * entropy)
        
        # Compute KL divergence for early stopping
        with torch.no_grad():
            old_action_dist = torch.distributions.Categorical(logits=old_log_probs.unsqueeze(-1))
            kl = torch.distributions.kl.kl_divergence(old_action_dist, action_dist).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy.item(),
            'kl_divergence': kl.item()
        }
    
    def _prepare_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare state tensor for network input."""
        # Convert game state to tensor representation
        # This would be implemented based on the actual game state format
        if isinstance(state, torch.Tensor):
            return state
        else:
            # Convert dict/numpy to tensor
            return torch.FloatTensor(state).to(self.device)
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'update_count': self.update_count,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        self.total_steps = checkpoint['total_steps']

class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.goal_texts = []
        
    def add(self, 
            state: Dict[str, torch.Tensor],
            action: int,
            reward: float,
            next_state: Dict[str, torch.Tensor],
            done: bool,
            log_prob: float,
            value: float,
            goal_text: str = ""):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.goal_texts.append(goal_text)
        
        # Compute next value (simplified - would need network forward pass)
        self.next_values.append(0.0 if done else value)
        
        # Maintain capacity
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            self.next_values.pop(0)
            self.goal_texts.pop(0)
    
    def get_batch(self, 
                  indices: List[int], 
                  advantages: torch.Tensor, 
                  returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get batch of experiences."""
        batch_states = [self.states[i] for i in indices]
        batch_actions = torch.tensor([self.actions[i] for i in indices]).to(self.device)
        batch_log_probs = torch.tensor([self.log_probs[i] for i in indices]).to(self.device)
        batch_goal_texts = [self.goal_texts[i] for i in indices]
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'log_probs': batch_log_probs,
            'advantages': advantages[indices],
            'returns': returns[indices],
            'goal_texts': batch_goal_texts
        }
    
    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_values.clear()
        self.goal_texts.clear()
    
    def __len__(self):
        return len(self.states)
    
    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor(self._rewards, dtype=torch.float32).to(self.device)
    
    @rewards.setter
    def rewards(self, value):
        self._rewards = value
    
    @property
    def values(self) -> torch.Tensor:
        return torch.tensor(self._values, dtype=torch.float32).to(self.device)
    
    @values.setter
    def values(self, value):
        self._values = value
    
    @property
    def dones(self) -> torch.Tensor:
        return torch.tensor(self._dones, dtype=torch.float32).to(self.device)
    
    @dones.setter
    def dones(self, value):
        self._dones = value
    
    @property
    def next_values(self) -> torch.Tensor:
        return torch.tensor(self._next_values, dtype=torch.float32).to(self.device)
    
    @next_values.setter
    def next_values(self, value):
        self._next_values = value

class A2CAgent:
    """
    Advantage Actor-Critic agent as alternative to PPO.
    Simpler implementation for comparison and debugging.
    """
    
    def __init__(self, 
                 network: nn.Module,
                 config: RLConfig,
                 device: str = 'cpu'):
        self.network = network.to(device)
        self.config = config
        self.device = device
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(
            list(self.network.grid_encoder.parameters()) + 
            list(self.network.goal_encoder.parameters()) +
            list(self.network.policy_head.parameters()),
            lr=config.learning_rate
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.network.value_head.parameters()),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer = deque(maxlen=config.buffer_size)
        
    def act(self, state: Dict[str, torch.Tensor], goal_text: str) -> Tuple[int, float, float]:
        """Select action using current policy."""
        self.network.eval()
        
        with torch.no_grad():
            grid_state = self._prepare_state(state).unsqueeze(0).to(self.device)
            output = self.network(grid_state, goal_text)
            
            policy_logits = output['policy_logits']
            value = output['value']
            
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store experience for training."""
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks."""
        if len(self.buffer) < self.config.mini_batch_size:
            return {}
        
        # Sample batch
        batch = list(self.buffer)
        self.buffer.clear()
        
        # Process batch
        states, actions, rewards, next_states, dones, log_probs, values = zip(*batch)
        
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32).to(self.device)
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards_tensor, dones)
        advantages = returns - values_tensor
        
        # Update networks
        actor_loss, critic_loss = self._update_networks(
            states, actions, advantages, returns, log_probs_tensor
        )
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'mean_return': returns.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def _compute_returns(self, rewards: torch.Tensor, dones: List[bool]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return * (0 if dones[t] else 1)
            returns[t] = running_return
        
        return returns
    
    def _update_networks(self, states, actions, advantages, returns, old_log_probs):
        """Update actor and critic networks."""
        self.network.train()
        
        # Forward pass for all states
        policy_losses = []
        value_losses = []
        
        for i, (state, action, advantage, return_val, old_log_prob) in enumerate(
            zip(states, actions, advantages, returns, old_log_probs)):
            
            grid_state = self._prepare_state(state).unsqueeze(0).to(self.device)
            goal_text = "collect red cube"  # Simplified
            
            output = self.network(grid_state, goal_text)
            
            policy_logits = output['policy_logits']
            value = output['value']
            
            # Actor loss
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            new_log_prob = action_dist.log_prob(torch.tensor(action).to(self.device))
            actor_loss = -(new_log_prob * advantage).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(value, return_val.unsqueeze(0))
            
            policy_losses.append(actor_loss)
            value_losses.append(critic_loss)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        torch.stack(policy_losses).mean().backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        torch.stack(value_losses).mean().backward()
        self.critic_optimizer.step()
        
        return torch.stack(policy_losses).mean().item(), torch.stack(value_losses).mean().item()
    
    def _prepare_state(self, state) -> torch.Tensor:
        """Prepare state tensor."""
        if isinstance(state, torch.Tensor):
            return state
        else:
            return torch.FloatTensor(state).to(self.device)

# Utility functions for training
def compute_nash_equilibrium_values(agents: List[PPOAgent], 
                                   evaluation_tasks: List[Any]) -> Dict[str, float]:
    """
    Compute Nash equilibrium values for score normalization.
    This is a simplified version - in practice would require more sophisticated game theory.
    """
    # Placeholder implementation
    # Would compute equilibrium values by having agents play against each other
    return {agent_id: 0.0 for agent_id in range(len(agents))}

def normalize_scores(scores: Dict[str, float], 
                    nash_values: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores using Nash equilibrium values."""
    normalized = {}
    for agent_id, score in scores.items():
        baseline = nash_values.get(agent_id, 0.0)
        normalized[agent_id] = (score - baseline) / (abs(baseline) + 1e-8)
    return normalized
