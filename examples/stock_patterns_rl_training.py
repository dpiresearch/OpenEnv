#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reinforcement Learning Training Example for Stock Patterns Environment.

This demonstrates a complete RL training loop using a simple DQN agent
to learn trading strategies on stock chart patterns.

The agent learns to:
1. Identify when to buy stocks
2. Hold positions optimally
3. Sell at the right time
4. Maximize profits across different patterns

Usage:
    # First, start the server:
    python -m envs.stock_patterns_env.server.app
    
    # Then run this training script:
    python examples/stock_patterns_rl_training.py
    
    # Or with custom settings:
    python examples/stock_patterns_rl_training.py --episodes 500 --difficulty 0.3
"""

import sys
import argparse
from pathlib import Path
from collections import deque
import random
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

# Try to import numpy, use basic Python if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not found. Using basic Python implementation (slower).")


class SimpleQAgent:
    """
    Simple Q-Learning agent for trading stock patterns.
    
    This is a basic implementation for demonstration purposes.
    For production, consider using PyTorch/TensorFlow with DQN or PPO.
    """
    
    def __init__(self, state_size=20, action_size=3, learning_rate=0.001, 
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_size: Number of features in state representation
            action_size: Number of possible actions (Hold=0, Buy=1, Sell=2)
            learning_rate: Learning rate for Q-value updates
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Simple Q-table (for discrete states) or feature approximation
        self.q_table = {}
        
    def get_state_key(self, state):
        """Convert state to hashable key for Q-table."""
        # Discretize prices into bins for Q-table lookup
        if len(state) == 0:
            return (0,) * 5
        
        # Use last 5 price changes (discretized)
        recent_prices = state[-5:] if len(state) >= 5 else [0] * (5 - len(state)) + state
        
        # Discretize to bins: -2 (big drop), -1 (small drop), 0 (flat), 1 (small rise), 2 (big rise)
        discretized = []
        for i in range(len(recent_prices)):
            if i == 0:
                discretized.append(0)
            else:
                change = recent_prices[i] - recent_prices[i-1]
                if change < -0.02:
                    discretized.append(-2)
                elif change < -0.005:
                    discretized.append(-1)
                elif change > 0.02:
                    discretized.append(2)
                elif change > 0.005:
                    discretized.append(1)
                else:
                    discretized.append(0)
        
        return tuple(discretized[-5:])  # Last 5 discretized changes
    
    def act(self, state, position=0, valid_actions=None):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state (list of prices)
            position: Current position (0=neutral, 1=long)
            valid_actions: List of valid actions (None = all valid)
        
        Returns:
            Action to take (0=Hold, 1=Buy, 2=Sell)
        """
        # Filter valid actions based on position
        if valid_actions is None:
            if position == 0:
                valid_actions = [0, 1]  # Can hold or buy
            else:
                valid_actions = [0, 2]  # Can hold or sell
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best known action
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            # Initialize Q-values for new state
            self.q_table[state_key] = [0.0] * self.action_size
        
        # Get Q-values and choose best valid action
        q_values = self.q_table[state_key]
        best_action = max(valid_actions, key=lambda a: q_values[a])
        
        return best_action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train on a batch of experiences from replay buffer.
        
        Returns:
            Average loss for this batch
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            # Initialize Q-values if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_size
            
            # Q-learning update
            current_q = self.q_table[state_key][action]
            
            if done:
                target_q = reward
            else:
                max_next_q = max(self.q_table[next_state_key])
                target_q = reward + self.gamma * max_next_q
            
            # Update Q-value
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
            
            # Track loss
            total_loss += abs(target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / self.batch_size
    
    def save(self, filepath):
        """Save Q-table to file."""
        import json
        data = {
            'q_table': {str(k): v for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.q_table = {eval(k): v for k, v in data['q_table'].items()}
        self.epsilon = data['epsilon']
        print(f"📂 Model loaded from {filepath}")


def train_agent(env, agent, episodes=100, max_steps=100, verbose=True):
    """
    Train the RL agent on the stock patterns environment.
    
    Args:
        env: StockPatternEnv instance
        agent: RL agent
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training statistics
    """
    # Training statistics
    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    losses = []
    
    print(f"\n{'='*60}")
    print(f"🎓 Training RL Agent on Stock Patterns")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Gamma: {agent.gamma}")
    print(f"Initial epsilon: {agent.epsilon:.2f}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset environment
        result = env.reset()
        
        total_reward = 0
        step = 0
        episode_loss = []
        
        while not result.done and step < max_steps:
            # Get current state
            state = result.observation.prices
            position = result.observation.position
            
            # Agent chooses action
            action = agent.act(state, position)
            
            # Execute action
            next_result = env.step(StockPatternAction(
                action_type="trade",
                trade_action=action
            ))
            
            # Get reward
            reward = next_result.reward or 0.0
            total_reward += reward
            
            # Store experience
            next_state = next_result.observation.prices
            agent.remember(state, action, reward, next_state, next_result.done)
            
            # Train agent (experience replay)
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            result = next_result
            step += 1
        
        # Get final state
        final_state = env.state()
        
        # Store statistics
        episode_rewards.append(total_reward)
        episode_returns.append(final_state.total_return)
        episode_lengths.append(step)
        if episode_loss:
            losses.append(sum(episode_loss) / len(episode_loss))
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_return = sum(episode_returns[-10:]) / min(10, len(episode_returns))
            avg_length = sum(episode_lengths[-10:]) / min(10, len(episode_lengths))
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Return (last 10): {avg_return:.2f}%")
            print(f"  Avg Length (last 10): {avg_length:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {len(agent.q_table)}")
            if losses:
                print(f"  Avg Loss: {losses[-1]:.4f}")
            print()
    
    elapsed_time = time.time() - start_time
    
    print(f"{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Time per episode: {elapsed_time/episodes:.2f}s")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print()
    
    return {
        'rewards': episode_rewards,
        'returns': episode_returns,
        'lengths': episode_lengths,
        'losses': losses,
        'time': elapsed_time,
    }


def evaluate_agent(env, agent, episodes=20, max_steps=100):
    """
    Evaluate trained agent performance.
    
    Args:
        env: StockPatternEnv instance
        agent: Trained RL agent
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Dictionary with evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"🎯 Evaluating Trained Agent")
    print(f"{'='*60}\n")
    
    # Temporarily disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    
    episode_rewards = []
    episode_returns = []
    winning_trades = 0
    total_trades = 0
    
    for episode in range(episodes):
        result = env.reset()
        total_reward = 0
        step = 0
        trades_this_episode = 0
        
        while not result.done and step < max_steps:
            state = result.observation.prices
            position = result.observation.position
            
            # Get best action (no exploration)
            action = agent.act(state, position)
            
            # Track trades
            if action in [1, 2]:  # Buy or Sell
                trades_this_episode += 1
            
            result = env.step(StockPatternAction(
                action_type="trade",
                trade_action=action
            ))
            
            reward = result.reward or 0.0
            total_reward += reward
            step += 1
        
        final_state = env.state()
        episode_rewards.append(total_reward)
        episode_returns.append(final_state.total_return)
        total_trades += trades_this_episode
        
        if total_reward > 0:
            winning_trades += 1
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Return={final_state.total_return:.2f}%, "
              f"Trades={trades_this_episode}, Pattern={final_state.pattern_name}")
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_return = sum(episode_returns) / len(episode_returns)
    win_rate = winning_trades / episodes * 100
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results")
    print(f"{'='*60}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Return: {avg_return:.2f}%")
    print(f"Win Rate: {win_rate:.1f}% ({winning_trades}/{episodes})")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades/Episode: {total_trades/episodes:.1f}")
    print()
    
    return {
        'avg_reward': avg_reward,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'rewards': episode_rewards,
        'returns': episode_returns,
    }


def plot_training_progress(stats):
    """Plot training statistics (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot rewards
        axes[0, 0].plot(stats['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Plot returns
        axes[0, 1].plot(stats['returns'])
        axes[0, 1].set_title('Portfolio Returns')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True)
        
        # Plot episode lengths
        axes[1, 0].plot(stats['lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Plot losses
        if stats['losses']:
            axes[1, 1].plot(stats['losses'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Avg Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('stock_patterns_training.png', dpi=150)
        print("📊 Training plots saved to: stock_patterns_training.png")
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib not available. Skipping plots.")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train RL agent on Stock Patterns')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per episode (default: 100)')
    parser.add_argument('--difficulty', type=float, default=0.5,
                       help='Pattern difficulty 0.0-1.0 (default: 0.5)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load pretrained model')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training progress')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='Environment server URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    print("🎯 Stock Patterns RL Training")
    print("=" * 60)
    
    try:
        # Connect to environment
        print(f"🔌 Connecting to environment at {args.url}...")
        env = StockPatternEnv(base_url=args.url)
        print("✅ Connected!\n")
        
        # Create agent
        print("🤖 Initializing RL agent...")
        agent = SimpleQAgent(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
        )
        
        # Load pretrained model if specified
        if args.load_model:
            agent.load(args.load_model)
        
        print("✅ Agent ready!\n")
        
        # Train agent
        stats = train_agent(
            env,
            agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=True
        )
        
        # Evaluate agent
        eval_stats = evaluate_agent(
            env,
            agent,
            episodes=args.eval_episodes,
            max_steps=args.max_steps
        )
        
        # Save model if specified
        if args.save_model:
            agent.save(args.save_model)
        
        # Plot results if requested
        if args.plot:
            plot_training_progress(stats)
        
        # Final summary
        print(f"{'='*60}")
        print(f"🎉 Training Complete!")
        print(f"{'='*60}")
        print(f"Training Episodes: {args.episodes}")
        print(f"Final Evaluation:")
        print(f"  Average Reward: {eval_stats['avg_reward']:.2f}")
        print(f"  Average Return: {eval_stats['avg_return']:.2f}%")
        print(f"  Win Rate: {eval_stats['win_rate']:.1f}%")
        print()
        
        if eval_stats['avg_return'] > 0:
            print("💰 Agent learned to make profitable trades!")
        else:
            print("📉 Agent needs more training to be profitable.")
        
        print("\n👋 Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m envs.stock_patterns_env.server.app")
        print("\nOr start with Docker:")
        print("  docker run -p 8000:8000 stock-patterns-env:latest")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    main()

