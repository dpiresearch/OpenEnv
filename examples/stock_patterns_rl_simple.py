#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple Reinforcement Learning Loop for Stock Patterns Environment.

This is a minimal, self-contained example that demonstrates the basic
RL training loop without requiring external dependencies.

It shows:
1. Basic RL loop structure
2. State representation
3. Action selection with exploration
4. Learning from rewards
5. Performance tracking

Usage:
    # Run directly (no server needed - uses environment locally):
    python examples/stock_patterns_rl_simple.py
    
    # Or with custom parameters:
    python examples/stock_patterns_rl_simple.py --episodes 50
"""

import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.stock_patterns_env.server.stock_patterns_environment import StockPatternEnvironment
from envs.stock_patterns_env.models import StockPatternAction


class SimpleRLAgent:
    """
    Ultra-simple Q-learning agent for demonstration.
    
    Uses a simple discretization strategy and tabular Q-learning.
    """
    
    def __init__(self, learning_rate=0.1, discount=0.95, epsilon=0.2):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_values = defaultdict(lambda: [0.0, 0.0, 0.0])  # [Hold, Buy, Sell]
    
    def get_state(self, prices, position):
        """
        Convert observation to simple state representation.
        
        Uses last 3 price movements (up/down) and current position.
        """
        if len(prices) < 4:
            return (0, 0, 0, position)
        
        # Get last 3 price changes
        recent = prices[-4:]
        movements = []
        for i in range(1, 4):
            if recent[i] > recent[i-1]:
                movements.append(1)  # Up
            else:
                movements.append(0)  # Down
        
        return tuple(movements + [position])
    
    def select_action(self, state, position):
        """Select action using epsilon-greedy policy."""
        # Determine valid actions based on position
        if position == 0:
            valid_actions = [0, 1]  # Hold or Buy
        else:
            valid_actions = [0, 2]  # Hold or Sell
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Choose best Q-value among valid actions
        q_vals = self.q_values[state]
        return max(valid_actions, key=lambda a: q_vals[a])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning rule."""
        current_q = self.q_values[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_values[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.q_values[state][action] += self.lr * (target_q - current_q)


def run_rl_training(episodes=50, max_steps=80, verbose=True):
    """
    Run the RL training loop.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Print progress
    """
    print("🎓 Simple RL Training on Stock Patterns")
    print("=" * 60)
    
    # Create environment (local, no server needed)
    env = StockPatternEnvironment(
        pattern_name=None,  # Random patterns
        difficulty=0.4,     # Medium difficulty
    )
    
    # Create agent
    agent = SimpleRLAgent(
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.2,
    )
    
    # Track performance
    all_rewards = []
    all_returns = []
    profitable_episodes = 0
    
    print(f"Training for {episodes} episodes...\n")
    
    # Training loop
    for episode in range(episodes):
        # Reset environment for new episode
        obs = env.reset()
        
        episode_reward = 0
        step = 0
        
        # Get initial state
        state = agent.get_state(obs.prices, obs.position)
        
        # Run episode
        while not obs.done and step < max_steps:
            # Agent selects action
            action = agent.select_action(state, obs.position)
            
            # Execute action in environment
            obs = env.step(StockPatternAction(
                action_type="trade",
                trade_action=action
            ))
            
            # Get reward
            reward = obs.reward or 0.0
            episode_reward += reward
            
            # Get next state
            next_state = agent.get_state(obs.prices, obs.position)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, obs.done)
            
            # Move to next state
            state = next_state
            step += 1
        
        # Track statistics
        final_return = env.state.total_return
        all_rewards.append(episode_reward)
        all_returns.append(final_return)
        
        if final_return > 0:
            profitable_episodes += 1
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
            avg_return = sum(all_returns[-10:]) / min(10, len(all_returns))
            win_rate = profitable_episodes / (episode + 1) * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Return (last 10): {avg_return:.2f}%")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  States learned: {len(agent.q_values)}")
            print()
    
    # Final statistics
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_return = sum(all_returns) / len(all_returns)
    win_rate = profitable_episodes / episodes * 100
    
    print(f"Overall Performance:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Return: {avg_return:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({profitable_episodes}/{episodes})")
    print(f"  States Learned: {len(agent.q_values)}")
    print()
    
    if avg_return > 0:
        print("💰 Agent learned to make profitable trades on average!")
    else:
        print("📉 Agent needs more training or hyperparameter tuning.")
    
    # Show some learned Q-values
    print("\n📚 Sample Learned Q-Values:")
    print("   State          ->  [Hold,  Buy, Sell]")
    print("-" * 50)
    for i, (state, q_vals) in enumerate(list(agent.q_values.items())[:5]):
        print(f"   {state} -> [{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, {q_vals[2]:6.2f}]")
    print()
    
    return {
        'rewards': all_rewards,
        'returns': all_returns,
        'win_rate': win_rate,
        'agent': agent,
    }


def test_trained_agent(agent, episodes=10):
    """
    Test the trained agent on new episodes.
    
    Args:
        agent: Trained agent
        episodes: Number of test episodes
    """
    print("\n" + "=" * 60)
    print("🎯 Testing Trained Agent")
    print("=" * 60)
    
    env = StockPatternEnvironment(difficulty=0.4)
    
    # Disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    test_returns = []
    wins = 0
    
    for ep in range(episodes):
        obs = env.reset()
        state = agent.get_state(obs.prices, obs.position)
        step = 0
        
        while not obs.done and step < 80:
            action = agent.select_action(state, obs.position)
            obs = env.step(StockPatternAction(action_type="trade", trade_action=action))
            state = agent.get_state(obs.prices, obs.position)
            step += 1
        
        final_return = env.state.total_return
        test_returns.append(final_return)
        
        if final_return > 0:
            wins += 1
        
        pattern_name = env.state.pattern_name
        print(f"  Test {ep+1}: Return={final_return:6.2f}%, Pattern={pattern_name}")
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    avg_return = sum(test_returns) / len(test_returns)
    win_rate = wins / episodes * 100
    
    print(f"\nTest Results:")
    print(f"  Average Return: {avg_return:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Simple RL training on Stock Patterns'
    )
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (default: 50)')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=80,
                       help='Max steps per episode (default: 80)')
    
    args = parser.parse_args()
    
    try:
        # Run training
        results = run_rl_training(
            episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=True
        )
        
        # Test trained agent
        test_trained_agent(results['agent'], episodes=args.test_episodes)
        
        print("=" * 60)
        print("🎉 All Done!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("• The agent learns Q-values for different price patterns")
        print("• Exploration (epsilon) helps discover good strategies")
        print("• The agent learns to buy low and sell high")
        print("• More episodes → better performance (usually)")
        print("\nNext Steps:")
        print("• Try adjusting learning_rate, discount, epsilon")
        print("• Increase episodes for better convergence")
        print("• Add more features to state representation")
        print("• Implement more sophisticated RL algorithms (DQN, PPO)")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

