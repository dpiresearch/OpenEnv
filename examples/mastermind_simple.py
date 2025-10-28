#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example of using Mastermind environment with a basic RL agent.

This demonstrates:
1. Connecting to the environment
2. Implementing a simple elimination-based strategy
3. Training over multiple episodes
4. Tracking performance metrics

The agent uses a constraint satisfaction approach, eliminating
impossible codes based on feedback from previous guesses.

Usage:
    # Start the server first:
    python -m envs.mastermind_env.server.app
    
    # Then run this script:
    python examples/mastermind_simple.py
"""

import sys
import random
import itertools
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.mastermind_env import MastermindEnv, MastermindAction


class SmartMastermindAgent:
    """
    A simple but effective Mastermind agent using constraint satisfaction.
    
    Strategy:
    1. Maintain a set of all possible codes
    2. After each guess, eliminate codes inconsistent with the feedback
    3. Pick the next guess from remaining possibilities
    """
    
    def __init__(self, code_length: int = 4, num_colors: int = 6):
        self.code_length = code_length
        self.num_colors = num_colors
        self.reset()
    
    def reset(self):
        """Reset agent for a new episode."""
        # Generate all possible codes
        self.possible_codes = list(
            itertools.product(range(self.num_colors), repeat=self.code_length)
        )
        self.guesses_made = []
        self.feedback_received = []
    
    def select_action(self) -> List[int]:
        """
        Select next guess.
        
        For the first guess, use a fixed starting guess.
        For subsequent guesses, pick from remaining possible codes.
        """
        if not self.guesses_made:
            # First guess: use a diverse starting point
            # [0, 0, 1, 1] is a good starting guess for classic Mastermind
            return [0, 0, 1, 1][:self.code_length] + [0] * max(0, self.code_length - 4)
        
        # Pick first remaining possible code
        if self.possible_codes:
            return list(self.possible_codes[0])
        else:
            # Fallback: random guess (shouldn't happen in a valid game)
            return [random.randint(0, self.num_colors - 1) for _ in range(self.code_length)]
    
    def update(self, guess: List[int], black_pegs: int, white_pegs: int):
        """
        Update agent's knowledge based on feedback.
        
        Args:
            guess: The guess that was made
            black_pegs: Number of correct colors in correct positions
            white_pegs: Number of correct colors in wrong positions
        """
        self.guesses_made.append(guess)
        self.feedback_received.append((black_pegs, white_pegs))
        
        # Eliminate inconsistent codes
        feedback = (black_pegs, white_pegs)
        self.possible_codes = [
            code for code in self.possible_codes
            if self._is_consistent(guess, feedback, code)
        ]
    
    def _is_consistent(
        self, 
        guess: List[int], 
        feedback: Tuple[int, int], 
        candidate: Tuple[int, ...]
    ) -> bool:
        """
        Check if a candidate code is consistent with the feedback.
        
        Args:
            guess: The guess that was made
            feedback: The (black_pegs, white_pegs) feedback received
            candidate: A candidate code to check
            
        Returns:
            True if the candidate could produce the given feedback
        """
        expected_feedback = self._calculate_feedback(guess, list(candidate))
        return expected_feedback == feedback
    
    def _calculate_feedback(
        self, 
        guess: List[int], 
        code: List[int]
    ) -> Tuple[int, int]:
        """
        Calculate what feedback would be given for a guess against a code.
        
        Args:
            guess: The guessed code
            code: The actual code
            
        Returns:
            Tuple of (black_pegs, white_pegs)
        """
        # Count black pegs (exact matches)
        black_pegs = sum(1 for i in range(self.code_length) if guess[i] == code[i])
        
        # Count white pegs (color matches in wrong positions)
        secret_colors = {}
        guess_colors = {}
        
        for i in range(self.code_length):
            if guess[i] != code[i]:  # Not an exact match
                # Count in secret
                secret_colors[code[i]] = secret_colors.get(code[i], 0) + 1
                # Count in guess
                guess_colors[guess[i]] = guess_colors.get(guess[i], 0) + 1
        
        # White pegs are the minimum overlap for each color
        white_pegs = 0
        for color, count in guess_colors.items():
            if color in secret_colors:
                white_pegs += min(count, secret_colors[color])
        
        return black_pegs, white_pegs


def run_episode(env: MastermindEnv, agent: SmartMastermindAgent, verbose: bool = True) -> dict:
    """
    Run a single episode.
    
    Args:
        env: The Mastermind environment
        agent: The agent to use
        verbose: Whether to print progress
        
    Returns:
        Dictionary with episode statistics
    """
    # Reset
    agent.reset()
    result = env.reset()
    
    if verbose:
        print(f"\n{'='*60}")
        print("🎯 New Game Started!")
        print(f"{'='*60}")
    
    # Play until done
    total_reward = 0
    step = 0
    
    while not result.done:
        # Agent selects action
        guess = agent.select_action()
        
        # Take action
        result = env.step(MastermindAction(guess=guess))
        
        # Update agent
        agent.update(guess, result.observation.black_pegs, result.observation.white_pegs)
        
        # Track reward
        reward = result.reward or 0
        total_reward += reward
        step += 1
        
        if verbose:
            print(f"\nStep {step}:")
            print(f"  Guess: {guess}")
            print(f"  Black pegs: {result.observation.black_pegs} ⚫")
            print(f"  White pegs: {result.observation.white_pegs} ⚪")
            print(f"  Reward: {reward:.1f}")
            print(f"  Remaining possibilities: {len(agent.possible_codes)}")
            print(f"  Attempts left: {result.observation.attempts_remaining}")
    
    # Episode finished
    if verbose:
        print(f"\n{'='*60}")
        if result.observation.game_phase == "won":
            print("🎉 CODE CRACKED! 🎉")
        else:
            print("😞 Out of attempts - Game Over")
        print(f"{'='*60}")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.1f}")
        
        # Get final state to see the secret code
        state = env.state()
        print(f"Secret code was: {state.secret_code}")
        print(f"All guesses: {result.observation.all_previous_guesses}")
    
    return {
        "steps": step,
        "total_reward": total_reward,
        "won": result.observation.game_phase == "won",
        "game_phase": result.observation.game_phase,
    }


def main():
    """Main training loop."""
    print("🎮 Mastermind RL Training Example")
    print("=" * 60)
    
    # Connect to environment server
    # Make sure server is running: python -m envs.mastermind_env.server.app
    try:
        env = MastermindEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error connecting to environment: {e}")
        print("\nMake sure the server is running:")
        print("  python -m envs.mastermind_env.server.app")
        print("\nOr start with Docker:")
        print("  docker run -p 8000:8000 mastermind-env:latest")
        return
    
    try:
        # Create agent
        agent = SmartMastermindAgent(code_length=4, num_colors=6)
        
        # Training parameters
        num_episodes = 10
        
        # Track statistics
        stats = {
            "wins": 0,
            "total_steps": 0,
            "total_reward": 0.0,
            "episodes": [],
        }
        
        print(f"\n🚀 Training for {num_episodes} episodes...")
        print(f"Strategy: Constraint Satisfaction (Elimination)")
        
        # Run episodes
        for episode in range(1, num_episodes + 1):
            verbose = (episode <= 3 or episode == num_episodes)  # Show first 3 and last
            
            if not verbose:
                print(f"\nEpisode {episode}...", end=" ", flush=True)
            else:
                print(f"\n{'#'*60}")
                print(f"# Episode {episode}/{num_episodes}")
                print(f"{'#'*60}")
            
            episode_stats = run_episode(env, agent, verbose=verbose)
            
            # Update statistics
            stats["episodes"].append(episode_stats)
            stats["total_steps"] += episode_stats["steps"]
            stats["total_reward"] += episode_stats["total_reward"]
            if episode_stats["won"]:
                stats["wins"] += 1
            
            if not verbose:
                status = "✅ WON" if episode_stats["won"] else "❌ LOST"
                print(f"{status} in {episode_stats['steps']} steps, reward: {episode_stats['total_reward']:.1f}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes: {num_episodes}")
        print(f"Wins: {stats['wins']} ({stats['wins']/num_episodes*100:.1f}%)")
        print(f"Avg Steps per Episode: {stats['total_steps']/num_episodes:.2f}")
        print(f"Avg Reward per Episode: {stats['total_reward']/num_episodes:.2f}")
        
        # Calculate average steps for winning episodes
        winning_episodes = [ep for ep in stats["episodes"] if ep["won"]]
        if winning_episodes:
            avg_winning_steps = sum(ep["steps"] for ep in winning_episodes) / len(winning_episodes)
            print(f"Avg Steps to Win: {avg_winning_steps:.2f}")
        
        print(f"\n💡 Strategy Performance:")
        print(f"   The elimination strategy guarantees finding the code")
        print(f"   if enough attempts are available. With 10 attempts and")
        print(f"   a 4-peg 6-color game, it should win 100% of the time!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always close the environment
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

