#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example of using Stock Patterns environment with OpenEnv.

This demonstrates:
1. Connecting to the environment
2. Observing price data
3. Making trading decisions
4. Identifying patterns
5. Tracking portfolio performance

Usage:
    # First, start the server:
    python -m envs.stock_patterns_env.server.app
    
    # Then run this example:
    python examples/stock_patterns_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.stock_patterns_env import StockPatternEnv, StockPatternAction


def visualize_prices(prices, pattern_name=None):
    """Simple ASCII visualization of price chart."""
    if not prices:
        return
    
    # Normalize prices for display
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price if max_price != min_price else 1
    
    # Create chart
    height = 10
    chart = []
    
    for i in range(height):
        line = []
        for price in prices[-30:]:  # Show last 30 prices
            normalized = (price - min_price) / price_range
            level = int(normalized * (height - 1))
            if height - 1 - i == level:
                line.append("●")
            elif height - 1 - i < level:
                line.append("│")
            else:
                line.append(" ")
        chart.append("".join(line))
    
    title = f"Pattern: {pattern_name}" if pattern_name else "Price Chart"
    print(f"\n{title}")
    print("─" * 50)
    for line in chart:
        print(line)
    print("─" * 50)


def example_trading():
    """Example: Trade a stock pattern."""
    print("📈 Stock Patterns Example - Trading")
    print("=" * 60)
    
    # Connect to environment server
    env = StockPatternEnv(base_url="http://localhost:8000")
    
    try:
        # Reset environment
        print("\n📍 Resetting environment...")
        result = env.reset()
        
        print(f"   Initial cash: ${result.observation.cash:.2f}")
        print(f"   Current price: ${result.observation.current_price:.2f}")
        
        # Simple trading strategy: buy low, sell high
        print("\n🎮 Trading the pattern...")
        step = 0
        position_held = False
        buy_price = 0.0
        
        while not result.done and step < 100:
            prices = result.observation.prices
            current_price = result.observation.current_price
            
            # Wait for enough data
            if len(prices) < 10:
                action = StockPatternAction(action_type="trade", trade_action=0)  # Hold
            
            # Simple strategy: buy if price is rising, sell if falling
            elif not position_held and len(prices) >= 5:
                # Check if uptrend
                if prices[-1] > prices[-5]:
                    action = StockPatternAction(action_type="trade", trade_action=1)  # Buy
                    position_held = True
                    buy_price = current_price
                    print(f"   Step {step + 1}: BUY at ${current_price:.2f}")
                else:
                    action = StockPatternAction(action_type="trade", trade_action=0)  # Hold
            
            elif position_held:
                # Sell if price drops or pattern is ending
                if prices[-1] < buy_price * 0.95 or result.observation.pattern_progress > 0.8:
                    action = StockPatternAction(action_type="trade", trade_action=2)  # Sell
                    position_held = False
                    print(f"   Step {step + 1}: SELL at ${current_price:.2f} (P/L: ${(current_price - buy_price) * (result.observation.portfolio_value / current_price):.2f})")
                else:
                    action = StockPatternAction(action_type="trade", trade_action=0)  # Hold
            else:
                action = StockPatternAction(action_type="trade", trade_action=0)  # Hold
            
            # Execute action
            result = env.step(action)
            
            # Show progress periodically
            if step % 20 == 0:
                print(f"   Step {step + 1}: Portfolio=${result.observation.portfolio_value:.2f}, Progress={result.observation.pattern_progress*100:.1f}%")
            
            step += 1
        
        # Episode finished
        print(f"\n✅ Trading complete!")
        print(f"   Total steps: {step}")
        print(f"   Final portfolio value: ${result.observation.portfolio_value:.2f}")
        
        # Get final state
        state = env.state()
        print(f"\n📊 Results:")
        print(f"   Pattern: {result.observation.pattern_name}")
        print(f"   Initial cash: ${state.initial_cash:.2f}")
        print(f"   Final value: ${result.observation.portfolio_value:.2f}")
        print(f"   Total return: {state.total_return:.2f}%")
        
        if state.total_return > 0:
            print(f"   Result: Profit! 💰")
        elif state.total_return < 0:
            print(f"   Result: Loss 📉")
        else:
            print(f"   Result: Break even ➖")
        
        # Visualize the pattern
        visualize_prices(
            [p * result.observation.current_price / prices[-1] for p in prices],
            result.observation.pattern_name
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m envs.stock_patterns_env.server.app")
        print("\nOr start with Docker:")
        print("  docker run -p 8000:8000 stock-patterns-env:latest")
    
    finally:
        env.close()


def example_pattern_identification():
    """Example: Identify which pattern is forming."""
    print("\n\n🔍 Stock Patterns Example - Pattern Identification")
    print("=" * 60)
    
    env = StockPatternEnv(base_url="http://localhost:8000")
    
    try:
        # Reset environment
        print("\n📍 Resetting environment...")
        result = env.reset()
        
        # Observe for a while
        print("\n👀 Observing price movements...")
        for i in range(20):
            result = env.step(StockPatternAction(action_type="trade", trade_action=0))
        
        # Show what we've observed
        prices = result.observation.prices
        print(f"   Observed {len(prices)} price points")
        print(f"   Pattern progress: {result.observation.pattern_progress*100:.1f}%")
        
        # Try to identify the pattern
        print("\n🤔 Trying to identify the pattern...")
        
        patterns = [
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "cup_and_handle",
            "double_top",
            "double_bottom",
            "ascending_triangle",
            "descending_triangle",
            "symmetrical_triangle",
            "flag",
            "pennant",
        ]
        
        for pattern in patterns:
            result = env.step(StockPatternAction(
                action_type="identify",
                pattern_guess=pattern
            ))
            
            if result.reward and result.reward > 0:
                print(f"   ✓ Correct! Pattern is: {pattern}")
                print(f"   Reward: +{result.reward:.0f}")
                break
            else:
                print(f"   ✗ Not {pattern}")
        
        # Get state
        state = env.state()
        print(f"\n📊 Statistics:")
        print(f"   Patterns seen: {state.total_patterns_seen}")
        print(f"   Correct identifications: {state.correct_identifications}")
        if state.total_patterns_seen > 0:
            accuracy = (state.correct_identifications / state.total_patterns_seen) * 100
            print(f"   Accuracy: {accuracy:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        env.close()


def main():
    """Run both examples."""
    print("🎯 Stock Patterns Environment Examples")
    print("=" * 60)
    print("\nThis demonstrates two ways to use the environment:")
    print("1. Trading: Buy and sell to profit from patterns")
    print("2. Identification: Recognize which pattern is forming")
    print()
    
    try:
        # Run trading example
        example_trading()
        
        # Run identification example
        example_pattern_identification()
        
        print("\n\n👋 All examples complete!")
        print("\nNext steps:")
        print("- Try different patterns by setting STOCK_PATTERN_NAME env var")
        print("- Adjust difficulty with STOCK_PATTERN_DIFFICULTY (0.0-1.0)")
        print("- Build your own trading strategies")
        print("- Train a pattern recognition model")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

