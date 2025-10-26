# Stock Patterns Environment

A reinforcement learning environment for simulating and trading common technical stock chart patterns.

## Overview

This environment generates synthetic stock price data that follows common technical analysis patterns. It's designed for training agents to:

1. **Recognize patterns**: Identify which technical pattern is forming (pattern recognition task)
2. **Trade patterns**: Execute trades to profit from pattern signals (trading task)
3. **Learn technical analysis**: Understand pattern characteristics and market behavior

## Supported Patterns

The environment simulates these 10 most common technical chart patterns:

### Reversal Patterns
1. **Head and Shoulders** - Bearish reversal pattern with three peaks
2. **Inverse Head and Shoulders** - Bullish reversal pattern (inverted)
3. **Double Top** - Bearish reversal with two peaks at similar levels
4. **Double Bottom** - Bullish reversal with two troughs at similar levels

### Continuation Patterns
5. **Cup and Handle** - Bullish continuation resembling a tea cup
6. **Ascending Triangle** - Bullish continuation with rising lows
7. **Descending Triangle** - Bearish continuation with falling highs
8. **Symmetrical Triangle** - Neutral continuation with converging trendlines
9. **Flag** - Short-term continuation after strong move
10. **Pennant** - Continuation with converging oscillations

## Quick Start

### Option 1: Local Development

```python
from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

# Start the server first:
# python -m envs.stock_patterns_env.server.app

# Connect to server
env = StockPatternEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
print(f"Initial prices: {result.observation.prices[-5:]}")

# Trade the pattern
result = env.step(StockPatternAction(action_type="trade", trade_action=1))  # Buy
print(f"Portfolio value: ${result.observation.portfolio_value:.2f}")

result = env.step(StockPatternAction(action_type="trade", trade_action=2))  # Sell
print(f"Reward: {result.reward:.2f}")

# Or identify the pattern
result = env.step(StockPatternAction(
    action_type="identify",
    pattern_guess="head_and_shoulders"
))
print(f"Correct! Reward: {result.reward}" if result.reward > 0 else "Incorrect")

env.close()
```

### Option 2: Docker

```python
from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

# Automatically starts container
env = StockPatternEnv.from_docker_image("stock-patterns-env:latest")

result = env.reset()
result = env.step(StockPatternAction(action_type="trade", trade_action=1))

env.close()
```

## Action Space

The environment supports two types of actions:

### Trading Actions
```python
StockPatternAction(
    action_type="trade",
    trade_action=0  # 0=Hold, 1=Buy, 2=Sell
)
```

### Pattern Identification Actions
```python
StockPatternAction(
    action_type="identify",
    pattern_guess="head_and_shoulders"  # Name of pattern
)
```

**Available pattern names:**
- `head_and_shoulders`
- `inverse_head_and_shoulders`
- `cup_and_handle`
- `double_top`
- `double_bottom`
- `ascending_triangle`
- `descending_triangle`
- `symmetrical_triangle`
- `flag`
- `pennant`

## Observation Space

Each observation contains:

```python
@dataclass
class StockPatternObservation:
    prices: List[float]              # Historical prices (normalized)
    volumes: List[float]             # Historical volumes (normalized)
    current_price: float             # Current stock price
    position: int                    # Position: -1=Short, 0=Neutral, 1=Long
    cash: float                      # Available cash
    portfolio_value: float           # Total portfolio value
    pattern_name: Optional[str]      # Pattern name (revealed after identification)
    pattern_progress: float          # Pattern completion (0.0 to 1.0)
    available_actions: List[str]     # Available action types
```

## Rewards

### Trading Rewards
- **Profitable trade**: Positive reward based on % profit (e.g., 10% profit = +10 reward)
- **Unprofitable trade**: Negative reward based on % loss (e.g., 5% loss = -5 reward)
- **Holding**: 0 reward

### Identification Rewards
- **Correct identification**: +100 reward
- **Incorrect identification**: -10 reward

## Configuration

### Environment Parameters

```python
env = StockPatternEnvironment(
    pattern_name="head_and_shoulders",  # Specific pattern or None for random
    difficulty=0.5,                     # 0.0=easy/clear, 1.0=hard/noisy
    window_size=50,                     # Number of historical prices to show
    initial_cash=10000.0,               # Starting cash amount
    mode="trade",                       # "trade" or "identify"
)
```

### Server Configuration

Set environment variables when running the server:

```bash
# Run with specific pattern
STOCK_PATTERN_NAME=cup_and_handle \
STOCK_PATTERN_DIFFICULTY=0.3 \
STOCK_PATTERN_MODE=trade \
python -m envs.stock_patterns_env.server.app

# Or with Docker
docker run -p 8000:8000 \
  -e STOCK_PATTERN_NAME=double_top \
  -e STOCK_PATTERN_DIFFICULTY=0.7 \
  stock-patterns-env:latest
```

## Building Docker Image

```bash
# First build the base image (if not already built)
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build stock patterns environment
docker build -t stock-patterns-env:latest \
  -f src/envs/stock_patterns_env/server/Dockerfile .

# Run the container
docker run -p 8000:8000 stock-patterns-env:latest
```

## Examples

### Example 1: Pattern Recognition Agent

```python
from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

env = StockPatternEnv(base_url="http://localhost:8000")

# Reset and observe
result = env.reset()

# Try to identify the pattern
patterns = [
    "head_and_shoulders",
    "cup_and_handle",
    "double_top",
    # ... etc
]

for pattern in patterns:
    result = env.step(StockPatternAction(
        action_type="identify",
        pattern_guess=pattern
    ))
    
    if result.reward > 0:
        print(f"✓ Correct! Pattern is: {pattern}")
        break
    else:
        print(f"✗ Not {pattern}")

env.close()
```

### Example 2: Simple Trading Strategy

```python
from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

env = StockPatternEnv(base_url="http://localhost:8000")

result = env.reset()
position = 0

while not result.done:
    prices = result.observation.prices
    
    if len(prices) < 5:
        # Not enough data, hold
        action = StockPatternAction(action_type="trade", trade_action=0)
    elif prices[-1] > prices[-5] and position == 0:
        # Price trending up, buy
        action = StockPatternAction(action_type="trade", trade_action=1)
        position = 1
    elif prices[-1] < prices[-5] and position == 1:
        # Price trending down, sell
        action = StockPatternAction(action_type="trade", trade_action=2)
        position = 0
    else:
        # Hold
        action = StockPatternAction(action_type="trade", trade_action=0)
    
    result = env.step(action)
    print(f"Step {env.state().step_count}: Value=${result.observation.portfolio_value:.2f}")

print(f"Final return: {env.state().total_return:.2f}%")
env.close()
```

### Example 3: Pattern Progress Monitoring

```python
from envs.stock_patterns_env import StockPatternEnv, StockPatternAction

env = StockPatternEnv(base_url="http://localhost:8000")
result = env.reset()

while not result.done:
    progress = result.observation.pattern_progress
    print(f"Pattern progress: {progress*100:.1f}%")
    
    # Just observe by holding
    result = env.step(StockPatternAction(action_type="trade", trade_action=0))

# Pattern revealed at the end
print(f"Pattern was: {result.observation.pattern_name}")
env.close()
```

## Use Cases

### 1. Pattern Recognition Training
Train a classifier to identify chart patterns from price sequences:
- Input: Historical price data
- Output: Pattern classification
- Reward: +100 for correct identification

### 2. Trading Strategy Development
Develop and test trading algorithms on patterned data:
- Input: Price, volume, portfolio state
- Output: Buy/Sell/Hold decisions
- Reward: Profit/loss from trades

### 3. Technical Analysis Research
Study pattern characteristics and statistical properties:
- Generate many instances of each pattern
- Analyze success rates and characteristics
- Compare difficulty levels

### 4. Multi-Agent Trading
Multiple agents competing or cooperating in the same market:
- Each agent sees the same pattern
- Different strategies compete
- Study market dynamics

## Advanced Features

### Custom Pattern Generation

The environment generates patterns with realistic characteristics:
- **Noise levels**: Controlled by difficulty parameter
- **Pattern duration**: Randomized within reasonable bounds
- **Amplitude variation**: Natural-looking price movements
- **Volume correlation**: Volume changes with price action

### Pattern Difficulty

Difficulty affects pattern clarity:
- **0.0 (Easy)**: Clear, textbook patterns with minimal noise
- **0.5 (Medium)**: Realistic patterns with moderate noise
- **1.0 (Hard)**: Noisy patterns requiring careful analysis

## API Reference

See the main [OpenEnv documentation](../../README.md) for HTTP API details.

### Endpoints

- `POST /reset` - Start new episode with random pattern
- `POST /step` - Execute action and get observation
- `GET /state` - Get current environment state
- `GET /health` - Health check

## Testing

```bash
# Test locally without Docker
cd src
python -m envs.stock_patterns_env.server.app &
python examples/stock_patterns_simple.py
```

## Contributing

When adding new patterns:

1. Add pattern name to `PATTERNS` list
2. Implement `_generate_<pattern_name>()` method
3. Update documentation
4. Test pattern generation

## License

BSD-style license. See [LICENSE](../../../LICENSE) file.

## References

- [Technical Analysis Patterns](https://www.investopedia.com/articles/technical/112601.asp)
- [Chart Pattern Recognition](https://www.investopedia.com/trading/introduction-to-technical-analysis/)
- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
uvicorn envs.stock_patterns_env.server.app:app --port 8001
```

### Patterns look wrong
- Check difficulty setting (lower = clearer patterns)
- Verify random seed for reproducibility
- Visualize patterns using matplotlib:

```python
import matplotlib.pyplot as plt

result = env.reset()
while not result.done:
    result = env.step(StockPatternAction(action_type="trade", trade_action=0))

prices = result.observation.prices
plt.plot(prices)
plt.title(f"Pattern: {result.observation.pattern_name}")
plt.show()
```

### Docker build fails
```bash
# Ensure base image is built first
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Check Docker daemon is running
docker info
```

