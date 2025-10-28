# Mastermind Environment

A classic code-breaking game implementation for reinforcement learning research.

## Overview

Mastermind is a code-breaking game where an agent attempts to guess a secret code within a limited number of attempts. After each guess, feedback is provided in the form of:
- **Black pegs**: Correct colors in correct positions
- **White pegs**: Correct colors in wrong positions

This environment provides a perfect testbed for:
- Sequential decision making
- Pattern recognition
- Logical reasoning strategies
- Information gathering under uncertainty

## Game Rules

1. A secret code of colored pegs is randomly generated
2. The agent makes guesses to crack the code
3. After each guess, feedback is provided:
   - Black peg: Correct color in correct position
   - White peg: Correct color in wrong position
4. The game ends when:
   - The code is correctly guessed (WIN)
   - Maximum attempts are exhausted (LOSS)

## Configuration

Default settings (can be customized via environment variables):

- **Code Length**: 4 pegs
- **Number of Colors**: 6 (0-5)
- **Maximum Attempts**: 10
- **Duplicate Colors**: Allowed

## Quick Start

### Start the server
python3 -m envs.mastermind_env.server.app &
### Run the game 
python3 examples/mastermind_simple.py

### Using Local Server

```python
from envs.mastermind_env import MastermindEnv, MastermindAction

# Connect to local server (must be running)
client = MastermindEnv(base_url="http://localhost:8000")

# Reset environment
result = client.reset()
print(f"Attempts remaining: {result.observation.attempts_remaining}")

# Make a guess
result = client.step(MastermindAction(guess=[0, 1, 2, 3]))
print(f"Black pegs: {result.observation.black_pegs}")
print(f"White pegs: {result.observation.white_pegs}")
print(f"Reward: {result.reward}")

# Continue playing...
while not result.done:
    # Your strategy here
    guess = [0, 1, 2, 3]  # Replace with your logic
    result = client.step(MastermindAction(guess=guess))

print(f"Game phase: {result.observation.game_phase}")
client.close()
```

### Using Docker

```python
from envs.mastermind_env import MastermindEnv, MastermindAction

# Automatically start container and connect
client = MastermindEnv.from_docker_image("mastermind-env:latest")

# Play the game
result = client.reset()
result = client.step(MastermindAction(guess=[0, 1, 2, 3]))
print(f"Feedback: {result.observation.black_pegs} black, {result.observation.white_pegs} white")

client.close()
```

## Action Space

**MastermindAction**:
- `guess`: List of integers representing color choices
  - Length must match `code_length` (default: 4)
  - Each integer must be in range [0, `num_colors`-1] (default: [0, 5])
  - Example: `[0, 1, 2, 3]`

## Observation Space

**MastermindObservation**:
- `black_pegs` (int): Number of correct colors in correct positions
- `white_pegs` (int): Number of correct colors in wrong positions
- `attempts_remaining` (int): Number of guesses left
- `all_previous_guesses` (List[List[int]]): History of all guesses made
- `all_previous_feedback` (List[Tuple[int, int]]): History of all feedback received
- `game_phase` (str): Current phase - "playing", "won", or "lost"
- `done` (bool): Whether the episode has ended
- `reward` (float): Reward for the current step

## Reward Structure

The reward function encourages both accuracy and efficiency:

- **Black pegs**: +10 points each
- **White pegs**: +1 point each
- **Step penalty**: -1 point per guess (encourages efficiency)
- **Solving the code**: +100 bonus + (5 × remaining attempts)
- **Running out of attempts**: -50 penalty

Example: Solving in 3 attempts = +100 + (5 × 7) = +135 bonus (plus feedback rewards)

## State Information

Access full environment state:

```python
state = client.state()
print(f"Episode ID: {state.episode_id}")
print(f"Code length: {state.code_length}")
print(f"Number of colors: {state.num_colors}")
print(f"Max attempts: {state.max_attempts}")
print(f"Current attempts: {state.current_attempts}")
print(f"Is solved: {state.is_solved}")
# Secret code is hidden from the agent during play
```

## Building and Running

### Build Docker Image

```bash
# First, build the base image (if not already built)
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Then build the Mastermind environment image
docker build -t mastermind-env:latest -f src/envs/mastermind_env/server/Dockerfile .
```

### Run Server Locally

```bash
# From the src directory
cd src
python -m envs.mastermind_env.server.app

# Or with uvicorn
uvicorn envs.mastermind_env.server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker run -p 8000:8000 mastermind-env:latest
```

### Custom Configuration

Use environment variables to customize the game:

```bash
# Run with custom settings
docker run -p 8000:8000 \
  -e MASTERMIND_CODE_LENGTH=5 \
  -e MASTERMIND_NUM_COLORS=8 \
  -e MASTERMIND_MAX_ATTEMPTS=12 \
  -e MASTERMIND_ALLOW_DUPLICATES=false \
  mastermind-env:latest
```

## Strategy Tips

Some approaches for solving Mastermind:

1. **Random Search**: Try random guesses (baseline)
2. **Knuth's Algorithm**: Minimax strategy that guarantees solving in ≤5 moves for classic 4-peg, 6-color game
3. **Genetic Algorithms**: Evolve candidate solutions based on feedback
4. **Constraint Satisfaction**: Eliminate impossible codes based on feedback
5. **Deep RL**: Learn an optimal policy through experience

## Example Strategies

### Simple Elimination Strategy

```python
import itertools
from envs.mastermind_env import MastermindEnv, MastermindAction

def is_consistent(guess, feedback, candidate):
    """Check if candidate is consistent with the feedback."""
    black, white = calculate_feedback(guess, candidate)
    return black == feedback[0] and white == feedback[1]

def calculate_feedback(guess, code):
    """Calculate black and white pegs."""
    black = sum(1 for i in range(len(guess)) if guess[i] == code[i])
    secret_colors = {}
    guess_colors = {}
    for i in range(len(guess)):
        if guess[i] != code[i]:
            secret_colors[code[i]] = secret_colors.get(code[i], 0) + 1
            guess_colors[guess[i]] = guess_colors.get(guess[i], 0) + 1
    white = sum(min(guess_colors.get(c, 0), secret_colors.get(c, 0)) for c in guess_colors)
    return black, white

# Initialize
client = MastermindEnv(base_url="http://localhost:8000")
result = client.reset()

# Generate all possible codes
code_length = 4
num_colors = 6
possible_codes = list(itertools.product(range(num_colors), repeat=code_length))

# Play game
while not result.done:
    # Pick first consistent code
    guess = list(possible_codes[0])
    result = client.step(MastermindAction(guess=guess))
    
    # Eliminate inconsistent codes
    feedback = (result.observation.black_pegs, result.observation.white_pegs)
    possible_codes = [c for c in possible_codes if is_consistent(guess, feedback, c)]

client.close()
```

## Testing

Test the environment directly:

```python
from envs.mastermind_env.server.mastermind_environment import MastermindEnvironment
from envs.mastermind_env import MastermindAction

# Create environment
env = MastermindEnvironment(code_length=4, num_colors=6, max_attempts=10)

# Test reset
obs = env.reset()
assert obs.attempts_remaining == 10
assert obs.game_phase == "playing"

# Test step
obs = env.step(MastermindAction(guess=[0, 1, 2, 3]))
assert 0 <= obs.black_pegs <= 4
assert 0 <= obs.white_pegs <= 4
assert obs.attempts_remaining == 9

print("✅ All tests passed!")
```

## API Endpoints

When running the server, the following endpoints are available:

- `GET /`: Web interface
- `POST /reset`: Reset the environment
- `POST /step`: Take an action
- `GET /state`: Get current state
- `GET /health`: Health check

## Contributing

Feel free to extend this environment with:
- Different game variants (e.g., Mastermind with more/fewer pegs)
- Advanced opponent strategies
- Additional reward shaping
- Visualization tools

## References

- [Mastermind (board game) - Wikipedia](https://en.wikipedia.org/wiki/Mastermind_(board_game))
- [Knuth's Mastermind Algorithm](https://en.wikipedia.org/wiki/Mastermind_(board_game)#Algorithms)

## License

BSD-style license. See LICENSE file in the root directory.

