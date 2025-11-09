# Numeric Dueling

A probabilistic, simultaneous-move, multi-round number-selection game designed to train and evaluate LLM capabilities in numerical reasoning, risk assessment, and strategic thinking under uncertainty.

Each round, you and your opponent simultaneously choose numbers from a range (e.g., 1-100). A random number R is drawn. If your choice exceeds R, you bust and lose the round. Otherwise, the winner is determined by comparing non-busted choices. The player with the highest total score after M rounds wins.


## Formal Game Definition

The game is defined as a tuple allowing for many variants:

```
G = { N, M, D, R_dist, BustRule, WinRule, ElimRule, Visibility }
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **N** | Integer | Number of players |
| **M** | Integer | Number of rounds |
| **D** | Range | Domain of valid numbers (e.g., 1-100) |
| **R_dist** | Distribution | Distribution for random number R (uniform, normal, custom) |
| **BustRule** | Function | Defines what happens when n_i > R |
| **WinRule** | Function | Determines winner if multiple players survive |
| **ElimRule** | Function | Defines per-round elimination rules |
| **Visibility** | Enum | Information revealed after each round |


## Modular Rule Functions

The game's modularity allows for many variants by swapping rule functions:

### BustRule(n_i, R) → {bust, safe, partial}

Defines the consequences of choosing a number relative to R:

- **Standard Bust**: If n_i > R, player busts (eliminated or loses round)
- **Soft Bust**: If n_i > R, player loses partial points but stays in game
- **Probabilistic Bust**: If n_i > R, bust with probability f(n_i - R)

### WinRule(survivors) → winner

Determines the winner among non-busted players:

- **Highest Wins**: max(n_i) among survivors wins
- **Closest to R**: winner is argmin(|n_i - R|), subject to n_i ≤ R
- **Cumulative Scoring**: survivors gain points proportional to n_i

### ElimRule(players) → eliminated_players

Defines per-round elimination:

- **None**: All players continue each round
- **Lowest Eliminated**: min(n_i) is eliminated per round
- **Hybrid**: Lowest eliminated unless someone busts

### Visibility → {full, partial, hidden}

Controls information revelation:

- **Full**: All player numbers and R revealed each round
- **Partial**: Only win/loss outcomes shown (not specific numbers)
- **Hidden**: Nothing revealed until game end


## Example Implementation: "Classic" Variant

For the first implementation, we'll demonstrate a simple, well-defined variant:

### Classic Numeric Dueling

```
N = 2                    # Two players (LLM vs opponent)
M = 5                    # Five rounds
D = [1, 100]             # Integers from 1 to 100
R_dist = Uniform(1, 100) # Random number uniformly distributed
BustRule = Standard      # Bust if n_i > R (lose that round)
WinRule = Highest        # Highest non-busted number wins round
ElimRule = None          # No elimination, play all 5 rounds
Visibility = Full        # All information revealed each round
```

### Scoring

- Each round: Winner gets 1 point, losers get 0
- If both bust: Both get 0
- If both safe: Higher number gets 1 point
- After 5 rounds: Player with most points wins the game

### Example Round

```
Round 3:
- Player 1 (LLM) chooses: 72
- Player 2 (Opponent) chooses: 55
- Random R drawn: 68

Result:
- Player 1 BUSTS (72 > 68) → 0 points
- Player 2 SAFE (55 ≤ 68) → 1 point

Revealed to players:
"Round 3 Results:
- You chose: 72
- Opponent chose: 55
- Random number was: 68
- Result: BUST! Opponent wins this round.
- Score: You: 1, Opponent: 2"
```


## Why This Environment for LLM Training

### Capabilities Exercised

1. **Numerical Reasoning**
   - Understanding integer ranges
   - Comparing magnitudes
   - Working with numerical feedback

2. **Probabilistic Thinking**
   - Understanding uniform distribution
   - Estimating likelihood of R values
   - Reasoning about random events

3. **Risk Assessment**
   - Higher numbers = higher reward (more likely to beat opponent)
   - Higher numbers = higher risk (more likely to bust)
   - Finding optimal risk/reward balance

4. **Strategic Reasoning**
   - Multi-agent game theory
   - Predicting opponent behavior
   - Adapting strategy based on opponent's patterns

5. **Multi-Step Planning**
   - Considering future rounds
   - Managing cumulative score
   - Adjusting aggression based on score differential

6. **Learning from Feedback**
   - Observing outcomes
   - Updating beliefs about opponent
   - Improving strategy over time

7. **Calibrated Confidence**
   - Knowing when to play conservatively (if ahead)
   - Knowing when to take risks (if behind)
   - Metacognitive awareness of uncertainty

### Real-World Applications

Many practical LLM applications require similar capabilities:
- **Financial analysis**: Risk/reward tradeoffs under uncertainty
- **Forecasting**: Probabilistic predictions with confidence intervals
- **Optimization**: Balancing competing objectives
- **Decision support**: Reasoning about uncertain outcomes
- **Quantitative problem-solving**: Working with numbers meaningfully


## Current Implementation Status

### ✅ Implemented Features

**Core Parameters:**
- ✅ **M** (num_rounds) - Configurable number of rounds
- ✅ **D** (range) - Configurable min/max number range
- ✅ **BustRule** - All three variants (Standard, Soft, Probabilistic)
- ✅ **WinRule** - All three variants (Highest, Closest, Cumulative)

**Game Infrastructure:**
- ✅ Session-based state management
- ✅ Dynamic prompt generation
- ✅ Pluggable opponent interface (Random, Fixed, Adaptive)
- ✅ Per-round verification and rewards
- ✅ Full game history tracking

**Current Defaults:**
- N = 2 players (hardcoded)
- R_dist = Uniform distribution (hardcoded)
- ElimRule = None (no elimination, hardcoded)
- Visibility = Full (all info revealed, hardcoded)

---

## Future Work

The following features are **not yet implemented** and should be added in future contributions:

### Core Game Parameters

#### 1. LLM vs LLM Support

**Status**: ❌ Not Implemented  
**Current**: Opponent must be random/fixed/adaptive (non-LLM)  
**Required Changes:**
- Add opponent type that makes API calls to another model server
- Handle asynchronous opponent moves
- Support symmetric LLM vs LLM training scenarios
- Update prompt generation for both players

**Priority**: High (enables self-play and multi-agent RL)

#### 2. Multi-Player Support (N > 2)

**Status**: ❌ Not Implemented  
**Current**: Hardcoded to 2 players  
**Required Changes:**
- Add `num_players` to GameConfig
- Support N-1 opponents
- Extend resolve_round() for N-player comparisons
- Clarify win conditions for 3+ players (single winner vs graduated scoring)
- Update prompts to show multiple opponents

**Priority**: Low (most RL scenarios are 1v1)

#### 3. Distribution Types (R_dist)

**Status**: ❌ Not Implemented  
**Current**: Hardcoded to Uniform(min, max)  
**Variants to Add:**
- Normal distribution (with mean, std dev)
- Exponential distribution
- Custom distributions from data
- Dynamic distributions (range changes based on prior results)

**Required Changes:**
- Add `distribution_type` and params to GameConfig
- Replace `random.randint()` with configurable distribution
- Handle edge cases (clamping to valid range)

**Priority**: Medium (changes optimal strategy significantly)

#### 4. Elimination Rules (ElimRule)

**Status**: ❌ Not Implemented  
**Current**: Hardcoded to None (all players play all rounds)  
**Variants to Add:**
- **Lowest Eliminated**: min(n_i) eliminated per round
- **Hybrid**: Lowest eliminated unless someone busts
- **Last Standing**: Elimination mode until one player remains

**Required Changes:**
- Track active vs eliminated players in GameState
- Skip eliminated players in resolve_round()
- Update prompts to show eliminations
- Modify game end condition

**Priority**: Low (fundamentally changes game dynamics)

#### 5. Visibility Modes

**Status**: ❌ Not Implemented  
**Current**: Hardcoded to Full (all information revealed)  
**Variants to Add:**
- **Partial**: Only win/loss outcomes shown (not numbers or R)
- **Hidden**: No information until game end
- **Asymmetric**: Different players see different information
- **Delayed**: Information revealed after N rounds
- **Noisy**: Information revealed with added noise

**Required Changes:**
- Add `visibility` to GameConfig
- Modify prompt generation to filter history based on visibility
- Adjust what information is revealed in RoundResult

**Priority**: Medium-High (incomplete information is interesting for training)

---

### Rule Enhancements

These extensions enhance existing rule types:

#### Bust Rule Variants
- **Independent R**: Each player gets their own R_i drawn independently (vs shared R)
- **Progressive penalty**: Penalty increases with consecutive busts
- **Grace period**: First bust is forgiven

#### Win Rule Variants  
- **Points-based**: Score = n_i if you survive (incentivizes higher numbers)
- **Margin bonus**: Extra points for winning by large margins
- **Cumulative multipliers**: Points accumulate across rounds with bonuses

#### Advanced Opponent Strategies
- **Learning opponents**: Opponent adapts based on LLM's patterns
- **Skill levels**: Easy/Medium/Hard opponent configurations
- **Mixed strategies**: Opponent randomly selects from multiple strategies

---

## Implementation Details

### Agent Workflow

For multi-round games, the agent follows this pattern:

**Game Setup (once):**
1. **seed_session** - Initialize game with configuration
   ```json
   POST /seed_session
   {
     "config": {
       "num_rounds": 5,
       "min_number": 1,
       "max_number": 100,
       "bust_rule": "standard",
       "win_rule": "highest",
       "opponent_type": "random"
     }
   }
   ```

**Per Round (repeat for each round):**

2. **get_prompt** - Get dynamically generated prompt for current round
   ```json
   POST /get_prompt
   {}
   ```
   Returns: Prompt with current game state, rules, history

3. **Send prompt to LLM** - Agent sends prompt to policy model

4. **verify** - Submit LLM response for verification
   ```json
   POST /verify
   {
     "responses_create_params": {...},
     "response": {LLM output}
   }
   ```
   Returns: Reward for this round, updated game state

5. **Repeat** steps 2-4 until game complete

**Why per-round verification?**
- Immediate reward signals for RL training
- Enables LLM vs LLM gameplay (each agent makes independent moves)
- Policy can adapt strategy based on previous rounds
- Aligns with multi-agent reinforcement learning patterns

### State Management

The server tracks:
- Current round number
- Score for each player
- History of all previous rounds (numbers chosen, R values, outcomes)
- Game status (ongoing, completed, winner)

### Prompt Generation

**Prompts are dynamically generated** by the server based on current game state and configuration.

The agent calls `/get_prompt` to receive a prompt that includes:
- Game rules adapted to current configuration (ranges, bust/win rules)
- Current round and scores
- Full history of previous rounds with outcomes
- Request for number choice with structured output format

**Example generated prompt (Classic variant, Round 3):**
```
You are playing Numeric Dueling. Choose a number from 1-100.

Rules:
- A random number R will be drawn from 1-100
- If your number > R, you BUST and get 0 points for that round
- Among non-busted players, the highest number wins 1 point
- Game lasts 5 rounds. Highest total score at the end wins.

Put your choice inside <choice></choice> tags. You can think step-by-step first.

Current Status:
- Round: 3/5
- Your score: 1.0
- Opponent score: 2.0

History:
Round 1: You chose 45, Opponent chose 62, R was 58. Opponent BUST. You won. (Points: You 1.0, Opp 0.0)
Round 2: You chose 70, Opponent chose 68, R was 75. Opponent won. (Points: You 0.0, Opp 1.0)

What number do you choose?
```

**Prompt adapts to all game variants:**
- Different number ranges (e.g., 1-50 instead of 1-100)
- Different bust rules (standard/soft/probabilistic)
- Different win rules (highest/closest/cumulative)
- Any number of rounds

### Output Format (LLM)

The LLM should output its choice in `<choice></choice>` tags for reliable parsing:

**Preferred format:**
```
I should be conservative since I'm behind. The average R is around 50...

<choice>55</choice>
```

**Parsing logic:**
1. **Primary**: Extract number from `<choice>NUMBER</choice>` tags (most reliable)
2. **Fallback**: Find any integer in valid range anywhere in response

**Validation:**
- Must be integer in valid range (e.g., 1-100)
- If extraction fails, player automatically loses the round (reward = 0.0)

**Example valid outputs:**
- `<choice>65</choice>`
- `Let me think... I'll play it safe. <choice>42</choice>`
- `<choice> 75 </choice>` (whitespace is okay)

**Example invalid outputs that trigger fallback:**
- `I choose 65` (no tags, but fallback will extract 65)
- `<choice>150</choice>` (out of range, extraction fails)

### Opponent Strategy

Initial implementation uses simple baseline opponents:
- **Random**: Chooses uniformly from 1-100
- **Fixed**: Always chooses same number (e.g., 60)
- **Adaptive**: Simple heuristic based on previous rounds


## Usage

### Option 1: Play as a Human (Interactive Demo)

Experience the game firsthand to understand its dynamics.

#### Setup (First Time Only)

```bash
# From the Gym repository root
cd Gym

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies if needed
uv sync --extra dev --group docs
```

#### Running the Interactive Client

**Terminal 1**: Start the server

```bash
cd Gym
source .venv/bin/activate

# Start the numeric_dueling server (defaults to port 5011)
ng_run "+config_paths=[resources_servers/numeric_dueling/configs/numeric_dueling_server_only.yaml]"
```

**Terminal 2**: Play the game (in a new terminal)

```bash
cd Gym
source .venv/bin/activate
cd resources_servers/numeric_dueling

# Connect to the server (default is 5011)
python client.py
# or explicitly specify port: python client.py --port 5011
```

#### What You Can Do

The interactive client offers:
- **5 preset variants**: Classic, Price is Right, High Stakes, Soft Landing, Risky Business
- **Custom configurations**: Set your own rules, ranges, and opponents
- **Visual feedback**: Round-by-round results
- **All opponent types**: Play against Random, Fixed, or Adaptive opponents

This is useful for:
- Understanding game dynamics before training LLMs
- Testing and debugging new rule variants
- Demonstrating the environment to others
- Verifying implementation correctness

### Option 2: Train/Evaluate LLMs

```bash
# Start resource server + model server
config_paths="resources_servers/numeric_dueling/configs/numeric_dueling.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# In another terminal: Run agent
cd Gym
source .venv/bin/activate
python responses_api_agents/simple_agent/client.py
```

## Data

Training and validation datasets consist of:
- Initial game states
- Expected model outputs (number choices)
- Verification results (win/loss/points)

See `data/` directory for:
- `train.jsonl` - Training dataset
- `validation.jsonl` - Validation dataset
- `example.jsonl` - 8 example game configurations covering all rule variants

## Licensing Information

**Code:** Apache 2.0  
**Data:** Apache 2.0

**Dependencies:**
- nemo_gym: Apache 2.0