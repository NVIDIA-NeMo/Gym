# Numeric Dueling Tests

This directory contains automated tests for the Numeric Dueling environment.

## Test Suite Overview

**Total: 42 tests (18 unit + 13 integration + 11 error handling)**

## Test Structure

- **Unit Tests (`TestResolveRound`)**: Deterministic tests with fixed R values, direct function calls, no HTTP overhead. Tests business logic in isolation.
- **Integration Tests (`TestApp`)**: Full request/response cycle via FastAPI TestClient, validates API contracts, tests session isolation.
- **Error Handling Tests (`TestErrorHandling`)**: Documents actual error behavior, tests Pydantic validation, verifies graceful degradation.

For detailed test coverage, see the checklist below.

## Running Tests

### Run all tests
```bash
cd /Users/cwing/Documents/Gym
pytest resources_servers/numeric_dueling/tests/test_app.py -v
```

### Run only unit tests
```bash
pytest resources_servers/numeric_dueling/tests/test_app.py::TestResolveRound -v
```

### Run only integration tests
```bash
pytest resources_servers/numeric_dueling/tests/test_app.py::TestApp -v
```

### Run only error handling tests
```bash
pytest resources_servers/numeric_dueling/tests/test_app.py::TestErrorHandling -v
```

### Run specific test
```bash
pytest resources_servers/numeric_dueling/tests/test_app.py::TestResolveRound::test_standard_bust_highest_player_wins -v
```

### Run with coverage
```bash
pytest resources_servers/numeric_dueling/tests/test_app.py --cov=resources_servers.numeric_dueling.app --cov-report=term-missing
```

## Test Coverage

### Unit Tests (Core Logic)

**`resolve_round()` function - Rule combinations:**
- Standard Bust + Highest: player wins, opponent wins, player busts, both bust (4 tests)
- Standard Bust + Closest: player closer, player over R, both over R (3 tests)
- Standard Bust + Cumulative: both survive, one busts (2 tests)
- Soft Bust + Highest: penalty calculation, penalty capping at 0.5 (2 tests)
- Soft Bust + Cumulative: proportional points with penalties (1 test)

**Edge Cases:**
- Choice exactly equals R (boundary)
- Choice equals R+1 (minimal overage)
- Both players choose same number (tie)
- Soft bust penalty exactly at 0.5 cap
- Cumulative scoring at 0.0 and 1.0 boundaries
- Different max_number ranges (1-200 scaling)
- Soft bust scales with max_number

**Total: 18 tests**

### Integration Tests (API Level)

**Game Flow:**
- All 9 rule combinations through API
- Multi-round game state persistence
- Session management with cookies
- Custom number ranges
- Both players bust scenario
- Game ending correctly after M rounds

**Opponent Types:**
- Fixed opponent (deterministic behavior)
- Adaptive opponent (adjusts based on history)
- Random opponent (covered in existing tests)

**Total: 13 tests**

### Error Handling Tests

**Error Scenarios:**
- Invalid session IDs (ValueError propagation)
- Missing session IDs (automatic session creation)
- Invalid choice types (Pydantic 422 validation)
- Missing player_choice field (422 validation)
- Invalid bust_rule enum (422 validation)
- Invalid win_rule enum (422 validation)
- Invalid opponent_type enum (422 validation)
- Missing config fields (defaults applied)
- Invalid number ranges (min > max crashes)
- Negative num_rounds (accepted with defaults)
- Malformed JSON (FastAPI 422 error)

**Total: 11 tests**

## Not Yet Tested (Future Work)

### Helper Functions
- `extract_number_from_response()`: with `<choice>` tags
- `extract_number_from_response()`: without tags, plain number
- `extract_number_from_response()`: malformed input
- `generate_game_prompt()`: verify prompts for each rule variant
- `generate_game_prompt()`: verify history inclusion

**Rationale:** These functions are implicitly tested via integration tests. Unit tests would provide marginal value.

### End-to-End Tests
- Full game with actual LLM via `/verify` endpoint
- Prompt generation → LLM response → parsing → scoring cycle
- Multiple LLMs playing against each other

**Rationale:** Requires LLM infrastructure. Manual testing via `client.py` provides adequate coverage for now.

## Contributing Tests

When adding new tests:
1. Follow existing naming conventions
2. Add docstrings explaining what's being tested
3. Update this README if adding new test categories
4. Ensure all tests pass before committing
5. Aim for >80% code coverage for new features