# Calendar Resources Server Tests

This directory contains tests for the Calendar Resources Server.

## Overview

The Calendar Resources Server is responsible for verifying calendar scheduling responses. It validates that scheduled events satisfy various constraints including:
- Time window constraints (min/max time)
- Temporal constraints (before, after, between, at)
- Event conflicts (no overlapping events)
- Duration matching
- Event count validation

## Test File Structure

### `test_app.py`

Main test suite containing unit tests for the `CalendarResourcesServer` verification logic.

#### Test Helper Methods

- **`_create_server()`**: Creates a CalendarResourcesServer instance with mock configuration
- **`_create_real_request(response_content, exp_cal_state, request_id)`**: Creates a CalendarVerifyRequest with NeMoGymResponse objects
- **`_run_verify_test(real_request, expected_reward)`**: Executes the verify method and asserts the expected reward

## Test Coverage

### Basic Functionality Tests

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_sanity` | Server instance creation | N/A |
| `test_valid_response_single_event` | Single event scheduled correctly | 1 |
| `test_valid_response_multiple_events` | Multiple events scheduled correctly | 1 |
| `test_no_events_expected` | Valid response when no events expected | 1 |

### Response Format Tests

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_think_tag_present` | Response contains `<think>` tags | 0 |
| `test_invalid_json_extraction` | Response contains invalid JSON | 0 |
| `test_empty_calendar_state` | Empty schedule when events expected | 0 |

### Event Count Validation

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_wrong_number_of_events_too_few` | Fewer events than expected | 0 |
| `test_wrong_number_of_events_too_many` | More events than expected | 0 |

### Conflict Detection

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_time_conflict_between_events` | Overlapping events | 0 |

### Temporal Constraint Tests

#### "before" Constraint
| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_constraint_before_valid` | Event ends before specified time | 1 |
| `test_constraint_before_violation` | Event ends after specified time | 0 |

#### "after" Constraint
| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_constraint_after_valid` | Event starts after specified time | 1 |
| `test_constraint_after_violation` | Event starts before specified time | 0 |

#### "between" Constraint
| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_constraint_between_valid` | Event within time window | 1 |
| `test_constraint_between_violation_starts_too_early` | Event starts before window | 0 |
| `test_constraint_between_violation_ends_too_late` | Event ends after window | 0 |

#### "at" Constraint
| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_constraint_at_valid` | Event starts at exact time | 1 |
| `test_constraint_at_violation` | Event starts at different time | 0 |

#### No Constraint
| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_constraint_none_valid` | Event with no constraint in valid time window | 1 |

### Time Window Tests

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_event_starts_before_min_time` | Event starts before min_time | 0 |
| `test_event_ends_after_max_time` | Event ends after max_time | 0 |

### Duration Tests

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_wrong_duration` | Event duration doesn't match expected | 0 |

### Complex Scenarios

| Test | Description | Expected Reward |
|------|-------------|-----------------|
| `test_complex_multi_event_schedule` | Multiple events with various constraints | 1 |

## Running Tests

### Run All Tests

```bash
pytest test_app.py
```

### Run Specific Test

```bash
pytest test_app.py::TestApp::test_valid_response_single_event
```

## Expected Calendar State Format

The `exp_cal_state` parameter defines the expected calendar configuration:

```python
{
    "0": {
        "event_id": 0,
        "duration": 60,  # in minutes
        "constraint": "between 2pm and 4pm",  # or "before", "after", "at", None
        "min_time": "10:00",  # earliest allowed start time
        "max_time": "16:00"   # latest allowed end time
    }
}
```

## Response Format

Valid calendar responses should be JSON arrays:

```json
[
    {
        "event_id": 0,
        "start_time": "2pm",
        "duration": 60
    }
]
```

## Reward System

- **Reward = 1**: All constraints satisfied, valid schedule
- **Reward = 0**: Constraint violation or invalid response

## Common Failure Scenarios

1. **Think Tags**: Responses containing `<think>` tags always receive reward 0
2. **JSON Parsing Errors**: Invalid JSON format results in reward 0
3. **Event Count Mismatch**: Number of scheduled events must match expected count
4. **Time Conflicts**: Events cannot overlap in time
5. **Constraint Violations**: All temporal constraints must be satisfied
6. **Time Window Violations**: Events must fit within min_time and max_time
7. **Duration Mismatch**: Event durations must exactly match expected durations