# CVDP Failure Analysis: Nemotron 3 Super

## 1. Executive Summary

**Nemotron 3 Super** achieves a **pass@1 of 27.81%** across 302 CVDP tasks (1510 rollouts, n=5). Performance drops steeply from easy (37.65%) to medium (16.43%) — a 21 pp cliff indicating a reasoning depth ceiling.

**Consistency profile**: 30 tasks (9.9%) always-pass, 160 tasks (53.0%) always-fail, 112 tasks (37.1%) flaky. The oracle ceiling (at least 1/5 passing) is 47.02%, yielding a **consistency gap of 19.21 pp** — meaning nearly as much performance is lost to inconsistency as to missing capability.

**Error type breakdown**: Of 1090 failed rollouts, **93.9% (1024) are simulation mismatches** — the RTL compiles and simulates, but produces wrong outputs. Syntax errors account for 2.7% (29), timeouts 2.8% (31), parse failures 0.5% (5), and compilation errors 0.1% (1). The model's primary failure mode is not code generation mechanics but **incorrect logic and algorithmic reasoning**.

**Per-CID pass rates**: cid016 (Bug Fixing) leads at 37.7%, followed by cid003 (Spec-to-RTL) at 31.5%, cid004 (Modification) at 29.8%, cid007 (Lint/QoR) at 25.0%, and cid002 (Code Completion) trailing at 21.1%. The cid002 weakness is driven by its 59.6% always-fail rate — the highest of any category.

**Causal narrative**: The model generates syntactically valid, compilable Verilog in >96% of attempts, but its RTL logic is wrong in the majority of cases. Controlling for CID and difficulty, failing rollouts produce larger RTL and consume more output tokens than passing ones across every CID/difficulty combination — the model spends more effort on problems it gets wrong. The weakest domains — Interfaces & Communication (74.4% always-fail), Control & FSMs (68.6%), and Memory & Buffering (64.5%) — require reasoning about concurrent state machines, protocol handshakes, and multi-cycle timing, which the model handles poorly.

> **Key limitations**: (1) Dominant failure mode is wrong logic in compilable RTL (93.9% of failures); (2) 53% of tasks never pass across 5 attempts — a fundamental capability gap; (3) 19.2 pp consistency gap suggests significant recoverable performance via better sampling; (4) Steep easy-to-medium dropoff (21 pp) across all CIDs indicates a reasoning complexity ceiling; (5) cid007 uniquely suffers from syntax errors (17.3% of its failures) and lint-check failures (all 75 lint failures are in cid007); (6) Failing rollouts consume +1100 to +2800 more output tokens than passing ones at the same CID/difficulty level.

---

## 2. Classification Overview

### 2.1 Error Type Distribution


| Failure Type        | Count    | % of 1090 Failed Rollouts | Description                                                              |
| ------------------- | -------- | ------------------------- | ------------------------------------------------------------------------ |
| SIMULATION_MISMATCH | 1024     | 93.9%                     | RTL compiles, simulation runs, assertions fail — wrong logic             |
| TIMEOUT             | 31       | 2.8%                      | Container exceeds 300s limit (infinite loops or non-terminating designs) |
| SYNTAX_ERROR        | 29       | 2.7%                      | iverilog cannot parse the generated Verilog                              |
| PARSE_FAILED        | 5        | 0.5%                      | No RTL extracted from model output                                       |
| COMPILATION_ERROR   | 1        | 0.1%                      | iverilog rejects elaboration (wrong module/port names)                   |
| INFRASTRUCTURE      | 0        | 0.0%                      | No infrastructure failures detected                                      |
| **Total**           | **1090** | **100%**                  |                                                                          |


**No incomplete RTL (missing `endmodule`) was detected across any rollout.**

### 2.2 Output Source Distribution

All 1510 rollouts produced output through the normal CODE_OUTPUT path — no REASONING_VERBATIM or REASONING_PARSED cases. The model reliably generates text through its primary output channel. Reasoning summaries are present in 1085/1090 failed rollouts (99.5%), indicating the model engages in structured reasoning before generating code, but this reasoning does not reliably translate into correct implementations.

### 2.3 Per-CID Failure Type Cross-Tabulation


| Failure Type        | cid002  | cid003  | cid004  | cid007  | cid016  | Total    |
| ------------------- | ------- | ------- | ------- | ------- | ------- | -------- |
| SIMULATION_MISMATCH | 360     | 259     | 186     | 115     | 104     | 1024     |
| SYNTAX_ERROR        | 0       | 1       | 2       | **26**  | 0       | 29       |
| TIMEOUT             | 10      | 7       | 1       | 8       | 5       | 31       |
| PARSE_FAILED        | 1       | 0       | 4       | 0       | 0       | 5        |
| COMPILATION_ERROR   | 0       | 0       | 0       | 1       | 0       | 1        |
| **Total failures**  | **371** | **267** | **193** | **150** | **109** | **1090** |


**Key finding**: 26 of 29 syntax errors (89.7%) concentrate in **cid007** (Lint/QoR Improvement). This is because cid007 tasks require modifying existing RTL for lint compliance and power/performance optimization. The model frequently introduces SystemVerilog constructs (inline variable declarations inside `always_comb`, `logic` type declarations within procedural blocks) that iverilog's `-g2012` mode rejects. Additionally, **all 75 lint-check failures** (where Verilator lint rejects the RTL) occur exclusively in cid007.

---

## 3. Domain Analysis

### 3.1 Per-Domain Performance


| Domain                          | Tasks   | Rollouts | pass@1     | Always-Fail | AF% (of Tasks) |
| ------------------------------- | ------- | -------- | ---------- | ----------- | -------------- |
| Interfaces & Communication      | 39      | 195      | 18.5%      | 29          | **74.4%**      |
| Control & FSMs                  | 35      | 175      | 13.1%      | 24          | **68.6%**      |
| Memory & Buffering              | 31      | 155      | 17.4%      | 20          | **64.5%**      |
| Signal Processing & Specialized | 33      | 165      | 15.8%      | 20          | 60.6%          |
| Arithmetic & Computation        | 44      | 220      | 21.4%      | 25          | 56.8%          |
| Encoding & Data Transform       | 42      | 210      | 39.5%      | 19          | 45.2%          |
| Other (misc. topics)            | 78      | 390      | 45.6%      | 23          | 29.5%          |
| **Total**                       | **302** | **1510** | **27.81%** | **160**     | **53.0%**      |


pass@1 = passing rollouts / total rollouts per domain. AF% = always-fail tasks / total tasks per domain.

**Interfaces & Communication** is the weakest domain: 74.4% of its tasks never pass. This includes AXI, APB, AHB protocol controllers and bus interfaces that require reasoning about multi-signal handshakes, ready/valid protocol timing, and address-decoded register maps. The model consistently fails to implement correct protocol state machines.

**Control & FSMs** has the lowest pass@1 (13.1%) — tasks like sorters, elevator controllers, and interrupt controllers require multi-state, multi-cycle algorithms that the model struggles to implement correctly.

### 3.2 Failure Class vs. Domain Cross-Tabulation


| Failure Type        | Arithmetic | Control | Encoding | Interfaces | Memory  | Other   | Signal Proc. | Total    |
| ------------------- | ---------- | ------- | -------- | ---------- | ------- | ------- | ------------ | -------- |
| SIMULATION_MISMATCH | 155        | 136     | 126      | 152        | 126     | 199     | 130          | 1024     |
| SYNTAX_ERROR        | 14         | 3       | 1        | 2          | 2       | 0       | 7            | 29       |
| TIMEOUT             | 4          | 8       | 0        | 5          | 0       | 13      | 1            | 31       |
| PARSE_FAILED        | 0          | 5       | 0        | 0          | 0       | 0       | 0            | 5        |
| COMPILATION_ERROR   | 0          | 0       | 0        | 0          | 0       | 0       | 1            | 1        |
| **Total failures**  | **173**    | **152** | **127**  | **159**    | **128** | **212** | **139**      | **1090** |


Simulation mismatch dominates uniformly across all domains — the model's weakness is **domain-agnostic incorrect logic**, not domain-specific compilation issues. Syntax errors show mild concentration in Arithmetic (14) and Signal Processing (7), likely because these domains involve more complex mathematical expressions where the model is more likely to use advanced SystemVerilog constructs.

---

## 4. Detailed Failure Analysis

### 4a. Simulation Mismatches (1024 rollouts, 93.9% of failures)

This is the primary analysis target. Sub-categorizing the 1024 simulation mismatches by the nature of the error in stderr:


| Sub-Category                             | Count | % of 1056 Sim Mismatches | Description                                                                      |
| ---------------------------------------- | ----- | ------------------------ | -------------------------------------------------------------------------------- |
| Test failure (no clear assertion detail) | ~581  | 55.0%                    | Tests fail but stderr shows only `FAILED` summary, not specific assertion values |
| Assertion with expected/got values       | 168   | 15.9%                    | Clear `expected X, got Y` messages showing specific value mismatches             |
| Other assertion failures                 | 154   | 14.6%                    | `assert False` or assertion without detailed expected/got                        |
| Lint check failure (cid007 only)         | 64    | 6.1%                     | Verilator lint rejects the RTL (even though it compiles in iverilog)             |
| AttributeError (missing signal)          | 42    | 4.0%                     | CocoTB cannot find a signal/port — model used different naming                   |
| ValueError (conversion failure)          | 35    | 3.3%                     | Signal values cannot be converted to int — typically X/Z propagation             |


**RTL Size and Token Usage**: Controlling for CID and difficulty, failing rollouts consistently produce larger RTL (+390 to +2253 chars median) and consume more output tokens (+1100 to +2800 median) than passing ones across every CID/difficulty combination. The model spends more tokens on problems it gets wrong.

No truncation was detected (0 rollouts hit the 32K token limit), so failures are not caused by output being cut off mid-generation.

#### Lint-Check Failures (cid007-Specific)

All 75 lint-check failures occur in cid007 (Lint/QoR Improvement). In these tasks, the model must fix lint warnings in existing RTL. The irony: the model "fixes" the code but introduces new lint violations or fails to address the original ones. 15 tasks account for all 75 failures (5/5 rollouts each), including:

- `cont_adder_0042`: 5/5 fail (all rollouts produce syntax errors + lint failures)
- `hill_cipher_0015`: 5/5 fail
- `IIR_filter_0019`: 5/5 fail
- `elevator_control_0033`: 5/5 fail

#### AttributeError Failures (42 rollouts)

These occur when CocoTB's test harness tries to access a signal that doesn't exist in the model's RTL. This means the model used a different port name, signal name, or module hierarchy than what the testbench expects — a naming/interface mismatch rather than a logic error.

#### ValueError Failures (35 rollouts)

`ValueError: Cannot convert to int` indicates X (unknown) or Z (high-impedance) values propagating to output signals when the testbench expects deterministic values. This typically means the model's reset logic is incomplete, leaving registers uninitialized, or there are undriven signals in the design.

### 4b. Syntax Errors (29 rollouts, 2.7% of failures)

26/29 syntax errors are in cid007. The root cause is the model introducing **inline variable declarations inside procedural blocks** — a SystemVerilog feature that iverilog's `-g2012` mode does not support:

```systemverilog
always_comb begin
    logic signed [DATA_WIDTH-1:0] thresh1 = THRESHOLD_VALUE_1;  // iverilog rejects this
    logic signed [DATA_WIDTH-1:0] thresh2 = THRESHOLD_VALUE_2;  // iverilog rejects this
```

iverilog produces: `/code/rtl/cont_adder.sv:35: syntax error` / `Malformed statement` for each inline declaration. These cascading errors cause massive error output (10+ error lines from a single root cause).

This is a single root cause expressed across 14 tasks and 26 rollouts: the model does not account for iverilog's limited SystemVerilog support when generating RTL modifications.

### 4c. Timeouts (31 rollouts, 2.8% of failures)

31 rollouts hit the 300-second container timeout, spread across 12 tasks:


| Task                  | Timeouts | Category | Domain     |
| --------------------- | -------- | -------- | ---------- |
| manchester_enc_0005   | 5/5      | cid016   | Encoding   |
| perf_counters_0001    | 5/5      | cid003   | Other      |
| ir_receiver_0001      | 4/5      | cid002   | Interfaces |
| Attenuator_0001       | 3/5      | cid002   | Other      |
| sorter_0059           | 3/5      | cid007   | Control    |
| elevator_control_0026 | 3/5      | cid002   | Control    |


Timeouts likely indicate the model generates RTL with **infinite loops or non-terminating FSMs** — e.g., a sorter that never asserts `done`, causing the testbench to wait indefinitely. The concentration in Control & FSM topics supports this: FSM designs with incorrect termination conditions naturally produce non-halting simulations.

### 4d. Parse Failures (5 rollouts, 0.5% of failures)

All 5 parse failures come from 2 tasks:

- `elevator_control_0006`: 4/5 rollouts parse-failed
- `elevator_control_0026`: 1/5 rollout parse-failed

Both are elevator control tasks where the model failed to produce extractable Verilog. With `container_exit_code=None` and stderr showing `"parse_failed: could not extract RTL from model output"`, these indicate the model's text output did not contain recognizable Verilog module definitions — possibly generating only natural language explanation instead of code.

---

## 5. Capability vs. Consistency

### 5.1 Oracle Ceiling

```
Oracle ceiling (pass >= 1/5):  47.02%  (142/302 tasks)
Actual pass@1:                 27.81%
Consistency gap:               19.21 pp
```

The 19.21 pp gap means that **if a perfect selection mechanism existed** (choosing the best rollout per task), performance would nearly double. This is a large consistency gap relative to the base rate, suggesting significant headroom from improved sampling or self-verification.

### 5.2 Consistency Distribution


| Pass Count        | Tasks   | % of 302 Tasks |
| ----------------- | ------- | -------------- |
| 0/5 (always-fail) | 160     | 53.0%          |
| 1/5               | 34      | 11.3%          |
| 2/5               | 25      | 8.3%           |
| 3/5               | 26      | 8.6%           |
| 4/5               | 27      | 8.9%           |
| 5/5 (always-pass) | 30      | 9.9%           |
| **Total**         | **302** | **100%**       |


The distribution is bimodal: tasks cluster at 0/5 (hard ceiling) and 4-5/5 (reliable), with a relatively flat middle. The 34 tasks at 1/5 represent the "barely capable" boundary — the model succeeds only with lucky sampling.

### 5.3 Flaky vs. Always-Fail Error Comparison

A "flaky" task is one where some rollouts pass and some fail (1/5 to 4/5 passing) — the model has partial capability but doesn't reliably produce correct solutions. Comparing the error sub-types between always-fail (0/5) and flaky task failures:


| Error Sub-Type       | Always-Fail (of 800) | Flaky (of 290) |
| -------------------- | -------------------- | -------------- |
| TestFail (generic)   | 425 (53.1%)          | 135 (46.6%)    |
| Assert: expected/got | 146 (18.2%)          | 63 (21.7%)     |
| Assert: other        | 82 (10.2%)           | 29 (10.0%)     |
| Lint failure         | 46 (5.8%)            | 24 (8.3%)      |
| ValueError (X/Z)     | 21 (2.6%)            | 19 (6.6%)      |
| AttributeError       | 35 (4.4%)            | 7 (2.4%)       |
| Syntax error         | 18 (2.2%)            | 7 (2.4%)       |


Always-fail total: 800 failing rollouts from 160 tasks. Flaky total: 290 failing rollouts from 112 tasks.

The distributions are similar, with one notable difference: **ValueError (X/Z propagation) is 2.5x more prevalent in flaky failures** (6.6% vs. 2.6%). This suggests that some flaky tasks fail because of non-deterministic signal initialization — the model sometimes gets the reset/initialization logic right and sometimes doesn't, producing X values that cause test failures in some rollouts but not others.

---

## 6. Concrete Failure Examples

### Example 1: Representative Simulation Mismatch — Sorting Engine (Reversed Output Order)

**Task**: `cvdp_copilot_sorter_0009` (cid002, easy, **0/5 pass**)

**Specification**: Complete a `sorting_engine` module implementing **brick sort** (odd-even sort). The module should sort elements in ascending order using a FSM with IDLE → LOAD → SORT → DONE states, performing pairwise compare-and-swap operations over N passes.

**What the model produced**: A complete FSM implementation with correct state transitions, correct data loading from `in_data`, and a correct compare-swap loop in the SORT state. However, the `out_data` construction is wrong:

```systemverilog
// Model's output construction (WRONG):
always @* begin
    out_data = 0;
    for (i = 0; i < N; i = i+1) begin
        out_data = {out_data, data_array[i]}; // shift left and add element
    end
end
```

This concatenation loop produces `out_data` with `data_array[0]` in the MSB position and `data_array[N-1]` in the LSB — the **reverse** of what the testbench expects (`data_array[0]` at LSB).

**Stderr evidence**: Every test assertion shows the output in descending order instead of ascending:

```
AssertionError: ERROR: DUT output=[25, 9, 2, 1] expected=[1, 2, 9, 25]
AssertionError: Sorted test failed, got [3, 2, 1, 0], expected [0, 1, 2, 3]
```

The sort algorithm itself works correctly (the values are sorted), but the **bit packing order is reversed**. The correct construction should use `out_data[i*WIDTH +: WIDTH] = data_array[i]`.

**Reasoning summary**: The model's reasoning correctly identified the FSM states, the brick sort algorithm, and the need to concatenate the array into `out_data`. It failed at the mechanical detail of bit-packing order — a subtle but critical implementation detail.

**Diagnosis**: This is a **correct-algorithm, wrong-interface** failure. The model understands the sorting algorithm but does not correctly map the internal array to the flattened output bus. This pattern (correct computation, wrong output formatting) likely accounts for a significant fraction of simulation mismatches.

---

### Example 2: Flaky Task Contrast — Dot Product Module (2/5 pass)

**Task**: `cvdp_copilot_dot_product_0002` (cid002, easy, **2/5 pass**)

**Specification**: Complete a dot product module that accumulates products of two input vectors using a 3-state FSM (IDLE → COMPUTE → OUTPUT), with 2-cycle output latency.

**Passing rollout (rollout 1)**: Uses a dedicated `out_latency` counter to track the 2-cycle output delay. In the COMPUTE state, the counter and accumulator are updated in the sequential `always_ff` block:

```systemverilog
// Passing: latency counter managed in always_ff
logic [1:0] out_latency;
// ...in sequential block:
//   Transitions from COMPUTE to OUTPUT, then counts 2 cycles before asserting valid
```

**Failing rollout (rollout 0)**: Uses `always_comb` to drive `next_state`, `dot_product_out`, and `dot_product_valid_out`. The problem: the output assignment `dot_product_out = 32'b0` is the default in the combinational block, which means the output is only driven correctly during the OUTPUT state. However, the failing rollout attempts to assign registered outputs from a combinational block, creating a race condition where `dot_product_valid_out` never gets asserted at the right time.

**Stderr**: `"Unexpected state: valid_out is not asserted"` — the testbench waits for `dot_product_valid_out` to go high after computation, but it never does. The module computes the correct dot product internally but fails to signal completion.

**Diagnosis**: The passing rollout correctly separates sequential state updates (including the output latency counter) from combinational next-state logic. The failing rollout conflates sequential and combinational assignments — a classic RTL design error where outputs that should be registered are instead driven combinationally with incorrect default values. This is the model's **boundary of understanding**: it sometimes gets the sequential/combinational separation right and sometimes doesn't.

---

### Example 3: Always-Fail from Weak Domain — APB GPIO Controller

**Task**: `cvdp_copilot_apb_gpio_0001` (cid003, medium, **0/5 pass**)

**Specification**: Design a GPIO module compatible with the APB protocol. Requires configurable GPIO width, bidirectional control, interrupt generation, synchronization flip-flops, and a register map with 7 addresses (input data, output data, output enable, interrupt enable, type, polarity, state).

**What the model produced**: A complete, well-structured module with:

- Correct port list matching the specification
- 2-stage synchronization flip-flops for `gpio_in`
- APB read/write state machine
- 7 internal registers at the correct addresses
- Interrupt generation logic with edge/level detection

**Stderr**: `TESTS=10 PASS=4 FAIL=6` — the module passes 4 of 10 tests, including basic read/write and reset tests, but fails 6 tests covering interrupt behavior and edge cases.

**Diagnosis**: The model correctly implements the APB bus interface, register map, and basic GPIO I/O — the "structural" aspects of the design. But it fails on the **interrupt generation logic** — specifically, the interaction between interrupt type (edge vs. level), polarity, and the state register's read-to-clear behavior. This requires reasoning about how multiple configuration registers interact to produce correct interrupt behavior across various edge cases, which is precisely the kind of multi-register, multi-condition reasoning the model struggles with on protocol tasks. The 4/10 test pass rate indicates **partial correctness**: the scaffolding is right, but the nuanced behavioral logic is wrong.

---

### Example 4: Syntax Error — Inline Variable Declarations in cid007

**Task**: `cvdp_copilot_cont_adder_0042` (cid007, easy, **0/5 pass**)

**Specification**: Modify a continuous adder module to fix lint warnings and improve code quality. The original code uses `always_comb` with threshold comparisons and accumulation logic.

**What the model produced**: The model attempted to improve the code by declaring local variables inside `always_comb`:

```systemverilog
always_comb begin
    // ...
    logic signed [DATA_WIDTH-1:0] thresh1 = THRESHOLD_VALUE_1;  // LINE 35
    logic signed [DATA_WIDTH-1:0] thresh2 = THRESHOLD_VALUE_2;  // LINE 36
    threshold_1_comb = (new_sum >= thresh1) || (new_sum <= -thresh1);
    threshold_2_comb = (new_sum >= thresh2) || (new_sum <= -thresh2);
```

**Stderr**:

```
/code/rtl/cont_adder.sv:35: syntax error
/code/rtl/cont_adder.sv:35: error: Malformed statement
/code/rtl/cont_adder.sv:36: syntax error
/code/rtl/cont_adder.sv:36: error: Malformed statement
```

iverilog's `-g2012` mode does not support inline variable declarations inside procedural blocks. The model should have declared `thresh1` and `thresh2` at the module level or used `localparam`.

**Additional failure layer**: Even when the syntax error is set aside, 4/5 rollouts also fail the Verilator lint check (`"Linting return errors"`), meaning the model's modifications introduce new lint violations rather than fixing the original ones. This double failure (syntax error preventing simulation + lint failure) makes cid007 tasks particularly challenging: the model must satisfy two validators (iverilog compilation + Verilator lint), and its modifications often satisfy neither.

**Diagnosis**: This is a **toolchain awareness failure**. The model uses valid SystemVerilog that would compile in commercial simulators (e.g., Synopsys VCS) but fails in the open-source iverilog toolchain used by CVDP. All 26 syntax errors in cid007 trace to this same root cause: the model doesn't constrain its output to the subset of SystemVerilog that iverilog supports.

---

## 7. Appendix

### A. Per-CID x Difficulty Pass Rates (Rollout-Level)


| CID       | Easy (pass/total) | Medium (pass/total) |
| --------- | ----------------- | ------------------- |
| cid002    | 78/240 (32%)      | 21/230 (9%)         |
| cid003    | 90/205 (44%)      | 33/185 (18%)        |
| cid004    | 57/150 (38%)      | 25/125 (20%)        |
| cid007    | 32/110 (29%)      | 18/90 (20%)         |
| cid016    | 48/105 (46%)      | 18/70 (26%)         |
| **Total** | **285/810 (35%)** | **135/700 (19%)**   |


### B. Always-Fail by CID (Task-Level)


| CID       | Always-Fail | Total Tasks | AF% (of Total Tasks) |
| --------- | ----------- | ----------- | -------------------- |
| cid002    | 56          | 94          | 59.6%                |
| cid004    | 31          | 55          | 56.4%                |
| cid007    | 21          | 40          | 52.5%                |
| cid003    | 38          | 78          | 48.7%                |
| cid016    | 14          | 35          | 40.0%                |
| **Total** | **160**     | **302**     | **53.0%**            |


### C. Execution Time


|                  | n    | Median | Mean  | Max    |
| ---------------- | ---- | ------ | ----- | ------ |
| Passing rollouts | 420  | 1.9s   | 3.9s  | 56.0s  |
| Failing rollouts | 1090 | 1.6s   | 11.3s | 322.7s |


The higher mean for failures (11.3s vs 3.9s) is skewed by the 31 timeout rollouts at ~300s. Median execution time is actually slightly lower for failures (1.6s vs 1.9s), because many failures are caught quickly by the first test assertion.

### D. Token Usage


|         | n    | Median Output Tokens | Mean Output Tokens | Max Output Tokens |
| ------- | ---- | -------------------- | ------------------ | ----------------- |
| Passing | 420  | 2,082                | 2,619              | 9,791             |
| Failing | 1090 | 4,042                | 4,788              | 26,301            |


No rollouts were truncated at the 32K token limit.

### E. RTL Size (Characters)


|         | n    | Median | Mean  |
| ------- | ---- | ------ | ----- |
| Passing | 420  | 2,044  | 2,480 |
| Failing | 1085 | 3,923  | 4,758 |


n for failing is 1085 (not 1090) because 5 parse-failed rollouts have no extracted RTL.

Controlling for CID and difficulty, the gap holds across every CID/difficulty combination (+390 to +2253 chars median).

### F. Timeout Tasks


| Task                       | Timeouts/5 | Category | Difficulty |
| -------------------------- | ---------- | -------- | ---------- |
| manchester_enc_0005        | 5/5        | cid016   | easy       |
| perf_counters_0001         | 5/5        | cid003   | easy       |
| ir_receiver_0001           | 4/5        | cid002   | easy       |
| Attenuator_0001            | 3/5        | cid002   | easy       |
| elevator_control_0026      | 3/5        | cid002   | medium     |
| sorter_0059                | 3/5        | cid007   | easy       |
| elevator_control_0033      | 2/5        | cid007   | medium     |
| gaussian_rounding_div_0022 | 2/5        | cid007   | medium     |
| vga_controller_0026        | 1/5        | cid007   | medium     |
| wb2ahb_0001                | 1/5        | cid003   | medium     |
| gcd_0001                   | 1/5        | cid003   | easy       |
| gcd_0009                   | 1/5        | cid004   | easy       |
