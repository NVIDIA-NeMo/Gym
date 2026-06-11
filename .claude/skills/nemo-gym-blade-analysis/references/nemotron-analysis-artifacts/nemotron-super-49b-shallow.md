# CVDP Agentic Heavy — Failure Analysis Report

**Model:** Nemotron Super 49b
**Rollouts:** 18 (6 tasks)
**pass@1:** 0.0%
**pass@k:** 0.0%
**Consistency:** 0.0%

## 1. Executive Summary

**Pipeline funnel:** 18 started → 9 wrote files (50.0%) → 7 compiled (38.9%) → 0 ran vvp (0.0%) → 0 passed (0.0%)

- Mean steps used: **11.7** (median: 14)
- Mean tokens: **126355**
- Mean Docker verification time: **7.4s**

### Self-Verification Behavior

- **33.3%** of rollouts read an existing testbench file
- **0.0%** ran vvp to self-test
- **0.0** mean compile→run feedback loops per rollout
- No rollouts reached vvp self-testing

### Comparison with Reference Model

| Metric | Nemotron Super 49b | GPT-5 | Delta |
|--------|----------:|---------------:|------:|
| pass@1 | 0.0% | 44.4% | -44.4% |
| pass@k | 0.0% | 66.7% | -66.7% |
| Consistency | 0.0% | 16.7% | -16.7% |
| File mod rate | 50.0% | 88.9% | -38.9% |
| Compile rate | 38.9% | 94.4% | -55.6% |
| vvp self-test rate | 0.0% | 50.0% | -50.0% |

### Per-Task Pass Rate Comparison

| Task ID | Nemotron Super 49b | GPT-5 | Delta |
|---------|----------:|---------------:|------:|
| cvdp_agentic_heavy_axi4_lite_0002 | 0% | 33% | -33% |
| cvdp_agentic_heavy_enso_0007 | 0% | 100% | -100% |
| cvdp_agentic_heavy_enso_0017 | 0% | 0% | 0% |
| cvdp_agentic_heavy_enso_0037 | 0% | 67% | -67% |
| cvdp_agentic_heavy_reckon_0001 | 0% | 0% | 0% |
| cvdp_agentic_heavy_ultraembedded_biriscv_0001 | 0% | 67% | -67% |

**Regressions (4):** cvdp_agentic_heavy_axi4_lite_0002, cvdp_agentic_heavy_enso_0007, cvdp_agentic_heavy_enso_0037, cvdp_agentic_heavy_ultraembedded_biriscv_0001

### Phase Distribution Shift

| Phase | Description | Nemotron Super 49b | GPT-5 |
|-------|-------------|----------:|---------------:|
| P0 | No tool usage | 1 (5.6%) | 0 (0.0%) |
| P1 | Explore only (ls/cat/pwd) | 8 (44.4%) | 1 (5.6%) |
| P2 | Wrote files (echo) | 2 (11.1%) | 0 (0.0%) |
| P3 | Compiled (iverilog) | 7 (38.9%) | 5 (27.8%) |
| P4 | Tested (vvp/simulation) | 0 (0.0%) | 4 (22.2%) |
| P5 | Passed verification | 0 (0.0%) | 8 (44.4%) |

### Behavioral Pattern Shift

| Pattern | Nemotron Super 49b | GPT-5 |
|---------|----------:|---------------:|
| B1:path_thrashing | 8 (44.4%) | 0 (0.0%) |
| B2:repeated_reads | 7 (38.9%) | 0 (0.0%) |
| B3:write_without_read | 2 (11.1%) | 3 (16.7%) |
| B4:no_verification | 2 (11.1%) | 0 (0.0%) |
| B5:step_exhaustion | 1 (5.6%) | 7 (38.9%) |
| B6:early_exit | 5 (27.8%) | 0 (0.0%) |
| B7:compile_no_run | 7 (38.9%) | 8 (44.4%) |

## 2. Workflow Phase Distribution

| Phase | Description | Count | % |
|-------|-------------|------:|---:|
| P0 | No tool usage | 1 | 5.6% |
| P1 | Explore only (ls/cat/pwd) | 8 | 44.4% |
| P2 | Wrote files (echo) | 2 | 11.1% |
| P3 | Compiled (iverilog) | 7 | 38.9% |
| P4 | Tested (vvp/simulation) | 0 | 0.0% |
| P5 | Passed verification | 0 | 0.0% |

**Bottleneck:** Among failed rollouts, most stall at **P1** (Explore only (ls/cat/pwd), 8 rollouts).

## 3. Pipeline Funnel

| Stage | Rollouts | % | Drop from previous |
|-------|--------:|---------:|-------------------:|
| Started | 18 | 100.0% | — |
| Wrote files (echo) | 9 | 50.0% | -9 (50.0%) |
| Compiled (iverilog) | 7 | 38.9% | -2 (22.2%) |
| Ran simulation (vvp) | 0 | 0.0% | -7 (100.0%) |
| Passed verification | 0 | 0.0% | — |

## 4. Tool Usage Breakdown

| Tool | Total Calls | Calls/Rollout | Error Rate |
|------|------------|---------------|------------|
| ls | 75 | 4.2 | 14.7% |
| cat | 79 | 4.4 | 78.5% |
| echo | 35 | 1.9 | 0.0% |
| pwd | 1 | 0.1 | 0.0% |
| iverilog | 20 | 1.1 | 70.0% |
| vvp | 0 | 0.0 | — |
| xcelium | 0 | 0.0 | — |

## 5. Tool Error Breakdown

| Error Type | Description | Count | % of All Errors |
|------------|-------------|------:|----------------:|
| E1:path_not_found | File/directory not found | 53 | 60.9% |
| E_other | Other error | 20 | 23.0% |
| E3:compilation_error | iverilog compilation error | 14 | 16.1% |

## 6. Behavioral Patterns

| Pattern | Description | Rollouts | % |
|---------|-------------|--------:|---------:|
| B1:path_thrashing | Tried 3+ path variations for the same file | 8 | 44.4% |
| B2:repeated_reads | Read the same file 3+ times without writing | 7 | 38.9% |
| B7:compile_no_run | Compiled (iverilog) but never ran simulation (vvp) | 7 | 38.9% |
| B6:early_exit | Gave text answer before using half the step budget | 5 | 27.8% |
| B4:no_verification | Wrote files but never compiled/tested | 2 | 11.1% |
| B3:write_without_read | Wrote to a file never read first | 2 | 11.1% |
| B5:step_exhaustion | Used all available steps (max_steps) | 1 | 5.6% |

## 7. Cross-Cutting Analysis

### 7.1 Task Type × Best Phase Reached

| Task Type | P0 | P1 | P2 | P3 | P4 | P5 | pass@k |
|-----------|------:|------:|------:|------:|------:|------:|-------:|
| CD | 0 | 1 | 0 | 3 | 0 | 0 | 0.0% |
| CM | 0 | 1 | 1 | 0 | 0 | 0 | 0.0% |

### 7.2 Project × Pass Rate

| Project | Rollouts | Pass Rate | Mean Steps | Mean Tokens |
|---------|--------:|---------:|-----------:|------------:|
| axi4_lite | 3 | 0.0% | 13 | 140521 |
| enso | 9 | 0.0% | 10 | 100662 |
| reckon | 3 | 0.0% | 16 | 168951 |
| ultraembedded_biriscv | 3 | 0.0% | 11 | 146672 |

## 8. Per-Task Breakdown

| Task ID | Type | pass@1 | Best Phase | Steps (mean) | Files Written | Patterns |
|---------|------|-------:|------------|-------------:|---------------|----------|
| cvdp_agentic_heavy_axi4_lite_0002 | CD | 0% | P3 | 13 | axi_ram_rd_if.sv, axi_ram_rd_if.sv, axi_ram_rd_if_tb.v | B1:path_thrashing, B2:repeated_reads, B3:write_without_read, B4:no_verification, B7:compile_no_run |
| cvdp_agentic_heavy_enso_0007 | CM | 0% | P1 | 6 | — | B6:early_exit |
| cvdp_agentic_heavy_enso_0017 | CD | 0% | P1 | 15 | — | B1:path_thrashing, B2:repeated_reads |
| cvdp_agentic_heavy_enso_0037 | CM | 0% | P2 | 9 | constants.sv, timestamp.v | B2:repeated_reads, B4:no_verification, B6:early_exit |
| cvdp_agentic_heavy_reckon_0001 | CD | 0% | P3 | 16 | spi_slave.v, spi_slave.v, spi_slave.v, spi_slave.v, spi_slave.v | B1:path_thrashing, B2:repeated_reads, B5:step_exhaustion, B7:compile_no_run |
| cvdp_agentic_heavy_ultraembedded_biriscv_0001 | CD | 0% | P3 | 11 | biriscv_defs.v, biriscv_divider.v | B1:path_thrashing, B2:repeated_reads, B3:write_without_read, B6:early_exit, B7:compile_no_run |

## 9. Fix Quality & Iteration Effectiveness

- Mean file rewrites per rollout: **1.5**
- Rollouts with improving vvp FAIL count: **0/18**
- Rollouts with zero-byte writes (accidental file erasure): **0/18**

## 10. Key Takeaways

- **44.4% of rollouts only explore (ls/cat) without writing.** The model reads code but doesn't apply fixes. Behavioral issue (B6: early exit) — addressable via RL.
- **7/18 rollouts compiled but never ran vvp (B7).** The model doesn't understand that iverilog only compiles — vvp is needed to execute. Behavioral issue — highest-leverage RL target.
- **Path thrashing is common** (8 rollouts). The model wastes steps trying path variations. Behavioral issue — addressable via RL or improved ls output.

**Task-level summary:** 0 always-pass, 0 sometimes-pass, 6 never-pass out of 6 tasks

**Overall:** pass@1=0.0%, file modification rate=50.0%, compile rate=38.9%, vvp rate=0.0%

---
