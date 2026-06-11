# CVDP Agentic Heavy — Failure Analysis Report

**Model:** Nemotron Super 49b
**Rollouts:** 18 (6 tasks)
**pass@1:** 0.0%
**pass@k:** 0.0%
**Consistency:** 0.0%

## 1. Executive Summary

Across 18 rollouts on 6 tasks:

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
The model explores but never writes fixes. It may be producing text-only answers instead of using echo.

## 3. Pipeline Funnel

Cumulative survival through each workflow stage:

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

## 11. Root Cause Classification

### Cross-Cutting Summary Table

| Bucket | Tasks | Avg pass rate | Top behavior codes | Improvement method |
|--------|------:|--------------:|-------------------|-------------------|
| NP behavioral | 3 (enso_0007, enso_0017, enso_0037) | 0% | B6 (early exit), B1 (path thrash), B2 (repeated reads) | RL |
| NP knowledge gap | 2 (axi4_lite_0002, biriscv_0001) | 0% | B7 (compile-no-run), B1, B3 | SFT + RL |
| NP behavioral + environment | 1 (reckon_0001) | 0% | B1, B2, B5, B7 | RL + EI mitigation |

### Per-Task Root Cause

| Task | Pass Rate | Root Cause | Evidence |
|------|-----------|------------|----------|
| enso_0007 | 0/3 | BI: Behavioral issue (early exit) | All 3 rollouts did only 6-7 steps (ls/cat), never wrote a file. B6 on all 3. Model explored then quit. |
| enso_0017 | 0/3 | BI: Behavioral issue (exploration loop) | All 3 rollouts did only ls/cat (13-16 steps), never wrote. R1 did 10 consecutive ls calls. |
| enso_0037 | 0/3 | BI: Behavioral issue (minimal engagement) | R0 did 1 step total. R1 wrote to WRONG paths (rtl/ instead of hardware/src/). R2 never wrote. |
| axi4_lite_0002 | 0/3 | KG: Knowledge gap (wrong extension + syntax) | All 3 rollouts wrote to `.sv` instead of `.v` (E6). Generated backtick-prefixed `module` keyword. Never compiled successfully. |
| biriscv_0001 | 0/3 | KG+BI: Knowledge gap + behavioral | R0 wrote correct file but 7 echo calls (thrashing), invalid defs (`1'b2`), missing `-I` flag. R1 produced 0 function calls. R2 had 11 iverilog attempts all failing on include paths. |
| reckon_0001 | 0/3 | BI+KG: Behavioral + knowledge gap | R0 wrote to correct file but hit syntax errors. R1/R2 wrote to wrong paths (rtl/, data/). All hit non-existent testbench references. |

---

## 12. Per-Task Deep Dive — Never-Pass Tasks (6)

### cvdp_agentic_heavy_enso_0007 (0/3 pass) — Behavioral Issue (Early Exit)

- **Root cause**: BI (B6: early exit) — the model explored briefly and gave up without attempting a fix
- **Golden patch file**: `hardware/src/parser.sv`
- **What model did**: All 3 rollouts performed only 6-7 tool calls (ls + cat). The model browsed the directory structure, read a few files, then stopped. Zero echo calls across all 3 rollouts.
- **Why it failed**: The model recognized this is a code modification task but never committed to writing a fix. It appears to have generated a text-based analysis rather than using the echo tool. With only 6-7 steps out of 20 available, the model left 65% of its step budget unused.
- **Contrast**: Claude Opus and GPT-5 both solve this task 3/3 in 13-36 steps. The fix (editing `parser.sv` to correct packet metadata parsing) is straightforward once the model reads the testbench error messages. Nemotron Super never even reads the testbench.
- **Improvement path**: RL to enforce tool use — the model must learn that text-only answers receive reward=0 and that echo is required to modify files.

### cvdp_agentic_heavy_enso_0017 (0/3 pass) — Behavioral Issue (Exploration Loop)

- **Root cause**: BI (B1: path thrashing, B2: repeated reads) — the model gets stuck in an ls/cat loop
- **Golden patch files**: `hardware/src/constants.sv`, `hardware/src/parser.sv`
- **What model did**: R0 (15 steps): 7 cat + 8 ls, no echo. R1 (13 steps): 2 cat + 11 ls — 10 consecutive ls calls browsing different directories without convergence. R2 (16 steps): 7 cat + 9 ls, no echo.
- **Why it failed**: The model explores the `hardware/` directory structure extensively but never transitions from "reading" to "writing" mode. It appears to be searching for something specific but can't find it, leading to repeated directory listings (B1).
- **Contrast**: Claude Opus solves this 1/3 in 21 steps; GPT-5 gets 0/3 but at least writes files in all 3 rollouts. Nemotron Super never even attempts an echo.
- **Improvement path**: RL to enforce the explore→write transition. A rule like "after reading 5+ files, you must attempt an echo" could break the exploration loop.

### cvdp_agentic_heavy_enso_0037 (0/3 pass) — Behavioral Issue (Minimal Engagement)

- **Root cause**: BI (B6: early exit, B4: no verification) — the model barely engages with the task
- **Golden patch files**: `hardware/src/constants.sv`, `hardware/src/timestamp.sv`
- **What model did**:
  - R0 (1 step): A single cat call. The model read one file and stopped. One step out of 20.
  - R1 (16 steps): Wrote 6 echo calls to `rtl/constants.sv` and `rtl/timestamp.v` — WRONG directory (`rtl/` instead of `hardware/src/`) and WRONG extension for timestamp (`.v` instead of `.sv`). Also generated backtick-prefixed content in some writes. Never compiled.
  - R2 (10 steps): Only ls/cat, never wrote.
- **Why it failed**: R0 and R2 show extreme disengagement — the model doesn't attempt a fix. R1 shows effort but writes to the wrong path, demonstrating it can't navigate the enso project's `hardware/src/` directory structure.
- **Contrast**: Claude Opus solves this 3/3 and GPT-5 solves it 2/3. Both consistently write to `hardware/src/`.
- **Improvement path**: RL for path discovery (use ls before echo to verify the target directory exists). The wrong-path issue (rtl/ vs hardware/src/) suggests the model defaults to a generic project layout assumption.

### cvdp_agentic_heavy_axi4_lite_0002 (0/3 pass) — Knowledge Gap

- **Root cause**: KG — the model generates Verilog with wrong file extension and invalid syntax
- **Golden patch file**: `rtl/axi_ram_rd_if.v`
- **What model did**: All 3 rollouts wrote to `rtl/axi_ram_rd_if.sv` (`.sv` instead of `.v` — E6 error). The model also generated code with backtick-prefixed `module` keywords (e.g., `` `module axi_ram_rd_if ``), which is invalid Verilog syntax. R1/R2 additionally referenced non-existent testbenches (`tb/axi_ram_rd_if_tb.v`).
- **Why it failed**: Two knowledge gaps compound: (1) the model doesn't know that the existing file uses `.v` extension and writes to `.sv` instead, meaning the original buggy `.v` file is never overwritten; (2) the generated Verilog has syntax errors (backtick prefix) that iverilog rejects.
- **Contrast**: GPT-5 solves this 1/3 by writing to the correct `.v` file. Claude Opus solves it 3/3. Both use the correct extension.
- **Improvement path**: SFT to teach the model to check existing file extensions before writing (or use ls to discover the actual filename). The backtick syntax issue suggests the model confuses Verilog preprocessor directives with module declarations.

### cvdp_agentic_heavy_ultraembedded_biriscv_0001 (0/3 pass) — Knowledge Gap + Behavioral

- **Root cause**: KG+BI — model writes to correct file but can't resolve include paths, and one rollout produces zero function calls
- **Golden patch file**: `src/core/biriscv_divider.v`
- **What model did**:
  - R0 (15 steps): 7 echo calls to `biriscv_divider.v` (correct file, heavy thrashing). Also wrote `biriscv_defs.v` with invalid constants (`1'b2`). One iverilog call failed with `Include file biriscv_defs.v not found` — missing `-I` flag.
  - R1 (0 steps): **Zero function calls.** The model produced text output only. The baseline (unmodified) code ran and failed.
  - R2 (19 steps): 2 echo calls to correct file. Then 11 consecutive iverilog attempts, all failing on include path issues. Tried `+incdir+` (segfault), `-I/code/src` (wrong dir), `-I/code/src-core` (typo with hyphen). Never found the correct `-I/code/src/core`.
- **Why it failed**: R0 and R2 found the correct file but couldn't compile it due to include path issues — a knowledge gap about iverilog's `-I` flag syntax and the project's include directory structure. R1 is a complete behavioral failure (P0: zero tool calls). R2's 11-attempt compilation loop is classic perseveration (B5+B7).
- **Contrast**: Claude Opus solves this 3/3 in 17-21 steps with clean compilation. GPT-5 solves it 2/3 — its passing rollouts also created custom testbenches that avoided the full-project compilation entirely.
- **Improvement path**: SFT on iverilog include path syntax (`-I` vs `+incdir+`). RL to prevent the 11-attempt compilation loop (perseveration). The zero-function-call rollout (R1) needs RL to enforce tool engagement.

### cvdp_agentic_heavy_reckon_0001 (0/3 pass) — Behavioral + Knowledge Gap

- **Root cause**: BI+KG — model writes to wrong file paths and generates syntactically invalid Verilog
- **Golden patch file**: `src/spi_slave.v`
- **What model did**:
  - R0 (14 steps): Wrote to correct file `/code/src/spi_slave.v` but with syntax errors. Iverilog: `tb/spi_slave_tb.v: No such file` + syntax errors.
  - R1 (14 steps): Wrote to WRONG path `/code/rtl/spi_slave.v` (rtl/ instead of src/). Same iverilog errors.
  - R2 (20 steps): Wrote to WRONG path `/code/data/spi_slave.v` (data/ instead of src/). Same iverilog errors. Used pwd to check working directory. Hit step exhaustion.
- **Why it failed**: Only R0 wrote to the correct file, but the fix had syntax errors. R1/R2 wrote to wrong directories (B1: path thrashing across rtl/, data/, src/). All 3 rollouts referenced a non-existent testbench `tb/spi_slave_tb.v`. The model can't reliably find the correct source path despite having ls available.
- **Contrast**: GPT-5 also 0/3 on this task (2/3 never wrote). Claude Sonnet uniquely solves it at 67%.
- **Improvement path**: RL to enforce ls-before-echo workflow (verify target directory). SFT on SPI slave design patterns. The path-finding issue is the proximate cause and is behavioral; the underlying fix quality is a separate knowledge gap.

---

## 13. Diagnostic Summary

**Nemotron Super 49b at 0.0% pass@1** represents a model with fundamental agentic workflow deficiencies. Unlike stronger models where failures are in RTL logic quality, Super 49b fails primarily because it can't execute the basic agent loop:

**Three dominant failure modes, in order of leverage:**

1. **Early exit / no engagement (B6: 28%, plus 7/18 rollouts with zero echo calls)**: The model explores briefly and gives up. On enso_0007 (which Opus/GPT-5/Sonnet all solve 3/3), Super 49b reads 3-7 files then stops. This is purely behavioral and the single highest-leverage RL target.

2. **Path thrashing (B1: 44%) and wrong file paths**: The model can't reliably navigate project directory structures. It writes to `rtl/` when the file is in `src/`, or uses `.sv` when the file is `.v`. On axi4_lite_0002, all 3 rollouts used the wrong extension. This is a knowledge gap about codebase navigation.

3. **Zero vvp usage (0% self-test rate)**: Across all 18 rollouts, the model never once ran vvp. It doesn't understand the compile→run loop at all. Combined with B7 (39%), this means even when the model compiles, it never checks whether its fix works.

**Causal chain**: Poor codebase navigation (B1) → wrong file paths → file not found errors (E1: 61% of errors) → model gives up early (B6) → no fix submitted → reward 0. Even when the model does write (50% file mod rate), it writes to wrong paths or with wrong syntax → compilation fails → never reaches vvp → no feedback → can't iterate.

**Gap to GPT-5 (+44pp)**: The gap is almost entirely behavioral. GPT-5 has 89% file mod rate vs 50%, 94% compile rate vs 39%, 50% vvp rate vs 0%. If Super 49b could match GPT-5's workflow discipline (through RL), it would likely achieve a non-zero pass rate even with its current RTL knowledge.

**Gap to Opus (+72pp)**: Closing this gap requires both behavioral improvements (RL for workflow) and substantial knowledge improvements (SFT for correct Verilog syntax, file extension handling, include path resolution).
