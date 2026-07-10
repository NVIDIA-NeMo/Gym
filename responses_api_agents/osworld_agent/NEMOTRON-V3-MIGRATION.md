# `nemotron-v3` OSWorld-to-Gym migration

## 当前运行结论

最终目标不是继续维护一份 OSWorld fork，而是：

```text
Gym omni-mini adapter
        +
clean xlang-ai/OSWorld main (83e8534)
        +
Colossus local Docker provider
```

内部 `osworld_internal/nemotron-v3` 只作为行为参考。NVCF、Singularity、
Apptainer 和 `remote_docker` 不进入当前运行链；WebArena 是下一阶段的独立
Gym integration。

Nemotron 3 Nano Omni 也不复用 Qwen3-Omni agent scaffold。二者共享 Gym 的
OpenAI-compatible model transport，但 prompt、输出协议、坐标语义和图片数量
限制不同。Gym 现在分别提供 `omni_mini_agent` 和 `qwen3_omni_agent`。

## Compared revisions

- Internal OSWorld branch: `nemotron-v3` at `b76dcf12`.
- Internal branch merge base: OSWorld `main` at `ef43297b`.
- Previous Gym dependency: JeffPengCoder/OSWorld `4c155eb1`.
- Clean upstream checkout: xlang-ai/OSWorld `main` at `83e8534`
  (2026-06-25).
- Internal delta: 112 commits, 111 files, 26,468 insertions and 188 deletions
  relative to its merge base.
- The previous Gym fork and current upstream main share `fe8c78e1`. The fork
  has four unique commits; upstream main has one newer Gemini integration
  commit.

## 调查观察与决策记录

下面每条都明确区分“观察到的事实”和“做出的决策”。

### 1. `nemotron-v3` 不是一个可整体搬运的小补丁集

观察：

- `origin/main` 是 `nemotron-v3` 的祖先，分支独有 112 个提交。
- 分支修改 111 个文件，混合 NVCF/Singularity provider、Nemotron/Qwen/Kimi
  agent、WebArena/WebVoyager/DOM、SLURM/vLLM 脚本、结果可视化和少量
  evaluator/任务数据修改。

决策：按“运行基础设施 / agent / evaluator / benchmark 与脚本”拆分，只把
OSWorld benchmark 运行所需行为迁移到 Gym adapter，不复制整个 fork。

### 2. 最终 OSWorld 基线改为干净 upstream main

观察：

- 之前 Gym 使用 JeffPengCoder/OSWorld `4c155eb1`。
- 它相对共同祖先 `fe8c78e1` 的四个提交分别是 `remote_docker`、submodule
  HTTPS URL、headless OpenCV/缺失依赖打包修补，以及 M3 merge。
- 当前 xlang-ai/OSWorld `main@83e8534` 已包含相同的官方 M3、Qwen3VL、
  Pointer、FastVM 和标准 Docker/DesktopEnv API，另有新的 Gemini 支持。
- 当前部署明确使用 Colossus 本机 Docker，不需要 `remote_docker`。

决策：Gym 依赖改为干净 upstream `83e8534`。可复现性要求保留精确 SHA；
该 SHA 是当前 main 的普通 upstream commit，不包含私有补丁。OpenCV、
`pynput`、`ag2` 和可选 SurferH submodule 的打包兼容由 Gym agent
`pyproject.toml` 处理，不改 OSWorld。

### 3. 适配边界放在 Gym agent

观察：Gym 已拥有 DesktopEnv 生命周期、Ray worker、policy endpoint、runner
registry、inline evaluator、超时和 rollout/result envelope。内部
`run_multienv_*.py` 重复实现进程池、拆分和结果目录。

决策：OSWorld 只负责 VM、task setup、action execution 和 evaluator；Gym
负责模型 transport、model-specific scaffold、并发、重试、超时、记录和结果
封装。

### 4. Omni Mini 不能直接复用 Qwen3-Omni scaffold

观察：

- Omni Mini 的正式模型是 Nemotron 3 Nano Omni。已有 Colossus probe 和真实
  rollout 证明其 hosted endpoint 输出 `## Action / ## Code`，代码为
  PyAutoGUI 或 `computer.wait/terminate`。
- 该 endpoint 实测每个 prompt 最多接受一张图；使用历史截图时第二步返回
  `At most 1 image(s) may be provided in one prompt.`。
- 它实测会输出 0–1 浮点坐标，例如 `pyautogui.click(0.500, 0.500)`。
- Qwen3VLAgent 要求 `<tool_call>`/`computer_use`，默认保留多张截图历史，
  并使用 Qwen 自己的 action/coordinate parser。

决策：仅复用 Gym model transport，不复用 Qwen agent scaffold。
`omni_mini_agent` 只发送当前截图，历史交互以有界文本加入 system message，
使用 Nemotron parser 并把 0–1 坐标投影到 1920×1080。真正的 Qwen3-Omni
仍使用独立 `qwen3_omni_agent`。

### 5. Colossus 使用 upstream Docker provider

观察：

- upstream main 内置 `docker` provider，负责 qcow2 下载、端口分配、容器
  启停、KVM 探测和 VM readiness。
- NVCF/Singularity 改动还要求 provider factory、DesktopEnv allowlist、特殊
  endpoint/port 构造和外部 function lifecycle。
- 仅把这些 provider class 复制到 Gym 不能形成可运行实现。

决策：当前唯一支持和验收的 provider 是同机 `docker`。NVCF/Singularity
不迁移，也不再把 `remote_docker` 当作依赖。Docker 镜像和 qcow2 在
Colossus worker 预置，避免并发首下载竞争。

### 6. Nemotron/Qwen 行为迁移方式不同

观察：

- Nemotron 模型协议不存在于 upstream OSWorld，需要 adapter-owned scaffold。
- upstream main 已有 Qwen3VLAgent；复制其完整实现会形成易漂移的第二份代码。
- 内部 Qwen 的关键增量是模型 transport、tool-call 检查、重试和相邻
  PyAutoGUI 合并。

决策：Nemotron v3 与 Omni Mini scaffold 由 Gym 持有；Qwen 复用 upstream
class，只在 Gym transport/execution 边界补足行为。

### 7. 无效模型输出不能记成成功

观察：内部 Qwen 对空响应曾返回 `DONE`，会把服务故障伪装成成功。

决策：Omni Mini 和 Qwen 都验证 finish reason 与动作格式；重试耗尽后返回
agent/model error，并令 Gym 设置 `mask_sample=True`，不伪造 `DONE`。

### 8. Setup cache 可完全下沉到 adapter

观察：cache 修改的本质是在 `env.reset()` 读取 setup 文件前，把预置文件放到
OSWorld 的 task cache 路径。

决策：Gym 在 reset 前为 OSWorld、OfficeWorld、SpreadsheetBench 和 PPTC
建立 symlink，不改 `DesktopEnv`。

### 9. evaluator 只做窄兼容

观察：

- CPU-only EasyOCR 和 Speechify 扩展名称是可解释、版本无关的兼容行为。
- Recreation.gov getter 是大段旧版本导航逻辑；VLC 变化是 task JSON setup
  顺序而非 adapter 行为。

决策：前两项放在 Gym runtime；不覆盖 upstream getter，也不在 adapter
隐式改 benchmark data。

### 10. Kimi、脚本和可视化不进入当前链路

观察：Kimi 的有效差异主要是直连 endpoint/reasoning fallback；SLURM/vLLM
脚本带内部 Lustre、container 和 account 路径；可视化脚本消费内部结果目录。

决策：endpoint 统一由 Gym model server 管理。当前 Omni Mini 目标不新增 Kimi
runner；集群启动和结果输出使用 Gym 标准工具。

### 11. WebArena 是下一步独立工作

观察：WebArena/WebVoyager/DOM 自己管理 CDP/Xvfb、tab context、captcha、
reset、judge 和 action tools，不遵循 OSWorld DesktopEnv contract。

决策：不放进 `osworld_agent`。下一阶段建立独立 Gym environment/agent
integration。

### 12. 验证必须区分逻辑覆盖与真实运行

观察：

- adapter 纯逻辑、upstream API contract 和假运行时可在本地验证。
- 2026-07-03 的真实 Omni Mini smoke 已证明 Gym + Docker + 单图协议可运行，
  但当时 agent venv 仍选择了旧的 `4c155eb1`。
- 2026-07-04 已在 Colossus host1 用 `OSWorld/main@83e8534` 完成新的真实
  Docker smoke。Chrome 任务经过 reset、7 步模型交互、PyAutoGUI execution、
  upstream evaluator、MP4 recording 和 container teardown，最终 score 为 1.0。
- 第一次启动在 VM 创建前失败：复用的旧 model-server venv 缺少当前 Gym
  已声明的 `anthropic`。这是 `skip_venv_if_present=true` 跳过依赖同步造成的
  deployment drift，不是 adapter 或 clean OSWorld 的运行失败。改用依赖完整的
  server venv 后，同一源码和同一任务通过。

决策：单元测试通过不等于 clean-main E2E。部署新 checkout 时必须同步 server
venv；只有已做 import preflight 时才允许设置 `skip_venv_if_present=true`。
本次 smoke 已满足 clean-main 核心 E2E 验收，但单个 Chrome task 不代表全部
应用、getter、cache 类型或并发规模都已抽样。

### 13. 本地 vLLM 与 Gym/OSWorld 生命周期分离

观察：Gym 已有 `vllm_model`，能代理外部 OpenAI-compatible Chat Completions、
转发单张截图，并把 vLLM `nemotron_v3` reasoning parser 的独立字段重新包装给
adapter agent。`local_vllm_model` 则把模型进程放进同一 Ray cluster，会让模型随
Gym/OSWorld 重启，且把 B200 和 Docker/KVM 主机耦合。Prenyx 按整节点 feature
调度 B200，不提供单卡 GRES；个人 home 只有 50 GB，BF16 模型与容器必须放到
`/lustre/fsw/general_sa/jepeng`。

决策：Prenyx 八卡 B200 节点以 TP8 运行 vLLM 0.20.0 和固定 revision 的官方 BF16
checkpoint；Colossus 继续运行 Gym + clean OSWorld Docker/KVM。Gym 使用独立的
external-vLLM config，并对齐 `internal-osworld-adapter-nano-omni` public-BF16 TP8 实验的三图
history、4096 max tokens、64000 context、32 max sequences、internal prompt、完整
Thought+Action+Code 文本历史、100 steps 和 5 秒等待。首次连接必须依次通过 `/models`、单图
Chat Completions、1-task VM smoke，之后才启动 361-task no-GDrive run。若 Colossus 不能直接访问
Prenyx compute-node 8000 端口，则建立受控 SSH tunnel，不把服务暴露到公网。

## 2026-07-04 clean-main 实机 smoke 记录

本次 smoke **只验证 Omni Mini**：唯一启动的 runner 是
`omni_mini_agent`，唯一调用的模型是
`nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`。没有启动或测试
`qwen3_omni_agent`、`nemotron_v3_agent` 或其他模型/agent。下面提到的
agent venv 只是 `osworld_agent` 服务的 Python 依赖环境，不代表切换了 agent。

| 项目 | 实际值 |
| --- | --- |
| Colossus worker | `dl325-0200.ipp2a2.colossus.nvidia.com`（host1） |
| Gym checkout | `omni-mini-adapter` worktree，基于 `origin/feature/osworld@6a1cf08b` |
| OSWorld checkout | clean `xlang-ai/OSWorld main@83e8534451ba8b3ab6477448ef3f0a8e563f05be` |
| provider / acceleration | upstream `docker` / KVM |
| model | `nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| runner | `omni_mini_agent`，单张当前 screenshot + 有界文本历史 |
| task | Chrome `bb5e4c0d-f964-439c-97b6-bdb9747de3f4`，将 Bing 设为默认搜索引擎 |
| result | `reward=1.0`, `score=1.0`, `mask_sample=false`, `finished=true`, `error=null` |
| trajectory | 6 个 GUI action + 第 7 步显式 `computer.terminate(status='success')` |
| recording | H.264, 1920×1080, 30 fps, 34.57 s, 1,104,659 bytes |
| teardown | 完成后 Docker containers=0，Gym/Ray/QEMU/collector 相关进程=0 |

这条轨迹同时实测了模型协议转换：例如模型的
`pyautogui.click(0.988,0.083)` 被执行为 `pyautogui.click(1897, 90)`；其余五个
0–1 坐标也都投影到 1920×1080。日志中没有多图拒绝、traceback、HTTP 4xx/5xx
或 parser error。OSWorld evaluator 重启 Chrome 后读到 `Microsoft Bing`，返回
1.0，而不是仅依赖 agent 自报成功。

本地保留的 smoke artifacts 位于
`smoke-artifacts/2026-07-04-clean-main-omni-mini/`。关键 SHA-256：

- `rollouts.jsonl`: `2a77e25e5d90e89dfec42bd11ff0d1756104796bff168bf7458bf4eb15c45de6`
- `driver.log`: `d97e7424ce84df10772039e1f6312d99a70c6e8de2e613160f822d3c959d10ec`
- rollout MP4: `4d228c14b9ba20399d4108e2b6d1966921b177217cedc375f35725aa89b554bb`

## 2026-07-05 Omni Mini 369-task 全量运行

本次只运行 `omni_mini_agent`。`test_all.jsonl` 从同一 clean upstream
`OSWorld/main@83e8534` 生成，包含 369 行和 369 个唯一 task；远端输入
SHA-256 为 `3be0746d59fc5a311447e3e36241381c022932c624315b851fef2c28ecbf5ac9`。
运行在同一 Colossus host1 上以 4 个 upstream Docker/KVM VM 并发执行，并启用
`resume_from_cache=true` 和固定视频抽样 seed。

启动阶段即确认一个不属于 adapter 的 upstream 外部依赖：`test_all` 中 8 个
Google Drive multi-app task 使用 `_googledrive_setup`，需要未随 OSWorld 仓库
分发的真实 `evaluation_examples/settings/googledrive/client_secrets.json`。host1
没有该凭据，因此这些 task 在 reset 阶段被 Gym 正确标成
`mask_sample=true`，不会混入模型失败分数。这 8 条恰好是 upstream
`test_all`（369）与 `test_nogdrive`（361）的差集；其余 361 条继续运行。

决策：保持用户要求的 369 行全量输入，不偷偷改成 361；在最终汇总中分别报告
有效 rollout、Google Drive credential-masked rollout 和其他失败。长跑使用同一
output/materialized-input 路径断点续跑，避免重做已完成 task。

最终结果：2:24:08 内完成 369/369，361 条有效、8 条 Google Drive credential
masked、failure sidecar 为 0。有效样本中 71 条 reward=1、290 条 reward=0，
有效样本的原始 OSWorld evaluator 平均分为 0.2019412；Gym adapter 把小于 1 的
部分分二值化后，有效 training reward 平均为 0.1966759，aggregate 再将 masked
行按 0 纳入后为 0.1924119。
16 个抽样 MP4 均已生成，结束后 Docker/QEMU/Gym 进程清零。

全量运行还暴露了 adapter/model contract 的主要后续项：日志记录了 982 次
`missing an Action or Code section` 可重试解析失败，其中 115 个 step 五次重试
全部耗尽并返回 `FAIL`。单图约束完全生效（零次多图拒绝），也没有 HTTP
4xx/5xx 或 server-exit；因此当前最值得改进的是 Omni Mini 输出格式遵循率，
而不是 OSWorld provider 或图片传输链路。

## 功能覆盖矩阵

| 功能 | clean upstream + Gym adapter | 结论 |
| --- | --- | --- |
| Docker VM 创建、端口、KVM、readiness、关闭 | upstream OSWorld | 覆盖 |
| task reset/setup 与 PyAutoGUI execution | upstream OSWorld | 覆盖 |
| Omni Mini prompt、单图限制、文本历史、reasoning/action parser、0–1 坐标 | Gym `omni_mini_agent` | 覆盖 |
| Nemotron v3 internal scaffold | Gym `nemotron_v3_agent` | 不属于本次 Omni Mini smoke |
| Qwen3-Omni prompt/parser | upstream Qwen3VLAgent + Gym transport/retry | 不属于本次 Omni Mini smoke |
| Ray 并发、step/task timeout、错误 mask、result envelope | Gym | 覆盖 |
| setup cache 预置 | Gym reset 前 staging | 覆盖 |
| EasyOCR CPU isolation、Speechify alias | Gym evaluator wrapper | 覆盖 |
| evaluator 与 benchmark task data | clean upstream OSWorld | 覆盖；不覆盖旧 getter/data patch |
| rollout MP4 | upstream controller + Gym lifecycle | 已在 clean-main Docker smoke 实测 |
| NVCF/Singularity/Apptainer/remote Docker | 不需要 | 明确不在当前目标 |
| WebArena/WebVoyager/DOM | 下一阶段独立 integration | 本轮不覆盖 |
| 内部 SLURM/vLLM/可视化脚本 | Gym 工具替代 | 不复制 |

## Remaining validation boundary

clean-main + Gym adapter 的核心运行闭环已经完成实机确认。现在可以说：这个组合
不依赖特殊 OSWorld fork，能够真实启动 upstream Docker VM、驱动 Omni Mini 完成
多步任务，并由 upstream evaluator 给分和生成录像。

仍不能把一条成功 smoke 外推为“全部 benchmark 功能已回归”。若要发布全量运行
结论，还应抽样 Chrome 之外的 LibreOffice、GIMP、VS Code、VLC、Thunderbird、
multi-app 和 OS tasks，至少覆盖一次 setup-cache staging、OCR/Speechify evaluator
兼容以及 `concurrency>1`。这里的后续验证仍指 Omni Mini，不包含其他模型或
runner。NVCF/Singularity/remote Docker 和 WebArena 不属于这次已约定的目标范围。
