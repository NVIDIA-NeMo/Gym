# VisualWebArena validation — 2026-11-12

The full maintained population completed with **231.0/908 raw reward = 25.44052863436123%** and **231/908 native binary full successes = 25.44052863436123%**. The complete original policy-request/response/native-turn/image audit passed. **Manual original-frame visual review is pending**; the final review queue contains27 episodes and54 original first/last frames, including both the original evaluator-inconclusive task436 episode and its conclusive recovery. This is the technically audited result, with visual admission still open.

| Setting | Actual value |
| --- | --- |
| Tested Gym commit | f8031c55e638491f4d0b39890714d553ed1fc09e |
| Run record | 2026-11-10-visualwebarena-pr2918-full-908-r1 |
| PR head when this report was prepared | a0fbc21ca936911c42b8893851b097fa30911884 |
| Measured source scope | The score above belongs to tested commit f8031c55e638491f4d0b39890714d553ed1fc09e; later PR-head changes are outside this measured run. |
| Historical measured commit | 301c577eb61457057d5cd34807073a4f4506b8fe |
| Historical raw and binary results | 219/908 = 24.118942731277532% for both definitions |
| Current minus historical | +12 raw reward / +12 binary successes / +1.3215859030837005 percentage points |
| All908 paired transitions | 78 improvements;66 regressions;764 unchanged; contingency00=611,01=78,10=66,11=153 |
| All raw-delta bins | {"-1.0":66,"0.0":764,"1.0":78} |

## Model, tokenizer, template and parsers

| Setting | Actual value |
| --- | --- |
| Checkpoint | Nano Omni tuned iter_0004622; frozen17-shard checkpoint |
| Architecture | ["NemotronH_Nano_VL_V2"] |
| Weight dtype and actual engine dtype | BF16 / torch.bfloat16 |
| Quantization | None |
| Weight bytes | 64783882136 |
| Complete weight manifest SHA256 | 3c311dc440e4d0b378086e75d395d9a4414c641505ee04bca6b9aef67885d02f |
| Checkpoint config SHA256 | 770d1d5b0fc87b6df693702fb3eaaaa599664041f3986c1d98a1d158710be8b0 |
| Served model alias | vllm_local; full actual serve argv below |
| Tokenizer | Frozen reference nano-tokenizer, mounted at /opt/public-v3-tokenizer |
| Tokenizer implementation | PreTrainedTokenizerFast |
| Tokenizer model_max_length metadata | 262144 |
| Effective server context / max_model_len | 64000 |
| Tokenizer flags | add_bos_token=false;add_eos_token=false;add_prefix_space=false;clean_up_tokenization_spaces=false |
| Tokenizer special tokens (beginning / end / unknown) | <s> / <\|im_end\|> / <unk>;image token ID18 |
| Actual chat-template file | nemotron_v3.chat_template.keep_history.jinja |
| Actual template selection | Explicit --chat-template /opt/policy-template/nemotron_v3.chat_template.keep_history.jinja; overrides the tokenizer-bundled chat_template.jinja |
| Actual request chat_template_kwargs | {"truncate_history_thinking":false} |
| History policy | Preserve thinking/text history; agent keeps3 recent screenshots plus original task images; no automatic text-history compaction |
| Tool-call parser | qwen3_coder; vllm/tool_parsers/qwen3coder_tool_parser.py |
| Reasoning parser | nano_v3 |
| Reasoning plugin | --reasoning-parser-plugin /opt/nemotron-v3-assets/nano_v3_reasoning_parser.py |
| Transport reasoning flags | {"uses_reasoning_parser":true,"uses_interleaved_reasoning":true} |
| Tool selection and batching | enable_auto_tool_choice=true;tool_choice=auto;parallel_tool_calls=true |
| Structured outputs | xgrammar backend;enable_in_reasoning=false |
| Integrated multimodal processor | nano_nemotron_vl.py from source-integrated-a4d716976-mixed-resolution-r2; exact SHA256 below |

The complete protocol asset identities are listed individually. The tokenizer-bundled template and explicitly selected policy template have separate byte counts and hashes.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| VLLM_SOURCE_OVERLAY/nano_nemotron_vl.py | 90066 | cd139df785ef89a136af18e5e6a24e776f5b62cee364dc7e7c2a3f6a0da0e37a |
| VLLM_SOURCE_OVERLAY/qwen3coder_tool_parser.py | 32732 | bc42cfa4d25e4658fdb00b32ace01541d98e0534accb41cd572a492de2c6fd0f |
| VLLM_RUNTIME_SHIM/README.md | 1009 | 7da0159d2c05abad050e6b7aa978dd30e20d42e8740f09f6678bd83dc0723568 |
| VLLM_RUNTIME_SHIM/SHIM_MANIFEST.sha256 | 159 | dbcde8a99429c2fc09fff642340da8d7a1afd7eb078222835bc3087283124892 |
| VLLM_RUNTIME_SHIM/SHIM_READY.env | 180 | 646f42a79b0292926c92ab19290aed9b08346b81dce85edec4bb23cb8153cb80 |
| VLLM_RUNTIME_SHIM/sitecustomize.py | 1818 | ad762e19e098ab4ce930534a3f9b0ef5b45387d3ed86ad303e62425febfafb89 |
| PUBLIC_V3_TOKENIZER_RUNTIME/chat_template.jinja | 10503 | 43a92aacc2e169bb37055e4d8b0df31edfdc8657d8c387fcf98254cb246e626a |
| PUBLIC_V3_TOKENIZER_RUNTIME/special_tokens_map.json | 420 | e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53 |
| PUBLIC_V3_TOKENIZER_RUNTIME/tokenizer.json | 17078217 | e422eeb52ad6b4974e3a5b097087005a42a1dbe80a5608f70bfb6ee82a68504d |
| PUBLIC_V3_TOKENIZER_RUNTIME/tokenizer_config.json | 187994 | a4a4d85e69d46ee6c1cd387951e15416b996a6e3e57874e625eea6d6945b37ae |
| POLICY_CHAT_TEMPLATE/nemotron_v3.chat_template.keep_history.jinja | 10505 | 1df972dd4d3240f8fadab10a815e7b4b678664565fb12623289140976a3c0e53 |
| SERVING_ASSETS/nano_v3_reasoning_parser.py | 799 | 7cac9c39900d87fc4b19a0f99f26a2422712c6f67733806d9e67f0a33081e5a1 |
| SERVING_ASSETS/serve_wrapper.py | 515 | 9132e1e980d69ac39397c1617fc685b4bce440a5af07b3b166fe57861c419a38 |
| SERVING_ASSETS/serve_wrapper_codex_local.py | 604 | d49f1ea6ab6c432a96aaffec276ba5df5ddf11bd9ac185b75703c07aaecf9c1d |

<details>
<summary>Complete checkpoint model config</summary>

```json
{
  "architectures": [
    "NemotronH_Nano_VL_V2"
  ],
  "auto_map": {
    "AutoConfig": "configuration.NemotronH_Nano_VL_V2_Config",
    "AutoModel": "modeling.NemotronH_Nano_VL_V2",
    "AutoModelForCausalLM": "modeling.NemotronH_Nano_VL_V2"
  },
  "max_sequence_length": 131072,
  "downsample_ratio": 0.5,
  "force_image_size": 512,
  "patch_size": 16,
  "use_thumbnail": true,
  "eos_token_id": 11,
  "model_type": "NemotronH_Nano_VL_V2",
  "ps_version": "v2",
  "template": "n5h_5p5_nanov2",
  "torch_dtype": "bfloat16",
  "image_tag_type": "internvl",
  "img_context_token_id": 18,
  "video_context_token_id": 131081,
  "img_context_token": "<image>",
  "video_context_token": "<video>",
  "img_start_token": "<img>",
  "img_end_token": "</img>",
  "vit_hidden_size": 1280,
  "projector_hidden_size": 20480,
  "norm_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "norm_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "video_pruning_rate": 0.7,
  "sound_context_token_id": 27,
  "sound_context_token": "<so_embedding>",
  "sound_config": {
    "model_type": "parakeet",
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "intermediate_size": 4096,
    "conv_kernel_size": 9,
    "convolution_bias": false,
    "subsampling_conv_channels": 256,
    "subsampling_conv_kernel_size": 3,
    "subsampling_conv_stride": 2,
    "subsampling_factor": 8,
    "num_mel_bins": 128,
    "projection_hidden_size": 4096,
    "projection_bias": false,
    "sampling_rate": 16000
  },
  "llm_config": {
    "architectures": [
      "NemotronHForCausalLM"
    ],
    "auto_map": {
      "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
      "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
    },
    "model_type": "nemotron_h",
    "bos_token_id": 1,
    "chunk_size": 128,
    "conv_kernel": 4,
    "expand": 2,
    "eos_token_id": 11,
    "pad_token_id": 0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.55.4",
    "attention_bias": false,
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_dropout": 0.0,
    "hidden_size": 2688,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "initializer_range": 0.02,
    "intermediate_size": 1856,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 64,
    "mamba_proj_bias": false,
    "max_position_embeddings": 262144,
    "mlp_bias": false,
    "mlp_hidden_act": "relu2",
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "n_group": 1,
    "n_groups": 8,
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "norm_eps": 1e-05,
    "norm_topk_prob": true,
    "num_attention_heads": 32,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 52,
    "num_key_value_heads": 2,
    "num_logits_to_keep": 1,
    "partial_rotary_factor": 1.0,
    "rescale_prenorm_residual": true,
    "residual_in_fp32": false,
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "sliding_window": null,
    "ssm_state_size": 128,
    "tie_word_embeddings": false,
    "time_step_floor": 0.0001,
    "time_step_limit": [
      0.0,
      1e+30
    ],
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "topk_group": 1,
    "use_bias": false,
    "use_cache": true,
    "use_conv_bias": true,
    "use_mamba_kernels": true,
    "vocab_size": 131072
  },
  "vision_config": {
    "auto_map": {
      "AutoConfig": "nvidia/C-RADIOv2-H--hf_model.RADIOConfig",
      "AutoModel": "nvidia/C-RADIOv2-H--hf_model.RADIOModel"
    },
    "adaptor_configs": {},
    "adaptor_names": null,
    "architectures": [
      "RADIOModel"
    ],
    "args": {
      "aa": null,
      "amp": true,
      "amp_dtype": "bfloat16",
      "amp_impl": "native",
      "aug_repeats": 0,
      "aug_splits": 0,
      "bn_eps": null,
      "bn_momentum": null,
      "cache_dir": null,
      "channels_last": false,
      "checkpoint_hist": 10,
      "chk_keep_forever": 100,
      "class_map": "",
      "clip_grad": null,
      "clip_mode": "norm",
      "cls_token_per_teacher": true,
      "coco_annotations_file": "/datasets/coco2017-adlsa/annotations/captions_val2017.json",
      "coco_image_dir": "/datasets/coco2017-adlsa/val2017",
      "color_jitter": 0.4,
      "cooldown_epochs": 0,
      "cpe_max_size": 2048,
      "crd_loss": false,
      "crd_loss_weight": 0.8,
      "crop_pct": null,
      "cutmix": 0.0,
      "cutmix_minmax": null,
      "dataset_download": false,
      "debug_full_knn": false,
      "decay_epochs": 90,
      "decay_milestones": [
        90,
        180,
        270
      ],
      "decay_rate": 0.1,
      "depchain": true,
      "dist_bn": "reduce",
      "dist_norm_weight": 0.0,
      "distributed": true,
      "drop": 0.0,
      "drop_block": null,
      "drop_connect": null,
      "drop_path": null,
      "dtype": "bfloat16",
      "epoch_repeats": 0.0,
      "eval": false,
      "eval_metric": "knn_top1",
      "eval_teacher": false,
      "eval_teacher_only": false,
      "eval_throughput": false,
      "fast_norm": false,
      "fd_loss_fn": "MSE",
      "feature_normalization": "SHIP_NORM",
      "feature_summarizer": "cls_token",
      "feature_upscale_factor": null,
      "force_new_wandb_id": false,
      "force_spectral_reparam": true,
      "freeze_bn": false,
      "fsdp": false,
      "fuser": "",
      "gp": null,
      "grad_accum_steps": 1,
      "grad_checkpointing": false,
      "head_init_bias": null,
      "head_init_scale": null,
      "head_warmup": 5,
      "head_weight_decay": 0.001,
      "hflip": 0.5,
      "img_size": null,
      "in_chans": null,
      "initial_checkpoint": null,
      "input_size": null,
      "interpolation": "",
      "layer_decay": null,
      "local_rank": 0,
      "log_interval": 50,
      "log_mlflow": false,
      "log_wandb": true,
      "loss_auto_balance": false,
      "lr_base": 0.1,
      "lr_base_scale": "",
      "lr_base_size": 256,
      "lr_cycle_decay": 0.5,
      "lr_cycle_limit": 1,
      "lr_cycle_mul": 1.0,
      "lr_k_decay": 1.0,
      "lr_noise": null,
      "lr_noise_pct": 0.67,
      "lr_noise_std": 1.0,
      "mean": null,
      "mesa": false,
      "min_lr": 0,
      "mixup": 0.0,
      "mixup_mode": "batch",
      "mixup_off_epoch": 0,
      "mixup_prob": 1.0,
      "mixup_switch_prob": 0.5,
      "mlp_hidden_size": 1520,
      "mlp_num_inner": 3,
      "mlp_version": "v2",
      "model": "vit_huge_patch16_224",
      "model_kwargs": {},
      "model_norm": false,
      "momentum": 0.9,
      "no_aug": false,
      "no_ddp_bb": true,
      "no_prefetcher": false,
      "no_resume_opt": false,
      "num_classes": null,
      "opt_betas": null,
      "opt_eps": null,
      "patience_epochs": 10,
      "pin_mem": false,
      "prefetcher": true,
      "pretrained": false,
      "rank": 0,
      "ratio": [
        0.75,
        1.3333333333333333
      ],
      "recount": 1,
      "recovery_interval": 0,
      "register_multiple": 10,
      "remode": "pixel",
      "reprob": 0.0,
      "reset_loss_state": false,
      "resplit": false,
      "save_images": false,
      "scale": [
        0.5,
        1.0
      ],
      "sched": "cosine",
      "seed": 42,
      "smoothing": 0.1,
      "spectral_heads": false,
      "spectral_reparam": false,
      "split_bn": false,
      "start_epoch": null,
      "std": null,
      "stream_teachers": true,
      "sync_bn": false,
      "synchronize_step": false,
      "teachers": [
        {
          "fd_normalize": false,
          "feature_distillation": true,
          "input_size": 378,
          "model": "ViT-H-14-378-quickgelu",
          "name": "clip",
          "pretrained": "dfn5b",
          "type": "open_clip",
          "use_summary": true
        },
        {
          "fd_normalize": false,
          "feature_distillation": true,
          "input_size": 378,
          "model": "ViT-SO400M-14-SigLIP-384",
          "name": "siglip",
          "pretrained": "webli",
          "type": "open_clip",
          "use_summary": true
        },
        {
          "fd_normalize": false,
          "feature_distillation": true,
          "input_size": 378,
          "model": "dinov2_vitg14_reg",
          "name": "dino_v2",
          "type": "dino_v2",
          "use_summary": true
        },
        {
          "fd_normalize": false,
          "feature_distillation": true,
          "input_size": 1024,
          "model": "vit-h",
          "name": "sam",
          "type": "sam",
          "use_summary": false
        }
      ],
      "torchcompile": null,
      "torchscript": false,
      "train_interpolation": "random",
      "train_split": "train",
      "tta": 0,
      "use_coco": false,
      "use_multi_epochs_loader": false,
      "val_ema_only": false,
      "val_split": "val",
      "vflip": 0.0,
      "vitdet_version": 1,
      "wandb_entity": "",
      "wandb_job_type": "",
      "wandb_name": "",
      "wandb_project": "",
      "warmup_lr": 1e-05,
      "warmup_prefix": false,
      "worker_seeding": "all",
      "workers": 8,
      "world_size": 256,
      "min_num_patches": 1024,
      "max_num_patches": 13312
    },
    "feature_normalizer_config": null,
    "inter_feature_normalizer_config": null,
    "max_resolution": 2048,
    "patch_size": 16,
    "preferred_resolution": [
      768,
      768
    ],
    "torch_dtype": "bfloat16",
    "version": "radio_v2.5-h",
    "vitdet_window_size": null
  }
}
```

</details>

## Serving and inference environment

| Setting | Actual value |
| --- | --- |
| Topology | 1 effective replica;8×NVIDIA H10080GB HBM3;TP8;DP1;PP1 |
| vLLM source | a4d716976; source-integrated-a4d716976-mixed-resolution-r2 |
| Installed vLLM distribution | 0.1.dev1+ga4d716976.d20260112.precompiled |
| Actual V1 engine version | 0.1.dev1+ga4d716976 |
| PyTorch runtime reported by CUDA admission | 2.9.0+cu128 |
| CUDA runtime reported by admission | 12.8 |
| Transformers | 4.57.3 |
| tokenizers | 0.22.2 |
| Triton | 3.5.0 |
| xgrammar | 0.1.27 |
| Inference Python | /usr/bin/python3.12; image patchlevel.h reports3.12.3 |
| Host NVIDIA kernel module | 535.216.03 |
| Architecture | x86_64 |
| Serving image | pytorch25.06-moe-avlm-eval-1217-vllm-gpu.sqsh |
| Serving image bytes | 45476511744 |
| Serving image SHA256 | 7dac78869a1f5f328a79654db75b86a569199745833346c86781fe5d33e52539 |
| Frozen static serving assets | 3682 files; exact original file identities reverified per admitted generation |
| GPU memory utilization | 0.9 |
| max_num_seqs | 32 |
| Swap space | 8 |
| KV cache dtype | auto |
| Mamba SSM cache dtype | float32 |
| enforce_eager | false |
| Prefix caching | false |
| Chunked prefill | true |
| Engine seed shown by initialization | 0 |
| Request seed | omitted |
| Float32 matmul | VLLM_FLOAT32_MATMUL_PRECISION=ieee;TORCH_ALLOW_TF32_CUBLAS_OVERRIDE unset |
| Runtime shim | disable-broken-deepep-r1; all4 frozen shim file identities listed above |
| Multimodal limit flag | --limit-mm-per-prompt omitted, retaining inherited serving behavior |
| Published generations with verified actual engine initialization | [1,4,5,6,7,8,9,10,11,12,13,14,15] |
| Generation identity admission | Every admitted generation retains17 frozen weight shards,3682 static files and all16 checked service/protocol fields; generation15 is the final published service |

<details>
<summary>Complete actual vLLM serve command, read from the running process</summary>

```sh
/usr/bin/python /usr/local/bin/vllm serve /opt/model --served-model-name vllm_local wa-vwa-20260912-r1-iter0004622 --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --data-parallel-size 1 --max-model-len 64000 --gpu-memory-utilization 0.9 --max-num-seqs 32 --swap-space 8 --allowed-local-media-path / --tokenizer /opt/public-v3-tokenizer --chat-template /opt/policy-template/nemotron_v3.chat_template.keep_history.jinja --reasoning-parser-plugin /opt/nemotron-v3-assets/nano_v3_reasoning_parser.py --reasoning-parser nano_v3 --tool-call-parser qwen3_coder --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --structured-outputs-config.backend=xgrammar --structured-outputs-config.enable_in_reasoning=false
```

</details>

<details>
<summary>Serving environment variables from the executed launcher</summary>

```sh
export PYTHONPATH=/opt/vllm-runtime-shim:/opt/vllm-source
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WEBVOYAGE_DISABLE_BROKEN_DEEPEP=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_FLOAT32_MATMUL_PRECISION=ieee
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
```

</details>

## Request sampling, agent and browser environment

| Setting | Actual value |
| --- | --- |
| Temperature | 0.1 |
| top_p | 0.95 |
| Request-level output-token cap | omitted: resolved max_output_tokens=null is not serialized as an additional policy cap |
| Request top_k / min_p / repetition or presence/frequency penalties | not explicitly sent |
| Agent protocol | nano_omni_toolcall |
| max_steps | 100 |
| max_image_history | 3 |
| Observation text | none |
| Agent and resource task-image byte limits | 33554432 |
| Task499 original image bytes | 27551037 |
| max_parse_retries | 2 |
| Invalid-tool-call retry/repair | {"nano_omni_retry_invalid_tool_calls":false,"nano_omni_parse_retry_feedback":false,"nano_omni_parse_retry_temperature":null} |
| Missing-tool-call retry delay secs | 1.0 |
| Tool-call limit | null |
| Max computer actions per model turn | 20 |
| Model-turn infrastructure retries / delay | {"max_retries":19,"delay_secs":5.0} |
| Request/run timeouts secs | {"resources_request_timeout_secs":1800,"model_request_timeout_secs":1200,"seed_request_timeout_secs":1800.0,"judge_request_timeout_secs":300.0,"close_request_timeout_secs":30.0,"run_timeout_secs":14400} |
| Transport endpoint continuity | {"endpoint_stale_grace_s":300.0,"endpoint_connection_retries":8,"endpoint_connection_retry_timeout_s":null,"endpoint_check_interval_s":10.0,"route_around_failing_endpoints":false,"endpoint_failure_threshold":3,"endpoint_retry_after_s":60.0} |
| Browser | Headed Chromium;Playwright1.55.0;PyAutoGUI;Xvfb1920×1080×24 |
| Viewport | [1920,1080] |
| Action delay secs | 2.0 |
| terminate_on_action_error | true |
| Browser sessions per worker | 1 |
| Site pool | local_locks |
| Record video | false |
| Browser runtime image tag | wa-vwa/browser-qualified:webarena-f622a316fd0f |
| Browser runtime image ID | sha256:a7fe67e1ab44e2b72c39ff8e001cb1a6c1000c75f013c94cfdf7b27ed710255b |
| Browser runtime build-source SHA256 | f622a316fd0fa4c60e9e1e57664486d10e1136686533d0b5457d652405351887 |
| Application source inside runtime | Fresh independent root and4 role environments from the tested f8031c55e source, mounted read-only; current committed locks |
| Gym/agent Python | CPython3.13.14-linux-x86_64-gnu |
| uv | 0.13.4 |
| Runtime preparation | uv sync --locked --extra dev;gym env prefetch current committed role locks |
| Execution backend | local_process;ray_enabled=false;3 admitted concurrent browser workers;per-site concurrency1 |
| Native judge | us/azure/openai/gpt-4.1 at inference-api.nvidia.com;native evaluator HTTP path |
| Judge generation and timeout | temperature0;top_p1;max_tokens768;HTTP timeout120s |

The four installed role environments are independent. All four complete package inventories and all607 rows from the running inference process are retained in the evidence record. Core runtime versions are shown per role without conflating the environments.

| Package | legacy_agent | web_agent | vllm_model transport | webarena_browser |
| --- | --- | --- | --- | --- |
| nemo-gym | 0.7.0rc0 | 0.7.0rc0 | 0.7.0rc0 | 0.7.0rc0 |
| openai | 2.44.0 | 2.44.0 | 2.44.0 | 2.44.0 |
| httpx | 0.28.1 | 0.28.1 | 0.28.1 | 0.28.1 |
| omegaconf | 2.3.0 | 2.3.1 | 2.3.0 | 2.3.1 |
| pydantic | 2.13.4 | 2.14.0 | 2.13.4 | 2.14.0 |
| pydantic-settings | 2.14.2 | 2.15.0 | 2.14.2 | 2.15.0 |
| pydantic_core | 2.46.4 | 2.50.0 | 2.46.4 | 2.50.0 |
| playwright | absent | absent | absent | 1.55.0 |
| pillow | absent | 12.3.0 | absent | 12.3.0 |
| tiktoken | 0.14.0 | 0.14.0 | 0.14.0 | 0.14.0 |
| uvicorn | 0.37.0 | 0.54.0 | 0.37.0 | 0.54.0 |
| uvloop | 0.21.0 | 0.23.0 | 0.21.0 | 0.23.0 |

## Population, sites, schedule and integrity

| Setting | Actual value |
| --- | --- |
| Population | 908 maintained tasks;runtime IDs0–907;341 distinct original task-image files |
| Prepared input bytes / SHA256 | 1166528 / 504c75655b1afa81d3d2d202470a838ac74383acb1c071604590da6550837a5f |
| Upstream maintained source revision | 6a2977939b157b0ab9de7799bb089c721f1ac115 |
| Scheduling | 480 per-site read tasks;12 cross-site read tasks alone;404 per-site write tasks;12 cross-site write tasks alone;492 read tasks precede416 write tasks |
| Per-site worker/task concurrency | 1 |
| Concurrent independent sites | 3 |
| Writable-site lifecycle | Fresh owned replicas at full-run start;no per-task resets;writable site state is not reused from smoke |
| Collector-order disclosure | Historical301c577 used tqdm.as_completed(map(_post_subroutine,examples)), which did not guarantee coroutine start order. Currentf8031c55 uses _EagerCompletionIterator with asyncio.ensure_future creation in input order. Historical result permutation was not replayed. All908 order-comparison rows are retained. |
| Native selection | First conclusive valid result per task, including zero reward;all attempts retained |
| Recovery | One original task436 evaluator read timeout was preserved as inconclusive;after all four phases an append-only native recovery produced valid0. All907 previously selected valid lines remain byte-identical. |
| Selected validity | 908 unique;no missing/unexpected/duplicate-valid IDs;no selected masks,evaluator errors or cleanup failures |
| Original archive seal | 42385 files;manifest SHA2569557dfe3842124e87b1b4c001c4a88e4de30c7099728cedf509ed55a0351add9 |
| Full wire audit | 40774 requests =40762 responses +12 retained paired terminal error events;every request has a terminal event |
| Native coverage | All908 selected native model-turn populations plus1 retained evaluator-error original are covered;all9 instrumented process counters contiguous from1 |
| Image/protocol verification | Every policy input image byte and frozen prompt/tool/history/sampling contract audited |
| Technical audit SHA256 | ec8101dfebe08737e34f201e1adbde170d1a21341fa420523c921d06fd7e1fdb |
| Visual queue integrity | 27 episodes/54 original frames/21498835 bytes;all frame bytes independently rehashed;16 frame observations reviewed (14 first,2 last);2 cases complete with valid native zeros corroborated;38 frame observations still pending |

| Site image | Exact Docker image ID |
| --- | --- |
| shopping | sha256:ccff8c1772be884313edad94136d2a4048020300a0fc169781c50a02aa8bd206 |
| reddit | sha256:0a0c002b4dd089db8a22064886cb31f89d5ac493d90f6c41074639d56c6f4fbe |
| classifieds | sha256:a2a794da92f62a8d7ffd02314e4fab40ba6c7fc08f568371608f88f0ef605e43 |
| classifieds_db | sha256:f61944ff3f2961363a4d22913b2ac581523273679d7e14dd26e8db8c9f571a7e |

All22 group rows are below. Full task membership, all original columns and every distribution bin remain in `baseline-comparison-all-groups.json`;all908 full pairs remain in `baseline-comparison-all-908-rows.json`.

| Dimension | Label | Count | Historical raw | Current raw | Historical binary | Current binary | Every raw-delta bin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| eval_types | page_image_query | 17 | 5.0 | 6.0 | 5 | 6 | {"0.0":14,"-1.0":1,"1.0":2} |
| eval_types | program_html | 383 | 72.0 | 87.0 | 72 | 87 | {"0.0":336,"1.0":31,"-1.0":16} |
| eval_types | program_html+page_image_query | 16 | 2.0 | 2.0 | 2 | 2 | {"1.0":1,"-1.0":1,"0.0":14} |
| eval_types | program_html+page_image_query+url_match | 1 | 0.0 | 0.0 | 0 | 0 | {"0.0":1} |
| eval_types | string_match | 265 | 90.0 | 94.0 | 90 | 94 | {"-1.0":24,"0.0":213,"1.0":28} |
| eval_types | string_match+program_html | 3 | 0.0 | 0.0 | 0 | 0 | {"0.0":3} |
| eval_types | string_match+url_match | 3 | 1.0 | 0.0 | 1 | 0 | {"0.0":2,"-1.0":1} |
| eval_types | url_match | 204 | 48.0 | 41.0 | 48 | 41 | {"0.0":165,"1.0":16,"-1.0":23} |
| eval_types | url_match+page_image_query | 1 | 0.0 | 0.0 | 0 | 0 | {"0.0":1} |
| eval_types | url_match+program_html | 5 | 1.0 | 1.0 | 1 | 1 | {"0.0":5} |
| eval_types | url_match+program_html+page_image_query | 1 | 0.0 | 0.0 | 0 | 0 | {"0.0":1} |
| eval_types | url_match+string_match | 9 | 0.0 | 0.0 | 0 | 0 | {"0.0":9} |
| phase | non_state_change | 492 | 144.0 | 140.0 | 144 | 140 | {"0.0":398,"-1.0":49,"1.0":45} |
| phase | state_change | 416 | 75.0 | 91.0 | 75 | 91 | {"0.0":366,"1.0":33,"-1.0":17} |
| site_group | classifieds | 221 | 57.0 | 56.0 | 57 | 56 | {"0.0":178,"1.0":21,"-1.0":22} |
| site_group | cross-site | 24 | 3.0 | 3.0 | 3 | 3 | {"0.0":22,"1.0":1,"-1.0":1} |
| site_group | reddit | 197 | 31.0 | 28.0 | 31 | 28 | {"0.0":172,"1.0":11,"-1.0":14} |
| site_group | shopping | 466 | 128.0 | 144.0 | 128 | 144 | {"0.0":392,"-1.0":29,"1.0":45} |
| stage | 1 | 480 | 142.0 | 138.0 | 142 | 138 | {"0.0":388,"-1.0":48,"1.0":44} |
| stage | 2 | 12 | 2.0 | 2.0 | 2 | 2 | {"1.0":1,"0.0":10,"-1.0":1} |
| stage | 3 | 404 | 74.0 | 90.0 | 74 | 90 | {"0.0":354,"1.0":33,"-1.0":17} |
| stage | 4 | 12 | 1.0 | 1.0 | 1 | 1 | {"0.0":12} |

The complete run record `2026-11-10-visualwebarena-pr2918-full-908-r1` retains full908 selected results and historical pairs,all22 group memberships and distributions,all40774 request observations and their terminal events,original invalid attempts,all native-call coverage rows,all42385 original file hashes,all54 unchanged extracted frames,all9 worker-admitted noncredential configuration sections,the complete checkpoint config,every frozen asset identity,the four complete role package inventories and all607 serving-package metadata rows. Original credential values and screenshots are excluded from this public text. The model/configuration evidence is source-bound and adds no policy or judge attempts.

## Subsequent adapter-fix validation

The VisualWebArena reference wrapper converts evaluator exceptions into zero
plus an error message. The Gym adapter now raises `EvaluatorInfrastructureError`
for that error result so the existing failure protocol masks the sample.
Conclusive zero scores still remain valid. Regression cases cover judge timeout,
disconnection and malformed verdicts through the real native evaluator. The
browser-instance test fixture also waits for its HTTP health response before use.

Local validation: **364 passed, 0 failed, 0 skipped** across the browser component
and the core Web suite. All repository pre-commit hooks passed.

A separate qualification used the latest PR runtime source plus this fix,
OpenAI SDK 2.45.0, the existing Nano Omni iter_0004622 service and two synthetic
tasks outside the maintained population. Both real policy requests completed.
The positive task returned a valid reward of 1. The injected unavailable-judge
task produced a masked, invalid failure record. The worker wrapper exited 1
because it deliberately treats the injected failed row as incomplete. The
qualification checked that expected nonzero exit. The original failed startup
with stale SDK package metadata and the later successful qualification are
retained separately. No maintained task was rerun or changed.

The complete scalar qualification receipt follows; raw logs, results and images
remain in the run record. The 231/908 score above still measures `f8031c55e`.

The inline comments identify public file hashes flagged by the entropy detector.
The unannotated JSON receipt remains in the run record.

```jsonc
{
  "observed_utc": "2026-11-12T04:13:06Z",
  "status": "PASS",
  "checks": {
    "one_success_result": true,
    "one_retained_failure_result": true,
    "positive_id": true,
    "positive_reward_one": true,
    "positive_valid": true,
    "positive_unmasked": true,
    "negative_id": true,
    "negative_invalid": true,
    "negative_masked": true,
    "negative_error_identifies_evaluator_infrastructure": true,
    "pipeline_nonzero_expected_for_injected_failure": true,
    "two_real_model_requests_complete": true
  },
  "successful_model_turns": 1,
  "successful_environment_steps": 1,
  "injected_failure_model_turns": 0,
  "injected_failure_environment_steps": 0,
  "evaluator_source_sha256": "270ca92c5c788edee9b1a84589af4123c3af10fe93da57e5952cf8bfa8080c92", // pragma: allowlist secret - Public SHA-256 artifact identity
  "source_authority_sha256": "eff5ee7339b3a78a3421b154a763b99e9f037ab131ec7c128be2d3b8b50b3b2e", // pragma: allowlist secret - Public SHA-256 artifact identity
  "latest_sdk": "2.45.0",
  "real_policy_unchanged": true,
  "formal_908_results_untouched": true,
  "raw_logs_and_images_remain_on_cluster": true,
  "qualification_inputs_sha256": "79677989e94ba32c17dbf2cc1f01b3e24a5913d5f4def6d413111cdcc374ef04", // pragma: allowlist secret - Public SHA-256 artifact identity
  "successful_raw_result_sha256": "0f817b5f6ee27043638536c30683174c93039f0ea5db71f672b042fe95a0092a", // pragma: allowlist secret - Public SHA-256 artifact identity
  "injected_failure_raw_result_sha256": "112bad6df84565a5ec5981b6acb355294cfab3398280bf3616cf03b867f30358", // pragma: allowlist secret - Public SHA-256 artifact identity
  "task_owned_fault_server_stopped": true,
  "actual_model_request_completions": 2
}
```
