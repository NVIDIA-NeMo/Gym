(training-nemo-rl-grpo-nemotron-3-nano-30b)=

# Nemotron 3 Nano 30B Multi-Node Training

After completing the standard multi-node training tutorial with Nemotron Nano 9B v2, you may want to scale up to larger models like Nemotron 3 Nano 30B. This tutorial walks through the complete setup for distributed training across 32 nodes using Slurm and Ray.

:::{card}

**Goal**: Train Nemotron 3 Nano 30B on 32 nodes using GRPO with proper multi-node Ray cluster coordination.

^^^

**In this section, you will**:

1. Set up the Nemotron 3 Nano 30B training environment
2. Download and prepare the training dataset
3. Configure the launch script for multi-node coordination
4. Submit and monitor the 32-node training job

:::

:::{button-ref} training-nemo-rl-grpo-multi-node-training
:color: secondary
:outline:
:ref-type: ref

← Previous: Multi-Node Training
:::

---

## Prerequisites

Make sure you have:

- ✅ Completed the {doc}`Multi-Node Training <multi-node-training>` tutorial
- ✅ Access to Slurm cluster with enroot/pyxis container support
- ✅ Access to NeMo RL container: `nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano`
- ✅ Understanding of Ray distributed computing framework
- ✅ Sufficient storage space (~110GB for model, data, and cache; checkpoints and logs accumulate with each run)

---

## 1. Initial Setup

### 1.1 Set Workspace Directory

Choose a location with sufficient space (~110GB minimum):

```bash
# Set workspace directory (adjust to your cluster's large storage)
# Examples: /scratch/$USER, /work/$USER, /data/$USER, /lustre/.../users/$USER
WORKSPACE=/path/to/large/storage/$USER

# Verify space available
df -h $WORKSPACE
```

**✅ Success Check**: Directory has at least 200GB available space.

---

### 1.2 Clone the Repository

Clone the Nemotron 3 Nano v3 branch of NeMo RL:

```bash
cd $WORKSPACE
git clone --recurse-submodules -b nano-v3 https://github.com/NVIDIA-NeMo/RL.git RL-nano-v3
cd RL-nano-v3
```

**✅ Success Check**: Repository cloned with nano-v3 branch checked out.

---

---

---

### 1.3 Prepare Container Image

**Option A: Use Registry Path Directly (Recommended for First Run)**

Use the container directly from NVIDIA Container Registry:

```bash
# No preparation needed - will be pulled automatically during job execution
CONTAINER=docker://nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano
```

This is the simplest approach but adds ~5-10 minutes to job startup time for first use.

---

**Option B: Pre-Pull Container (Optional - For Faster Job Startup)**

For faster job startup on subsequent runs, pre-pull and convert to .sqsh format:

**Step 1: Get NGC API Key**

1. Go to https://org.ngc.nvidia.com/setup/api-keys
2. Generate an API key
3. Configure enroot credentials:

```bash
mkdir -p ~/.config/enroot
echo "machine nvcr.io login \$oauthtoken password <YOUR_API_KEY>" >> ~/.config/enroot/.credentials
```

**Step 2: Pull Container Using Sbatch**

Due to head node restrictions, pull the container from a compute node:

Create `pull_container.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=enroot-import
#SBATCH --account=<your_account>
#SBATCH --partition=<partition_name>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=enroot-import-%j.out

# Set workspace directory (adjust to your cluster's large storage)
WORKSPACE=/path/to/large/storage/$USER

ENROOT_CACHE_PATH=$WORKSPACE/.cache/enroot

enroot import -o "$WORKSPACE/nemo-rl.v0.4.0.nemotron_3_nano.sqsh" \
    "docker://nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano"
```

Submit the job:

```bash
sbatch pull_container.sh
```

**Step 3: Use Local Container**

Update your launch script to use the local .sqsh file:

```bash
CONTAINER=$WORKSPACE/nemo-rl.v0.4.0.nemotron_3_nano.sqsh
```

**✅ Success Check**: Container file exists (~15GB) or registry path configured.


### 1.4 Install uv Tool

Install uv (which includes uvx) for downloading HuggingFace models and datasets:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (uv installs to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uvx --version
```

**✅ Success Check**: Command shows uv version number.

---

### 1.5 Download and Process Training Data

Download and process the dataset on a compute node (head nodes have limited memory):

Create `prepare_data.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=prepare-data
#SBATCH --account=<your_account>
#SBATCH --partition=<partition_name>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=prepare-data-%j.out

# Data directory
DATA_DIR=${WORKSPACE}/RL-nano-v3/data/

# Download dataset
uvx --from huggingface-hub hf download nvidia/Nemotron-3-Nano-RL-Training-Blend \
    --repo-type dataset \
    --local-dir ${DATA_DIR}

# Fill in placeholders
chmod +x ${DATA_DIR}/create_nanov3_jsonl.py
${DATA_DIR}/create_nanov3_jsonl.py --input ${DATA_DIR}/train.jsonl --output ${DATA_DIR}/train-full.jsonl

# Split: reserve last 1000 rows for validation
head -n -1000 ${DATA_DIR}/train-full.jsonl > ${DATA_DIR}/train-split.jsonl
tail -n 1000 ${DATA_DIR}/train-full.jsonl > ${DATA_DIR}/val-split.jsonl

# Verify split
wc -l ${DATA_DIR}/train-split.jsonl ${DATA_DIR}/val-split.jsonl
```

Submit the job (exports WORKSPACE and HF_TOKEN to the compute node):

```bash
sbatch --export=WORKSPACE,HF_TOKEN prepare_data.sh
```

:::{note}
**Why use a compute node?** The `create_nanov3_jsonl.py` script is memory-intensive and may fail on head nodes which have resource limits. Running on a compute node ensures sufficient memory.
:::

**✅ Success Check**: Job completes and creates `train-split.jsonl` and `val-split.jsonl`.

---

### 1.6 Download Model

Download the Nemotron 3 Nano 30B model:

```bash
cd $WORKSPACE/RL-nano-v3

uvx --from huggingface-hub hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --repo-type model \
    --local-dir model
```

**✅ Success Check**: Model files downloaded (~59GB total) to `model/` directory.

---

### 1.7 Verify Setup

Confirm all components are in place:

```bash
cd $WORKSPACE/RL-nano-v3

# Check directory structure
ls -lh
# Expected: data/, model/, examples/, nemo_rl/, etc.

# Check data files
ls -lh data/train-split.jsonl data/val-split.jsonl

# Check model size and key files
du -sh model/
# Expected: ~59GB

# Verify essential model files exist
ls model/config.json model/*.safetensors
# Should show config.json and 13 safetensors files
```

**✅ Success Check**: All directories and files present with correct sizes.

---

## 2. Create Launch Script

Create a launcher script that properly handles multi-node Ray coordination:

```bash
cd $WORKSPACE/RL-nano-v3
```

Create `launch_nemotron_training.sh`:

```bash
#!/bin/bash
# Nemotron 3 Nano 30B Multi-Node Training Launcher

# Configuration
HOST_BASE=$WORKSPACE  # Or your preferred base directory
NUM_NODES=32

# Paths
DATA_DIR=${HOST_BASE}/RL-nano-v3/data
MODEL_CHECKPOINT=${HOST_BASE}/RL-nano-v3/model
CONFIG_PATH=${HOST_BASE}/RL-nano-v3/examples/nemo_gym/grpo_nanov3.yaml
LOG_DIR=${HOST_BASE}/RL-nano-v3/logs
CKPT_DIR=${HOST_BASE}/RL-nano-v3/checkpoints
CACHE_DIR=${HOST_BASE}/RL-nano-v3/.cache

# Training command with shared cache directory
TRAINING_CMD="cd ${HOST_BASE}/RL-nano-v3 && \
mkdir -p ${LOG_DIR} ${CKPT_DIR} ${CACHE_DIR} && \
export HF_HOME=${CACHE_DIR}/huggingface && \
export TRANSFORMERS_CACHE=${CACHE_DIR}/huggingface && \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_CHECKPOINT} \
    data.train_jsonl_fpath=${DATA_DIR}/train-split.jsonl \
    data.validation_jsonl_fpath=${DATA_DIR}/val-split.jsonl \
    ++logger.log_dir=${LOG_DIR} \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    ++checkpointing.enabled=True \
    ++checkpointing.checkpoint_dir=${CKPT_DIR} \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=8"

echo "Submitting ${NUM_NODES}-node training job..."
echo "Using BASE_LOG_DIR: ${HOST_BASE}/nemoRL/nemo-rl"

# Submit job
BASE_LOG_DIR=${HOST_BASE}/nemoRL/nemo-rl \
COMMAND="$TRAINING_CMD" \
CONTAINER="docker://nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano" \
MOUNTS="${HOST_BASE}:${HOST_BASE}" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=<your_account> \
    --job-name=nemotron-nano-30b \
    --time=8:00:00 \
    --gres=gpu:8 \
    --chdir=/tmp \
    --output=${LOG_DIR}/slurm-%j.out \
    --error=${LOG_DIR}/slurm-%j.err \
    ${HOST_BASE}/nemoRL/nemo-rl/ray.sub
```

Make it executable:

```bash
chmod +x launch_nemotron_training.sh
```

:::{tip}
**Key Configuration Points:**

- `NUM_NODES=32`: 32 nodes × 8 GPUs = **256 GPUs total**
- `--chdir=/tmp`: Sets a neutral working directory for the job
- `HF_HOME` and `TRANSFORMERS_CACHE`: Set to shared storage so all nodes can access model conversions
- `BASE_LOG_DIR`: Specifies where Ray cluster logs will be written
- `--account`: Replace `<your_account>` with your Slurm account name
- `--time=8:00:00`: Adjust based on your cluster's limits
:::

**✅ Success Check**: Script created and executable.

---

## 3. Submit Training Job

:::{important}
**Run the launch script from a neutral directory like `/tmp`** to ensure consistent container working directory behavior across different cluster configurations.
:::

```bash
# Run from /tmp for best compatibility
cd /tmp
bash $WORKSPACE/RL-nano-v3/launch_nemotron_training.sh
```

**Expected output:**

```
Submitting 32-node training job...
Using BASE_LOG_DIR: /home/user/nemoRL/nemo-rl
Submitted batch job 9453356
```

**✅ Success Check**: Job submitted successfully with job ID returned.

---

## 4. Monitor Job Status

Monitor your submitted job:

```bash
# Check if job is running (replace JOBID with your job number)
squeue --job=JOBID

# Detailed status
squeue --job=JOBID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

**Note:** For job state codes (PD, R, CD, etc.), see [Slurm documentation](https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES).

**✅ Success Check**: Job transitions from `PD` to `R` state.

---

## 5. Monitor Training Progress

### 5.1 Check Ray Cluster Logs

Wait 1-2 minutes for Ray cluster to initialize, then check logs:

```bash
# Set your job ID
JOBID=your_job_id

# Check if Ray head started
ls $WORKSPACE/nemoRL/nemo-rl/${JOBID}-logs/

# Verify Ray head is ready
cat $WORKSPACE/nemoRL/nemo-rl/${JOBID}-logs/STARTED_RAY_HEAD
# File should exist if Ray initialized successfully

# View training execution (most important)
tail -100 $WORKSPACE/nemoRL/nemo-rl/${JOBID}-logs/ray-driver.log

# Follow training progress live
tail -f $WORKSPACE/nemoRL/nemo-rl/${JOBID}-logs/ray-driver.log
```

---

### 5.2 Verify Ray Cluster Formation

Check that all Ray actors are online:

```bash
# For 32-node job (32 nodes × 8 GPUs = 256 actors)
grep "Number of actors online: 256/256" $WORKSPACE/RL-nano-v3/logs/slurm-${JOBID}.out
```

**✅ Success Check**: All actors online (256/256 for 32 nodes).

---

### 5.3 Watch Training Metrics

Monitor rollout collection progress:

```bash
# Watch rollout collection
tail -f $WORKSPACE/nemoRL/nemo-rl/${JOBID}-logs/ray-driver.log | grep "Collecting rollouts"

# Example output:
# Collecting rollouts:  21%|██        | 428/2048 [02:01<05:42, 4.73it/s]
# Collecting rollouts:  25%|██▌       | 512/2048 [03:15<08:12, 3.12it/s]
```

Check TensorBoard logs:

```bash
# List experiment directories
ls -ltr $WORKSPACE/RL-nano-v3/logs/

# Check TensorBoard events
find $WORKSPACE/RL-nano-v3/logs/exp_*/tensorboard/ -name "*.tfevents.*" -mmin -5
```

**✅ Success Check**: Rollout percentage increasing steadily, TensorBoard events being written.

---

## 6. Troubleshooting

### Issue: Job Stays in Pending (PD) State

**Check the reason:**

```bash
squeue --job=JOBID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

**Common reasons:**
- `(Priority)`: Waiting in queue for resources
- `(Resources)`: Not enough nodes available
- `(QOSMaxNodePerUserLimit)`: Exceeds node limit

**Solution:** Wait for resources, or adjust job parameters.

---

### Issue: Ray Head Doesn't Start

**Symptom:** No `STARTED_RAY_HEAD` file in logs directory.

**Check Ray head log:**

```bash
cat $WORKSPACE/nemoRL/nemo-rl/JOBID-logs/ray-head.log
```

**Solution:** Check logs for errors related to container startup or resource allocation.

---

### Issue: Training Crashes with Cache Errors

**Symptom:** `FileNotFoundError` mentioning `run_config.yaml` in ray-driver.log.

**Check logs:**

```bash
grep "FileNotFoundError.*run_config.yaml" $WORKSPACE/nemoRL/nemo-rl/JOBID-logs/ray-driver.log
```

**Root cause:** Model conversion saved to local node cache, inaccessible to other nodes.

**Solution:** Verify shared cache directories are set in `TRAINING_CMD`:

```bash
export HF_HOME=${CACHE_DIR}/huggingface
export TRANSFORMERS_CACHE=${CACHE_DIR}/huggingface
```

---

## 7. Key Technical Details

### Why Ray.sub?

Without `ray.sub`, each node would start its own independent Ray cluster. The `ray.sub` script from NeMo RL:

1. Starts a Ray head on the first node
2. Connects all worker nodes to that head
3. Creates a unified distributed cluster
4. Manages placement groups for GPU actors

### Why Shared Cache?

HuggingFace Transformers converts models to Megatron format on first use:
- Without shared cache: Each node converts independently → race conditions
- With shared cache: Rank 0 converts once, all nodes share the result

---

## 8. File Structure Reference

After setup, your directory structure should look like:

```
$HOME/
├── RL-nano-v3/                      # Project root
│   ├── data/
│   │   ├── train-split.jsonl       # Training data
│   │   └── val-split.jsonl         # Validation data
│   ├── model/                       # Nemotron 3 Nano 30B model (~63GB)
│   ├── examples/nemo_gym/
│   │   ├── grpo_nanov3.yaml        # Training config
│   │   └── run_grpo_nemo_gym.py    # Training script
│   ├── logs/                        # Training outputs
│   │   ├── exp_*/                  # Experiment directories
│   │   ├── slurm-*.out            # Slurm stdout
│   │   └── slurm-*.err            # Slurm stderr
│   ├── checkpoints/                # Saved model checkpoints
│   ├── .cache/                     # Shared HuggingFace cache
│   └── launch_nemotron_training.sh # Launch script
└── nemoRL/nemo-rl/
    ├── ray.sub                      # Ray orchestration script
    └── JOBID-logs/                  # Ray cluster logs
        ├── STARTED_RAY_HEAD        # Ray ready sentinel
        ├── ray-head.log            # Head node log
        ├── ray-worker-*.log        # Worker node logs
        └── ray-driver.log          # Training execution log
```

---

## Next Steps

Congratulations! You've successfully set up and launched Nemotron 3 Nano 30B training on a 32-node cluster using Ray and Slurm.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: /environment-tutorials/creating-training-environment
:link-type: doc

Create your own resource server with custom tools and verification logic.
:::

::::
