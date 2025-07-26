- [NeMo-Gym](#nemo-gym)
- [Setup](#setup)
- [Development](#development)

# NeMo-Gym
# Setup
Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install NeMo Gym
```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

Install NeMo Gym as a developer or contributor
```bash
uv pip install -e ".[dev]"
```

# Development
Lint
```bash
ruff check --fix
```

Format
```bash
ruff format
```
