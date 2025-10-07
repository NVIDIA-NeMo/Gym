# NeMo Gym

NeMo Gym orchestrates scalable rollout collection for LLM training. It provides both the infrastructure to systematically generate agent interaction data and a large collection of NVIDIA and community curated environments with verification logic, making it easy to generate large-scale, high-quality data for RL workflows using the training framework of your choice.

TODO:
- add link to hugging face data
- add mention that we use this infra for nemotron training

## 🏆 Why NeMo Gym?

- **Fast Development** - Less boilerplate, more innovation - get from idea to working agent quickly
- **Production-Ready** - Async architecture designed for high-throughput training workloads  
- **Configuration-Driven** - Swap models, resources, and environments via YAML without touching code
- **Unified Patterns** - Standardized interfaces across the fragmented agent ecosystem
- **Smart Orchestration** - Agents automatically coordinate model↔resource calls with async efficiency
- **Focus on Logic, Not Plumbing** - Built-in server framework handles infrastructure concerns

## 🚀 Quick Start

### New to NeMo Gym?
Follow our **[Tutorial Series](tutorials/README.md)** for a progressive learning experience:
- **Getting Started**: Setup and first agent interaction
- **Basics**: Verification, rollouts, and training fundamentals  
- **Advanced**: Custom environments, deployment, and scaling

### Quick Installation
```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs
```

### Run Your First Agent
Start with **[Understanding Concepts](tutorials/01-concepts.md)**, then follow **[Setup & Installation](tutorials/02-setup.md)** for hands-on implementation.

**TLDR**:
```bash
# Configure API access
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Start servers and run agent
config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

python responses_api_agents/simple_agent/client.py
```

## 📖 Documentation

- **[Tutorials](tutorials/README.md)** - Progressive learning path from basics to advanced topics
- **[Contributing](CONTRIBUTING.md)** - Developer setup, testing, and contribution guidelines
- **[API Documentation](docs/)** - Technical reference and API specifications
 

## 🤝 Community & Support

We'd love your contributions! Here's how to get involved:

- **[Report Issues](https://github.com/NVIDIA-NeMo/Gym/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/NVIDIA-NeMo/Gym/discussions)** - Community Q&A and ideas
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute code, docs, or new environments