# NeMo Gym Tutorials

## Tutorial Structure

- **Getting Started**: Learn core concepts and run your first agent in NeMo Gym
- **Training**: Understand verification and collect rollouts for RL, SFT, and DPO
- **Resource Servers**: Build custom environments, tools, and verification systems
- **Advanced Operations**: Master configuration, testing, deployment, and scaling


## Getting Started

### **[00 - Key Terminology](00-terminology.md)**
Essential vocabulary and definitions for building RL workflows with NeMo Gym. Quick reference for terms you'll encounter when generating rollouts, training agents, and working with the platform. 

### **[01 - Understanding Core Concepts](01-concepts.md)**
Learn NeMo Gym's core abstractions: Models, Resources, and Agents. Understand how they work together.

### **[02 - Setup and Installation](02-setup.md)** 
Get NeMo Gym installed and servers running with your first successful agent interaction.

### **[03 - Interacting with Agents in NeMo Gym](03-your-first-agent.md)**
Break down the agent workflow step-by-step and experiment with different inputs. This covers the fundamentals you need before beginning verification, rollout collection and training.

## Training

### **[04 - Verifying Agent Results](04-verifying-results.md)**
Understand how NeMo Gym evaluates agent performance and what verification means for training. Learn about different verification patterns and how they drive agent improvement.

### **[05 - Rollout Collection Fundamentals](05-rollout-collection.md)**
Master NeMo Gym's rollout generation system - the foundation for understanding agent behavior, creating training data, and evaluating performance. Learn to systematically capture complete agent interactions.

<!-- TODO: Add link [06 - Collecting Rollouts for Reinforcement Learning](06-rl-rollout-collection.md) -->
### **06 - Collecting Rollouts for Reinforcement Learning (Coming soon!)**
*Coming soon* - Generate rollouts for real-time RL training. Learn online rollout collection, integration with RL frameworks, and continuous agent improvement through environmental feedback.

### **[07 - Collecting Rollouts for SFT and DPO Training](07-sft-dpo-rollout-collection.md)**
Use generated rollouts to create training data for supervised fine-tuning and direct preference optimization. Learn data preparation and quality filtering.


## Resource Servers

<!-- TODO: Add link [08 - Building Custom Resource Servers](08-custom-resource-servers.md) -->
### **08 - Building Custom Resource Servers**
*Coming soon* - Learn to create your own tools and verification systems. Over time we plan to add more tutorials that demonstrate integration with MCP and Docker, multi-task training and how to perform dynamic prompting.


## Advanced Operations

### **[09 - Configuration Management](09-configuration-guide.md)**
Master NeMo Gym's flexible configuration system to handle different environments, secrets, and deployment scenarios.
