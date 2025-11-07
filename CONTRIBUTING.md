# Contributing To NeMo-Gym

Welcome to NeMo Gym! This guide covers everything you need to contribute to the NeMo Gym codebase.

## Types of Contributions We Welcome

We're excited about contributions that expand NeMo Gym's capabilities and improve the developer experience:

### High Priority Contributions

**New Environments (Resource Servers)**
- Novel task domains (coding, reasoning, tool use, games, etc.)
- Real-world integrations (browsers, databases, APIs, file systems)
- Educational environments for agent training research
- Domain-specific verification logic and benchmarks

**RL Framework Integrations** 
- Connectors to RL libraries (TRL, SkyRL, Slime, etc.)
- Training loop implementations and examples
- Reward signal processing and optimization
- Multi-agent training support

### Always Welcome

**Documentation & Tutorials**
- New tutorial topics and advanced guides
- Code examples and best practices
- API documentation improvements
- Video tutorials and learning resources

**Bug Fixes**
- Performance improvements
- Error handling and edge cases
- Configuration and setup issues
- CI/CD pipeline fixes

**Features & Enhancements**
- New agent capabilities and tools
- Developer experience improvements
- Testing and debugging utilities
- Configuration and deployment features

### ⚠️ **Special Considerations**

**Bug Reports**
- Detailed reproduction steps highly appreciated
- Include environment details and minimal examples
- **Environment Behavior Changes**: Changing existing environment behavior should be minimized as it requires releasing a new version and makes results hard to compare across versions

**Breaking Changes**
- Discuss major API changes in issues before implementing
- Provide migration guides for breaking changes
- Consider backward compatibility when possible

---

**Not sure where to start?** Check our [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) or create a new issue to discuss your idea!



## Resource Server Verification Process

### What is the "Verified" flag?

The `verified` flag in resource server configs indicates that the environment has demonstrated value through training runs and/or shows meaningful improvement on target benchmarks. This quality signal helps identify production-ready environments.

### Verification Process

1. **Submit PR** with `verified: false` in your resource server config YAML.

Please note any config staged without a `verified` flag will automatically be updated to contain `verified: false` via pre-commit hook.
```yaml
your_server:
   resources_servers:
      your_server:
         entrypoint: app.py
         domain: coding
         verified: false  # Set to false by default when submitting environment for review
```


2. **Include W&B Links** in the PR description.

3. Reviewer Approval
A maintainer will:
- Review the W&B training run
- Verify the improvement claims
- Approve the PR if verification evidence is sufficient

### Default Status

- All resource servers should **start as `verified: false`** by default until a reviewer approves and updates it to `true`
- If a server's verification is later disputed or training runs cannot be accessed or reproduced, maintainers may revert it to `verified: false`
- The verified flag is reflected in the README resource server tables



## Quick Start for Contributors

### Development Setup
```bash
# Clone and set up development environment
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev --group docs

# Install pre-commit hooks (required for contributors)
pre-commit install
```

### Development Commands

**Run NeMo Gym Tests**:
```bash
ng_dev_test                                    # Run all NeMo Gym core tests
ng_test_all                                    # Run all server tests
ng_test +entrypoint=responses_api_agents/simple_agent  # Test single server
```

**View Test Coverage**:
```bash
coverage html                                  # Generate HTML coverage report
```

**Configuration Debugging**:
```bash
ng_dump_config                                 # Dump config as NeMo Gym sees it
```

## Contributing Resource Servers

If you're contributing a new resource server to NeMo Gym, follow this comprehensive checklist:

### Quality Control Checklist

1. **Necessary information to be included in the merge request**:
   1. Corresponding dataset on the spreadsheet
   2. Description of the prompt: What is the source? Which domain does it cover?
   3. Description of the environment, if any
   4. Description of the verifier: How is it verified and have you checked correctness?
   5. Legal approval status? If synthetically generated with open models, note this

2. **Simple correctness check**: After implementing your resources_server:
   1. The command you used to run the server for the uploaded data
   2. The resulting rollout and judges (include 5 examples here for people to understand better the data samples, and to ensure reward here is correct.)
   3. Other additional notes for running the server properly with the new PR.
3. Test: Please follow the guideline here to implement your own test and run test for your environment. Tests are strongly encouraged and you must have at least one test for every server you make. Test coverage is not explicitly required which means that YOU ARE RESPONSIBLE FOR YOUR OWN SERVER CORRECTNESS AND FUNCTION.
4. Reward Profiling: Please run inference on your prompts and environments (a ~500 small subset is OK) on two models:
   1. Qwen 3 30B A3B
   2. Qwen 3 235B Instruct (for agents/instruction following) or Qwen 3 235B Thinking (for math/coding)
   3. Generate 16 responses per prompt, report reward distribution
   4. [If using tool calling] Provide tool call metrics and correlation with rewards

5. **Training-based correctness check** [After Nemo Gym + Nemo RL integration]:
   - Train with GRPO on Qwen 30B A3B Instruct and Qwen 235B Instruct
   - Include training accuracy curve and test benchmark accuracy curve
   - See [Resource Server Verification Process](#resource-server-verification-process) section

6. **PR Check and Review**:
   1. Assign team member for reproduction and review
   2. Verify all above steps 1-5
   3. Check correctness of 5 examples  
   4. Re-run procedure to ensure reproducibility
   5. Ping @banghuaz-nvidia @bxyu-nvidia after greenlight

## Contributing RL Framework Integrations

This section is coming soon!
<!-- 
### Integration Requirements
- [ ] Interface specifications
- [ ] Reward signal processing
- [ ] Training loop integration points

### Testing Requirements  
- [ ] Unit tests for integration layer
- [ ] Training convergence tests
- [ ] Performance benchmarks

### Documentation Requirements
- [ ] Integration tutorial
- [ ] API documentation
- [ ] Example training scripts

### Performance Considerations
- [ ] Memory usage patterns
- [ ] GPU utilization optimization
- [ ] Distributed training support -->


## CI/CD Requirements

### What CI/CD Do I Need to Pass?

All contributions must pass these automated checks:

**Required Checks**:
- **Unit Tests**: All existing tests must pass
- **Build Docs**: Documentation must build without errors  
- **Copyright Check**: All files must have proper copyright headers
- **DCO Signing**: All commits must be signed off
- **Pre-commit Hooks**: Code formatting and linting

**Test Requirements**:
- At least one test per server you contribute
- Tests must run via `ng_test +entrypoint=your_server_path`
- Use pytest for async testing patterns

### Build Docs CI Failures

If the build-docs check fails:

1. **Check for broken links**:
   ```bash
   # Test documentation locally by running doc build commands
   cd docs
   uv run --frozen --only-group docs sphinx-build --fail-on-warning --builder html . _build/html
   ```

2. **Common issues**:
   - Missing docstrings in public functions
   - Broken markdown links in README or tutorials
   - Invalid rst/sphinx syntax

3. **Fix and test locally** before pushing

### Copyright Header Errors

**Error**: "Found files with missing copyright"

**Solution**: Add this header to all new Python files:
```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## DCO and Commit Signing Setup

All NeMo Gym contributions require **commit signing** and **DCO sign-off**. This section will get you set up quickly.

### Quick Setup

**1. Configure Git Signing (Required)**
```bash
# Set your identity (use your GitHub email)
git config --global user.name "Your Name"
git config --global user.email "your@github-email.com"

# Enable commit signing
git config --global commit.gpgsign true

# Use SSH signing (recommended - simpler than GPG)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
```

**2. Add Signing Key to GitHub**
1. Go to [GitHub Settings → SSH and GPG keys](https://github.com/settings/keys)
2. Under "SSH keys" section, find "Signing keys" subsection
3. Click "New SSH key" → Set type to "Signing Key"
4. Copy your public key: `cat ~/.ssh/id_ed25519.pub`
5. Paste the **entire output** (including `ssh-ed25519` prefix)

**3. Test Your Setup**
```bash
# Test commit signing
git commit --allow-empty -s -m "Test commit signing"

# Verify it's signed (should show "S" for signed)
git log --show-signature -1
```

### IDE Integration

**VSCode Setup**
Create/update `.vscode/settings.json` in your project:
```json
{
    "git.enableCommitSigning": true,
    "git.alwaysSignOff": true
}
```

**Other IDEs**: Enable "Git commit signing" and "Always sign-off" in your Git settings.

### Making Signed Commits

**Every commit must be signed off** using the `-s` flag:
```bash
# Standard workflow
git add .
git commit -s -m "Your commit message"

# The -s flag adds this line automatically:
# Signed-off-by: Your Name <your@email.com>
```

### Troubleshooting

**Problem**: `error: gpg failed to sign the data`
```bash
# Check if key exists and is correct format
ls -la ~/.ssh/id_ed25519.pub

# Verify git config
git config --list | grep -E "(user|gpg|signingkey)"

# Test SSH key
ssh-add -l
```

**Problem**: "Signing key not found on GitHub"
- Ensure you copied the **complete** public key including `ssh-ed25519` prefix
- Add the key as a **Signing Key** (not Authentication Key) on GitHub
- Wait a few minutes for GitHub to process the new key

**Problem**: IDE not signing commits
- Restart your IDE after configuring Git
- Check IDE Git settings match the command line configuration
- Try committing via command line first to verify setup

**Problem**: "DCO sign-off missing"
```bash
# Fix the last commit (if not pushed yet)
git commit --amend -s --no-edit

# For multiple commits, use interactive rebase
git rebase -i HEAD~3  # Adjust number as needed
```

### Alternative: GPG Signing

If you prefer GPG over SSH signing:

```bash
# Generate GPG key (if you don't have one)
gpg --full-generate-key

# List keys and copy the key ID
gpg --list-secret-keys --keyid-format=long

# Configure Git to use GPG
git config --global user.signingkey YOUR_GPG_KEY_ID
# Don't set gpg.format (defaults to gpg)

# Add GPG key to GitHub
gpg --armor --export YOUR_GPG_KEY_ID
# Copy output to GitHub Settings → SSH and GPG keys → GPG keys
```

### Understanding DCO

**DCO (Developer Certificate of Origin)** certifies that you have the right to submit your contribution. By signing off, you agree to these terms:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

**Any contribution with unsigned commits will be rejected.**

## Common Issues and Troubleshooting

### Testing Issues

**Problem**: Tests fail locally but not in CI
- Check Python version (3.12+ required)
- Ensure all dependencies installed: `uv sync --extra dev`
- Run in clean environment

**Problem**: Async test failures
- Use `pytest-asyncio` for async tests
- Mark async tests with `@pytest.mark.asyncio`
- Ensure proper fixture cleanup

### Pre-commit Hook Failures

**Problem**: Pre-commit hooks fail
```bash
# Fix common issues
pre-commit run --all-files        # Run all hooks manually
pre-commit autoupdate             # Update hook versions
```

**Common fixes**:
- `ruff check --fix .` for linting
- `ruff format .` for formatting  
- Add copyright headers to new files

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** with tests
3. **Run local checks**: `ng_dev_test && pre-commit run --all-files`
4. **Commit with signoff**: `git commit -s -m "Your message"`
5. **Push and create PR**: Ensure all CI checks pass
6. **Address review feedback** and iterate

---

**Questions?** Check existing issues or create a new one for guidance.
