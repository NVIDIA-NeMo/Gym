(development-setup)=

# Development Setup

This guide covers everything you need to set up your development environment and submit contributions to NeMo Gym.

## Quick Start

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

## Development Commands

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

---

## CI/CD Requirements

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

1. **Test documentation locally**:
   ```bash
   cd docs
   uv run --frozen --only-group docs sphinx-build --fail-on-warning --builder html . _build/html
   ```

2. **Common issues**:
   - Missing docstrings in public functions
   - Broken markdown links in README or tutorials
   - Invalid rst/sphinx syntax

### Copyright Header Errors

**Error**: "Found files with missing copyright"

**Solution**: Add this header to all new Python files:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

---

## DCO and Commit Signing

All NeMo Gym contributions require **commit signing** and **DCO sign-off**.

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

---

## Common Issues

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

---

## Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** with tests
3. **Run local checks**: `ng_dev_test && pre-commit run --all-files`
4. **Commit with signoff**: `git commit -s -S -m "Your message"`
5. **Push and create PR**: Ensure all CI checks pass
6. **Address review feedback** and iterate

---

**Questions?** Check existing issues or create a new one for guidance.

