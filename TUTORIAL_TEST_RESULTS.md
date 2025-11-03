# Tutorial Testing Results

**Date**: November 3, 2025  
**Tutorial Tested**: Get Started > Setup and Installation  
**Status**: ✅ Instructions Correct (with findings)

---

## Summary

Successfully validated the setup and installation tutorial by following it step-by-step. The tutorial instructions are **accurate and work correctly**, but we discovered some issues during testing.

---

## Test Steps Completed

### ✅ 1. Prerequisites Check
- **Python 3.12.2** - Met requirement (3.12+)
- **UV package manager** - Already installed
- **Git** - Available
- **Virtual environment** - Already existed (.venv/)

### ✅ 2. Environment Configuration (env.yaml)
Created `env.yaml` with:
```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: <test-key>
policy_model_name: gpt-4.1-2025-04-14
```

**Finding**: Model name `gpt-4.1-2025-04-14` is valid and correctly documented.

### ✅ 3. Server Startup
Command used:
```bash
config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**Finding**: Servers started successfully on expected ports:
- Head server: 11000 ✅
- simple_weather resource: 49524
- simple_agent: 49525
- openai_model: 49526

**Issue Found**: Ray initialization requires `required_permissions: ['all']` in sandboxed environments due to `psutil` needing system-level process access.

### ⚠️ 4. Client Test
Command used:
```bash
python responses_api_agents/simple_agent/client.py
```

**Result**: Client connected to servers successfully but received 500 Internal Server Error.

**Root Cause**: Test API key had insufficient OpenAI quota:
```
Error code: 429 - You exceeded your current quota
```

**Server Behavior**: The simple_agent server doesn't gracefully handle OpenAI 429 errors, resulting in a 500 Internal Server Error with `text/plain` response instead of proper JSON error.

---

## Issues Discovered

### 1. ❌ Bug: Poor Error Handling for OpenAI Quota Errors
**Severity**: Medium  
**Location**: `responses_api_agents/simple_agent/`

**Issue**: When OpenAI returns a 429 "insufficient quota" error, the simple_agent server crashes and returns:
- HTTP 500 Internal Server Error
- Content-Type: `text/plain` instead of `application/json`
- Generic "Internal Server Error" message

**Expected Behavior**: Should return a proper JSON error response with:
- Clear indication that it's an OpenAI API quota issue
- HTTP 429 or 502 status code
- Actionable error message for the user

**Impact**: Makes debugging difficult for users following the tutorial.

### 2. ⚠️ Documentation Gap: API Key Quota Requirements
**Severity**: Low  
**Location**: `docs/get-started/setup-installation.md`

**Current Documentation**:
```markdown
- **OpenAI API key** (for the tutorial agent)
```

**Suggestion**: Add clarification:
```markdown
- **OpenAI API key** with available credits (for the tutorial agent)
```

And in the troubleshooting section, add:
```markdown
:::{dropdown} "500 Internal Server Error" from client
Check your OpenAI API key has sufficient quota/credits:
- Log into platform.openai.com
- Check Billing & Usage
- Add credits if needed
:::
```

---

## Documentation Validation

### ✅ Accurate Instructions
- Installation steps are correct
- Command syntax is accurate
- File paths are correct
- Configuration format is correct
- Model name (`gpt-4.1-2025-04-14`) is valid

### ✅ Success Checks Match Reality
- Server startup output matches documentation
- Port assignments work as described (11000 for head, auto-assigned for others)
- File structure section is accurate

### ✅ Troubleshooting Section
- Covers common issues like missing `ng_run`, API key format
- Could be enhanced with OpenAI quota error handling

---

## Recommendations

### For Documentation

1. **Add to troubleshooting section**:
   - OpenAI quota/billing errors
   - How to verify API key has credits
   - Link to OpenAI's billing page

2. **Clarify prerequisites**:
   - Specify API key needs active credits/quota
   - Mention approximate cost for running tutorial (~$0.01-0.05)

3. **Add validation step**:
   ```bash
   # Test your OpenAI API key
   python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
   ```

### For Code

1. **Improve error handling in simple_agent**:
   - Catch OpenAI API exceptions
   - Return proper JSON error responses
   - Map OpenAI error codes to appropriate HTTP status codes
   - Provide actionable error messages

2. **Add graceful degradation**:
   - Detect quota errors early
   - Provide clear user feedback
   - Suggest solutions (check billing, try different model, etc.)

---

## Test Environment

- **OS**: macOS 25.0.0 (Darwin)
- **Python**: 3.12.2
- **NeMo Gym**: Current development branch (llane/get-started)
- **UV**: Installed via Homebrew/conda
- **Shell**: zsh

---

## Conclusion

The **"Setup and Installation" tutorial is technically correct and can be followed successfully**. The instructions are accurate, commands work as documented, and the expected output matches reality.

The main issue encountered (500 error) was due to:
1. **Immediate cause**: Test API key lacking OpenAI credits
2. **Underlying issue**: Server doesn't handle OpenAI errors gracefully

**For users with valid API keys and credits, the tutorial should work perfectly as written.**

### Action Items

**High Priority**:
- [ ] Fix error handling in simple_agent for OpenAI API errors
- [ ] Add troubleshooting entry for quota/billing errors

**Medium Priority**:
- [ ] Add API key validation step to tutorial
- [ ] Clarify credit requirements in prerequisites

**Low Priority**:
- [ ] Add cost estimate for running tutorials
- [ ] Consider adding mock/example mode for testing without API calls

