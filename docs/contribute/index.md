(contribute-index)=

# Contribute to NeMo Gym

Welcome! We are excited to have you contribute to NeMo Gym. Whether you are adding new training environments, integrating RL frameworks, improving documentation, or fixing bugs, your contributions help advance RL training.

---

## Ways to Contribute

:::{dropdown} {octicon}`server;1.5em;sd-mr-1` Add Environments

New environments expand Gym's training capabilities.

- Novel training environments (coding, reasoning, tool use, games, and so on)
- Benchmark integrations (such as SWE-Bench and Tau Bench)

→ Refer to {ref}`contribute-environments` for guidance.
:::

:::{dropdown} {octicon}`workflow;1.5em;sd-mr-1` Integrate RL Frameworks

Have a preferred training framework? Help integrate it with NeMo Gym.

→ Refer to {ref}`training-framework-integration` for integration guides.
:::

:::{dropdown} {octicon}`book;1.5em;sd-mr-1` Improve Documentation

- New tutorials and advanced guides
- Code examples and best practices
- API documentation improvements
:::

:::{dropdown} {octicon}`bug;1.5em;sd-mr-1` Report Bugs & Fix Issues

- Found a bug? [Open an issue](https://github.com/NVIDIA-NeMo/Gym/issues/new?template=bug.md) with reproduction steps.
- Want to fix one? Check [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) for bugs to tackle.
:::

:::{dropdown} {octicon}`tools;1.5em;sd-mr-1` Features & Enhancements

Have an idea for a new feature or improvement? [Open an issue](https://github.com/NVIDIA-NeMo/Gym/issues/new?template=feature.md) to start a discussion!
:::

---

## Use of AI and LLM Tools

We encourage contributors to use AI coding assistants (Copilot, Cursor, Claude, ChatGPT, and so on) where they genuinely help. However, AI assistance does not replace human understanding, judgment, and accountability.

### Guiding Principle

**If the human effort required to create a pull request is less than the effort required for maintainers to review it, that contribution should not be submitted.**

You are responsible for every line of code you submit, regardless of whether you or an AI tool wrote it.

### What We Expect

- **Understand your changes**: You must be able to explain and debug every line in your PR. Treat AI output as code from an untrusted source that requires your review.
- **Self-review and test**: Before requesting review, read through the diff carefully, run the test suite (`pytest`), and run pre-commit checks locally. Never treat AI-generated code as ready to merge without your own verification.
- **Keep PRs focused**: AI tools sometimes make "drive-by improvements" to unrelated code. Strip out any changes that are not directly relevant to the task at hand.
- **Verify correctness of AI-generated tests**: AI-generated tests can appear to pass while testing nothing meaningful. Ensure assertions are substantive and cover the intended behavior.

### AI Attribution

When AI tools generate a substantial portion of your contribution, add an `Assisted-by:` trailer to your commit message:

```bash
git commit -s -S -m "Add reward function for code evaluation

Assisted-by: GitHub Copilot"
```

This is not required for routine autocomplete suggestions, only for cases where AI generated significant code blocks, logic, or documentation.

### What We Will Close

We will close pull requests and issues that appear to be low-effort, AI-generated submissions. Common indicators include:

- Boilerplate or generic code that ignores project conventions
- PRs that do not pass CI or pre-commit checks
- Descriptions or comments that are clearly unreviewed LLM output
- Bulk "improvements" with no corresponding issue or discussion

This is not about policing tool usage. It is about maintaining the quality bar that the community and maintainers depend on.

---

## Before You Start

All contributions require signed commits and DCO sign-off. Refer to {ref}`development-setup` for environment setup, CI/CD requirements, and commit signing instructions.

**Not sure where to start?** Check our [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) or create a new issue to discuss your idea.
