---
name: nemo-gym-release-notes
description: Write and review user-focused NeMo Gym release notes and README news entries. Use when drafting, editing, prioritizing, or validating a release summary, release-note section, changelog highlights, or release announcement.
---

# NeMo Gym Release Notes

Write concise release notes that explain what users can now accomplish and why it matters.

## Build an evidence base

1. Identify the previous release tag and exact release branch or commit.
2. Treat that bounded range as authoritative. Exclude changes outside it, regardless of branch.
3. Read the preceding release notes for structure, voice, bullet length, punctuation, and emphasis.
4. For each candidate, record the supporting commit or PR, user outcome, shipped scope, and intended section.
5. Verify contributor counts and first-time status from the same range, excluding bots.
6. Check ambiguous claims against the implementing PR, code, tests, and docs.
7. Distinguish a PR's motivating use case from the interface contract actually shipped.

## Work in dependency order

1. Draft all detailed release-note sections.
2. Self-review every bullet with the checklist below, then reread the full draft for duplication, inconsistent terminology, misplaced items, weak priority, and contradictions.
3. Recheck corrected claims against the evidence base.
4. Derive the Release Summary and Highlights from the approved sections.
5. Update README news last as a condensed mirror of the approved Highlights.

## Write for users

- Lead with the outcome; name the mechanism afterward.
- Replace implementation jargon with the task it enables, or define the term on first use.
- Omit command syntax unless it is essential to the user action; leave procedural details to the documentation.
- Explain unfamiliar environments or integrations instead of listing names alone.
- Give migrations enough context for users to understand what changed, whether they are affected, and what action to take.
- State safety or isolation limits explicitly. For example, host-process execution is not sandbox isolation.
- Describe only shipped support. Distinguish infrastructure for defining a capability from ready-made implementations, and provider-neutral architecture from providers that implement it.
- Keep documentation additions in a documentation section rather than presenting them as product capabilities.

## Keep each bullet coherent

- Give each bullet one main user benefit.
- Group details only when they support the same outcome; split unrelated capabilities even if they share a component or PR.
- Check that conjunctions and punctuation do not imply a relationship or causality that does not exist.
- Avoid overloaded terms such as “orchestration”; name the layer or workflow precisely.
- Place each item in the section matching its user workflow, not its internal code ownership.
- Ensure each section title accurately describes its contents.
- Prefer short natural sentences and match the preceding release's punctuation.

## Use emphasis sparingly

- Default to plain text in ordinary feature bullets.
- Use bold only when it materially improves scanning, typically for contributor handles or category labels in dense lists.
- Avoid bold prefixes and repeated product names.
- Match the preceding release rather than adding emphasis for visual variety.

## Prioritize

Use release goals and user evidence first. As tie-breakers, prefer broad workflows, required migrations, reliability and data correctness, common integrations, then specialized controls. Move niche flags below capabilities that affect more users. Remove low-value implementation details when a detailed changelog will accompany the notes.

## Update README news

- Keep only the newest release above the collapsed “Previous News” block. Move the former current entry to the top of that block without duplicating it, preserving reverse chronological order.
- Copy the Highlights list, preserving its order, scope, and meaning; shorten wording when useful rather than copying mechanically.
- Do not copy the introductory summary paragraph or detailed sections.
- Include the release date and release link, and do not introduce claims absent from the approved highlights.

## Review checklist

For every bullet, ask:

- Is it inside the release diff?
- What can the user now do, avoid, trust, or debug?
- Is the scope explicit, and are absolute claims such as “complete,” “reliable,” or “without loss” justified by the evidence?
- Does the sentence combine unrelated changes?
- Does its grammar imply a relationship that does not exist?
- Is architecture or infrastructure being mistaken for a shipped implementation?
- Would an unfamiliar reader understand the named benchmark or mechanism?
- Is this important enough for release notes rather than the detailed changelog?
- Is it ordered appropriately within its section?
- Does its terminology, punctuation, and emphasis match the preceding release?
- Is the claim duplicated elsewhere without serving the summary?

## Final validation

1. Recheck every claim and contributor against the bounded release range.
2. Search for prohibited or deferred launch terminology.
3. Confirm section and bullet priority against the release goals.
4. Confirm README news matches the approved Highlights in order, scope, and meaning.
5. Remove temporary review markers.
6. Run the Fern check, scoped pre-commit hooks, IDE lint checks, and `git diff --check`.
7. Review the final diff for unrelated edits.

## References

Read [section-template.md](section-template.md) before choosing or renaming sections.
Read [examples.md](examples.md) when drafting wording or resolving a review finding.
