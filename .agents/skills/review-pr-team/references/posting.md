# Staging and publishing a review on GitHub

Start every command with `source "$HOME/.cache/gym-pr-review/pr-<PR>/state.env"` (SKILL.md §1.1).

## Why the review is staged without a body

When a pending review is created through the API with a body, the GitHub UI's "Submit review" dialog
opens with an empty text box. Submitting from the UI then replaces the body with that empty box, and
GitHub refuses to edit an empty body afterwards. Inline comments survive.

So always stage the pending review with inline comments only. If the review has a summary
(SKILL.md §4.3), save it as `$STATE_DIR/summary.md` and post it after publishing.

## Build the payload

Build the request with Python, so markdown and backticks need no shell escaping. Give each comment
an `_expect` value: a short piece of text that must appear on the lines the comment is anchored to,
such as the function name or the variable the finding is about. The audit uses it to catch a comment
attached to the wrong line, and removes it before posting.

```python
import json, os
from pathlib import Path

state = Path.home() / ".cache/gym-pr-review/pr-<PR>"
head_sha = (state / "state.env").read_text().split("HEAD_SHA=")[1].split()[0]
comments = [
    {
        "path": "nemo_gym/x.py",
        "line": 42,
        "side": "RIGHT",
        "body": "<finding>\n\n```suggestion\n<fix>\n```",
        "_expect": "total",
    },
    # A comment on several lines adds "start_line": 40 and "start_side": "RIGHT".
]
review = {"commit_id": head_sha, "comments": comments}  # no "event" and no "body": the review stays pending
(state / "payload.json").write_text(json.dumps(review))
```

## Audit before posting

The audit script checks every comment and prints one row per comment. Show the table to the
reviewer. Do not post while any row fails, and run the audit again before adding a comment later.

```bash
python3 .agents/skills/review-pr-team/scripts/audit_review_payload.py \
  --payload "$STATE_DIR/payload.json" --worktree "$WORKTREE" --diff "$STATE_DIR/pr.diff" \
  --out "$STATE_DIR/review-post.json"
```

| Check | Fails when |
|---|---|
| `hunk` | the commented lines are not all inside one changed section (hunk) of the PR's diff. GitHub rejects the whole review if any comment is outside the diff. |
| `anchor` | the commented lines at the PR head do not contain the comment's `_expect` text |
| `perm` | a link to code points at a branch instead of a commit, or a `file:line` reference is not a link |
| `sugg` | applying the suggestion block to the file removes a `def`, `class`, or `return` that the replaced lines had |
| `sugg` | the file with the block applied no longer parses (Python or YAML) |
| `sugg` | the file with the block applied fails `ruff format --check` |
| `mermaid` | (warning only) a mermaid diagram is present. Render it once with mermaid-cli (`npx -y @mermaid-js/mermaid-cli`) or GitHub's preview before posting. |

Two checks need the lead's judgment rather than the script:

- Compare each final comment with `$STATE_DIR/drafts.json` from before the follow-up passes. Every
  link and every suggestion block must be unchanged.
- Read each comment once as the author would, without the investigation behind it.

## Stage the pending review

GitHub allows one pending review per user per PR, and it may hold the reviewer's own drafts. Check
first:

```bash
gh api --paginate repos/NVIDIA-NeMo/Gym/pulls/<PR>/reviews --jq '.[] | select(.state=="PENDING") | .id'
```

If none exists, post the audited payload:

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/<PR>/reviews --method POST --input "$STATE_DIR/review-post.json" \
  --jq '{id, node_id, html_url}'
```

Then write the review record, which a later re-review reads:

```json
{"review_id": 123, "node_id": "PRR_...", "head_sha": "<HEAD_SHA>", "base_sha": "<BASE_SHA>",
 "verdict": "comment", "summary_posted": false}
```

Save it as `$STATE_DIR/review.json`. Set `summary_posted` to `null` when the review has no summary.

Show the reviewer the review's `html_url` and the verdict they chose. Staging does not authorize
publishing. Wait for the reviewer to say publish, or to say they will publish themselves.

## Add a comment to a pending review

The REST API cannot add to an existing pending review, so use GraphQL with the review's `node_id`.
This mutation reads `line` against the PR's current head, not the review's commit. Recompute the
anchor at the current head, and run the audit again first.

```python
import json
from pathlib import Path

state = Path.home() / ".cache/gym-pr-review/pr-<PR>"
query = (
    "mutation($reviewId: ID!, $body: String!, $path: String!, $line: Int!, $startLine: Int) {"
    " addPullRequestReviewThread(input: {pullRequestReviewId: $reviewId, body: $body, path: $path,"
    " line: $line, side: RIGHT, startLine: $startLine, startSide: RIGHT}) { thread { id } } }"
)
variables = {"reviewId": "<node_id>", "body": "<comment>", "path": "<file>", "line": 42, "startLine": None}
(state / "add_comment.json").write_text(json.dumps({"query": query, "variables": variables}))
```

```bash
gh api graphql --input "$STATE_DIR/add_comment.json"
```

## Edit or cancel

- Edit a staged comment with
  `gh api repos/NVIDIA-NeMo/Gym/pulls/comments/<comment-id> --method PATCH -F body=@<file>`. Only `-F`
  reads `@file`; `-f body=@file` posts the literal text. Read the body back after editing.
- Cancel only when the reviewer asks. Deleting the pending review also deletes any drafts the
  reviewer added in the UI.
- Before any change, re-read the pending review from GitHub. A comment that disappeared was deleted
  on purpose. A comment you do not recognize is the reviewer's.

## Publish

**From the GitHub UI (the usual path).** Remind the reviewer which verdict they chose. When they say
it is published, confirm the review is no longer pending. If there is a summary, post it and set
`summary_posted` to `true` in the review record:

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/<PR>/reviews/<review_id> --jq .state
gh pr comment <PR> --repo NVIDIA-NeMo/Gym --body-file "$STATE_DIR/summary.md"
```

**Through the API, when the reviewer asks.** Submit with the chosen verdict as the event
(`APPROVE`, `COMMENT`, or `REQUEST_CHANGES`). Add `-F body=@"$STATE_DIR/summary.md"` only when there
is a summary:

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/<PR>/reviews/<review_id>/events --method POST -f event=COMMENT
```

## Reply to existing threads

Post approved replies after the review is published:

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/<PR>/comments/<comment-id>/replies --method POST -F body=@<file>
```

Check who wrote a thread before replying. Never "correct" someone else's comment as if it were yours.
