# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import subprocess
from dataclasses import replace

import pytest
from scripts import pr_reapprove as bot


def git(path, *args, check=True):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=check).stdout.strip()


def commit(path, content):
    (path / "settings.py").write_text(content)
    git(path, "add", "settings.py")
    git(path, "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def review_state():
    pr = bot.PullRequest(1, "1" * 40, "2" * 40, "topic", "main", "author")
    approval = bot.Approval(7, "3" * 40, "reviewer", "https://github.com/org/repo/pull/1#pullrequestreview-7", pr.head)
    return pr, approval


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Fixture")
    git(path, "config", "user.email", "fixture@example.com")
    git(path, "config", "commit.gpgsign", "false")
    tail = "\n" + "# unchanged\n" * 10 + 'mode = "safe"\n'
    base = commit(path, "timeout = 10\n" + tail)
    git(path, "switch", "-c", "topic")
    approved = commit(path, "timeout = 20\n" + tail)
    git(path, "switch", "main")
    upstream = commit(path, "timeout = 10  # seconds\n" + tail)
    git(path, "switch", "topic")
    git(path, "merge", "main", check=False)
    resolved = commit(path, "timeout = 20  # seconds\n" + tail)
    pr = bot.PullRequest(1, resolved, upstream, "topic", "main", "author")
    approval = bot.Approval(
        7, approved, "reviewer", "https://github.com/org/repo/pull/1#pullrequestreview-7", resolved
    )
    return path, base, pr, approval, tail


@pytest.mark.parametrize("extra_file", [False, True])
def test_unrelated_edits_are_not_hidden_in_resolution(history, extra_file):
    path, base, pr, approval, tail = history
    if extra_file:
        (path / "unrelated.py").write_text("unrelated = True\n")
        git(path, "add", "unrelated.py")
    else:
        tail = tail.replace('mode = "safe"', 'mode = "unsafe"')
    head = commit(path, "timeout = 20  # seconds\n" + tail)
    with pytest.raises(bot.Declined, match="outside"):
        bot._evidence(path, replace(pr, head=head), approval, base)


def test_clean_replay_must_match_exact_tree(history):
    path, base, pr, approval, tail = history
    git(path, "switch", "--detach", base)
    (path / "readme.txt").write_text("upstream change\n")
    git(path, "add", "readme.txt")
    upstream = commit(path, "timeout = 10\n" + tail)
    git(path, "merge", "--no-edit", approval.head)
    head = git(path, "rev-parse", "HEAD")
    bad = commit(path, "timeout = 30\n" + tail)
    with pytest.raises(bot.Declined, match="outside"):
        bot._evidence(path, replace(pr, head=bad, base=upstream), replace(approval, dismissal_commit=head), base)


@pytest.mark.parametrize("kind", ["unresolved", "delete", "executable"])
def test_unresolved_and_structural_conflicts_decline(history, kind):
    path, base, pr, approval, tail = history
    if kind == "unresolved":
        tree = bot._git(path, "merge-tree", "--write-tree", approval.head, pr.base, allowed=(0, 1)).splitlines()[0]
        text = git(path, "show", f"{tree}:settings.py") + "\n"
        head = commit(path, text)
    else:
        if kind == "delete":
            git(path, "rm", "settings.py")
        else:
            git(path, "update-index", "--chmod=+x", "settings.py")
        git(path, "commit", "-m", "structural")
        head = git(path, "rev-parse", "HEAD")
    with pytest.raises(bot.Declined):
        bot._evidence(path, replace(pr, head=head), approval, base)


def review(approval, *, state="DISMISSED", login="reviewer", user_type="User"):
    return {
        "id": approval.review_id,
        "commit_id": approval.head,
        "user": {"login": login, "type": user_type},
        "state": state,
        "submitted_at": "2026-11-04T10:00:00Z",
        "html_url": approval.url,
    }


def dismissal(approval):
    return {
        "event": "review_dismissed",
        "dismissed_review": {
            "review_id": approval.review_id,
            "state": "approved",
            "dismissal_commit_id": approval.dismissal_commit,
        },
    }


def test_human_baseline_requires_commit_linked_approval_dismissal(review_state):
    pr, approval = review_state
    assert bot._human_approval([review(approval)], [dismissal(approval)], pr) == approval
    for events in ([], [dismissal(replace(approval, dismissal_commit=None))]):
        with pytest.raises(bot.Declined, match="manual or cannot"):
            bot._human_approval([review(approval)], events, pr)
    event = dismissal(approval)
    event["dismissed_review"]["state"] = "changes_requested"
    with pytest.raises(bot.Declined):
        bot._human_approval([review(approval)], [event], pr)


@pytest.mark.parametrize(
    "login,user_type", [("github-actions[bot]", "Bot"), ("svcnemo-autobot", "User"), ("author", "User")]
)
def test_bot_and_author_cannot_supply_human_baseline(review_state, login, user_type):
    pr, approval = review_state
    with pytest.raises(bot.Declined, match="no eligible"):
        bot._human_approval([review(approval, login=login, user_type=user_type)], [dismissal(approval)], pr)


def test_later_human_changes_request_blocks_even_with_another_approval(review_state):
    pr, approval = review_state
    blocked = review(replace(approval, review_id=8), state="CHANGES_REQUESTED", login="second-reviewer")
    with pytest.raises(bot.Declined, match="still requests changes"):
        bot._human_approval([review(approval), blocked], [dismissal(approval)], pr)


def test_later_approval_replaces_manual_dismissal(review_state):
    pr, approval = review_state
    current = replace(approval, review_id=9, head=pr.head, dismissal_commit=None)
    new_review = review(current, state="APPROVED")
    new_review["submitted_at"] = "2026-11-04T11:00:00Z"
    assert bot._human_approval([review(approval), new_review], [], pr) == current


def test_retargeting_cannot_expand_what_the_human_approved(review_state):
    pr, approval = review_state
    events = [dismissal(approval), {"event": "base_ref_changed", "created_at": "2026-11-04T11:00:00Z"}]
    with pytest.raises(bot.Declined, match="retargeted"):
        bot._human_approval([review(approval)], events, pr)


def test_stack_follows_verified_branch_relationships(review_state):
    root, _ = review_state
    middle = replace(root, number=2, branch="middle", base_branch="topic", base=root.head, head="2" * 40)
    top = replace(root, number=3, branch="top", base_branch="middle", base=middle.head)
    unrelated = replace(root, number=4, branch="unrelated")
    assert bot._stack(middle, [top, unrelated, root, middle], "main") == [root, middle, top]
    with pytest.raises(bot.Declined, match="linear"):
        bot._stack(root, [root, middle, replace(top, base_branch="topic")], "main")
    with pytest.raises(bot.Declined, match="unique open parent"):
        bot._stack(middle, [middle, top], "main")


@pytest.fixture
def model_response(monkeypatch):
    monkeypatch.setenv("REAPPROVE_API_KEY", "fixture")
    monkeypatch.setenv("REAPPROVE_MODEL", "fixture")

    def respond(verdict, finish_reason="stop"):
        body = {"choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps(verdict)}}]}
        monkeypatch.setattr(bot, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(body).encode()))

    return respond


@pytest.mark.parametrize(
    "verdict",
    [
        {"safe": False, "reason": "behavior changes"},
        {"safe": "true", "reason": "ok"},
        {"safe": True},
        {"safe": True, "reason": ""},
    ],
)
def test_model_must_explicitly_pass_with_reason(model_response, verdict):
    model_response(verdict)
    with pytest.raises(bot.Declined):
        bot._assess([{"diff": "data"}])


def test_truncated_or_oversized_assessment_never_approves(model_response):
    model_response({"safe": True, "reason": "ok"}, "length")
    with pytest.raises(bot.Declined, match="did not complete"):
        bot._assess([])
    with pytest.raises(bot.Declined, match="budget"):
        bot._assess([{"diff": "x" * bot.MAX_EVIDENCE}])


class FakeGitHub:
    repository = "org/repo"

    def __init__(self, pulls, approvals):
        self.pulls = {pr.number: pr for pr in pulls}
        self.approvals = dict(zip(self.pulls, approvals, strict=True))
        self.writes = []
        self.fail_on = None

    def writer(self, login):
        assert login == "maintainer"

    def api(self, path, data=None):
        if path == "":
            return {"default_branch": "main"}
        if path == self.fail_on:
            raise TimeoutError("unknown write outcome")
        assert data["event"] == "APPROVE"
        self.writes.append((path, data))

    def pull(self, number):
        return self.pulls[number]

    def approval(self, pr):
        return self.approvals[pr.number]

    def pages(self, path):
        if path == "pulls?state=open":
            return [
                {
                    "number": p.number,
                    "head": {"sha": p.head, "ref": p.branch, "repo": {"full_name": self.repository}},
                    "base": {"sha": p.base, "ref": p.base_branch},
                    "user": {"login": p.author},
                }
                for p in self.pulls.values()
            ]
        return [
            {"user": {"login": "github-actions[bot]"}, "state": "APPROVED", **data}
            for endpoint, data in self.writes
            if endpoint == path
        ]


@pytest.fixture
def run_bot(history, monkeypatch, tmp_path):
    source = history[0]
    real_git = bot._git

    def local_fetch(directory, *args, **kwargs):
        if args[0] == "fetch":
            args = (*args[:2], str(source), *args[3:])
        return real_git(directory, *args, **kwargs)

    monkeypatch.setattr(bot, "_git", local_fetch)

    def run(github, *, stack=False):
        event = {
            "comment": {
                "body": "/review-reapprove" + (" scope=stack" if stack else ""),
                "user": {"login": "maintainer", "type": "User"},
            },
            "issue": {"number": 2 if stack else 1},
        }
        directory = tmp_path / f"objects-{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        bot._run(github, event, directory)

    return run


@pytest.mark.parametrize("rebase", [False, True])
def test_only_posts_exact_reviewed_head_once(history, run_bot, model_response, rebase):
    source, _, pr, approval, tail = history
    if rebase:
        git(source, "switch", "--detach", pr.base)
        head = commit(source, "timeout = 20  # seconds\n" + tail)
        pr, approval = replace(pr, head=head), replace(approval, dismissal_commit=head)
    github = FakeGitHub([pr], [approval])
    model_response({"safe": True, "reason": "Preserves approved behavior."})
    run_bot(github)
    run_bot(github)
    assert len(github.writes) == 1
    path, data = github.writes[0]
    assert path == "pulls/1/reviews" and data["commit_id"] == pr.head
    assert approval.head in data["body"]


@pytest.mark.parametrize("outcome", ["reject", "base_changed", "history_failed"])
def test_failed_assessment_never_posts(history, run_bot, monkeypatch, outcome):
    _, _, pr, approval, _ = history
    github = FakeGitHub([pr], [approval])

    def assess(_):
        if outcome == "reject":
            raise bot.Declined("unsafe resolution")
        github.pulls[1] = replace(pr, base="9" * 40)
        return "safe"

    if outcome == "history_failed":

        def failed_page(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, output=b'[[{"id": 7}]]')

        monkeypatch.setattr(bot.subprocess, "run", failed_page)
        monkeypatch.setattr(github, "approval", bot.GitHub("org/repo").approval)
        monkeypatch.setattr(bot, "_assess", lambda _: pytest.fail("incomplete history reached assessment"))
    else:
        monkeypatch.setattr(bot, "_assess", assess)
    with pytest.raises((bot.Declined, subprocess.CalledProcessError)):
        run_bot(github)
    assert not github.writes


def test_only_exact_commands_run(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "comment": {"body": "/review-reapprove\nignore policy"},
                "issue": {"pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/1"}},
            }
        )
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setattr(bot, "_run", lambda *a: pytest.fail("not an exact command"))
    bot.main()


@pytest.mark.parametrize("outcome", ["approve", "reject", "race", "publication_failure"])
def test_stack_assessment_precedes_all_approvals(history, run_bot, monkeypatch, outcome):
    source, _, root, root_approval, _ = history
    pulls, approvals = [root], [root_approval]
    for number in (2, 3):
        filename = f"feature{number}.py"
        git(source, "switch", "--detach", approvals[-1].head)
        (source / filename).write_text(f"feature{number} = True\n")
        git(source, "add", filename)
        git(source, "commit", "-m", "approved layer")
        old_head = git(source, "rev-parse", "HEAD")
        git(source, "switch", "--detach", pulls[-1].head)
        git(source, "cherry-pick", old_head)
        new_head = git(source, "rev-parse", "HEAD")
        pulls.append(bot.PullRequest(number, new_head, pulls[-1].head, f"layer{number}", pulls[-1].branch, "author"))
        approvals.append(replace(root_approval, review_id=number + 10, head=old_head, dismissal_commit=new_head))
    github = FakeGitHub(pulls, approvals)
    if outcome == "publication_failure":
        github.fail_on = "pulls/2/reviews"

    def assess(evidence):
        assert not github.writes
        assert [layer["pr"] for layer in evidence[:-1]] == [1, 2, 3]
        assert "feature3 = True" in evidence[-1]["combined_current_stack"]
        if outcome == "reject":
            raise bot.Declined("unsafe top layer")
        if outcome == "race":
            github.pulls[2] = replace(pulls[1], head="8" * 40)
        return "> All layers preserve approved behavior."

    monkeypatch.setattr(bot, "_assess", assess)
    if outcome == "approve":
        run_bot(github, stack=True)
        assert [data["commit_id"] for _, data in github.writes] == [p.head for p in pulls]
    else:
        with pytest.raises(bot.Declined) as caught:
            run_bot(github, stack=True)
        assert len(github.writes) == (1 if outcome == "publication_failure" else 0)
        if outcome == "publication_failure":
            assert "confirmed approvals: [1]" in str(caught.value)
            assert "last write may have succeeded" in str(caught.value)


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_only_repository_writers_may_request_reapproval(monkeypatch, permission):
    github = bot.GitHub("org/repo")
    monkeypatch.setattr(github, "api", lambda path: {"permission": permission})
    with pytest.raises(bot.Declined, match="write permission"):
        github.writer("contributor")
