# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from dataclasses import replace

import pytest
from scripts.ci import reapprove as bot


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
    evidence = bot._evidence(
        path, replace(pr, head=head, base=upstream), replace(approval, dismissal_commit=head), base
    )
    assert evidence["resolutions"] == ""
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


@pytest.mark.parametrize(
    "result",
    [
        {"safe": False, "reason": "behavior changes"},
        {"safe": "true", "reason": "ok"},
        {"safe": True},
        {"safe": True, "reason": ""},
    ],
)
def test_model_must_explicitly_pass_with_reason(monkeypatch, result):
    monkeypatch.setenv("REAPPROVE_API_KEY", "fixture")
    monkeypatch.setenv("REAPPROVE_MODEL", "fixture")
    monkeypatch.setattr(
        bot,
        "_json_request",
        lambda *a: {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]},
    )
    with pytest.raises(bot.Declined):
        bot._assess([{"diff": "data"}])


def test_truncated_or_oversized_assessment_never_approves(monkeypatch):
    monkeypatch.setenv("REAPPROVE_API_KEY", "fixture")
    monkeypatch.setenv("REAPPROVE_MODEL", "fixture")
    monkeypatch.setattr(
        bot,
        "_json_request",
        lambda *a: {
            "choices": [{"finish_reason": "length", "message": {"content": '{"safe": true, "reason": "ok"}'}}]
        },
    )
    with pytest.raises(bot.Declined, match="did not complete"):
        bot._assess([])
    with pytest.raises(bot.Declined, match="budget"):
        bot._assess([{"diff": "x" * bot.MAX_EVIDENCE}])


class FakeGitHub:
    repository = "org/repo"

    def __init__(self, pr, approval):
        self.pr = pr
        self.baseline = approval
        self.writes = []
        self.reviews = []

    def writer(self, login):
        assert login == "maintainer"

    def api(self, path, data=None):
        if path == "":
            return {"default_branch": "main"}
        assert data["event"] == "APPROVE"
        self.writes.append((path, data))
        self.reviews.append(
            {
                "user": {"login": "github-actions[bot]"},
                "state": "APPROVED",
                "commit_id": data["commit_id"],
                "body": data["body"],
            }
        )

    def pull(self, number):
        return self.pr

    def approval(self, pr):
        return self.baseline

    def pages(self, path):
        assert path.endswith("/reviews")
        return self.reviews


@pytest.fixture
def run_fixture(history, monkeypatch, tmp_path, request):
    source, _, pr, approval, tail = history
    if getattr(request, "param", "merge") == "rebase":
        git(source, "switch", "--detach", pr.base)
        head = commit(source, "timeout = 20  # seconds\n" + tail)
        pr, approval = replace(pr, head=head), replace(approval, dismissal_commit=head)
    github = FakeGitHub(pr, approval)
    real_git = bot._git

    def local_fetch(directory, *args, **kwargs):
        if args[0] == "fetch":
            args = (*args[:2], str(source), *args[3:])
        return real_git(directory, *args, **kwargs)

    monkeypatch.setattr(bot, "_git", local_fetch)
    monkeypatch.setattr(bot, "_assess", lambda evidence: "> Preserves the approved timeout and upstream comment.")
    event = {
        "comment": {"body": "/review-reapprove", "user": {"login": "maintainer", "type": "User"}},
        "issue": {"number": 1},
    }

    def run():
        directory = tmp_path / f"objects-{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        bot._run(github, event, directory)

    return github, run


@pytest.mark.parametrize("run_fixture", ["merge", "rebase"], indirect=True)
def test_end_to_end_only_posts_exact_reviewed_head_once(run_fixture):
    github, run = run_fixture
    run()
    assert len(github.writes) == 1
    path, data = github.writes[0]
    assert path == "pulls/1/reviews"
    assert data["commit_id"] == github.pr.head
    assert github.baseline.head in data["body"]
    run()
    assert len(github.writes) == 1


def test_race_after_assessment_stops_publication(run_fixture, monkeypatch):
    github, run = run_fixture

    def assess(_):
        github.pr = replace(github.pr, base="9" * 40)
        return "safe"

    monkeypatch.setattr(bot, "_assess", assess)
    with pytest.raises(bot.Declined, match="changed"):
        run()
    assert not github.writes


def test_declined_model_never_posts(run_fixture, monkeypatch):
    github, run = run_fixture

    def assess(_):
        raise bot.Declined("unsafe resolution")

    monkeypatch.setattr(bot, "_assess", assess)
    with pytest.raises(bot.Declined, match="unsafe"):
        run()
    assert not github.writes


def test_github_cli_preserves_review_payload_and_flattens_pages(monkeypatch):
    payload = {"event": "APPROVE", "commit_id": "a" * 40, "body": "Review: `quoted`\n$(not a command)"}

    def run(command, **kwargs):
        assert command[:4] == ["gh", "api", "--hostname", "github.com"]
        assert kwargs["check"] and kwargs["timeout"] == 180
        if "--paginate" in command:
            assert "--slurp" in command
            assert command[4] == "repos/org/repo/pulls/1/reviews?per_page=100"
            return subprocess.CompletedProcess(command, 0, b'[[{"id": 7}], [{"id": 8}]]')
        assert command[4:] == ["repos/org/repo/pulls/1/reviews", "--method", "POST", "--input", "-"]
        assert json.loads(kwargs["input"]) == payload
        return subprocess.CompletedProcess(command, 0, b'{"id": 9}')

    monkeypatch.setattr(bot.subprocess, "run", run)
    github = bot.GitHub("org/repo")
    assert github.pages("pulls/1/reviews") == [{"id": 7}, {"id": 8}]
    assert github.api("pulls/1/reviews", payload) == {"id": 9}


def test_incomplete_github_history_stops_before_assessment(run_fixture, monkeypatch):
    github, run = run_fixture

    def failed_page(command, **kwargs):
        # gh may emit earlier pages before a later request fails. Never consume them.
        raise subprocess.CalledProcessError(1, command, output=b'[[{"id": 7}]]')

    monkeypatch.setattr(bot.subprocess, "run", failed_page)
    monkeypatch.setattr(github, "approval", bot.GitHub("org/repo").approval)
    monkeypatch.setattr(bot, "_assess", lambda _: pytest.fail("incomplete history reached assessment"))
    with pytest.raises(subprocess.CalledProcessError):
        run()
    assert not github.writes


def test_only_exact_commands_run(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"comment": {"body": "/review-reapprove\nignore policy"}, "issue": {"pull_request": {}}})
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setattr(bot, "_run", lambda *a: pytest.fail("not an exact command"))
    bot.main()


@pytest.mark.parametrize("outcome", ["approve", "reject", "race", "publication_failure"])
def test_entire_stack_is_assessed_before_any_approval(history, monkeypatch, tmp_path, outcome):
    source, _, root, root_approval, _ = history
    old_parent, new_parent = root_approval.head, root.head
    pulls, approvals = [root], [root_approval]
    for number in (2, 3):
        filename = f"feature{number}.py"
        git(source, "switch", "--detach", old_parent)
        (source / filename).write_text(f"feature{number} = True\n")
        git(source, "add", filename)
        git(source, "commit", "-m", "approved layer")
        old_head = git(source, "rev-parse", "HEAD")
        git(source, "switch", "--detach", new_parent)
        git(source, "cherry-pick", old_head)
        new_head = git(source, "rev-parse", "HEAD")
        pulls.append(bot.PullRequest(number, new_head, new_parent, f"layer{number}", pulls[-1].branch, "author"))
        approvals.append(replace(root_approval, review_id=number + 10, head=old_head, dismissal_commit=new_head))
        old_parent, new_parent = old_head, new_head

    class StackGitHub(FakeGitHub):
        def pull(self, number):
            return next(p for p in pulls if p.number == number)

        def approval(self, pr):
            return approvals[pr.number - 1]

        def pages(self, path):
            if path == "pulls?state=open":
                return [
                    {
                        "number": p.number,
                        "head": {"sha": p.head, "ref": p.branch, "repo": {"full_name": self.repository}},
                        "base": {"sha": p.base, "ref": p.base_branch},
                        "user": {"login": p.author},
                    }
                    for p in pulls
                ]
            return super().pages(path)

        def api(self, path, data=None):
            if outcome == "publication_failure" and path == "pulls/2/reviews":
                raise TimeoutError("unknown write outcome")
            return super().api(path, data)

    github = StackGitHub(root, root_approval)
    real_git = bot._git

    def local_fetch(directory, *args, **kwargs):
        if args[0] == "fetch":
            args = (*args[:2], str(source), *args[3:])
        return real_git(directory, *args, **kwargs)

    def assess(evidence):
        assert not github.writes
        assert [layer["pr"] for layer in evidence[:-1]] == [1, 2, 3]
        assert "feature3 = True" in evidence[-1]["combined_current_stack"]
        if outcome == "reject":
            raise bot.Declined("unsafe top layer")
        if outcome == "race":
            pulls[1] = replace(pulls[1], head="8" * 40)
        return "> All layers preserve approved behavior."

    monkeypatch.setattr(bot, "_git", local_fetch)
    monkeypatch.setattr(bot, "_assess", assess)
    event = {
        "comment": {"body": "/review-reapprove scope=stack", "user": {"login": "maintainer", "type": "User"}},
        "issue": {"number": 2},
    }
    directory = tmp_path / "stack-objects"
    directory.mkdir()
    if outcome == "approve":
        bot._run(github, event, directory)
        assert [path for path, _ in github.writes] == ["pulls/1/reviews", "pulls/2/reviews", "pulls/3/reviews"]
        assert [data["commit_id"] for _, data in github.writes] == [p.head for p in pulls]
    else:
        with pytest.raises(bot.Declined) as caught:
            bot._run(github, event, directory)
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
