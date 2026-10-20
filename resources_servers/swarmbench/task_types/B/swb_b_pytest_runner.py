# Usage: python -I -B swb_b_pytest_runner.py OUT_JSON ROOT < {"ids": [...], "paths": [...], "args": [...]}
# Runs the given pytest node ids with ROOT as cwd and rootdir (vendored pytest, `paths` prepended to sys.path) and
# writes {"outcomes": {nodeid: passed|failed|skipped|error}, "messages": {nodeid: text}} to OUT_JSON.
# Ids that never ran are absent from "outcomes".
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Collector:
    def __init__(self):
        self.out = {}
        self.msgs = {}

    def pytest_runtest_logreport(self, report):
        prev = self.out.get(report.nodeid)
        if report.failed:
            outcome = "failed" if report.when == "call" else "error"
        elif report.skipped:
            outcome = "skipped"
        else:
            outcome = "passed"
        if prev in ("failed", "error") or (prev == "skipped" and outcome == "passed"):
            return
        self.out[report.nodeid] = outcome
        if report.failed:
            self.msgs[report.nodeid] = str(report.longrepr)[-4000:]

    def pytest_collectreport(self, report):
        if report.failed:
            key = report.nodeid or "<collect>"
            self.out[key] = "error"
            self.msgs[key] = str(report.longrepr)[-4000:]


def main():
    out_path, root = sys.argv[1], os.path.realpath(sys.argv[2])
    job = json.load(sys.stdin)
    for p in reversed(job.get("paths", [])):
        sys.path.insert(0, os.path.realpath(p))
    sys.path.insert(0, os.path.join(HERE, "_vendor"))
    os.chdir(root)
    sys.setrecursionlimit(5000)
    import pytest

    import io
    col = Collector()
    real_out, real_err = sys.stdout, sys.stderr
    buf = io.StringIO()
    code = None
    sys.stdout = sys.stderr = buf
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--rootdir", root, "-c", os.devnull, "--assert=plain",
                     "--continue-on-collection-errors"] + job.get("args", []) + job["ids"], plugins=[col])
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    if not col.out:  # nothing ran (e.g. conftest import error): keep pytest's own report
        col.msgs["<session>"] = f"pytest exit {code}: " + buf.getvalue()[-3000:]
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"outcomes": col.out, "messages": col.msgs}, f)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
