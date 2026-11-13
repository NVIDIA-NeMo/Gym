# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required non-Python assets for frozen Enroot runtimes."""
from pathlib import Path
import os

REPAIR_REVISION = 'v5_failure_repairs_20261027'

UID_HELPERS = ('gosu', 'id', 'runuser', 'service', 'setpriv', 'su', 'su-exec', 'sudo')

def validate_runtime_assets(root=None):
    package = Path(root)/'saas_bench' if root is not None else Path(__file__).parent
    required = [package/'uid_wrappers'/name for name in UID_HELPERS]
    missing = [str(p) for p in required if not p.is_file() or not os.access(p, os.X_OK)]
    if missing:
        raise RuntimeError('Missing or non-executable runtime UID helpers: ' + ', '.join(missing))
    return {'uid_helpers': list(UID_HELPERS), 'passed': True}

def source_fingerprint(manifest):
    import hashlib
    import json
    frozen = {k: v for k, v in manifest['sha256'].items() if k.startswith('runtime_snapshot/')}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def require_app_canaries(root, apps):
    """A bulk launch needs actual access tests from this exact frozen source."""
    import json
    import hashlib
    root = Path(root)
    config_hash = hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    fingerprint = source_fingerprint(json.loads((root/'manifest.json').read_text()))
    for app in sorted(set(apps)):
        report_path = root/'runtime_canaries'/app/'result.json'
        if not report_path.is_file():
            raise ValueError('Fresh startup/auth canary required before bulk launch: '+app)
        report = json.loads(report_path.read_text())
        if not (report.get('passed') is True and report.get('app') == app and
                report.get('runtime_source_sha256') == fingerprint and
                report.get('campaign_config_sha256') == config_hash):
            raise ValueError('Startup/auth canary failed or belongs to another frozen runtime: '+app)
