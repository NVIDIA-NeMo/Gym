# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 main hooks/artifacts, main stop, then sidecar hooks/artifacts."""

import json
import shlex
from functools import partial
from pathlib import Path

from resources_servers.terminal_bench_4.transfers import artifact_metadata_path, download_dir, download_file


async def collect(environment, directory, diagnostics):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    task = environment.task.config
    shared_logs = getattr(environment, "shared_logs", None)
    collected_artifacts = task.collected_artifacts
    entries, claims = [], []

    async def hooks(main):
        for hook in task.verifier.collect:
            if (hook.service == "main") != main:
                continue
            record = {"operation": "collect_hook", "service": hook.service, "command": hook.command}
            try:
                result = await environment.exec(
                    hook.command,
                    service=hook.service,
                    timeout_sec=int(hook.timeout_sec),
                    user=hook.user,
                )
                record.update(return_code=result.return_code, stdout=result.stdout, stderr=result.stderr)
            except Exception as exc:
                record["error"] = str(exc)
            diagnostics.append(record)

    async def artifacts(main):
        for artifact in collected_artifacts:
            if (artifact.service in (None, "main")) != main:
                continue
            target = directory / artifact.host_path
            record = {
                "source": artifact.source,
                "destination": "artifacts/" + artifact.host_path.as_posix(),
                "service": artifact.service,
                "exclude": list(artifact.exclude),
                "type": "file" if Path(artifact.source).suffix else "directory",
                "status": "skipped",
            }
            # Preserve the reference's same-source re-collection exception.
            collision = any(
                source != artifact.source
                and (claimed == target or claimed in target.parents or target in claimed.parents)
                for claimed, source in claims
            )
            if not collision:
                claims.append((target, artifact.source))
                try:
                    kind = await environment.exec(
                        f"test -d {shlex.quote(artifact.source)}",
                        service=artifact.service,
                    )
                    if kind.return_code:
                        kind = await environment.exec(
                            f"test -d {shlex.quote(artifact.source)}",
                            service=artifact.service,
                            user="root",
                        )
                    record["type"] = "directory" if kind.return_code == 0 else "file"
                except Exception:
                    pass
                try:
                    sandbox = environment.sandbox(artifact.service)
                    metadata_path = artifact_metadata_path(directory, artifact.host_path)
                    if record["type"] == "directory":
                        shared_archive = (
                            shared_logs.collection_archive(artifact, collected_artifacts) if shared_logs else None
                        )
                        digest = await download_dir(
                            sandbox,
                            artifact.source,
                            target,
                            exclude=artifact.exclude,
                            exec_command=partial(environment.exec, service=artifact.service),
                            shared_archive=shared_archive,
                            metadata_path=metadata_path,
                        )
                        if shared_archive:
                            shared_logs.retain_archive(digest)
                    else:
                        record["exclude"] = []
                        await download_file(sandbox, artifact.source, target, metadata_path=metadata_path)
                    record["status"] = "ok"
                except Exception as exc:
                    record["status"] = "failed"
                    diagnostics.append({"operation": "collect_artifact", "source": artifact.source, "error": str(exc)})
            entries.append(record)
            (directory / "manifest.json").write_text(json.dumps(entries, indent=2))

    await hooks(True)
    await artifacts(True)
    sidecars = {a.service for a in task.artifacts if a.service not in (None, "main")}
    sidecars |= {hook.service for hook in task.verifier.collect if hook.service != "main"}
    if sidecars:
        try:
            await environment.stop_main()
        except Exception as exc:
            diagnostics.append({"operation": "stop_main", "error": str(exc)})
        await hooks(False)
        await artifacts(False)
    return entries
