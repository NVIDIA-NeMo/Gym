#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 FAILED_JOB_LOG" >&2
    exit 2
fi

readonly failed_job_log="$1"
if [[ ! -r "${failed_job_log}" ]]; then
    echo "Failed job log is not readable: ${failed_job_log}" >&2
    exit 2
fi

# Keep this list limited to runner and transport failures. Product assertions, dependency
# conflicts, and other deterministic failures must remain visible rather than being retried.
readonly retryable_pattern='(server certificate verification failed|x509: certificate signed by unknown authority|SSL certificate problem|Could not resolve host|Temporary failure in name resolution|Connection reset by peer|Connection timed out|TLS handshake timeout|No space left on device|Insufficient runner disk space|The (self-hosted |hosted )?runner lost communication|The runner has received a shutdown signal)'

if LC_ALL=C grep -Eiq -- "${retryable_pattern}" "${failed_job_log}"; then
    echo "Retrying failed jobs after a recognized runner or transport failure."
    exit 0
fi

echo "Failed jobs do not contain a recognized retryable infrastructure failure."
exit 1
