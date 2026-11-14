# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only document identities explicitly supplied by the pinned task rubric.

The factual ID/filename pairs below are transcribed from the parenthesized
filenames in ``match_criteria`` in the MIT-licensed Harvey task data:
https://github.com/harveyai/harvey-labs/blob/
9e9fa7fe126103de1fceb2227b7c70bf59b67f73/tasks/litigation-dispute-resolution/
categorize-document-production-set-by-relevance-and-privilege/task.json

Only explicitly identified documents are included. The rubric does not identify
DOC_004, DOC_006, or DOC_021 by filename, so their identities are not guessed.
"""

CORRECTION_VERSION = "production-document-identities-v1"
SOURCE_REVISION = "9e9fa7fe126103de1fceb2227b7c70bf59b67f73"  # pragma: allowlist secret
TASK_ID = "litigation-dispute-resolution/categorize-document-production-set-by-relevance-and-privilege"

_DOCUMENT_IDS = {
    "DOC_001": "vasquez-env-compliance-memo.docx",
    "DOC_002": "chandrasekaran-ack-email.eml",
    "DOC_003": "vasquez-pip.docx",
    "DOC_005": "vasquez-termination-letter.docx",
    "DOC_007": "simmons-vasquez-situation-email.eml",
    "DOC_008": "chandrasekaran-burrell-litrisik-email.eml",
    "DOC_009": "alderwood-engagement-letter.docx",
    "DOC_010": "alderwood-draft-audit-report.docx",
    "DOC_011": "vasquez-separation-agreement.docx",
    "DOC_012": "chandrasekaran-meeting-notes-aug28.docx",
    "DOC_013": "plaintiff-demand-letter.docx",
    "DOC_014": "litigation-hold-notice.docx",
    "DOC_015": "chandrasekaran-fwd-burrell-memo.eml",
    "DOC_016": "vasquez-performance-summary.docx",
    "DOC_017": "vasquez-invention-assignment.docx",
    "DOC_018": "mpc7x-original-patent.docx",
    "DOC_019": "mpc7x-continuation-app.docx",
    "DOC_020": "inventor-list-email-chain.eml",
    "DOC_022": "burrell-stanton-joint-defense.eml",
    "DOC_023": "vasquez-benefits-enrollment.docx",
    "DOC_024": "simmons-bellworth-ops-metrics.eml",
    "DOC_025": "bellworth-reply-adding-gc.eml",
}


def document_id_aliases(*, task_id: str | None, source_revision: str | None) -> dict[str, str]:
    """Return verified identities only for the task and revision that define them."""
    if task_id == TASK_ID and source_revision == SOURCE_REVISION:
        return _DOCUMENT_IDS.copy()
    return {}
