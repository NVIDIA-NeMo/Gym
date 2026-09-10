# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign tools for AutomationBench."""

from automationbench.tools.zapier.docusign.envelope import (
    docusign_add_envelope_cc,
    docusign_add_envelope_signer,
    docusign_create_envelope,
    docusign_create_envelope_from_template,
    docusign_create_signature_request,
    docusign_find_envelope_info,
    docusign_find_envelope_recipients,
    docusign_list_envelopes,
    docusign_send_envelope,
    docusign_void_envelope,
)
from automationbench.tools.zapier.docusign.template import (
    docusign_find_template,
    docusign_find_template_documents,
    docusign_list_templates,
)
from automationbench.tools.zapier.docusign.workspace import (
    docusign_add_user_to_workspace,
    docusign_create_workspace,
    docusign_find_workspaces,
)

__all__ = [
    # Envelope tools
    "docusign_create_signature_request",
    "docusign_add_envelope_cc",
    "docusign_add_envelope_signer",
    "docusign_create_envelope",
    "docusign_create_envelope_from_template",
    "docusign_list_envelopes",
    "docusign_send_envelope",
    "docusign_void_envelope",
    "docusign_find_envelope_info",
    "docusign_find_envelope_recipients",
    # Template tools
    "docusign_find_template",
    "docusign_find_template_documents",
    "docusign_list_templates",
    # Workspace tools
    "docusign_create_workspace",
    "docusign_find_workspaces",
    "docusign_add_user_to_workspace",
]
