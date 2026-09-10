# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Canva tools for AutomationBench.

Tools map 1:1 with Zapier Canva actions.
"""

from automationbench.tools.zapier.canva.assets import (
    canva_create_asset_upload_job,
    canva_get_asset_upload_job,
    canva_get_design_autofill_job,
    canva_get_design_export_job,
    canva_get_design_import_job,
)
from automationbench.tools.zapier.canva.designs import (
    canva_create_design,
    canva_create_design_autofill_job,
    canva_create_design_export_job,
    canva_create_design_import_job,
    canva_export_design,
    canva_find_design,
    canva_list_brand_templates,
    canva_list_designs,
    canva_move_folder_item,
    canva_new_design,
)

__all__ = [
    # Design tools
    "canva_create_design",
    "canva_find_design",
    "canva_create_design_export_job",
    "canva_create_design_import_job",
    "canva_create_design_autofill_job",
    "canva_move_folder_item",
    "canva_export_design",
    "canva_list_designs",
    "canva_list_brand_templates",
    "canva_new_design",
    # Asset tools
    "canva_create_asset_upload_job",
    "canva_get_asset_upload_job",
    "canva_get_design_export_job",
    "canva_get_design_import_job",
    "canva_get_design_autofill_job",
]
