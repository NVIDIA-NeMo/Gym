# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer-scenario models and schema loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


SCENARIO_SCHEMA_PATH = Path(__file__).resolve().parent / "prompts" / "customer_scenario_collection_schema.json"


class CustomerScenario(BaseModel):
    """
    Customer scenario. A definition of a customer who is contacting a customer service representative. This includes a general description of the customer, and information about the specific situation the customer is in and the tasks they are trying to complete.
    """

    customer_persona: str = Field(
        description=(
            "Customer's persona. This information defines the customer in general, "
            "not the specific situation they are in."
        ),
        validation_alias=AliasChoices("customer_persona", "persona"),
    )
    reason_for_contact: str = Field(
        description=("The reason for the customer to contact the customer service representative."),
        validation_alias=AliasChoices("reason_for_contact", "reason_for_call"),
    )
    customer_details: str = Field(
        description=(
            "Specific details about the customer that are relevant to the "
            "customer's tasks. This should be information that the customer "
            "can provide to the representative when the representative asks "
            "for details about the customer's tasks."
        )
    )
    # Keep the field nullable without adding a literal default to the generated schema.
    unknown_info: Optional[str] = Field(default_factory=lambda: None)
    task_instructions: str = Field(
        description=(
            "Instructions for the customer about the specific situation the "
            "customer is in, the tasks they are trying to complete, and how to "
            "interact with the customer service representative."
        )
    )
    representative_domain: SkipJsonSchema[Optional[str]] = Field(
        description="The domain of the customer service representative.",
        default=None,
    )
    outside_policy_scope: SkipJsonSchema[Optional[bool]] = Field(
        description=(
            "A value that is true if the action that the customer service "
            "representative should take in response to the customer's request is "
            "not covered in the policy (and the representative should transfer the "
            "customer to a human agent), or false if the action that should be "
            "taken is covered in the policy."
        ),
        default=None,
    )

    def create_tuple(self) -> tuple[str, str, str, Optional[str], str]:
        return (
            self.customer_persona,
            self.reason_for_contact,
            self.customer_details,
            self.unknown_info,
            self.task_instructions,
        )


class CustomerScenarioCollection(BaseModel):
    """
    Customer scenario collection. A collection of customer scenarios.
    """

    scenarios: list[CustomerScenario] = Field(
        description="An array that contains the customer scenarios in the collection."
    )


def generated_scenario_schema_json() -> str:
    return json.dumps(CustomerScenarioCollection.model_json_schema())


def scenario_schema_json() -> str:
    return SCENARIO_SCHEMA_PATH.read_text(encoding="utf-8")
