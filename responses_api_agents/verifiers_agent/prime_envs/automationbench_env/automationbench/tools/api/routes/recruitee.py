# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Recruitee REST API routing layer for api_fetch.

Dispatches incoming REST-style Recruitee requests to their matching implementation
functions. Modeled after the Recruitee API (https://docs.recruitee.com).
"""

from automationbench.tools.api.impl.recruitee import (
    recruitee_candidates_apply,
    recruitee_candidates_create,
    recruitee_candidates_custom_fields_create,
    recruitee_candidates_delete,
    recruitee_candidates_delete_cover_letter,
    recruitee_candidates_delete_cv,
    recruitee_candidates_get,
    recruitee_candidates_list,
    recruitee_candidates_notes_create,
    recruitee_candidates_notes_delete,
    recruitee_candidates_notes_list,
    recruitee_candidates_search,
    recruitee_candidates_update,
    recruitee_candidates_update_cv,
    recruitee_departments_create,
    recruitee_departments_delete,
    recruitee_departments_list,
    recruitee_interview_events_create,
    recruitee_interview_events_delete,
    recruitee_interview_events_list,
    recruitee_interview_events_update,
    recruitee_interview_sub_delete,
    recruitee_interview_templates_delete,
    recruitee_interview_templates_list,
    recruitee_locations_delete,
    recruitee_locations_list,
    recruitee_offers_create,
    recruitee_offers_delete,
    recruitee_offers_get,
    recruitee_offers_list,
    recruitee_offers_update,
    recruitee_placements_create,
    recruitee_placements_delete,
    recruitee_placements_update,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Offers
    ("GET", r"recruitee/v1/c/([^/]+)/offers$", "offers_list"),
    ("POST", r"recruitee/v1/c/([^/]+)/offers$", "offers_create"),
    ("GET", r"recruitee/v1/c/([^/]+)/offers/([^/]+)$", "offers_get"),
    ("PATCH", r"recruitee/v1/c/([^/]+)/offers/([^/]+)$", "offers_update"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/offers/([^/]+)$", "offers_delete"),
    # Candidates
    ("GET", r"recruitee/v1/c/([^/]+)/candidates$", "candidates_list"),
    ("POST", r"recruitee/v1/c/([^/]+)/candidates$", "candidates_create"),
    ("GET", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)$", "candidates_get"),
    ("PATCH", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)$", "candidates_update"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)$", "candidates_delete"),
    # Candidate sub-resources
    ("PATCH", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)/update_cv$", "candidates_update_cv"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)/delete_cv$", "candidates_delete_cv"),
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/candidates/([^/]+)/delete_cover_letter$",
        "candidates_delete_cover_letter",
    ),
    ("GET", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)/notes$", "candidates_notes_list"),
    ("POST", r"recruitee/v1/c/([^/]+)/candidates/([^/]+)/notes$", "candidates_notes_create"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/notes/([^/]+)$", "candidates_notes_delete"),
    (
        "POST",
        r"recruitee/v1/c/([^/]+)/custom_fields/candidates/([^/]+)/fields$",
        "candidates_custom_fields_create",
    ),
    ("GET", r"recruitee/v1/c/([^/]+)/search/new/candidates$", "candidates_search"),
    # Candidate apply (careers site)
    ("POST", r"recruitee/v1/offers/([^/]+)/candidates$", "candidates_apply"),
    # Placements
    ("POST", r"recruitee/v1/c/([^/]+)/placements$", "placements_create"),
    ("PATCH", r"recruitee/v1/c/([^/]+)/placements/([^/]+)$", "placements_update"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/placements/([^/]+)$", "placements_delete"),
    # Departments
    ("GET", r"recruitee/v1/c/([^/]+)/departments$", "departments_list"),
    ("POST", r"recruitee/v1/c/([^/]+)/departments$", "departments_create"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/departments/([^/]+)$", "departments_delete"),
    # Locations
    ("GET", r"recruitee/v1/c/([^/]+)/locations$", "locations_list"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/locations/([^/]+)$", "locations_delete"),
    # Interview Events
    ("GET", r"recruitee/v1/c/([^/]+)/interview/events$", "interview_events_list"),
    ("POST", r"recruitee/v1/c/([^/]+)/interview/events$", "interview_events_create"),
    ("PATCH", r"recruitee/v1/c/([^/]+)/interview/events/([^/]+)$", "interview_events_update"),
    ("DELETE", r"recruitee/v1/c/([^/]+)/interview/events/([^/]+)$", "interview_events_delete"),
    # Interview Templates
    ("GET", r"recruitee/v1/c/([^/]+)/interview/templates$", "interview_templates_list"),
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/interview/templates/([^/]+)$",
        "interview_templates_delete",
    ),
    # Interview sub-resource deletes
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/interview/schedules/([^/]+)$",
        "interview_schedules_delete",
    ),
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/interview/request_links/([^/]+)$",
        "interview_request_links_delete",
    ),
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/interview/result_requests/([^/]+)$",
        "interview_result_requests_delete",
    ),
    (
        "DELETE",
        r"recruitee/v1/c/([^/]+)/interview/meeting_rooms/([^/]+)$",
        "interview_meeting_rooms_delete",
    ),
]

_HANDLERS = {
    # Offers
    "offers_list": lambda w, ids, p, b: recruitee_offers_list(w, company_id=ids[0], **p),
    "offers_create": lambda w, ids, p, b: recruitee_offers_create(w, company_id=ids[0], **b),
    "offers_get": lambda w, ids, p, b: recruitee_offers_get(
        w, company_id=ids[0], offer_id=ids[1], **p
    ),
    "offers_update": lambda w, ids, p, b: recruitee_offers_update(
        w, company_id=ids[0], offer_id=ids[1], **b
    ),
    "offers_delete": lambda w, ids, p, b: recruitee_offers_delete(
        w, company_id=ids[0], offer_id=ids[1]
    ),
    # Candidates
    "candidates_list": lambda w, ids, p, b: recruitee_candidates_list(w, company_id=ids[0], **p),
    "candidates_create": lambda w, ids, p, b: recruitee_candidates_create(
        w, company_id=ids[0], **b
    ),
    "candidates_get": lambda w, ids, p, b: recruitee_candidates_get(
        w, company_id=ids[0], candidate_id=ids[1], **p
    ),
    "candidates_update": lambda w, ids, p, b: recruitee_candidates_update(
        w, company_id=ids[0], candidate_id=ids[1], **b
    ),
    "candidates_delete": lambda w, ids, p, b: recruitee_candidates_delete(
        w, company_id=ids[0], candidate_id=ids[1]
    ),
    # Candidate sub-resources
    "candidates_update_cv": lambda w, ids, p, b: recruitee_candidates_update_cv(
        w, company_id=ids[0], candidate_id=ids[1], **b
    ),
    "candidates_delete_cv": lambda w, ids, p, b: recruitee_candidates_delete_cv(
        w, company_id=ids[0], candidate_id=ids[1]
    ),
    "candidates_delete_cover_letter": lambda w, ids, p, b: recruitee_candidates_delete_cover_letter(
        w, company_id=ids[0], candidate_id=ids[1]
    ),
    "candidates_notes_list": lambda w, ids, p, b: recruitee_candidates_notes_list(
        w, company_id=ids[0], candidate_id=ids[1], **p
    ),
    "candidates_notes_create": lambda w, ids, p, b: recruitee_candidates_notes_create(
        w, company_id=ids[0], candidate_id=ids[1], **b
    ),
    "candidates_notes_delete": lambda w, ids, p, b: recruitee_candidates_notes_delete(
        w, company_id=ids[0], note_id=ids[1]
    ),
    "candidates_custom_fields_create": lambda w,
    ids,
    p,
    b: recruitee_candidates_custom_fields_create(w, company_id=ids[0], candidate_id=ids[1], **b),
    "candidates_search": lambda w, ids, p, b: recruitee_candidates_search(
        w, company_id=ids[0], **p
    ),
    # Candidate apply
    "candidates_apply": lambda w, ids, p, b: recruitee_candidates_apply(w, offer_slug=ids[0], **b),
    # Placements
    "placements_create": lambda w, ids, p, b: recruitee_placements_create(
        w, company_id=ids[0], **b
    ),
    "placements_update": lambda w, ids, p, b: recruitee_placements_update(
        w, company_id=ids[0], placement_id=ids[1], **b
    ),
    "placements_delete": lambda w, ids, p, b: recruitee_placements_delete(
        w, company_id=ids[0], placement_id=ids[1]
    ),
    # Departments
    "departments_list": lambda w, ids, p, b: recruitee_departments_list(w, company_id=ids[0], **p),
    "departments_create": lambda w, ids, p, b: recruitee_departments_create(
        w, company_id=ids[0], **b
    ),
    "departments_delete": lambda w, ids, p, b: recruitee_departments_delete(
        w, company_id=ids[0], department_id=ids[1]
    ),
    # Locations
    "locations_list": lambda w, ids, p, b: recruitee_locations_list(w, company_id=ids[0], **p),
    "locations_delete": lambda w, ids, p, b: recruitee_locations_delete(
        w, company_id=ids[0], location_id=ids[1]
    ),
    # Interview Events
    "interview_events_list": lambda w, ids, p, b: recruitee_interview_events_list(
        w, company_id=ids[0], **p
    ),
    "interview_events_create": lambda w, ids, p, b: recruitee_interview_events_create(
        w, company_id=ids[0], **b
    ),
    "interview_events_update": lambda w, ids, p, b: recruitee_interview_events_update(
        w, company_id=ids[0], event_id=ids[1], **b
    ),
    "interview_events_delete": lambda w, ids, p, b: recruitee_interview_events_delete(
        w, company_id=ids[0], event_id=ids[1]
    ),
    # Interview Templates
    "interview_templates_list": lambda w, ids, p, b: recruitee_interview_templates_list(
        w, company_id=ids[0]
    ),
    "interview_templates_delete": lambda w, ids, p, b: recruitee_interview_templates_delete(
        w, company_id=ids[0], template_id=ids[1]
    ),
    # Interview sub-resource deletes
    "interview_schedules_delete": lambda w, ids, p, b: recruitee_interview_sub_delete(
        w, company_id=ids[0], resource_type="schedule", resource_id=ids[1]
    ),
    "interview_request_links_delete": lambda w, ids, p, b: recruitee_interview_sub_delete(
        w, company_id=ids[0], resource_type="request_link", resource_id=ids[1]
    ),
    "interview_result_requests_delete": lambda w, ids, p, b: recruitee_interview_sub_delete(
        w, company_id=ids[0], resource_type="result_request", resource_id=ids[1]
    ),
    "interview_meeting_rooms_delete": lambda w, ids, p, b: recruitee_interview_sub_delete(
        w, company_id=ids[0], resource_type="meeting_room", resource_id=ids[1]
    ),
}


route_recruitee = make_router(_ROUTES, _HANDLERS)
