#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

: 'Check CODEOWNERS team approvals from a trusted target-branch checkout.'
set -euo pipefail

provider=${1:?Usage: check-review-approvals.sh github PR_NUMBER | gitlab}
github_api=https://api.github.com
tmp=$(mktemp -d)
trap 'rm -r -- "$tmp"' EXIT

api() {
    : 'Call either provider using its CI token.'
    local token=${GH_TOKEN:?GH_TOKEN needs GitHub team membership access}
    if [[ $1 != "$github_api/"* ]]; then token=${GITLAB_TOKEN:?}; fi
    curl --fail --silent --show-error --max-time 30 -H "Authorization: Bearer $token" "$@"
}

pages() {
    : 'Read every page of a provider list as JSON lines.'
    local page=1 response
    while :; do
        response=$(api "$1?per_page=100&page=$page")
        jq -e 'type == "array"' <<< "$response" > /dev/null
        jq -c '.[]' <<< "$response"
        [[ $(jq length <<< "$response") == 100 ]] || break
        ((page += 1))
    done
}

publish() {
    : 'Publish the verdict on the exact request head and, on GitLab, its pipeline.'
    if [[ $provider == github ]]; then
        jq -n --arg state "$1" --arg url "$run_url" \
            '{state:$state, context:"Package approvals", target_url:$url}' > "$tmp/status.json"
    else
        jq -n --arg state "$1" --arg url "$run_url" --argjson pipeline "$pipeline" \
            '{state:$state, name:"Package approvals", target_url:$url, pipeline_id:$pipeline}' > "$tmp/status.json"
    fi
    api "$status_url" --request POST -H 'Content-Type: application/json' --data-binary "@$tmp/status.json" > /dev/null
}

if [[ $provider == github ]]; then
    repository=${GITHUB_REPOSITORY:-NVIDIA-NeMo/Gym}
    request_url="$github_api/repos/$repository/pulls/${2:?PR number required}"
    request=$(api "$request_url")
    head_sha=$(jq -er '.head.sha' <<< "$request")
    author=$(jq -er '.user.id' <<< "$request")
    status_url="$github_api/repos/$repository/statuses/$head_sha"
    failure_state=failure
    run_url="${GITHUB_SERVER_URL:-https://github.com}/$repository/actions/runs/${GITHUB_RUN_ID:-}"
else
    [[ $provider == gitlab ]]
    number=${CI_MERGE_REQUEST_IID:-$(jq -er 'select(.object_kind == "merge_request") | .object_attributes.iid' "${TRIGGER_PAYLOAD:?}")}
    project_url="${CI_API_V4_URL:?}/projects/${CI_PROJECT_ID:?}"
    request_url="$project_url/merge_requests/$number"
    for ((attempt=0; attempt<12; attempt++)); do
        request=$(api "$request_url")
        pipeline=$(jq -r 'select(.head_pipeline.sha == .sha) | .head_pipeline.id // empty' <<< "$request")
        [[ -n $pipeline ]] && break
        sleep 5
    done
    [[ -n $pipeline ]]
    jq -e --arg branch "${CI_COMMIT_BRANCH:?}" '.target_branch == $branch' <<< "$request" > /dev/null
    head_sha=$(jq -er '.sha' <<< "$request")
    author=$(jq -er '.author.id' <<< "$request")
    status_url="$project_url/statuses/$head_sha"
    failure_state=failed
    run_url=${CI_JOB_URL:-}
fi
code=0 state=success
trap '
    code=$?
    trap - EXIT
    if ((code != 0)); then state=$failure_state; fi
    publish "$state"
    rm -r -- "$tmp"
    exit "$code"
' EXIT
publish pending

if [[ $provider == github ]]; then
    pages "$request_url/files" > "$tmp/files.jsonl"
    [[ $(jq -s length "$tmp/files.jsonl") == "$(jq .changed_files <<< "$request")" ]]
    jq -j '.filename, "\u0000", (.previous_filename // empty | ., "\u0000")' "$tmp/files.jsonl" > "$tmp/paths"
    approved=$(pages "$request_url/reviews" | jq -sc --arg head "$head_sha" --argjson author "$author" '
        map(select(.state == "APPROVED" or .state == "CHANGES_REQUESTED" or .state == "DISMISSED")) |
        sort_by(.id) | group_by(.user.id) | map(last) |
        [.[] | select(.state == "APPROVED" and .commit_id == $head and .user.id != $author) | .user.id]')
else
    api "$project_url/approvals" | jq -e '.reset_approvals_on_push == true' > /dev/null
    jq -e '.detailed_merge_status != "checking" and .detailed_merge_status != "approvals_syncing"' <<< "$request" > /dev/null
    pages "$request_url/diffs" > "$tmp/files.jsonl"
    [[ $(jq -s length "$tmp/files.jsonl") == "$(jq -r .changes_count <<< "$request")" ]]
    jq -j '.old_path, "\u0000", .new_path, "\u0000"' "$tmp/files.jsonl" > "$tmp/paths"
    approved=$(api "$request_url/approvals" | jq -c --argjson links "${REVIEWER_IDENTITIES_JSON:?}" --argjson author "$author" '
        [.approved_by[].user.id | select(. != $author) as $id |
         $links[] | select(.gitlab_user_id == $id) | .github_user_id] | unique')
fi

git init -q "$tmp/match"
awk '/^# suggest: / {sub(/^# suggest: /, ""); hints[++h]=$0; next}
     /^[^#[:space:]]/ {native[++n]=$0}
     END {for (i=1;i<=h;i++) print hints[i]; for (i=1;i<=n;i++) print native[i]}' \
    .github/CODEOWNERS > "$tmp/rules"
owners_by_rule=('')
: > "$tmp/match/.git/info/exclude"
while read -r pattern owners; do
    if [[ $pattern != /* && ${pattern%/} != */* ]]; then pattern="**/$pattern"; fi
    if [[ $pattern == */ ]]; then pattern+='**'; fi
    printf '%s\n' "$pattern" >> "$tmp/match/.git/info/exclude"
    owners_by_rule+=("$owners")
    if [[ $pattern != */\*\* ]]; then
        printf '%s/**\n' "$pattern" >> "$tmp/match/.git/info/exclude"
        owners_by_rule+=("$owners")
    fi
done < "$tmp/rules"
printf '%s\n' '!*/' >> "$tmp/match/.git/info/exclude"
while IFS= read -r -d '' path; do
    parent=${path%/*}
    if [[ $parent == "$path" ]]; then parent=.; fi
    mkdir -p -- "$tmp/match/$parent"
done < "$tmp/paths"
git -C "$tmp/match" -c core.excludesFile=/dev/null check-ignore --no-index -v -z --stdin < "$tmp/paths" > "$tmp/matches"
mapfile -d '' -t matches < "$tmp/matches"
declare -A required=() rosters=()
for ((index=0; index<${#matches[@]}; index+=4)); do required[${matches[index+1]}]=1; done
result=0
for rule in "${!required[@]}"; do
    satisfied=false
    for team in ${owners_by_rule[rule]}; do
        [[ $team =~ ^@[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$ ]]
        if [[ -z ${rosters[$team]:-} ]]; then
            org_team=${team#@}
            rosters[$team]=$(pages "$github_api/orgs/${org_team%/*}/teams/${org_team#*/}/members" | jq -sc '[.[].id] | sort')
        fi
        members=${rosters[$team]}
        if [[ $members == '[2382532,54727607]' ]] || \
            jq -e --argjson approved "$approved" '(. - (. - $approved)) | length > 0' <<< "$members" > /dev/null; then
            satisfied=true
        fi
    done
    printf '%s: %s\n' "$satisfied" "${owners_by_rule[rule]}"
    if [[ $satisfied == false ]]; then result=1; fi
done
[[ $(api "$request_url" | jq -r '.head.sha // .sha') == "$head_sha" ]]
exit "$result"
