# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Origin: NVIDIA Agentic IF strict-v2.2; NVIDIA-authored code, Apache-2.0.
# Original file: run_scripts/validate_agentic_if_response_judge_v2_2.py
# First tracked package revision: df90e68292130bb21ffe8a023fc5d0a6e1be0878
# Public source archive (exact publication revision):
# https://github.com/ericyang0/agentic-if-benchmark/tree/b9b32049d48b3296614df71968a634c07779d5c7
# Archived file: resources_servers/agentic_if_realworld/benchmark/policy/validate_response_judge.py
#!/usr/bin/env python3
"""Binary response-judge contract validation; no model calls or label changes."""
import argparse
import hashlib
import json
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).parent / 'judge_output.schema.json'
TEXT_CODES = {'IF-FORMAT', 'IF-COVERAGE', 'IF-CITE', 'IF-LANG', 'IF-NEWINSTR',
              'IF-SCOPE', 'IF-LENGTH'}
NATIVE_CODES = TEXT_CODES | {'IF-TOOLPREF', 'IF-TOOLGROUND', 'IF-SCHEMA',
                             'IF-PLAN', 'IF-REASONACT'}
TEXT_ACTION_CODES = TEXT_CODES | {'IF-TOOLPREF', 'IF-PLAN', 'IF-REASONACT'}
SCOPES = {'native_replay': 'native_next_turn',
          'transcript_text_replay': 'transcript_text_constraint_proxy',
          'single_turn_text_replay': 'single_turn_text_observable_constraints'}
VERSION = 'agentic-if-generated-response-judge-v2.2-strict'


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def contract_sha(constraints):
    # Result receipts are computed after generation; definitions and rule choices
    # are frozen before it. Each result is independently input/response-bound.
    return sha([{k: v for k, v in constraint.items() if k != 'deterministic_check'}
                for constraint in constraints])


def aggregate(verdicts):
    verdicts = list(verdicts)
    if not verdicts or any(v not in {'pass', 'fail'} for v in verdicts):
        raise ValueError('Aggregation requires nonempty binary verdicts')
    return 'fail' if 'fail' in verdicts else 'pass'


def summarize_records(records, expected_item_ids):
    """Full declared denominator; incomplete runs cannot expose a complete accuracy."""
    expected = list(expected_item_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Expected item IDs must be nonempty and unique')
    observed = {}
    for record in records:
        identifier = record['item_id']
        if identifier not in expected or identifier in observed:
            raise ValueError('Unexpected or duplicate result item')
        status = record['scoring_status']
        if status == 'scored':
            if (record.get('validation_status') != 'valid' or
                    record.get('policy_version') != VERSION or
                    record.get('item_verdict') not in {'pass', 'fail'}):
                raise ValueError('Only validated strict binary grades may be scored')
            observed[identifier] = record['item_verdict']
        elif status == 'not_scored':
            if (record.get('model_verdict') is not None or
                    record.get('item_verdict') is not None or not record.get('reason')):
                raise ValueError('Not-scored items need a reason and no model verdict')
            observed[identifier] = 'not_scored'
        else:
            raise ValueError('Unsupported scoring status')
    passed = sum(v == 'pass' for v in observed.values())
    failed = sum(v == 'fail' for v in observed.values())
    unresolved = len(expected) - passed - failed
    return {'policy_version': VERSION, 'expected_items': len(expected),
            'pass': passed, 'fail': failed, 'not_scored': unresolved,
            'unattempted': len(expected) - len(observed),
            'scoring_coverage': (passed + failed) / len(expected),
            'complete_accuracy': passed / len(expected) if unresolved == 0 else None,
            'complete': unresolved == 0}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def input_strings(request, mode):
    # Do not ground support in arbitrary request settings or hash metadata.
    yield from strings(request['messages'])
    yield from strings(request.get('tools', []))
    if mode in {'transcript_text_replay', 'single_turn_text_replay'}:
        for message in request['messages']:
            content = message.get('content', '')
            marker = 'CAPTURED HISTORICAL CONTEXT (JSON):\n'
            if isinstance(content, str) and marker in content:
                yield from strings(json.loads(content.split(marker, 1)[1]))


def response_fields(message):
    if message.get('role') != 'assistant':
        raise ValueError('Response must be a delivered assistant message')
    content = message.get('content')
    if isinstance(content, list):
        if any(not isinstance(p, dict) or p.get('type') != 'text' or
               not isinstance(p.get('text'), str) for p in content):
            raise ValueError('Unsupported non-text response content')
        content = '\n'.join(p['text'] for p in content)
    for field, value in [('content', content), ('refusal', message.get('refusal')),
                         ('reasoning_content', message.get('reasoning_content'))]:
        if value is not None and not isinstance(value, str):
            raise ValueError(f'Unsupported response {field}')
    calls = message.get('tool_calls', [])
    if not isinstance(calls, list):
        raise ValueError('Unsupported tool-call transport')
    return {'content': content or '', 'refusal': message.get('refusal') or '',
            'tool_calls': encode(calls),
            'reasoning_content': message.get('reasoning_content') or ''}


def validate_payload(payload):
    """Mechanical preflight; eligibility flags require independent upstream review."""
    mode = payload['benchmark_mode']
    if mode not in SCOPES:
        raise ValueError('Unsupported benchmark mode')
    receipt = payload['inference_binding']
    request_hash = sha(payload['input_request'])
    response_hash = sha(payload['response'])
    if not (payload['bindings_verified'] is True and
            payload['input_sha256'] == request_hash == receipt['input_sha256'] and
            payload['response_sha256'] == response_hash == receipt['response_sha256'] and
            receipt['id'] == payload['item_id'] and
            receipt['benchmark_manifest_sha256'] == payload['benchmark_manifest_sha256']):
        raise ValueError('Technical/binding failure: no model verdict')
    response = payload['response']
    if response['status'] != 'complete':
        raise ValueError('Incomplete delivery: no model verdict')
    if response['finish_reason'] not in {'stop', 'tool_calls'}:
        raise ValueError('Unsupported or truncated finish: no model verdict')
    fields = response_fields(response['message'])
    if mode == 'single_turn_text_replay' and (
            payload['input_request'].get('tools') or response['message'].get('tool_calls')):
        raise ValueError('Single-turn text mode cannot assert native tool interaction')
    if response['finish_reason'] == 'tool_calls' and not response['message'].get('tool_calls'):
        raise ValueError('Tool-call finish without delivered tool calls')
    constraints = payload['constraints']
    if payload['constraint_contract_sha256'] != contract_sha(constraints):
        raise ValueError('Frozen constraint contract hash mismatch')
    if not isinstance(constraints, list) or not constraints:
        raise ValueError('Missing supplied constraints')
    supplied = {c['constraint_id']: c for c in constraints}
    if len(supplied) != len(constraints):
        raise ValueError('Duplicate supplied constraints')
    dependencies = {}
    for source in constraints:
        deps = source.get('depends_on', [])
        if (not isinstance(deps, list) or any(not isinstance(d, str) for d in deps)
                or len(set(deps)) != len(deps) or any(d not in supplied for d in deps)):
            raise ValueError('Dependencies must be unique supplied constraint IDs')
        dependencies[source['constraint_id']] = deps
    visiting, visited = set(), set()

    def visit(identifier):
        if identifier in visiting:
            raise ValueError('Constraint dependency cycle')
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in dependencies[identifier]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in supplied:
        visit(identifier)
    visible_input = list(input_strings(payload['input_request'], mode))
    allowed = (NATIVE_CODES if mode == 'native_replay' else
               TEXT_ACTION_CODES if mode == 'single_turn_text_replay' else TEXT_CODES)
    for source in constraints:
        if not isinstance(source['constraint_id'], str) or not source['constraint_id']:
            raise ValueError('Constraint ID must be a nonempty string')
        question = source.get('evaluation_question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Frozen evaluation question required')
        code = source['subtype']
        if code not in allowed:
            raise ValueError('Unsupported subtype in this replay mode')
        if source.get('eligibility_verified') is not True:
            raise ValueError('Constraint eligibility not verified')
        if mode == 'single_turn_text_replay' and (
                source.get('observation_scope') != 'delivered_text' or
                source.get('visibility_verified') is not True):
            raise ValueError('Single-turn text assertions need verified delivered-text visibility')
        origin = source.get('support_origin', 'input')
        if origin not in {'input', 'response_reasoning'}:
            raise ValueError('Unsupported support origin')
        if origin == 'response_reasoning' and code not in {'IF-PLAN', 'IF-REASONACT'}:
            raise ValueError('Own reasoning cannot restore an external instruction')
        support_fields = visible_input if origin == 'input' else [fields['reasoning_content']]
        quotes = source['support_quotes']
        if not isinstance(quotes, list) or not quotes or any(
                not isinstance(q, str) or not q or not any(q in text for text in support_fields)
                for q in quotes):
            raise ValueError('Support quote not grounded in its declared source')
        if code in {'IF-TOOLPREF', 'IF-TOOLGROUND', 'IF-SCHEMA'}:
            if mode == 'single_turn_text_replay':
                registry = payload.get('text_tool_registry')
                quote = payload.get('text_tool_registry_quote')
                if (not isinstance(registry, list) or not isinstance(quote, str) or not quote or
                        not any(quote in text for text in visible_input) or
                        json.loads(quote) != registry or
                        payload.get('text_tool_registry_sha256') != sha(registry)):
                    raise ValueError('Original embedded text tool registry must be exactly grounded')
            elif not isinstance(payload['input_request'].get('tools'), list):
                raise ValueError('Current tool registry missing')
        if code == 'IF-SCHEMA' and not payload['input_request']['tools']:
            raise ValueError('IF-SCHEMA requires a registered function schema')
        check = source.get('deterministic_check')
        rule = source.get('deterministic_rule')
        if rule is not None and check is None:
            raise ValueError('Frozen deterministic rule requires a bound result receipt')
        if check is not None:
            if (not isinstance(rule, dict) or
                    rule.get('rule_id') != check['rule_id'] or
                    rule.get('rule_version') != check['rule_version']):
                raise ValueError('Deterministic check must match the frozen rule')
            if (check['verdict'] not in {'pass', 'fail'} or
                    not isinstance(check['rule_id'], str) or not check['rule_id'] or
                    not isinstance(check['rule_version'], str) or not check['rule_version'] or
                    check['input_sha256'] != request_hash or
                    check['response_sha256'] != response_hash):
                raise ValueError('Invalid or stale deterministic check')
    return supplied, fields


def validate_review(review, payload):
    supplied, fields = validate_payload(payload)
    jsonschema.validate(review, json.loads(SCHEMA_PATH.read_text()))
    if review['item_id'] != payload['item_id'] or review['benchmark_mode'] != payload['benchmark_mode']:
        raise ValueError('Item/mode mismatch')
    for key in ('input_sha256', 'response_sha256', 'constraint_contract_sha256',
                'benchmark_manifest_sha256'):
        if review[key] != payload[key]:
            raise ValueError('Review binding mismatch: ' + key)
    mode = payload['benchmark_mode']
    scope = SCOPES[mode]
    if review['scoring_scope'] != scope:
        raise ValueError('Replay scope mismatch')
    returned = {c['constraint_id']: c for c in review['constraints']}
    if len(returned) != len(review['constraints']) or set(returned) != set(supplied):
        raise ValueError('Omitted, duplicate, or invented constraint')
    subtype_verdicts = {}
    for identifier, result in returned.items():
        source = supplied[identifier]
        violated = result['violation_established']
        if violated != (result['verdict'] == 'fail'):
            raise ValueError('Established violation must fail; full compliance must pass')
        severity = result.get('deviation_severity')
        if severity is not None and ((violated and severity == 'none') or
                                     (not violated and severity != 'none')):
            raise ValueError('Severity is diagnostic and cannot excuse a violation')
        if result['subtype'] != source['subtype']:
            raise ValueError('Invented or remapped subtype')
        if result['support_quote'] not in source['support_quotes'] or not result['support_quote']:
            raise ValueError('Support quote not from supplied contract')
        if result['support_origin'] != source.get('support_origin', 'input'):
            raise ValueError('Support origin mismatch')
        if result['verdict'] == 'pass' and any(
                returned[d]['verdict'] == 'fail' for d in source.get('depends_on', [])):
            raise ValueError('Dependent constraint passed despite a failed prerequisite')
        if source.get('deterministic_check') is not None:
            if result['verdict'] != source['deterministic_check']['verdict']:
                raise ValueError('Judge verdict conflicts with deterministic check')
        evidence = result['response_evidence']
        field = evidence['field']
        if mode == 'single_turn_text_replay' and field == 'tool_calls':
            raise ValueError('Single-turn text evidence must use delivered text, not native calls')
        quote = evidence['quote']
        if quote and quote not in fields[field]:
            raise ValueError('Evidence quote is not from the new response')
        if not evidence['expected'] or not evidence['observation']:
            raise ValueError('Evidence requires expected behavior and observation')
        kind = evidence['kind']
        if result['verdict'] == 'fail':
            if kind == 'conflicting_span':
                if not quote:
                    raise ValueError('Conflict requires an exact new response quote')
            elif kind == 'missing_required_element':
                if quote:
                    raise ValueError('Omission must not fabricate an absence quote')
            else:
                raise ValueError('Failure requires conflict or omission evidence')
        elif kind == 'satisfying_span':
            if not quote:
                raise ValueError('Pass requires a satisfying response quote')
        elif kind == 'absence_of_prohibited_element':
            if quote:
                raise ValueError('Absence must not fabricate a response quote')
        else:
            raise ValueError('Pass requires satisfaction or permitted absence evidence')
        subtype_verdicts.setdefault(result['subtype'], []).append(result['verdict'])
    verdict = aggregate(c['verdict'] for c in returned.values())
    if review['item_verdict'] != verdict:
        raise ValueError('Inconsistent item aggregation')
    return {'validation_status': 'valid', 'scoring_status': 'scored', 'policy_version': VERSION,
            'item_id': payload['item_id'], 'item_verdict': verdict,
            'benchmark_mode': mode, 'scoring_scope': scope,
            'subtype_verdicts': {k: aggregate(v) for k, v in subtype_verdicts.items()},
            'scored_constraints': len(returned),
            'passed_constraints': sum(c['verdict'] == 'pass' for c in returned.values()),
            'failed_constraints': sum(c['verdict'] == 'fail' for c in returned.values()),
            'semantic_or_human_validation_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--review', type=Path)
    group.add_argument('--check-payload', action='store_true')
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload.read_text())
        if args.check_payload:
            supplied, _ = validate_payload(payload)
            result = {'validation_status': 'eligible', 'policy_version': VERSION,
                      'item_id': payload['item_id'], 'model_verdict': None,
                      'constraints': len(supplied), 'semantic_or_human_validation_performed': False}
        else:
            result = validate_review(json.loads(args.review.read_text()), payload)
    except (ValueError, KeyError, TypeError, AttributeError, OSError,
            jsonschema.ValidationError) as error:
        print(json.dumps({'scoring_status': 'not_scored', 'model_verdict': None,
                          'policy_version': VERSION,
                          'error_stage': 'admission' if args.check_payload else 'admission_or_review',
                          'error': str(error).splitlines()[0]}, ensure_ascii=False))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
