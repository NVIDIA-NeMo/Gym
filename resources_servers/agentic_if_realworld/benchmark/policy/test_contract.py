# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Origin: NVIDIA Agentic IF strict-v2.2; NVIDIA-authored code, Apache-2.0.
# Original file: tests/test_agentic_if_response_judge_v2_2.py
# First tracked package revision: df90e68292130bb21ffe8a023fc5d0a6e1be0878
"""Synthetic strict-scoring contracts; no source traces or empirical judge claims."""
import copy
import json
from pathlib import Path

import jsonschema
import pytest

from run_scripts import validate_agentic_if_response_judge_v2_2 as judge


def rebind(payload, review):
    payload['input_sha256'] = judge.sha(payload['input_request'])
    payload['response_sha256'] = judge.sha(payload['response'])
    payload['constraint_contract_sha256'] = judge.contract_sha(payload['constraints'])
    payload['inference_binding'] = {
        'id': payload['item_id'], 'input_sha256': payload['input_sha256'],
        'response_sha256': payload['response_sha256'],
        'benchmark_manifest_sha256': payload['benchmark_manifest_sha256']}
    for key in ('input_sha256', 'response_sha256', 'constraint_contract_sha256',
                'benchmark_manifest_sha256'):
        review[key] = payload[key]


def case():
    payload = {
        'item_id': 'synthetic-strict', 'benchmark_mode': 'native_replay',
        'input_request': {'messages': [{'role': 'user', 'content': 'Return exactly OK.'}]},
        'response': {'status': 'complete', 'finish_reason': 'stop',
                     'message': {'role': 'assistant', 'content': 'OK.'}},
        'bindings_verified': True, 'benchmark_manifest_sha256': judge.sha('synthetic-manifest'),
        'constraints': [{'constraint_id': 'literal-001', 'subtype': 'IF-FORMAT',
                         'support_quotes': ['Return exactly OK.'],
                         'evaluation_question': 'Does the complete output equal OK exactly?',
                         'eligibility_verified': True, 'support_origin': 'input'}]}
    review = {
        'policy_version': judge.VERSION, 'item_id': payload['item_id'],
        'benchmark_mode': 'native_replay', 'scoring_scope': 'native_next_turn',
        'item_verdict': 'fail',
        'constraints': [{'constraint_id': 'literal-001', 'subtype': 'IF-FORMAT',
                         'support_quote': 'Return exactly OK.', 'support_origin': 'input',
                         'verdict': 'fail', 'violation_established': True,
                         'deviation_severity': 'minor',
                         'rationale': 'One extra period violates the exact literal.',
                         'response_evidence': {'kind': 'conflicting_span', 'field': 'content',
                                               'quote': 'OK.', 'expected': 'OK',
                                               'observation': 'An extra period was delivered.'}}]}
    rebind(payload, review)
    return payload, review


@pytest.mark.parametrize('severity', ['minor', 'major', 'not_assessed'])
def test_violation_fails_regardless_of_severity(severity):
    payload, review = case()
    review['constraints'][0]['deviation_severity'] = severity
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'


def test_severity_assessment_is_optional():
    payload, review = case()
    del review['constraints'][0]['deviation_severity']
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'


@pytest.mark.parametrize('verdict,established,severity', [
    ('pass', True, 'minor'), ('pass', False, 'minor'), ('fail', False, 'minor'),
    ('fail', True, 'none')])
def test_known_violation_cannot_be_waived_or_scored_inconsistently(verdict, established, severity):
    payload, review = case()
    review['constraints'][0].update(verdict=verdict, violation_established=established,
                                    deviation_severity=severity)
    review['item_verdict'] = verdict
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        judge.validate_review(review, payload)


def test_exact_compliance_passes():
    payload, review = case()
    payload['response']['message']['content'] = 'OK'
    result = review['constraints'][0]
    result.update(verdict='pass', violation_established=False, deviation_severity='none',
                  rationale='Exact literal matches.')
    result['response_evidence'].update(kind='satisfying_span', quote='OK',
                                      observation='Exact literal matches.')
    review['item_verdict'] = 'pass'
    rebind(payload, review)
    assert judge.validate_review(review, payload)['item_verdict'] == 'pass'


@pytest.mark.parametrize('field', ['input_sha256', 'response_sha256',
                                 'constraint_contract_sha256', 'benchmark_manifest_sha256'])
def test_stale_review_bindings_rejected(field):
    payload, review = case()
    review[field] = '0' * 64
    with pytest.raises(ValueError, match='Review binding mismatch'):
        judge.validate_review(review, payload)


def test_same_quote_does_not_allow_reuse_for_different_response():
    payload, review = case()
    stale = copy.deepcopy(review)
    payload['response']['message']['content'] = 'OK. More text.'
    rebind(payload, review)
    with pytest.raises(ValueError, match='Review binding mismatch'):
        judge.validate_review(stale, payload)


def test_changed_frozen_question_is_rejected():
    payload, review = case()
    payload['constraints'][0]['evaluation_question'] = 'Is this mostly compliant?'
    with pytest.raises(ValueError, match='contract hash mismatch'):
        judge.validate_review(review, payload)


@pytest.mark.parametrize('question', [None, '', '  '])
def test_missing_frozen_evaluation_question_blocks_admission(question):
    payload, review = case()
    payload['constraints'][0]['evaluation_question'] = question
    rebind(payload, review)
    with pytest.raises(ValueError, match='evaluation question'):
        judge.validate_payload(payload)


@pytest.mark.parametrize('verdict', ['unknown', 'needs_adjudication', 'partial'])
def test_no_third_model_verdict(verdict):
    payload, review = case()
    review['constraints'][0]['verdict'] = verdict
    with pytest.raises(jsonschema.ValidationError):
        judge.validate_review(review, payload)


@pytest.mark.parametrize('status,finish', [('missing_response', 'stop'),
                                        ('complete', 'length'), ('technical_error', None)])
def test_delivery_failures_have_no_model_grade(status, finish):
    payload, review = case()
    payload['response'].update(status=status, finish_reason=finish)
    rebind(payload, review)
    with pytest.raises(ValueError, match='delivery|truncated'):
        judge.validate_review(review, payload)


def test_partial_item_cannot_pass_by_omitting_one_requirement():
    payload, review = case()
    source = copy.deepcopy(payload['constraints'][0])
    source['constraint_id'] = 'literal-002'
    payload['constraints'].append(source)
    rebind(payload, review)
    with pytest.raises(ValueError, match='Omitted'):
        judge.validate_review(review, payload)


def test_deterministic_check_cannot_be_overruled_with_pass():
    payload, review = case()
    payload['constraints'][0]['deterministic_rule'] = {'rule_id': 'exact-literal', 'rule_version': '1'}
    payload['constraints'][0]['deterministic_check'] = {
        'verdict': 'fail', 'rule_id': 'exact-literal', 'rule_version': '1',
        'input_sha256': payload['input_sha256'], 'response_sha256': payload['response_sha256']}
    review['item_verdict'] = 'pass'
    result = review['constraints'][0]
    result.update(verdict='pass', violation_established=False, deviation_severity='none')
    result['response_evidence']['kind'] = 'satisfying_span'
    rebind(payload, review)
    with pytest.raises(ValueError, match='conflicts with deterministic'):
        judge.validate_review(review, payload)


def test_response_specific_check_does_not_change_frozen_definition_hash():
    payload, review = case()
    source = payload['constraints'][0]
    source['deterministic_rule'] = {'rule_id': 'exact-literal', 'rule_version': '1'}
    rebind(payload, review)
    frozen = payload['constraint_contract_sha256']
    source['deterministic_check'] = {
        'verdict': 'fail', 'rule_id': 'exact-literal', 'rule_version': '1',
        'input_sha256': payload['input_sha256'], 'response_sha256': payload['response_sha256']}
    assert judge.contract_sha(payload['constraints']) == frozen
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'


def test_undeclared_or_changed_deterministic_rule_is_rejected():
    payload, review = case()
    payload['constraints'][0]['deterministic_check'] = {
        'verdict': 'fail', 'rule_id': 'post-hoc-counter', 'rule_version': '1',
        'input_sha256': payload['input_sha256'], 'response_sha256': payload['response_sha256']}
    with pytest.raises(ValueError, match='frozen rule'):
        judge.validate_review(review, payload)


def test_predeclared_deterministic_check_cannot_be_silently_omitted():
    payload, review = case()
    payload['constraints'][0]['deterministic_rule'] = {'rule_id': 'exact-literal', 'rule_version': '1'}
    rebind(payload, review)
    with pytest.raises(ValueError, match='requires a bound result'):
        judge.validate_review(review, payload)


def test_forbidden_native_tool_call_fails_without_tool_execution():
    payload, review = case()
    instruction = 'Do not call erase.'
    payload['input_request']['messages'][0]['content'] = instruction
    payload['input_request']['tools'] = [{'type': 'function', 'function': {
        'name': 'erase', 'parameters': {'type': 'object', 'properties': {}}}}]
    calls = [{'id': 'call-1', 'type': 'function', 'function': {'name': 'erase', 'arguments': '{}'}}]
    payload['response'].update(finish_reason='tool_calls', message={
        'role': 'assistant', 'content': None, 'tool_calls': calls})
    payload['constraints'][0].update(subtype='IF-TOOLPREF', support_quotes=[instruction],
                                    evaluation_question='Did the response avoid calling erase?')
    result = review['constraints'][0]
    result.update(subtype='IF-TOOLPREF', support_quote=instruction)
    result['response_evidence'].update(field='tool_calls', quote=judge.encode(calls),
                                      expected='No erase call', observation='Forbidden call emitted.')
    rebind(payload, review)
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'


def test_unassessable_constraint_blocks_whole_item_outside_judge_json():
    payload, review = case()
    payload['constraints'][0]['eligibility_verified'] = False
    rebind(payload, review)
    with pytest.raises(ValueError, match='eligibility'):
        judge.validate_review(review, payload)


def test_cli_operational_rejection_never_becomes_model_failure(tmp_path, monkeypatch, capsys):
    payload, review = case()
    payload['response']['status'] = 'missing_response'
    rebind(payload, review)
    path = tmp_path / 'payload.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setattr('sys.argv', ['validator', '--payload', str(path), '--check-payload'])
    with pytest.raises(SystemExit) as error:
        judge.main()
    assert error.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result['model_verdict'] is None and result['scoring_status'] == 'not_scored'


def test_all_twelve_definitions_preserved_and_strict_threshold_explicit():
    root = Path(__file__).resolve().parents[1]
    for path in [root / 'docs/agentic_if_response_judge_v2_2.md',
                 root / 'run_scripts/schemas/agentic_if_response_judge_v2_2.md']:
        text = path.read_text()
        sections = {line.split(' — ')[0].removeprefix('### ') for line in text.splitlines()
                    if line.startswith('### IF-')}
        assert sections == judge.NATIVE_CODES
        assert 'including a minor, local or incidental violation' in text
        assert 'Source admission and response scoring are separate decisions' in text


def test_incomplete_grading_does_not_publish_complete_accuracy():
    payload, review = case()
    grade = judge.validate_review(review, payload)
    summary = judge.summarize_records([grade], [payload['item_id'], 'missing'])
    assert summary['complete_accuracy'] is None
    assert summary['expected_items'] == 2 and summary['not_scored'] == 1
    assert summary['unattempted'] == 1 and summary['scoring_coverage'] == 0.5
    operational = {'item_id': 'missing', 'scoring_status': 'not_scored',
                   'model_verdict': None, 'reason': 'delivery_timeout'}
    summary = judge.summarize_records([grade, operational], [payload['item_id'], 'missing'])
    assert summary['complete_accuracy'] is None and summary['unattempted'] == 0


def test_complete_binary_grading_allows_point_accuracy():
    payload, review = case()
    grade = judge.validate_review(review, payload)
    summary = judge.summarize_records([grade], [payload['item_id']])
    assert summary['complete'] is True and summary['complete_accuracy'] == 0.0


def test_duplicate_or_technical_failure_cannot_inflate_score():
    payload, review = case()
    grade = judge.validate_review(review, payload)
    with pytest.raises(ValueError, match='duplicate'):
        judge.summarize_records([grade, grade], [payload['item_id']])
    operational = {'item_id': payload['item_id'], 'scoring_status': 'not_scored',
                   'model_verdict': 'fail', 'reason': 'delivery_timeout'}
    with pytest.raises(ValueError, match='no model verdict'):
        judge.summarize_records([operational], [payload['item_id']])


def text_action_case():
    payload, review = case()
    payload['benchmark_mode'] = review['benchmark_mode'] = 'single_turn_text_replay'
    review['scoring_scope'] = 'single_turn_text_observable_constraints'
    payload['constraints'][0].update(observation_scope='delivered_text', visibility_verified=True)
    rebind(payload, review)
    return payload, review


def test_single_turn_text_checks_do_not_claim_native_replay():
    payload, review = text_action_case()
    result = judge.validate_review(review, payload)
    assert result['item_verdict'] == 'fail'
    assert result['scoring_scope'] == 'single_turn_text_observable_constraints'


@pytest.mark.parametrize('key,value', [('visibility_verified', False),
                                     ('observation_scope', 'unseen_execution')])
def test_text_action_visibility_is_required(key, value):
    payload, review = text_action_case()
    payload['constraints'][0][key] = value
    rebind(payload, review)
    with pytest.raises(ValueError, match='visibility'):
        judge.validate_review(review, payload)


def test_text_tool_policy_requires_exact_embedded_original_registry():
    payload, review = text_action_case()
    registry = [{'type': 'function', 'function': {
        'name': 'erase', 'parameters': {'type': 'object', 'properties': {}}}}]
    quote = json.dumps(registry, indent=2)
    instruction = 'Do not call erase.'
    payload['input_request']['messages'][0]['content'] = instruction + '\nTools:\n' + quote
    payload.update(text_tool_registry=registry, text_tool_registry_quote=quote,
                   text_tool_registry_sha256=judge.sha(registry))
    payload['response']['message']['content'] = '{"name":"erase","arguments":{}}'
    payload['constraints'][0].update(subtype='IF-TOOLPREF', support_quotes=[instruction],
                                    evaluation_question='Does the emitted action avoid erase?')
    result = review['constraints'][0]
    result.update(subtype='IF-TOOLPREF', support_quote=instruction)
    result['response_evidence'].update(quote=payload['response']['message']['content'],
                                      expected='No erase action', observation='Forbidden action in text.')
    rebind(payload, review)
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'
    payload['text_tool_registry'] = []
    with pytest.raises(ValueError, match='registry'):
        judge.validate_review(review, payload)


@pytest.mark.parametrize('subtype', ['IF-SCHEMA', 'IF-TOOLGROUND'])
def test_text_mode_does_not_unlock_native_only_assertions(subtype):
    payload, review = text_action_case()
    payload['constraints'][0]['subtype'] = subtype
    review['constraints'][0]['subtype'] = subtype
    rebind(payload, review)
    with pytest.raises(ValueError, match='Unsupported subtype'):
        judge.validate_review(review, payload)


@pytest.mark.parametrize('subtype', ['IF-PLAN', 'IF-REASONACT'])
def test_text_mode_can_check_visible_recorded_commitments(subtype):
    payload, review = text_action_case()
    commitment = 'I will output exactly OK in the next message.'
    payload['input_request']['messages'].insert(0, {'role': 'assistant', 'content': commitment})
    payload['constraints'][0].update(subtype=subtype, support_quotes=[commitment])
    review['constraints'][0].update(subtype=subtype, support_quote=commitment)
    rebind(payload, review)
    assert judge.validate_review(review, payload)['item_verdict'] == 'fail'


def test_native_tool_calls_cannot_be_smuggled_into_text_mode():
    payload, review = text_action_case()
    payload['response']['message']['tool_calls'] = [{'id': 'unexpected-native-call'}]
    rebind(payload, review)
    with pytest.raises(ValueError, match='native tool interaction'):
        judge.validate_review(review, payload)
