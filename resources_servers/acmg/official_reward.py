# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored byte for byte from the `nvidia/acmg_variant_benchmark` Hugging Face dataset, file `training/reward.py`,
# revision 59fb412e7e39e8b209f431fce522458587ca8fec. Upstream file SHA-256:
# e2a80a9d3d0cc5051d73c9d6ea6245e4e2272ea5ef00cb6375a978b152666f1f
# Source: https://huggingface.co/datasets/nvidia/acmg_variant_benchmark/blob/59fb412e7e39e8b209f431fce522458587ca8fec/training/reward.py
# Authored in https://gitlab-master.nvidia.com/caof/acmg_classification_benchmark as `scripts/acmg_reward.py`; byte-identical
# at commit 94dd961 ("Reward: add the evidence score and the rules scheme").
# The dataset is NVIDIA-authored and private; its card declares `license: other`. This file is excluded from ruff in
# pyproject.toml so the checksum keeps holding, and resources_servers/acmg/tests/test_scoring.py verifies it.
# Do not edit anything below the marker line; change `scoring.py` instead.
# --- upstream file begins below, byte for byte ---
#!/usr/bin/env python3
"""Standalone reward for ACMG/AMP variant classification answers (no other project imports).

A response reasons in prose and ends with two lines, read tolerantly (markdown, `PM2_Supporting` or
`PM2 (Supporting)`, full label names, or a JSON answer):

    CRITERIA: <code> <strength>; <code> <strength>; ...
    CLASSIFICATION: <P, LP, VUS, LB or B>

`reward(response, reference, arm)` scores one response against a reference from `training/rl.jsonl` (or a
benchmark answer key) and returns a dict of components in [0, 1]. `None` marks a component that does not
apply (a reference whose label is not scored). `scalar()` combines them into one number, under one of two schemes:

- default: 0.5 label + 0.3 criterion F1 + 0.2 self-consistency (or any `weights`).
- rules: rule_gate x (0.4 label + 0.6 evidence_score). evidence_score = max(0, 1 - d / 8), where d is the distance
  between the returned criteria and the key in ACMG/AMP evidence points (Supporting 1, Moderate 2, Strong 4, Very
  Strong or stand-alone 8; a criterion the key does not accept counts double). rule_gate is 1 when the label follows
  from the response's own criteria under the rules and 0.25 otherwise. A right label reached with the wrong evidence
  earns little; the right evidence combined correctly earns close to 1.

From the command line (`python acmg_reward.py --help`):

    python acmg_reward.py --references data/answers.jsonl --responses responses.jsonl --summary
    python acmg_reward.py --references training/rl.jsonl --responses samples.jsonl --output scores.jsonl
    python acmg_reward.py --references training/rl.jsonl --responses samples.jsonl --scheme rules --summary

Mirrors the scoring in scripts/acmg_anchor_benchmark.py and scripts/acmg_real_anchor.py.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

LABELS = ('P', 'LP', 'VUS', 'LB', 'B')
POINTS = {'Supporting': 1, 'Moderate': 2, 'Strong': 4, 'VeryStrong': 8}
EVIDENCE_POINTS = {**POINTS, 'StandAlone': 8}
# The rules scheme: label share, gate for a label that does not follow from the response's own criteria, the distance
# in points at which the evidence score reaches zero, and the cost multiplier for a criterion the key does not accept.
RULES = {'label_share': 0.4, 'gate': 0.25, 'scale': 8.0, 'over_call': 2.0}
V, S, M, U = 'VeryStrong', 'Strong', 'Moderate', 'Supporting'
DEFAULT_STRENGTH = {'PVS': V, 'PS': S, 'PM': M, 'PP': U, 'BA': 'StandAlone', 'BS': S, 'BP': U}
ALIASES = {'supporting': U, 'supp': U, 'sup': U, 'moderate': M, 'mod': M, 'strong': S, 'verystrong': V, 'vs': V,
    'standalone': 'StandAlone'}
CODE = re.compile(r'(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])')
PAIR = re.compile(r'(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])(?![0-9])'
    r'(?:[\s_:\-–(\[]*(stand[\s_-]*alone|very[\s_-]*strong|strong|moderate|supporting|supp?|mod|vs)(?![a-z]))?', re.I)
JSON_PAIR = re.compile(r'"code"\s*:\s*"([^"]*)"\s*,\s*"strength"\s*:\s*"([^"]*)"', re.I)
CLASS = re.compile(r'classification(?:\s+(?:is|would be|remains))?[\W_]{0,8}(likely[\s_-]+pathogenic|likely[\s_-]+benign|'
    r'(?:variant of )?uncertain significance|pathogenic|benign|vus|lp|lb|p|b)(?![a-z])', re.I)
CRITERIA_HEAD = re.compile(r'^[\W_]*criteria[\W_]*?[:=]', re.I | re.M)
LABEL_WORDS = {'likely pathogenic': 'LP', 'likely benign': 'LB', 'variant of uncertain significance': 'VUS',
    'uncertain significance': 'VUS', 'pathogenic': 'P', 'benign': 'B', 'vus': 'VUS', 'lp': 'LP', 'lb': 'LB', 'p': 'P', 'b': 'B'}
# ACMG/AMP 2015 Table 5. Each criterion fills one requirement at its own or a weaker strength.
P_RULES = [{V: 1, S: 1}, {V: 1, M: 2}, {V: 1, M: 1, U: 1}, {V: 1, U: 2}, {S: 2}, {S: 1, M: 3}, {S: 1, M: 2, U: 2}, {S: 1, M: 1, U: 4}]
LP_RULES = [{V: 1, M: 1}, {S: 1, M: 1}, {S: 1, U: 2}, {M: 3}, {M: 2, U: 2}, {M: 1, U: 4}]
SVI2020_LP = {V: 1, U: 1}
B_RULES, LB_RULES = [{S: 2}], [{S: 1, U: 1}, {U: 2}]


# ---------------------------------------------------------------------------- parsing

def label_of(text):
    return LABEL_WORDS.get(re.sub(r'[\s_-]+', ' ', text.strip().lower()))


def criteria_of(text):
    """{code: strength} in a stretch of text; a bare code takes its ACMG default strength. First mention wins."""
    found = {}
    for m in PAIR.finditer(text):
        code, strength = m.group(1).upper(), m.group(2)
        strength = ALIASES.get(re.sub(r'[\s_-]+', '', strength.lower())) if strength else DEFAULT_STRENGTH['PVS' if code == 'PVS1' else code[:2]]
        found.setdefault(code, strength)
    return found


def parse_answer(text):
    """(label or None, {code: strength} or None when no criteria list is given)."""
    text = re.sub(r'```(?:json)?', '', text or '')
    start, end = text.find('{'), text.rfind('}')
    try: obj = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError: obj = None
    if isinstance(obj, dict) and isinstance(obj.get('classification'), str) and label_of(obj['classification']):
        items = obj.get('criteria') if isinstance(obj.get('criteria'), list) else []
        return label_of(obj['classification']), criteria_of('; '.join(f"{c.get('code', '')} {c.get('strength', '')}" for c in items if isinstance(c, dict)))
    labels = list(CLASS.finditer(text)); label = label_of(labels[-1].group(1)) if labels else None
    if JSON_PAIR.search(text): return label, criteria_of('; '.join(f'{c} {s}' for c, s in JSON_PAIR.findall(text)))
    heads = list(CRITERIA_HEAD.finditer(text))
    if not heads: return label, None
    stop = labels[-1].start() if labels and labels[-1].start() > heads[-1].end() else len(text)
    return label, criteria_of(text[heads[-1].end():stop])


# ---------------------------------------------------------------------------- combining

def meets(counts, need):
    spare = 0
    for strength in (V, S, M, U):
        spare += counts.get(strength, 0) - need.get(strength, 0)
        if spare < 0: return False
    return True


def tavtigian(total):
    return 'P' if total >= 10 else 'LP' if total >= 6 else 'VUS' if total >= 0 else 'LB' if total >= -6 else 'B'


def total_points(criteria, stand_alone=0):
    return sum({**POINTS, 'StandAlone': stand_alone}.get(s, 0) * (1 if c.startswith('P') else -1) for c, s in criteria.items())


def standard_classes(criteria):
    """Labels derivable from the criteria under any standard system: ACMG/AMP 2015, the same with the SVI 2020
    Very Strong + Supporting rule, or the 2020 point scale. Contradictory categorical evidence may be read as VUS
    or with the sufficient side prevailing. None if the criteria are invalid (BA1 not stand-alone, or the reverse)."""
    if any((c == 'BA1') != (s == 'StandAlone') for c, s in criteria.items()) or any(s not in (*POINTS, 'StandAlone') for s in criteria.values()):
        return None
    has_p = any(c.startswith('P') for c in criteria); has_b = any(c.startswith('B') for c in criteria)
    if 'BA1' in criteria: return {'B'} | ({'VUS'} if has_p else set())
    classes = {tavtigian(total_points(criteria))}
    p = Counter(s for c, s in criteria.items() if c.startswith('P')); b = Counter(s for c, s in criteria.items() if c.startswith('B'))
    for svi in (False, True):
        path = any(meets(p, n) for n in P_RULES); lp = any(meets(p, n) for n in LP_RULES) or (svi and meets(p, SVI2020_LP))
        pathogenic = 'P' if path else 'LP' if lp else None
        benign = 'B' if any(meets(b, n) for n in B_RULES) else 'LB' if any(meets(b, n) for n in LB_RULES) else None
        if (pathogenic and benign) or not (pathogenic or benign): classes.add('VUS'); continue
        classes.add(pathogenic or benign)
        if (pathogenic and has_b) or (benign and has_p): classes.add('VUS')
    return classes


GENERAL = [{'system': 'categorical', 'class': cls, 'conditions': [{'side': side, 'strength': st, 'op': '>=', 'value': n} for st, n in need.items()]}
    for cls, side, needs in (('P', 'Pathogenic', P_RULES), ('LP', 'Pathogenic', LP_RULES + [SVI2020_LP]), ('B', 'Benign', B_RULES), ('LB', 'Benign', LB_RULES))
    for need in needs]


def document_classes(criteria, combining):
    """Labels the criteria give under a rule document's combining rules (`reference['combining']`).

    Categorical counts read as 'at least' (a stronger criterion fills a weaker slot); each side takes its strongest
    class; mixed evidence also allows VUS. Point rules are read literally and on the 2020 scale. BA1 decides Benign
    (VUS also allowed against pathogenic criteria; under points it may count as -8)."""
    systems, rules = combining.get('systems') or [], combining.get('rules') or []
    counts = {side: Counter(s for c, s in criteria.items() if c[0] == side[0]) for side in ('Pathogenic', 'Benign')}
    total = total_points(criteria, stand_alone=8)
    if any(c == 'BA1' or (c.startswith('B') and s == 'StandAlone') for c, s in criteria.items()):
        return {'B'} | ({'VUS'} if counts['Pathogenic'] else set()) | ({tavtigian(total)} if {'points', 'point_scale'} & set(systems) else set())
    compare = {'>=': lambda x, v: x >= v, '<=': lambda x, v: x <= v, '==': lambda x, v: x == v, '>': lambda x, v: x > v, '<': lambda x, v: x < v}

    def categorical(rule_list):
        def met(rule):
            need = defaultdict(Counter)
            for c in rule['conditions']:
                if c['side'] is None or c['strength'] not in (V, S, M, U): return False
                need[c['side']][c['strength']] += int(c['value'])
            return all(meets(counts[side], need[side]) for side in need)
        hits = {x['class'] for x in rule_list if met(x)}
        p = 'P' if 'P' in hits else 'LP' if 'LP' in hits else None
        b = 'B' if 'B' in hits else 'LB' if 'LB' in hits else None
        found = {x for x in (p, b) if x}
        if 'VUS' in hits or not found or (p and counts['Benign']) or (b and counts['Pathogenic']): found.add('VUS')
        return found

    classes = set()
    if 'categorical' in systems: classes |= categorical([x for x in rules if x['system'] == 'categorical'])
    if 'general' in systems or not systems: classes |= categorical(GENERAL)
    if 'points' in systems:
        classes |= {x['class'] for x in rules if x['system'] == 'points' and all(
            compare[c['op']](total if c['side'] is None else counts[c['side']][c['strength']], c['value']) for c in x['conditions'])}
    if 'points' in systems or 'point_scale' in systems: classes.add(tavtigian(total))
    return classes


# ---------------------------------------------------------------------------- reward

def evidence_distance(returned, reference, over_call=RULES['over_call']):
    """How far the returned criteria are from the key, in ACMG/AMP evidence points. A code the key accepts costs the
    gap to its nearest accepted strength; a code it does not accept costs its points times `over_call`; a required
    code not returned costs its smallest accepted strength. 0 means the criteria match a reading of the key."""
    accepted = defaultdict(set)
    for code, strength in reference.get('acceptable') or []: accepted[code].add(strength)
    d = 0.0
    for code, strength in returned.items():
        points = EVIDENCE_POINTS.get(strength, 0)
        if code in accepted: d += min(abs(points - EVIDENCE_POINTS.get(s, 0)) for s in accepted[code])
        else: d += over_call * points
    for code, strengths in (reference.get('required') or {}).items():
        if code not in returned: d += min(EVIDENCE_POINTS.get(s, 0) for s in strengths)
    return d


def reward(response, reference, arm, scale=RULES['scale'], over_call=RULES['over_call']):
    """Score one response. `reference` has label, label_scoring, accepted_labels, required ({code: [strengths]}),
    acceptable ([[code, strength], ...]) and, for the with-rules arm, combining. Components:

    - answered: a classification was found.
    - label: the classification is an accepted label (None when the reference's label is not scored).
    - criterion_precision / criterion_recall / criterion_f1: returned (code, strength) pairs that are acceptable;
      required codes returned at an accepted strength. Returning nothing when nothing is required scores 1.
    - exact_criteria: nothing unacceptable returned and every required criterion returned.
    - self_consistent: the label follows from the response's own criteria, under the rule document's combining rules
      in the with-rules arm (when the reference carries them), under any standard system otherwise.
    - evidence_distance: the criteria's distance from the key in evidence points (`evidence_distance`; None without an
      answer), and evidence_score = max(0, 1 - distance / scale)."""
    label, found = parse_answer(response)
    answered = label is not None
    returned = (found or {}) if answered else {}
    required = {c: set(s) for c, s in (reference.get('required') or {}).items()}
    acceptable = {tuple(x) for x in reference.get('acceptable') or []}
    good = sum((c, s) in acceptable for c, s in returned.items())
    hit = sum(returned.get(c) in s for c, s in required.items())
    precision = good / len(returned) if returned else (1.0 if not required else 0.0)
    recall = hit / len(required) if required else (1.0 if not returned else precision)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scored = reference.get('label_scoring', 'scored') != 'excluded'
    distance = evidence_distance(returned, reference, over_call) if answered else None
    consistent = None
    if answered and found is not None:
        classes = document_classes(returned, reference['combining']) if arm == 'with_rules' and reference.get('combining') else standard_classes(returned)
        consistent = None if classes is None else label in classes
    return {'answered': float(answered),
        'label': (float(answered and label in (reference.get('accepted_labels') or [reference.get('label')]))) if scored else None,
        'criterion_precision': precision if answered else 0.0, 'criterion_recall': recall if answered else 0.0,
        'criterion_f1': f1 if answered else 0.0,
        'exact_criteria': float(answered and good == len(returned) and hit == len(required)),
        'self_consistent': float(bool(consistent)),
        'evidence_distance': distance, 'evidence_score': max(0.0, 1 - distance / scale) if answered else 0.0}


def reference_from_answer(row):
    """A reference from one row of the benchmark's `data/answers.jsonl` (Hugging Face export)."""
    return {'label': row['label'], 'label_scoring': row['label_scoring'], 'accepted_labels': row['accepted_labels'],
        'required': {x['code']: x['strengths'] for x in row['required']}, 'acceptable': [[x['code'], x['strength']] for x in row['acceptable']],
        'combining': row.get('combining_rules')}


def scalar(components, weights=None, scheme='default', label_share=RULES['label_share'], gate=RULES['gate']):
    """One number in [0, 1], zero without an answer.

    default: a weighted mean of components, by default 0.5 label + 0.3 criterion F1 + 0.2 self-consistency. A label
    that is not scored drops out and the other weights are renormalised.
    rules: (label_share x label + (1 - label_share) x evidence_score), multiplied by `gate` when the label does not
    follow from the response's own criteria. Without a scored label, the evidence score alone."""
    if not components['answered']: return 0.0
    if scheme == 'rules':
        core = components['evidence_score'] if components.get('label') is None else label_share * components['label'] + (1 - label_share) * components['evidence_score']
        return core * (1.0 if components['self_consistent'] else gate)
    if scheme != 'default': raise ValueError('scheme is default or rules')
    weights = weights or {'label': 0.5, 'criterion_f1': 0.3, 'self_consistent': 0.2}
    used = {k: w for k, w in weights.items() if components.get(k) is not None}
    return sum(w * components[k] for k, w in used.items()) / sum(used.values())


# ---------------------------------------------------------------------------- command line

ARMS = ('with_rules', 'without_rules')
COMPONENTS = ('answered', 'label', 'criterion_precision', 'criterion_recall', 'criterion_f1', 'exact_criteria', 'self_consistent',
    'evidence_distance', 'evidence_score')


def load_references(path):
    """{id or case_id: (reference, arm or None)} from training/rl.jsonl or a benchmark data/answers.jsonl."""
    refs = {}
    with open(path, encoding='utf-8') as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip(): continue
            row = json.loads(line)
            if 'reference' in row and 'id' in row: refs[row['id']] = (row['reference'], row.get('arm'))
            elif 'required' in row and 'case_id' in row: refs[row['case_id']] = (reference_from_answer(row), None)
            else: raise SystemExit(f'{path}:{n}: not a reference row; expected training rl.jsonl or benchmark answers.jsonl')
    return refs


def parse_weights(text):
    """'label=0.5,criterion_f1=0.3,self_consistent=0.2' -> {component: weight}."""
    weights = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        name, _, value = part.partition('=')
        if name not in COMPONENTS[1:]: raise argparse.ArgumentTypeError(f'unknown component {name!r}; use {", ".join(COMPONENTS[1:])}')
        try: weights[name] = float(value)
        except ValueError: raise argparse.ArgumentTypeError(f'the weight for {name} must be a number, not {value!r}') from None
    if not weights or sum(weights.values()) <= 0: raise argparse.ArgumentTypeError('weights must sum to more than zero')
    return weights


def summarize(rows):
    """Mean over responses of each component and of the combined reward. The label mean covers scored labels only, and
    criterion F1 is averaged per response (the benchmark report instead pools criteria across responses)."""
    out = {'responses': len(rows)}
    for name in ('reward', *COMPONENTS):
        values = [x[name] for x in rows if x.get(name) is not None]
        out[name] = round(sum(values) / len(values), 4) if values else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='acmg_reward.py', description='Score ACMG/AMP variant classification answers against reference answers.',
        epilog='examples:\n'
            '  python acmg_reward.py --references data/answers.jsonl --responses responses.jsonl --summary\n'
            '  python acmg_reward.py --references training/rl.jsonl --responses samples.jsonl --output scores.jsonl\n\n'
            'Each response line names its case with "id" ("<case_id>:<arm>", as in the reference files) or with "case_id"\n'
            'and "arm". Arms are with_rules and without_rules.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--references', required=True, type=Path, metavar='PATH',
        help='training/rl.jsonl (prompts with references) or data/answers.jsonl (benchmark answer keys)')
    parser.add_argument('--responses', required=True, type=Path, metavar='PATH', help='JSONL file with one model response per line')
    parser.add_argument('--response-field', default='response', metavar='NAME', help='field holding the response text (default: response)')
    parser.add_argument('--arm', choices=ARMS, help='arm for response lines that do not name one')
    parser.add_argument('--weights', type=parse_weights, metavar='SPEC',
        help='components and weights for the combined reward (default: label=0.5,criterion_f1=0.3,self_consistent=0.2)')
    parser.add_argument('--scheme', choices=('default', 'rules'), default='default',
        help='default: weighted components (see --weights); rules: rule_gate x (label_share x label + (1 - label_share) x evidence_score)')
    parser.add_argument('--label-share', type=float, default=RULES['label_share'], metavar='X', help='rules: share of the label term (default: 0.4)')
    parser.add_argument('--gate', type=float, default=RULES['gate'], metavar='X',
        help="rules: multiplier when the label does not follow from the response's own criteria (default: 0.25)")
    parser.add_argument('--evidence-scale', type=float, default=RULES['scale'], metavar='POINTS',
        help='distance in evidence points at which the evidence score reaches zero (default: 8, one Very Strong criterion)')
    parser.add_argument('--over-call', type=float, default=RULES['over_call'], metavar='X',
        help='cost multiplier for a criterion the key does not accept (default: 2)')
    parser.add_argument('--output', type=Path, metavar='PATH', help='write per-response scores as JSONL to this file (default: standard output)')
    parser.add_argument('--summary', action='store_true', help='print mean scores overall and by arm (to standard error when scores go to standard output)')
    args = parser.parse_args(argv)
    if args.scheme == 'rules' and args.weights: parser.error('--weights applies to the default scheme only')
    refs = load_references(args.references)
    scored, problems, seen = [], Counter(), set()
    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    try:
        with open(args.responses, encoding='utf-8') as handle:
            for n, line in enumerate(handle, 1):
                if not line.strip(): continue
                try: x = json.loads(line)
                except ValueError: problems['line is not JSON'] += 1; continue
                rid, case, arm = x.get('id'), x.get('case_id'), x.get('arm')
                if rid and not case: case, _, arm_in_id = str(rid).partition(':'); arm = arm or arm_in_id or None
                found = refs.get(rid) or refs.get(case)
                if found is None: problems['no reference for the case'] += 1; continue
                reference, ref_arm = found; arm = ref_arm or arm or args.arm
                if arm not in ARMS: problems['no arm (add "arm" or use --arm)'] += 1; continue
                text = x.get(args.response_field)
                parts = reward(text if isinstance(text, str) else '', reference, arm, args.evidence_scale, args.over_call)
                row = {'id': rid or f'{case}:{arm}', 'arm': arm, 'reward': round(scalar(parts, args.weights, args.scheme, args.label_share, args.gate), 4), **parts}
                scored.append(row); seen.add(rid if rid in refs else case)
                out.write(json.dumps(row) + '\n')
    finally:
        if args.output: out.close()
    report = sys.stdout if args.output else sys.stderr
    if args.summary:
        summary = {'overall': summarize(scored), **{arm: summarize([x for x in scored if x['arm'] == arm]) for arm in ARMS if any(x['arm'] == arm for x in scored)},
            'references_without_a_response': len(set(refs) - seen), 'skipped_responses': dict(problems)}
        print(json.dumps(summary, indent=1), file=report)
    elif problems:
        print(json.dumps({'skipped_responses': dict(problems)}), file=sys.stderr)
    return 0 if scored else 1


if __name__ == '__main__':
    sys.exit(main())
