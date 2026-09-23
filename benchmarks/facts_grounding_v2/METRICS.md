# FACTS Grounding v2 metrics and protocol

## Scope

FACTS Grounding measures whether a model can produce a long-form answer using only a supplied
document while still addressing the user's request. This adapter implements the official v2
judge protocol for the 856-row public Kaggle release. It does not include the private holdout
set or submit results to the Kaggle leaderboard.

The source of record is Kaggle's FACTS Grounding Public Examples dataset, version 17. The
prepare script pins and verifies the public source files before producing NeMo Gym rows.
The December 2024 Hugging Face mirror has 860 rows; four rows are absent from the pinned
public release and are not scored.

## Judging protocol

The benchmark keeps eligibility and groundedness separate:

1. Gemini 2.5 Flash, then GPT-5, independently writes a baseline answer and rates whether the
   policy response follows the request. The first well-formed rating other than Major Issue(s)
   determines eligibility. An unparseable eligibility rating follows the official protocol
   and is treated as eligible.
2. Each judge labels the response sentence by sentence for support from the supplied context.
   A judge's answer is grounded when at least one sentence is parsed and none is
   not_supported. A response with no parseable grounding sentence is not grounded.
3. The adjusted factuality reward is zero for an ineligible response; otherwise it is the mean
   of the two judges' grounded/not-grounded verdicts. The unadjusted groundedness score is
   retained separately. Both judges run even for ineligible responses, matching the v2
   reference protocol.

## Metric dictionary

| Metric | Definition | Denominator / notes |
|---|---|---|
| factuality_score | Mean adjusted reward; the primary public-set score | All scored answers; report judge-failed rows separately |
| unadjusted_factuality_score | Mean grounding verdict with ineligible answers included | All answers with grounding verdicts |
| eligibility_rate | Fraction of answers judged eligible | All answers |
| grounded_rate_eligible/<judge> | Eligible answers that judge found fully supported | Eligible answers |
| grounded_all_judges_rate_eligible | Eligible answers grounded by both judges | Eligible answers |
| judge_disagreement_rate_eligible | Eligible answers with exactly one grounded verdict | Eligible answers |
| eligibility_decided_by/<judge> | Fraction whose eligibility was settled by that judge | All answers |
| eligibility_invalid_rate | Fraction with an unparseable eligibility rating | All answers |
| grounding_parse_empty_rate/<judge> | Fraction with no parsed sentence-level verdict | Answers graded by that judge |
| grounding_unparseable_line_rate/<judge> | Fraction with at least one unparseable sentence line | Answers graded by that judge |
| factuality_score_ci95_low/high | Bootstrap 95% interval for mean adjusted reward | Fixed seed and 2,000 resamples |
| generation_empty_rate, generation_truncated_rate | Empty and output-cap-truncated generations | All answers |
| factuality_score/<slice>/<value> | Adjusted score for a domain, high-level type, or task-type slice | Rows in that slice |

Judge transport failures are routed to the failure sidecar and excluded from the score until
retried. Unparseable judge content is not a transport failure: it follows the official
eligibility or grounding parsing rules and is reflected in the corresponding diagnostics.

## Row mapping

| Source field or operation | NeMo Gym representation |
|---|---|
| Kaggle full_prompt | One user message in responses_create_params.input; no prompt template is applied |
| user_request and context_document | Task data consumed by the eligibility and grounding judges |
| system_instruction, domain, type, high_level_type | Preserved task data and result-slice metadata |
| Official eligibility decision | eligibility_ratings, eligible, eligibility_deciding_judge |
| Sentence-level grounding decisions | judge_receipts, grounding_verdicts, grounding_label_counts, parse diagnostics |
| Official reward and component score | reward, grounding_score, unadjusted_score |

## Interpretation

- Report the public-set denominator and the two judge identities with every factuality score.
- A score of 0.5 for one eligible answer means the two judges disagreed on groundedness.
- Do not compare this public-only score directly with Kaggle leaderboard scores, which combine
  public and private examples and may use a different execution pipeline.
- Empty, truncated, missing, duplicate, and judge-failed rows are separate outcomes; do not
  silently remove them from the denominator or describe infrastructure failures as model errors.
