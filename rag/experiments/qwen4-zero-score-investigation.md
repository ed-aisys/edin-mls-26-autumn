# Remaining zero-score answers

## Objective and current evidence

Reduce genuinely failed answers while retaining the complete 350-question
practice evaluation. The original public judge, its raw outputs and all zero
scores remain unchanged. Removing difficult questions or replacing their scores
after inspection would invalidate comparisons with the baseline.

The [completed paired experiment](qwen4-answer-evaluation.md) reduced all-zero
judgements from 138 to 121 and increased the mean normalised score from 0.337381
to 0.381905. This is a limited absolute gain at higher processing cost. We have
not yet demonstrated that another intervention reduces the remaining failures.

The [complete diagnostic inventory](../results/qwen4-answers-3724290/zero-review-v1/triage.json)
retains every question. Among the 121 Qwen all-zero answers:

- 62 have no positively labelled record among the selected five.
- 29 include some, but not all, positively labelled records.
- 30 include all positively labelled records.

These groups describe record-ID coverage, not verified failure causes. Qrels
may be incomplete, and a selected record may not resolve an ambiguous question.
Even full labelled coverage does not establish a generation or judge error.

## Frozen review sample

The [sampling script](../extensions/prepare_qwen_zero_review.py) selects six
questions from each zero-score group and six positive-score controls, using
seed 20261117. It produces 24 paired questions and 48 answer packets. Each packet
contains the ASR transcript, candidate answer, original selected evidence and
public gold evidence. It hides method names and automatic scores and shuffles
the two methods within each pair. This is partial blinding: content can reveal
the selection method, and reviewers know the study concerns answer failures.

- [Review packets](../results/qwen4-answers-3724290/zero-review-v1/review-packets.json).
- [Review key](../results/qwen4-answers-3724290/zero-review-v1/review-key.json):
  keep separate until the initial review is complete.
- [Summary and source hashes](../results/qwen4-answers-3724290/zero-review-v1/summary.json).

Reviewer fields remain empty. Review answerability from selected evidence,
correctness of the core answer, unsupported claims, confused conditions and
missing important points before consulting the original judge explanation.
Record overlapping causes where necessary. Two reviewers should complete
independent reviews and discuss disagreements; report the agreement counts.
The balanced diagnostic sample does not estimate population failure rates.

## Follow-up experiments

1. Complete the blinded review and then compare reviewers' findings with the
   original judge explanations. Keep disputed scores in the public benchmark;
   report manual findings separately. Known explanation reversals warrant
   inspection but do not determine replacement scores.
2. Freeze a small gold-context control on the same 24 diagnostic questions.
   Supply positively labelled records to the unchanged EuroLLM generator and
   prompt, preserving transcript questions and generation settings. Compare
   against the saved Qwen evidence condition. Gold context uses evaluation
   labels and serves only as a diagnostic, not a deployable method or a strict
   mathematical answer-quality upper bound. Its scores will not support a
   population improvement claim because sampling depends on prior scores.
3. Choose one intervention from the observed failures. If missing useful
   evidence dominates, test evidence selection within the fixed candidate
   pool. If selected evidence answers the question but generated answers add
   unsupported details or confuse conditions, test a prompt that asks for a
   direct evidence-supported answer, preserves jurisdiction/date conditions
   and qualifies missing information. Keep retrieval fixed for that control.
   If judge disagreement dominates, prioritise evaluation reliability and
   report the disputed cases without tuning to the judge's mistakes.
4. Freeze the chosen intervention and its configuration before a fresh paired
   full-practice run. Retain all 350 questions, the same public evaluator and
   a matched control. Report all three score dimensions, paired intervals,
   zero-to-positive and positive-to-zero transitions, input/output tokens and
   processing time. Account for generation/judge variation with a small
   predeclared repeatability check. Do not rerun or tune on the already seen
   dev split during this investigation.

Success requires a lower all-zero rate together with improved overall scores
and an explicit cost comparison. A method that rescues some zeros but damages
other answers is not an overall improvement by default. An average dimension
score of 2 can be an aspiration; the current results do not guarantee it.

## ASR connection

The current comparison holds ASR transcripts fixed and isolates downstream
evidence selection. If manual review identifies transcription-induced ambiguity,
a separate clean-question versus frozen-ASR-transcript control can test whether
that changes retrieval and answers while holding RAG settings fixed. The current
data do not establish that ASR caused any particular zero judgement. Reuse of
cached transcripts also excludes ASR from the reported downstream timing.

## Execution status

Completed: publishable full results, offline answer/cache audit, all-question
triage, frozen review sample and reproducibility tests. Pending: human review,
gold-context generation/judging, intervention selection and full paired
validation. No additional model or ELM calls were made to prepare this PR.
