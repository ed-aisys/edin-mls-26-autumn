Use this directory for investigation-specific code if useful. You are not required
to place all modifications here; substantive changes may naturally involve src/.

## Evidence-selection investigation

### Qwen retrieval and answer experiments

[The Qwen retrieval report](../experiments/qwen06-retrieval.md) includes the
completed 350-query 0.6B and 4B comparisons. Both runs preserve the same frozen
top-100 candidates; the primary Qwen arm uses source-text prefixes matched to
MiniLM's visible input. The secondary arm allows longer input. Record quality
and latency separately.

- [evaluate_qwen_reranker.py](evaluate_qwen_reranker.py): download a pinned
  0.6B/4B checkpoint, then score frozen candidates on an allocated GPU.
- [audit_qwen_retrieval.py](audit_qwen_retrieval.py): recompute retrieval metrics
  from published rankings and verify query coverage and stable tie ordering.
- [compare_qwen_sizes.py](compare_qwen_sizes.py): verify shared inputs and
  reproduce paired size/input comparisons from the two completed runs.
- [run_qwen_answer_experiment.py](run_qwen_answer_experiment.py): freeze dense5
  and 4B top-5 contexts, generate/resume answers, and summarise results from the
  unchanged public answer evaluator. The
  [answer report](../experiments/qwen4-answer-evaluation.md) records the completed
  350-query paired experiment and its earlier labelled checkpoint.
- [audit_qwen_answers.py](audit_qwen_answers.py): verify all 700 saved answers,
  judge cache keys, raw judgements, score means and paired intervals offline.
- [prepare_qwen_zero_review.py](prepare_qwen_zero_review.py): retain all 350
  questions in a diagnostic inventory and freeze stratified, blinded review
  packets. Keep the review key separate until reviewers finish. The
  [zero-score investigation](../experiments/qwen4-zero-score-investigation.md)
  specifies follow-up controls and criteria for accepting a change.

From the repository root, the published retrieval data supports these offline
checks without model downloads or ELM calls:

```bash
python rag/extensions/audit_qwen_retrieval.py rag/results/qwen06-retrieval-3724137/full/summary.json --audit
python rag/extensions/audit_qwen_retrieval.py rag/results/qwen4-retrieval-3724174/full/summary.json --audit
python rag/extensions/compare_qwen_sizes.py
python rag/extensions/audit_qwen_answers.py
# Choose a fresh directory; the published zero-review-v1 sample is frozen.
python rag/extensions/prepare_qwen_zero_review.py --output rag/results/qwen4-answers-3724290/zero-review-reproduction
```

Downloaded weights and tokenizer caches stay outside Git. Frozen result files
preserve their original bytes so checksum checks work across platforms.

The [pilot report](../experiments/selection-pilot-results.md) records the fixed sample,
five configurations, model revisions, input budget and evaluation limitations.
These extensions operate on saved ASR transcripts and candidates; they do not run ASR.

- [collect_retrieval_diagnostics.py](collect_retrieval_diagnostics.py): save complete top-100 rankings using cached transcripts and the original encoder/retriever.
- [prepare_selection_assets.py](prepare_selection_assets.py): download pinned reranker weights and the official generator tokenizer, on the login node.
- [evidence_selection.py](evidence_selection.py): preserve original top-k prompts or apply an explicit full-chat input budget, independent of qrels.
- [run_selection_experiment.py](run_selection_experiment.py): prepare frozen contexts with dense/re-ranked ordering, then generate/resume paired answers with token usage and latency metadata.
- [validate_reranker.py](validate_reranker.py): independently reproduce cached scores and check scoring direction.
- [summarize_selection_experiment.py](summarize_selection_experiment.py): check matched coverage, tokens, selected-ID recall, observed/attributed costs and optional paired judge scores.
- [bm25_retriever.py](bm25_retriever.py) / [compare_retrieval_routes.py](compare_retrieval_routes.py): lexical BM25 and RRF vs dense (no ELM).
- [measure_e2e_latency.py](measure_e2e_latency.py): audio-to-answer wall-clock breakdown on a GPU node.
- [collect_dual_transcripts.py](collect_dual_transcripts.py) / [asr_rag_four_condition.py](asr_rag_four_condition.py): CUDA vs reference transcript × dense5 vs budget recall.
- [prepare_blind_review.py](prepare_blind_review.py): anonymised packets for manual answer review.
- [compare_reranker_models.py](compare_reranker_models.py): matched practice-only
  L6/L12 replacement control, with frozen candidates, synchronised timings,
  paired query-bootstrap intervals and all-query outputs (no generation/judge).
  Protocol and results: [RERANK_MODEL_CONTROL.md](../experiments/RERANK_MODEL_CONTROL.md).

For the completed frozen dev run and its summary-only audit, see
[DEV_VALIDATION.md](../experiments/DEV_VALIDATION.md).

### Report revision: summaries only

Do not rerun dev generation/judging or tune methods from dev scores. The completed
five-arm practice/dev summaries can be recomputed from local saved outputs:

```bash
~/mls-rag-env/bin/python rag/extensions/summarize_selection_experiment.py \
  --run rag/results/selection-run-full --judge-prefix judge_json_ \
  --split practice --output rag/results/selection-run-full/comparison.json
# For dev, substitute dev-validation and --split dev.
python3 rag/extensions/audit_report_evidence.py
```

The summary command replaces only its designated output. Judge files can be under
the run root or `answers/`; this experiment uses the `judge_json_` prefix.
Report generation latency uses `median_generation_ms`. Query bootstrap does not
estimate repeated-generation/judge uncertainty. The audit script checks document
prefixes, score discrepancies, zero-score explanations, correlations and maxima;
its output is `rag/results/report-evidence/summary.json`. Positive openings are
defined by whole-word `correctly`/`accurately` in the explanation's first sentence.
These checks do not establish judge error rates or causal effects.

Run model computations on allocated compute resources. From the repository root,
after installing course assets and the RAG environment:

```bash
# Download the pinned model/tokenizer assets on a machine with network access.
python rag/extensions/prepare_selection_assets.py

# Recover practice rankings from the existing frozen baseline transcripts.
python rag/extensions/collect_retrieval_diagnostics.py \
  --transcript-cache rag/results/icf-3713067/practice_transcripts.jsonl \
  --sample rag/experiments/selection-pilot-sample.jsonl \
  --saved-answers rag/results/icf-3713067/sample_answers.jsonl \
  --output rag/results/selection-rankings

# Prepare all five configurations before sending generation requests.
python rag/extensions/run_selection_experiment.py prepare \
  --sample rag/experiments/selection-pilot-sample.jsonl \
  --rankings rag/results/selection-rankings/rankings.jsonl \
  --output rag/results/selection-run/plan
# On a GPU node (no ASR in this step): add --rerank-device cuda for faster 100-pair scoring
python rag/extensions/run_selection_experiment.py generate \
  --plan rag/results/selection-run/plan \
  --output rag/results/selection-run/answers --max-new-queries 3

# After inspecting the first three queries, resume the same frozen plan.
python rag/extensions/run_selection_experiment.py generate \
  --plan rag/results/selection-run/plan \
  --output rag/results/selection-run/answers --resume

# Evaluate each arm under the same output-format setting.
for arm in dense5 dense20 rerank5 dense_budget rerank_budget; do
  python rag/scripts/evaluate_answers.py --split practice --allow-subset --json-mode \
    --candidates "rag/results/selection-run/answers/$arm.jsonl" \
    --out-dir "rag/results/selection-run/judge_json_$arm"
done
python rag/extensions/summarize_selection_experiment.py \
  --run rag/results/selection-run --judge-prefix judge_json_ \
  --output rag/results/selection-run/comparison.json
```

Generation and judging use ELM requests. The evaluator's default request remains
unchanged; optional JSON mode uses a separate cache namespace and records its
protocol in summaries. Use fresh preparation/output directories for new methods;
resume only when the source and frozen plan are unchanged.

Run the local adapter, selection and judge-protocol tests from the repository root:

```bash
python -m unittest discover -s rag/tests -v
```

A saved relevant document ID does not imply a
truncated document retained its useful span. Judge scores require checking against
the question and references; structured JSON does not ensure correct judgements.
