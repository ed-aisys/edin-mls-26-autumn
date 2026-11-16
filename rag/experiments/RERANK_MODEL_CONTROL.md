# Exploratory L6/L12 model-replacement control

Protocol written before running this control, after observing the L6 failures.
This is an exploratory diagnostic, not held-out confirmation or a depth-only
causal ablation. No dev inference, answer generation or judge calls are allowed.

## Purpose and model-card check

Does the retrieval degradation also appear with a second reranker?

- Existing: `cross-encoder/ms-marco-MiniLM-L6-v2`,
  revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Control: `cross-encoder/ms-marco-MiniLM-L12-v2`,
  revision `7b0235231ca2674cb8ca8f022859a6eba2b1c968`.
- The L12 model card identifies English MS MARCO passage ranking training,
  base model `microsoft/MiniLM-L12-H384-uncased`, Apache-2.0 licensing,
  and descending-score ranking with query/document inputs.
- Source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/blob/7b0235231ca2674cb8ca8f022859a6eba2b1c968/README.md
- The card's benchmark scores are not evidence about our coursework corpus.
  A model replacement may change learned weights/training/tokenization as well
  as depth; any improvement cannot be attributed to capacity alone.

## Fixed comparison

- All 350 practice questions, unchanged ASR transcripts and dense top-100
  candidates in `rag/results/selection-rankings-practice-full/rankings.jsonl`.
- No candidate filtering or query/document text editing.
- Query first, document second; max length 512; `truncation="only_second"`;
  batch size 32; model evaluation/inference mode; float32; TF32 disabled.
- Both models on the same otherwise idle RTX A6000 (physical GPU 1).
- Three full-query warmups per model, excluded from measurement.
- One timed rerank of all 100 candidates per model/query, with CUDA
  synchronization at both boundaries. Includes tokenization, host/device
  transfers, forward pass and score materialization, excludes model loading
  and sorting. Alternate L6-first/L12-first across the fixed query order.
- Stable descending-logit sort, dense position used to break ties.
- Recall@5, MRR@10, NDCG@10 use the existing course metric definitions and
  positive practice qrels; MRR/NDCG use the first ten reranked candidates,
  although a downstream evidence arm would select five.
- Retain every question, including ties and regressions.
- Three paired contrasts: L6 minus dense, L12 minus dense, L12 minus L6.
  Query bootstrap: 5,000 resamples, seed 20261116, percentile 95% intervals,
  no multiplicity adjustment. This excludes repeated-run uncertainty.
- Count qrel-positive query/document pairs removed/added in each contrast.
- Inspect the preselected CAGR (`q_b7390fb1b4a9ad79`) and TaxAct
  (`q_65e52727cacd73bb`) cases, without using them to select the model or
  replace all-query metrics.

## Run and outputs

```bash
CUDA_VISIBLE_DEVICES=1 ~/mls-rag-env/bin/python \
  rag/extensions/compare_reranker_models.py \
  --output rag/results/rerank-model-control-practice350
```

The output directory must be fresh. `config.json` freezes the protocol,
source/input hashes, package versions and model-asset hashes before inference.
`scores.jsonl` saves every model score/ranking and measured latency;
`per_query.json` retains all per-query metrics and transitions;
`cases.json` retains the preselected cases;
`summary.json` contains means, intervals, replacement counts and output hashes.
Original cached L6 rankings/scores are compared with the matched rerun to flag
numerical or implementation drift. No report text is changed by this command.

Interpret quality intervals at query level, not as proof of answer quality.
Qrel-negative documents are not necessarily irrelevant. Shared host activity,
single-pass timing and model-training differences limit generalization.

## Completed results

All 350 questions completed. The new L6 run exactly reproduces all 350 saved
full rankings and scores (maximum absolute logit difference zero).
Tokenizer vocabulary, serialization, configuration and special-token files
are identical across the two checkpoints; measured depth is 6 versus 12,
both with hidden size 384 and 12 attention heads. Identical tokenizers reduce
one confound but do not identify a causal depth effect.

| Arm | Recall@5 | MRR@10 | NDCG@10 | Mean rerank ms | Median rerank ms |
|---|---:|---:|---:|---:|---:|
| Dense | 0.421337 | 0.488746 | 0.420461 | — | — |
| L6 | 0.387689 | 0.453454 | 0.395219 | 167.35 | 168.77 |
| L12 | 0.410235 | 0.452749 | 0.396664 | 292.93 | 295.62 |

Paired query-bootstrap mean differences and 95% intervals:

| Contrast | Recall@5 delta [CI] | MRR@10 delta [CI] | NDCG@10 delta [CI] |
|---|---|---|---|
| L6 − dense | −0.03365 [−0.06865, 0.00064] | −0.03529 [−0.06974, −0.00081] | −0.02524 [−0.05283, 0.00263] |
| L12 − dense | −0.01110 [−0.04267, 0.02011] | −0.03600 [−0.06905, −0.00117] | −0.02380 [−0.05079, 0.00349] |
| L12 − L6 | +0.02255 [0.00217, 0.04272] | −0.00070 [−0.02075, 0.01955] | +0.00144 [−0.01305, 0.01500] |

These are exploratory, unadjusted intervals across several comparisons/metrics,
not confirmatory tests. In particular, the MRR intervals just below zero should
not be overstated as robust generalization evidence.

Evidence replacement at five, counting qrel-positive question/document pairs:

| Replacement | Removed | Added | Recall improved / worsened / unchanged queries |
|---|---:|---:|---|
| Dense → L6 | 83 | 47 | 35 / 63 / 252 |
| Dense → L12 | 67 | 42 | 31 / 47 / 272 |
| L6 → L12 | 17 | 28 | 26 / 15 / 309 |

Pair counts and macro recall differ in weighting because questions have different
numbers of qrel-positive documents. Equal recall can still involve document swaps.

### Preselected cases (all retained)

- CAGR: the two dense-related documents move from L6 ranks 11/38 to L12 ranks
  7/14. Recall@5 stays **0** for both rerankers versus dense **1**.
  L12 recovers one document within ten, so MRR@10 becomes 1/7, not a top-five
  repair. The beta, confirmation and Vitamin C snippets remain L12's top three.
- TaxAct/TurboTax: the two displaced documents move from L6 ranks 74/60 to L12
  ranks 76/50. Recall@5 stays **0.25** for both, versus dense **0.75**.
  The retained direct-comparison document stays second.

### Interpretation and verification

L12 provides an exploratory Recall@5 improvement over L6, but there is no
corresponding detectable MRR/NDCG improvement between the models. It does not
establish a retrieval advantage over dense: the Recall/NDCG intervals versus
dense cross zero, and the unadjusted MRR interval is negative. Its median
reranking latency is **1.75×** L6 in this matched single-pass measurement.
Thus changing this checkpoint partially recovers coverage, but does not yet
justify reranking adoption or proceeding to answer generation. Both checkpoints
still exhibit the two selected failure patterns. This neither identifies
capacity as the cause nor generalizes to all rerankers.

An independent read-back of saved scores recomputed all nine means, all nine
bootstrap intervals, all replacement counts, and both models' mean/median
timings. Input/output/source hashes matched. All **19** repository tests passed.
No dev operations, answer generation or judge calls were made, and the report
was not edited for this control. Full precision is in the output `summary.json`;
every selected and nonselected question is retained in `per_query.json` and
`scores.jsonl`.
