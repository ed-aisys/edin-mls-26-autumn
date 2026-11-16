# Qwen3-Reranker retrieval comparison: 0.6B and 4B

## 主要发现

完整的 350 题 practice 结果及逐题复算确认：在相同候选池和匹配原文片段下，
Recall@5 从 dense 的 **0.4213**，提升到 Qwen 0.6B 的 **0.4671**、
4B 的 **0.5262**。4B 的 MRR@10、NDCG@10 也高于 dense 和 0.6B。
4B 的平均重排耗时为 **5.74 秒/题**，0.6B 为 **1.62 秒/题**。
这是检索质量与成本的结果；尚未评估这两种 Qwen 配置的答案质量。

Protocol fixed on 2026-11-16, before inspecting Qwen scores. This is a practice
development experiment, not an untouched test-set result. No ASR inference,
answer generation, or ELM judging is performed.

## Question

Can a different pretrained reranker improve early evidence ranking over the
original dense ranking and the existing MiniLM reranker, within the same frozen
top-100 candidate sets? Does allowing more source text change the result?

## Fixed inputs and comparisons

- All 350 practice queries, using the original cached ASR transcripts and
  `diagnostic-retrieval-3713758/rankings.jsonl`.
- Frozen rankings SHA-256:
  `9b7982cde774cea800cf9eedada0dd5e81b94341f451d2e93a4ee97bb8fbdb22`.
- Qwen model: `Qwen/Qwen3-Reranker-0.6B`, revision
  `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
- MiniLM weights and revision are checked against the existing selection asset
  manifest. No training or prompt tuning on relevance labels.
- `dense`: unchanged frozen ranking.
- `minilm_512`: original MiniLM classification logits, original source text,
  document-only truncation at 512 pair tokens, float32.
- `qwen_matched` (primary Qwen arm): the original-character document prefix
  visible to MiniLM under its 512-token pair limit. Qwen receives this prefix
  with its official instruction/template, without further truncation.
- `qwen_native2048` (secondary arm): original source text, capped at 2048 Qwen
  tokens including the template. Only the document is truncated.
- Qwen uses BF16, SDPA, and last-position `yes` minus `no` logits. Ties retain
  original dense order. The instruction is the official default English search
  instruction, fixed before the run.

The matched comparison controls source-text availability approximately; it does
not equate tokenizations. It also changes architecture, training, tokenizer, and
precision, so it cannot isolate parameter count as the cause of any difference.

## Evaluation

Report macro mean Recall@5/10/20/100, MRR@10, and graded NDCG@10 using the
repository's metric functions. Reordering the same 100 candidates must preserve
Recall@100. Report paired changes, wins/losses/ties and descriptive query-bootstrap
95% intervals (2,000 resamples, seed 20261116). These intervals do not capture
generation, judge, repeated timing, or model-selection uncertainty.

Compute the per-query oracle Recall@5 as
`min(5, number of positive qrels in candidate pool) / number of positive qrels`.
This is a labelled diagnostic ceiling, never an inference method.

Record warm reranking wall time, median/p95, input lengths, truncation counts,
software/model/input hashes, and peak allocated GPU memory. Scoring time includes
tokenization, transfer, inference, and score copy; excludes model loading, output
I/O, and the common MiniLM-prefix extraction. Both models are resident together,
so peak memory is not an isolated per-model memory benchmark. Reranking timings
are not end-to-end audio-to-answer latency.

## Execution

- ICF Teaching job: `3724137`.
- Allocated node: `landonia11`; NVIDIA RTX A6000, 49,140 MiB reported.
- Existing CUDA ASR environment is reused; no baseline dependency upgrades.
- The four pure evaluation/template tests run before a two-query GPU smoke run,
  followed by the full practice run.
- Remote output: `rag/results/qwen06-retrieval-3724137/` and the corresponding
  Slurm log. Completion requires a full summary and successful Slurm exit.

Source: [runner](../extensions/evaluate_qwen_reranker.py),
[Slurm entry point](../scripts/icf_qwen_retrieval.sbatch),
[official Qwen model and scoring example](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

## 0.6B results

The full 350-query 0.6B run completed successfully (`3724137`, Slurm exit
`0:0`, 31m07s including startup and the two-query smoke run). Results were
downloaded and independently recomputed from all saved rankings and public
practice qrels. Query coverage, frozen transcripts, candidate identity, tie
ordering, and all six aggregate metrics passed verification.

- Dense: Recall@5 **0.421337**, Recall@10 0.479594, Recall@20 0.578524,
  Recall@100 0.769856, MRR@10 0.488746, NDCG@10 **0.420461**.
- MiniLM-512: Recall@5 **0.387689**, Recall@10 0.473867,
  Recall@20 0.547789, Recall@100 0.769856, MRR@10 0.453454,
  NDCG@10 **0.395219**. Mean/median/p95 reranking time:
  330.9/332.9/381.6 ms per query. Recall@5 exactly reproduces the earlier
  full-practice pool-100 MiniLM result; timing is from this new batch-8 run.
- Qwen matched source (primary): Recall@5 **0.467113**, Recall@10 0.549809,
  Recall@20 0.631344, Recall@100 0.769856, MRR@10 0.498499,
  NDCG@10 **0.454397**. Mean/median/p95 reranking time:
  1617.6/1629.7/1787.0 ms per query.
- Qwen native-2048 (secondary): Recall@5 **0.467514**, Recall@10 0.550904,
  Recall@20 0.633011, Recall@100 0.769856, MRR@10 0.492680,
  NDCG@10 **0.452019**. Mean/median/p95 reranking time:
  2018.9/2028.0/2570.6 ms per query.

For primary Qwen vs dense, mean Recall@5 improves by **4.58 percentage points**
(descriptive paired-bootstrap 95% interval **+1.55 to +7.77 points**): 59 queries
improve, 37 worsen, and 254 tie. NDCG@10 improves by **3.39 points**
(interval **+0.94 to +5.94 points**). MRR@10 improves by 0.98 points, but its
interval crosses zero (-2.18 to +4.17 points); evidence for that gain is weaker.

Mean primary Qwen scoring time is about 4.89 times the replayed MiniLM scoring
time. Native-2048 adds about 0.40 seconds/query to primary Qwen, while its mean
Recall@5 is only 0.04 percentage points higher and NDCG@10 is slightly lower.
This secondary comparison does not establish a worthwhile longer-input benefit.

The labelled top-5 oracle within the same candidate pools is 0.764417. All
methods preserve Recall@100. The two Qwen arms have identical ordered top-5
lists on 226/350 queries. Native-2048 truncates 4/35,000 documents; matched
inputs require no additional Qwen truncation. Character-prefix shortening can
also remove trailing whitespace, so it is not itself a token-truncation count.

Peak allocated GPU memory with both resident models is 1.71 GiB; the measured
full inference/output loop took 1437.9 seconds. These are retrieval-only
measurements. **No answer-quality or end-to-end latency improvement has yet
been demonstrated.** The result supports checkpoint choice as a relevant factor,
not parameter count as an isolated causal explanation.

Data: [full summary](../results/qwen06-retrieval-3724137/full/summary.json),
[saved rankings](../results/qwen06-retrieval-3724137/full/rankings.jsonl),
[independent audit](../results/qwen06-retrieval-3724137/full/independent-audit.json),
[frozen original runner](../results/qwen06-retrieval-3724137/runner_at_submission.py).

## 4B comparison protocol

The user requested `Qwen/Qwen3-Reranker-4B` as an additional comparison while
0.6B was still running. Its pinned revision is
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`. Use the same 350 queries,
top-100 candidate lists, MiniLM reference, official instruction/template,
matched-source and native-2048 arms, BF16, batch size 8, and A6000 GPU type.
The 4B job is scheduled after successful 0.6B completion. Compare quality and
measured costs without assuming that larger size guarantees a gain.

ICF job `3724174` was submitted with `afterok:3724137`, one RTX A6000,
2 CPUs, 24 GiB host RAM, and a 3-hour wall limit. Weights were downloaded and
SHA-256 checked before submission. Its output root is
`rag/results/qwen4-retrieval-3724174/`;
[4B Slurm entry point](../scripts/icf_qwen4_retrieval.sbatch).

The original 0.6B runner was saved remotely as
`rag/results/qwen06-retrieval-3724137/runner_at_submission.py` before extending
the CLI for both model sizes. Its SHA-256 is
`a5095af678928392380b4227cdff3430d27bfd1cdd647e5a244220b988858fa6`.
The running process and its saved configuration continue to describe the
original 0.6B implementation.

## 4B results

Job `3724174` completed with exit `0:0` in **1h24m29s**, including initialization
and the smoke run. The full inference/output loop took 4718.9 seconds. All 350
queries and all six metrics were independently recomputed successfully. Both
runs use identical transcripts, dense candidate orderings, and MiniLM orderings;
their input hashes, scoring templates, batch size and GPU type match.

- Qwen-4B matched source (primary): Recall@5 **0.526226**, Recall@10
  **0.620013**, Recall@20 **0.688168**, Recall@100 **0.769856**, MRR@10
  **0.595635**, NDCG@10 **0.534080**. Mean/median/p95 reranking time:
  **5743.4/5822.7/6420.3 ms** per query.
- Qwen-4B native-2048 (secondary): Recall@5 **0.530559**, Recall@10
  **0.625966**, Recall@20 **0.690549**, Recall@100 **0.769856**, MRR@10
  **0.599495**, NDCG@10 **0.539752**. Mean/median/p95 reranking time:
  **7264.5/7327.5/9323.7 ms** per query.

### Paired evidence and cost

Against dense, primary 4B improves Recall@5 by **10.49 percentage points**
(query-bootstrap 95% interval **+7.26 to +13.78 points**): 82 queries improve,
22 worsen and 246 tie. MRR@10 improves by **0.106889** (interval
**+0.072598 to +0.142935**); NDCG@10 improves by **0.113620** (interval
**+0.087612 to +0.139758**). Thus the observed benefit extends beyond Recall@5.

Directly pairing 4B against 0.6B under the matched-source protocol yields:

- Recall@5 difference **+0.059113**, 95% interval **+0.034639 to +0.084130**;
  62 queries improve, 19 worsen and 269 tie.
- MRR@10 difference **+0.097136**, interval **+0.066273 to +0.129838**.
- NDCG@10 difference **+0.079683**, interval **+0.058985 to +0.100887**.
- Mean matched-source scoring latency is about **3.55 times** 0.6B
  (5.74 vs 1.62 seconds/query). These are scoring costs, not ELM or end-to-end
  audio-to-answer times.

For 4B, native-2048 vs matched source adds **0.004333 Recall@5** (0.43 percentage
points), with a difference interval **-0.003571 to +0.013715**. Only 5 queries
improve and 3 worsen on that metric. NDCG@10 improves by a smaller **0.005672**
(interval **+0.001512 to +0.010327**), while mean scoring time increases by
**1.52 seconds/query**, approximately 26.5%. This is evidence for a modest
additional ranking benefit, not a large or clearly established Recall@5 gain.

The native-vs-matched and direct size intervals above use 2,000 paired query
resamples with `random.Random(20261116)` and linear percentile interpolation.
They are descriptive, unadjusted for multiple comparisons or adaptive model
selection. Intervals reported in the individual run summaries use NumPy's RNG;
the resampling implementations are recorded separately.

Peak allocated GPU memory is **8.88 GiB** with MiniLM and 4B both resident,
compared with 1.71 GiB for MiniLM and 0.6B. This is not isolated model memory.
Native-2048 truncates 4/35,000 documents; matched text is not truncated again by
Qwen. Matched/native ordered top-5 lists coincide on 223/350 queries.

### Interpretation and next experiment

The earlier MiniLM failure does not generalize to all rerankers. This experiment
shows that checkpoint choice affects early evidence selection in this system:
Qwen 0.6B improves over dense, and 4B improves further on the measured retrieval
metrics at higher cost. This comparison does not isolate parameter count from
training and architectural differences. Both Qwen sizes remain useful quality
and latency trade-offs; 4B does not dominate 0.6B on cost.

Recall@100 stays constant because no new candidates are retrieved. The result
concerns ordering the existing pool, not recovering documents outside it.
The approximately 0.7644 labelled top-5 oracle is a ceiling, not a deployable
method or an expectation that scaling alone will reach it.

Next evaluate paired answers for dense5, Qwen-0.6B top-5 and Qwen-4B top-5,
holding the generator and prompt protocol fixed and examining context-token
costs, answer quality, and end-to-end latency. Keep the matched-source Qwen
comparison primary. No answer generation or further GPU jobs were initiated
while analysing these retrieval results.

Data: [4B full summary](../results/qwen4-retrieval-3724174/full/summary.json),
[4B saved rankings](../results/qwen4-retrieval-3724174/full/rankings.jsonl),
[4B independent audit](../results/qwen4-retrieval-3724174/full/independent-audit.json),
[paired size and input comparison](../results/qwen4-retrieval-3724174/paired-size-comparison.json).
