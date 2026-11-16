# Qwen-4B top-5: paired answer evaluation

## Protocol (fixed before answer scores)

Question: does the observed Qwen-4B retrieval improvement transfer to better
answers from the unchanged EuroLLM generator?

- All 350 practice queries, using the same frozen ASR transcripts as the
  completed retrieval study. Dev is not rerun or used for tuning.
- Input rankings: `qwen4-retrieval-3724174/full/rankings.jsonl`.
- Primary arms: `dense5` and `qwen4_matched5`. The latter uses the predeclared
  `qwen_matched` ordering, not a new reranker run or a post-hoc question subset.
- Both arms pass the original full text of their selected five records to the
  same existing evidence prompt and system instruction. MiniLM-visible-prefix
  matching applied during reranker scoring only; it does not truncate generator
  evidence. No qrels are loaded while choosing records or building prompts.
- Generator: `utter-project/EuroLLM-22B-Instruct-2512`, temperature 0.1.
  No prompt changes, answer repair, or query rewriting.
- No context truncation. A 28,672-token input preflight ceiling stops execution
  on overflow. Input token counts are measured; this is a top-5 comparison,
  not a claim of equal token budgets.
- Fresh dense5 answers are generated because the ICF checkout lacks the earlier
  350-query raw dense5 answers and judgements. Earlier aggregate scores are
  historical context, not this experiment's paired control.
- Per-question arm order is deterministically shuffled before API calls.
  Incremental answers are saved with prompt hashes and resume fingerprints.
- Judge: unchanged `scripts/evaluate_answers.py`, EuroLLM-22B, JSON mode,
  one judgement per answer. Correctness, grounding and completeness remain on
  0–4 scales; normalised score is their sum divided by 12. Same public qrel gold
  reconstruction and transcript-as-question convention as previous experiments.
- Two questions per arm first check generation, normal stop completions, and
  valid scoring. Their judge cache is reused during full evaluation. Successful
  checks automatically continue to the full 700 answers and 700 judgements.
- Source/model/input/prompt hashes are frozen. Scoring rubric and evaluator
  source files are not edited.

Report paired score changes and query-bootstrap 95% intervals (2,000 resamples,
seed 20261116), per-dimension changes and wins/losses/ties. A single generation
and judgement per answer does not quantify repeat-run uncertainty. Same-model
generation/judging and incomplete gold evidence remain limitations; inspect
representative paired cases before interpreting mechanisms.

Measure fresh generation latency and API token usage. Reranker latency is
attributed from the preceding GPU run, not recomputed. No audio-to-answer
end-to-end latency claim follows from this cached-input experiment.

## Execution

- Initial ICF Teaching CPU job: **3724289**, 2 CPUs, 8 GiB host RAM, 4-hour limit.
- No GPU is requested: local work packs evidence; ELM serves generation/judging.
- Retained full-run job: **3724290**. Remote root:
  `rag/results/qwen4-answers-3724290/`.
- Retained log: `rag/results/qwen4-answers-3724290.log`.
- [Experiment runner](../extensions/run_qwen_answer_experiment.py),
  [Slurm entry point](../scripts/icf_qwen4_answers.sbatch),
  [existing evaluator](../scripts/evaluate_answers.py).

A second identical job (`3724290`) appeared under the same account shortly after
the original submission. To avoid continuing duplicate ELM calls without
interfering with an unconfirmed external submission, the assistant cancelled its
own job `3724289` and retained `3724290`. The decision was operational, before
full answer scores were available. Partial output from the cancelled job remains
saved; it is not merged into the retained experiment or selected by score.

## Results

Job `3724290` completed with Slurm exit `0:0` in **1h40m45s**, covering 350
answers and 350 judgements per arm. This wall time includes preparation,
generation and judging; it is not per-query inference latency.

The unchanged public evaluator gives dense5 a mean normalised answer score of
**0.337381** and Qwen-4B top-5 **0.381905**. The paired mean difference is
**+0.044524**, approximately 4.45 points on a 0-to-100 display scale, or a 13.2%
relative increase in the mean score. This score is not the percentage of
questions answered correctly.

The predeclared paired query bootstrap gives a 95% interval of
**+0.011190 to +0.078810**. The interval excludes zero under this single-run,
fixed-judge protocol. Across queries, 115 scores improve, 80 decline and 155 tie.
The interval describes query resampling uncertainty and does not include
repeated-generation/judge variability or adaptive model-selection uncertainty.

All three 0-to-4 dimensions increase:

- Correctness: **1.425714 to 1.585714**, mean difference +0.160000,
  95% interval +0.028571 to +0.297143.
- Grounding: **1.448571 to 1.648571**, difference +0.200000,
  interval +0.051429 to +0.351429.
- Completeness: **1.174286 to 1.348571**, difference +0.174286,
  interval +0.042857 to +0.308571.

Use the fresh dense5 control from this run for the comparison. Earlier dense5
scores came from different generations/judgements and are historical context.

### Retrieval benefit and remaining failures

The same paired questions previously showed Recall@5 increasing from 0.421337
to 0.526226. We now also observe a downstream answer-score gain. The generator,
prompt instructions and number of supplied records remain fixed, while selected
records and order change. Single stochastic generation/judging runs still limit
causal attribution.

Higher labelled recall does not guarantee a higher answer score on an individual
query. Of the 82 queries whose Recall@5 improves, 34 answer scores improve,
11 decline and 37 tie. When Recall@5 ties, answers can still change because the
selected documents or their order differ; generation/judge variation can also
contribute. These counts do not separate those mechanisms.

All-zero judgements decrease from **138/350 (39.43%)** to
**121/350 (34.57%)**. There are 53 zero-to-positive transitions and 36
positive-to-zero transitions. Full-score judgements increase from 38 to 50.
The remaining failures deserve inspection alongside the aggregate improvement.

An offline inventory divides the 121 Qwen all-zero questions into 62 with no
positive qrel in the selected five records, 29 with partial positive-qrel
coverage and 30 with full positive-qrel coverage. Full labelled coverage does
not prove that the answer is present, correctly interpreted or fairly judged.
The [zero-score investigation](qwen4-zero-score-investigation.md) freezes a
24-question diagnostic sample and specifies the next controls. All 350 questions
remain in the benchmark. These diagnostic materials do not establish a further
answer-quality improvement.

### Judge checks

The [offline audit](../results/qwen4-answers-3724290/answer-evaluation-audit.json)
verifies all query IDs, frozen prompts and selected records. It reconstructs gold
evidence from the course asset ZIP, verifies all cache keys, matches scores to
raw judge responses, recomputes normalised scores and means, and reproduces the
paired intervals and change counts. It makes no model or API calls.

Some judge explanations require manual review. For
`q_9a684db4ff009865`, the dense answer discusses investment selection, while the
gold evidence discusses settlement and unsettled funds. The judge explanation
attributes the settlement discussion to the candidate and says the gold does
not address selling/buying stocks. This reverses the contents of the two inputs.
It establishes a problem with that explanation, not the correct replacement
score or a population-wide judge error rate. The original scores remain intact.

Other zeros have substantive problems: for example, the Ireland pension answer
includes supported tax/bankruptcy points but adds claims unsupported by the gold
evidence. Its all-zero outcome warrants review for partial credit rather than
assuming either a wholly correct answer or a wholly reliable judgement.

### Cost

Mean measured generation time is **4.755 seconds** for dense5 and
**5.034 seconds** for Qwen-selected evidence. Mean server input tokens are
**1289.71** and **1267.31**, respectively, with all 700 server counts matching
the local estimates. Thus the higher score accompanies similar average input
length, not a large increase in context volume.

Adding previously measured rerank time gives an attributed selection-plus-
generation estimate of **4.757 seconds** for dense5 and **10.779 seconds** for
4B, approximately 2.27 times as long. The reranker itself costs a mean
5.743 seconds/query in its earlier GPU run. These combined values are not
measured end-to-end audio-to-answer latency; ASR and initial retrieval are cached.

The experiment supports a quality/cost trade-off: Qwen-4B improves public-judge
answer scores at higher processing cost. It does not establish an answer-quality
advantage over Qwen-0.6B, whose generated answers have not been evaluated here.
An optional next comparison is 0.6B top-5 answers under the same protocol.

Full data: [comparison](../results/qwen4-answers-3724290/comparison.json),
[dense5 scores](../results/qwen4-answers-3724290/judge_json_dense5/answer_judge_per_query.jsonl),
[4B scores](../results/qwen4-answers-3724290/judge_json_qwen4_matched5/answer_judge_per_query.jsonl),
[audit script](../extensions/audit_qwen_answers.py).

### Earlier checkpoint

The [2026-11-16 22:03:46 UTC checkpoint](../results/qwen4-answers-3724290/checkpoints/20261116T220346Z/checkpoint.json)
contains 240 completed answers per arm on the same 240 query IDs. The live files
held 241 dense5 answers and 240 Qwen answers at capture; the checkpoint excludes
the unmatched dense5 answer. It contains the original frozen configuration,
generation fingerprint and subset judge outputs, with file hashes. It records
the `in_progress_checkpoint` status at capture time; the full run above has
since completed.

Use the checkpoint for inspection and backup; resuming generation requires the
live run directory with its full plans and caches.
