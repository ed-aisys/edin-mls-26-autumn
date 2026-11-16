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

Full answer evaluation is still running. The two-question generation/judge
checks passed for both arms. Their scores are setup checks and do not estimate
the full experiment's answer quality.

The [2026-11-16 22:03:46 UTC checkpoint](../results/qwen4-answers-3724290/checkpoints/20261116T220346Z/checkpoint.json)
contains 240 completed answers per arm on the same 240 query IDs. The live files
held 241 dense5 answers and 240 Qwen answers at capture; the checkpoint excludes
the unmatched dense5 answer. It contains the original frozen configuration,
generation fingerprint and subset judge outputs, with file hashes. Its status
is `in_progress_checkpoint`, and full answer scoring is incomplete.

Continue the original job to obtain the 350-query paired answer comparison.
Use the checkpoint for inspection and backup; resuming generation requires the
live run directory with its full plans and caches. Retrieval improvements alone
do not establish answer-quality gains.
