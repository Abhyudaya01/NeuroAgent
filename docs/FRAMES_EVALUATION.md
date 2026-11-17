# FRAMES Evaluation

## Purpose

FRAMES (Factuality, Retrieval, And reasoning MEasurement Set) is a multi-hop RAG benchmark from Google DeepMind and Harvard. It contains 824 questions with gold answers and supporting Wikipedia articles. NeuroAgent uses it as an end-to-end stress test for retrieval, reasoning, source grounding, and retry behavior.

## Prepare A FRAMES Question Subset

Install dependencies, then prepare a 50-question subset:

```bash
pip install -r requirements.txt
python eval/frames.py --limit 50 --output eval/frames_questions_50.json
```

The loader uses a repo-local Hugging Face cache at `eval/cache/huggingface` by default.

If you already downloaded FRAMES as a CSV, JSON, or JSONL file, prepare it offline:

```bash
python eval/frames.py --input-file path/to/frames.csv --limit 50 --output eval/frames_questions_50.json
```

The generated file keeps:
- `id`
- `question`
- `answer`
- `reasoning_types`
- `wiki_links`

## Run Controlled Ablations

Run each configuration on the same FRAMES subset:

```bash
python eval/ablation.py --config react_baseline --questions eval/frames_questions_50.json
python eval/ablation.py --config neuro_no_memory --questions eval/frames_questions_50.json
python eval/ablation.py --config neuro_full --questions eval/frames_questions_50.json
```

Saved summaries are written to `eval/results/`. Each result includes the final report, internal confidence, retry counts, completion counts, and FRAMES gold-answer metadata when present.

## Analyze Failure Modes

After all three runs finish, analyze the saved summaries:

```bash
python eval/analyze_failures.py \
  eval/results/react_baseline_latest.json \
  eval/results/neuro_no_memory_latest.json \
  eval/results/neuro_full_latest.json \
  --output eval/results/frames_failure_analysis_latest.json
```

The analyzer reports:
- premature termination rate
- likely critic miscalibration cases
- average retry pressure
- example question IDs for inspection

## Interpretation Caveat

The current `answer_present` check is a lightweight gold-answer presence heuristic. It is useful for triage, but it is not a substitute for a full LLM-judge or human-verified FRAMES accuracy protocol. For resume and demo purposes, describe the benchmark as an ablation and failure-analysis pass, not as a state-of-the-art FRAMES submission.

## Checkpointed runs and usage measurement

Start with a one-question smoke test for each configuration. Use the same
`OPENAI_CHAT_MODEL` for the planner and critic across configurations.
Synthesis and retrieval-query rewriting currently use GPT-4o.

```bash
OPENAI_CHAT_MODEL=gpt-4o WANDB_MODE=disabled python eval/ablation.py \
  --config react_baseline --questions eval/frames_questions_50.json \
  --limit 1 --output-dir eval/results/frames_smoke
```

Run the full subset by omitting `--limit`. Use a separate output directory,
for example `eval/results/frames_50`. Repeat for `neuro_no_memory` and
`neuro_full`. `WANDB_MODE=disabled` keeps smoke tests local. Omit it to
use the configured W&B integration.

The runner saves an atomic checkpoint after each attempted question. Resume
an interrupted run using the same arguments plus `--resume PATH_TO_CHECKPOINT`.
A resume verifies configuration, question contents, model settings, agent and
evaluation source hashes, and retrieval-index contents. It skips already
recorded question IDs, including errors. Start a new run to retry errors.
The `latest` alias is updated only after the question loop finishes.
Authentication, permission, and quota failures stop the run with `blocked`
status and a nonzero CLI exit. Other question errors are recorded and the
runner continues.

Each question records elapsed seconds, task details, tool confidence, critic
confidence where available, and chat input/output tokens grouped by the
provider-reported model. Usage covers LangChain chat calls, including calls
completed before a question fails. Embedding usage, Tavily, and E2B charges
are outside that scope. Token counts provide a cost comparison without
assuming a price schedule. Dollar estimates require dated model prices and
external tool billing data.

Keep partial runs separate from completed comparisons. The runner's
`answer_presence_rate` remains a substring heuristic over the report.

## Judge-based answer scoring

Grade a completed run against its saved gold answers with a separate judge.
The judge receives the research question, reference answer, and final report.
It extracts the asserted answer and accepts semantically equivalent answers.
Judge calls have their own usage records, separate from agent usage.

```bash
python -m eval.scoring eval/results/frames_50/neuro_full_latest.json \
  --output eval/results/frames_50/neuro_full_grades.json --model gpt-4o
```

Repeat with the same judge for the other configurations. Scoring saves a
checkpoint after each verdict and resumes automatically when rerun with the
same output. Source contents, judge model, and rubric must match. Failed
agent responses count as incorrect. Judge request failures leave previously
saved verdicts intact and surface the error for retry.

`judge_accuracy` reports the fraction correct among scored gold questions.
Use it for comparisons only after scoring status is `completed` and all
intended benchmark questions have run. Review judge explanations manually
before making accuracy claims, especially ambiguous or numeric answers.

## Failure analysis definitions

`incomplete_output_rate` measures reports with missing planned task outputs.
`premature_termination_rate` requires a saved task index showing the run ended
before reaching the end of its plan. Older artifacts lack that observation,
so their premature termination rate is unavailable. Operational exceptions
are reported separately. Critic calibration checks require saved critic
scores. Historical tool-confidence averages cannot establish critic
calibration. Answer-presence-based calibration flags remain triage heuristics.

## Budget-limited comparisons

For GPT-4o runs, `--max-chat-cost-usd` adds a stop between questions. Estimates
use $2.50 per million input tokens and $10.00 per million output tokens,
checked on 2026-11-17 at https://developers.openai.com/api/docs/models/gpt-4o.
Cached inputs use the full input price, making this a conservative chat
estimate. The runner reserves at least $0.50, or twice the most expensive
observed question, before starting another question. An individual question
can exceed that reserve, so the option is an estimated spending limit.
Embedding and external-tool charges require separate accounting.

A budget stop saves `budget_limited` status and preserves the previous
completed `latest` file. Resume with a larger budget to continue, or compare
only the question IDs shared by every configuration. Keep budget-stopped
runs explicitly labeled in reported results.

Positive grades require a supporting `answer_quote` recovered from the report.
The source-span matcher preserves words and numeric values while tolerating
punctuation formatting. Every component of the predicted answer must appear in
that span. A multi-part answer may join supported components with "and". The
judge must assess an asserted answer rather than an incidental mention or a plan
to calculate it. A malformed positive quotation receives one format-correction
retry. Use a new grading output path for the revised validation policy. Existing
checkpoints have a different policy identity. See [FRAMES_REAUDIT.md](FRAMES_REAUDIT.md).

For list responses, `answer_parts` records exact item spans in the order asserted
by the report. Validation preserves this order and still requires a reported
answer anchor. Semantic equivalence decisions and any post-run adjudications
must be retained separately from raw grading artifacts. The fresh ten-question
pilot and its reviewed verdicts are documented in [FRESH_EVALUATION.md](FRESH_EVALUATION.md).
