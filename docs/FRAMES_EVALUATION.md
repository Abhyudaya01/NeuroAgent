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
