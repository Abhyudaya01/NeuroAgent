# Reproducibility Notes

## Goal

NeuroAgent is intended to be a reproducible research prototype, not just a local demo. This document summarizes the current reproducibility story and its limitations.

## Environment

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables for standard runs:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Optional:
- `OPENAI_CHAT_MODEL`
- `WANDB_API_KEY`
- `E2B_API_KEY`
- `NEURO_AGENT_API_KEY`

## Retrieval Memory

The retrieval index is generated from the markdown files in [data/seed_docs](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/data/seed_docs).

To rebuild:

```bash
python -m agent.tools.ingest --input-dir data/seed_docs
```

Generated artifacts:
- `faiss_index.index`
- `faiss_index.json`

These are derived files and may vary slightly depending on dependency versions and embedding backend behavior.

## Benchmark Inputs

Current benchmark files:
- [questions.json](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/eval/questions.json)
- [questions_50.json](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/eval/questions_50.json)

The 50-question file is intended as the main benchmark candidate for future aggregate comparisons.

## Running Ablations

Example commands:

```bash
python eval/ablation.py --config react_baseline --questions eval/questions.json
python eval/ablation.py --config neuro_no_memory --questions eval/questions.json
python eval/ablation.py --config neuro_full --questions eval/questions.json
```

For the larger benchmark:

```bash
python eval/ablation.py --config react_baseline --questions eval/questions_50.json
python eval/ablation.py --config neuro_no_memory --questions eval/questions_50.json
python eval/ablation.py --config neuro_full --questions eval/questions_50.json
```

Saved summaries are written to [eval/results](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/eval/results).

## What Is Reproducible Today

- graph execution logic
- benchmark question files
- saved JSON outputs for completed ablation runs
- test suite behavior
- API surface and deployment scaffolding

## What Is Still Variable

- LLM outputs are non-deterministic across time and model revisions
- web search results can drift based on indexing and recency
- rate-limit timing can alter latency and retry patterns
- benchmark scoring is still incomplete and not yet publication-grade

## Practical Recommendation

For internal comparisons, use:
- the same model
- the same question set
- the same environment keys
- the same retrieval corpus
- the same saved code revision

That will give the most stable comparisons currently possible within the project.
