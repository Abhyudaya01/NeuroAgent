# Contributing to NeuroAgent

Thanks for taking an interest in NeuroAgent.

This repository is structured as a research prototype, so contributions are especially valuable when they improve:
- reliability
- reproducibility
- benchmark quality
- source grounding
- evaluation rigor

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` from [`.env.example`](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/.env.example) and set the API keys you need.

## Before Opening a PR

Please run:

```bash
pytest
python -m py_compile agent/graph.py api/main.py eval/ablation.py
```

If your change affects retrieval memory, rebuild the FAISS index:

```bash
python -m agent.tools.ingest --input-dir data/seed_docs
```

## Contribution Priorities

The most useful contributions right now are:
- better benchmark scoring and evaluation metrics
- cleaner result aggregation in `eval/results`
- improved retrieval ranking and deduplication
- stronger synthesis prompts and output structure
- deployment and usability improvements
- additional high-quality seed documents with citations

## Coding Style

- Keep functions small and composable.
- Prefer explicit state transitions over hidden side effects.
- Do not silently swallow errors unless the code has a deliberate fallback path.
- When changing benchmark behavior, update tests and document the new assumptions.

## Research-Oriented Changes

If you change evaluation logic, please include:
- what changed
- why it improves measurement or reliability
- what comparison or metric it affects

That context is especially helpful for this repository because it is intended to support research-style iteration rather than only product development.
