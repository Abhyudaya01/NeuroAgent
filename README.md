# NeuroAgent

NeuroAgent is a modular research-oriented LLM agent for studying reliability in tool-using language model systems. It combines planning, evidence retrieval, critique, retries, and source-grounded synthesis in a single LangGraph workflow.

The project is designed less like a chatbot demo and more like a small experimental platform: components are separable, ablations are supported, run traces are logged, and local retrieval memory can be toggled on or off to study its effect on groundedness and retry behavior.

## Live Demo

- API: [https://neuro-agent-api.onrender.com/](https://neuro-agent-api.onrender.com/)
- Docs: [https://neuro-agent-api.onrender.com/docs](https://neuro-agent-api.onrender.com/docs)
- Health: [https://neuro-agent-api.onrender.com/health](https://neuro-agent-api.onrender.com/health)

## Why This Project Exists

Many LLM projects stop at one-shot prompting or a shallow RAG wrapper. NeuroAgent is built around a different question:

How do planning, retrieval memory, self-critique, and synthesis interact to improve reliability in LLM agents?

To explore that, NeuroAgent implements:
- a planner that decomposes research questions into executable tasks
- an executor that uses search, retrieval, synthesis, and code tools
- a Reflexion-style critic loop that evaluates intermediate outputs
- retry logic that rewrites retrieval or synthesis behavior based on criticism
- local retrieval memory backed by FAISS and seeded technical documents
- ablation configurations for baseline vs. no-memory vs. full-agent comparisons

## Current Capabilities

- Multi-stage LangGraph workflow with `planner -> executor -> critic -> report`
- Tool support for:
  - `web_search`
  - `document_retrieval`
  - `synthesis`
  - `code_interpreter`
- Retry-aware retrieval query reformulation using critic feedback
- Source-aware synthesis prompts that carry forward real URLs and filenames
- FastAPI deployment surface for live hosting
- Weights & Biases logging for task-level traces and final reports
- Evaluation scaffold for `react_baseline`, `neuro_no_memory`, and `neuro_full`
- FRAMES benchmark preparation and failure-mode analysis utilities

## Repository Layout

```text
NeuroAgent/
├── agent/
│   ├── graph.py                 # LangGraph wiring and run entry point
│   ├── llm_retry.py             # Shared OpenAI rate-limit retry helper
│   ├── state.py                 # Shared graph state contract
│   ├── nodes/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   └── report.py
│   └── tools/
│       ├── search.py
│       ├── retrieval.py
│       ├── code_interpreter.py
│       └── ingest.py
├── api/
│   ├── main.py                  # FastAPI deployment app
│   └── schemas.py               # API request/response models
├── data/seed_docs/              # Local technical corpus for retrieval memory
├── eval/
│   ├── ablation.py              # Benchmark/ablation runner
│   ├── analyze_failures.py      # Failure-mode analysis for saved benchmark runs
│   ├── frames.py                # FRAMES benchmark adapter
│   ├── questions.json           # Small starter question set
│   └── questions_50.json        # Expanded 50-question benchmark set
├── eval/results/                # Saved ablation outputs
├── frontend/
│   └── app.py                   # Optional UI workbench
├── tests/
│   ├── test_api.py
│   ├── test_ablation.py
│   ├── test_graph.py
│   └── test_llm_retry.py
├── render.yaml                  # Render deployment manifest
├── requirements.txt
└── README.md
```

## Architecture

NeuroAgent is intentionally modular.

1. The planner decomposes a research question into 3 to 7 structured tasks.
2. The executor routes each task to the best available tool.
3. The critic evaluates the output for factual grounding, source reliability, completeness, and specificity.
4. Weak outputs are retried with updated prompts or rewritten retrieval queries.
5. The report node composes a source-grounded final summary.

Additional design notes are documented in [ARCHITECTURE.md](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/docs/ARCHITECTURE.md).

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy [`.env.example`](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/.env.example) to `.env` and set the keys you need.

Required for most agent runs:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Optional but supported:
- `OPENAI_CHAT_MODEL`
- `WANDB_API_KEY`
- `E2B_API_KEY`
- `NEURO_AGENT_API_KEY`

### 3. Build the retrieval index

```bash
python -m agent.tools.ingest --input-dir data/seed_docs
```

This creates:
- [faiss_index.index](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/faiss_index.index)
- [faiss_index.json](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/faiss_index.json)

## Running the Agent

### CLI-style graph run

```bash
python -m agent.graph
```

### Python entry point

```python
from agent.graph import run_agent

state = run_agent(
    question="What are the best recent approaches for long-context memory in LLM agents?",
    session_id="demo-run",
)
print(state["final_report"])
```

## API and Local Hosting

NeuroAgent includes a minimal FastAPI service for local use and public deployment.

### Run locally

```bash
export NEURO_AGENT_API_KEY="your-secret-key"
uvicorn api.main:app --reload
```

Then open:
- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example API request

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question":"What are the best recent approaches for long-context memory in LLM agents?"}'
```

### Deploy on Render

This repository includes [render.yaml](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/render.yaml) for a minimal Render deployment.

Required deployment env vars:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `NEURO_AGENT_API_KEY`

Optional deployment env vars:
- `WANDB_API_KEY`
- `E2B_API_KEY`
- `OPENAI_CHAT_MODEL`

## Evaluation and Ablations

The repository includes an evaluation scaffold for three configurations:

- `react_baseline`
- `neuro_no_memory`
- `neuro_full`

Run the small benchmark:

```bash
python eval/ablation.py --config react_baseline --questions eval/questions.json
python eval/ablation.py --config neuro_no_memory --questions eval/questions.json
python eval/ablation.py --config neuro_full --questions eval/questions.json
```

Run the larger benchmark:

```bash
python eval/ablation.py --config react_baseline --questions eval/questions_50.json
python eval/ablation.py --config neuro_no_memory --questions eval/questions_50.json
python eval/ablation.py --config neuro_full --questions eval/questions_50.json
```

Saved outputs are written to [eval/results](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/eval/results).

### FRAMES benchmark path

Prepare a 50-question FRAMES subset:

```bash
python eval/frames.py --limit 50 --output eval/frames_questions_50.json
```

Then run the same ablation configs:

```bash
python eval/ablation.py --config react_baseline --questions eval/frames_questions_50.json
python eval/ablation.py --config neuro_no_memory --questions eval/frames_questions_50.json
python eval/ablation.py --config neuro_full --questions eval/frames_questions_50.json
```

Analyze failure modes across saved runs:

```bash
python eval/analyze_failures.py \
  eval/results/react_baseline_latest.json \
  eval/results/neuro_no_memory_latest.json \
  eval/results/neuro_full_latest.json \
  --output eval/results/frames_failure_analysis_latest.json
```

See [FRAMES_EVALUATION.md](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/docs/FRAMES_EVALUATION.md) for the methodology and caveats.

Current evaluation notes and reproducibility guidance are documented in [REPRODUCIBILITY.md](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/docs/REPRODUCIBILITY.md).

## Seed Memory Corpus

The local retrieval memory currently includes technical notes on:
- long-context memory
- memory architecture patterns
- hallucination reduction techniques
- Reflexion and self-evaluation
- RAG for agents
- MemGPT and episodic memory
- streaming inference and prompt compression
- reliability evaluation for agentic systems

These files live in [data/seed_docs](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/data/seed_docs).

## Logging and Observability

NeuroAgent logs:
- planner task tables
- critic scores
- retries
- final confidence
- final report HTML

using Weights & Biases when `WANDB_API_KEY` is configured.

## Testing

Run the full test suite:

```bash
pytest
```

The current suite covers:
- planner behavior
- executor routing
- critic retry logic
- search and retrieval formatting
- API behavior
- ablation output persistence
- rate-limit retry handling

## Current Status

NeuroAgent is best described as a working research prototype.

What is implemented:
- modular agent pipeline
- retrieval memory
- Reflexion-style critique loop
- source-grounded synthesis
- API deployment surface
- ablation runner and saved result artifacts

What is still in progress:
- stronger automatic scoring
- broader benchmark analysis
- richer result aggregation
- publication-quality evaluation metrics

## Contributing

Please see [CONTRIBUTING.md](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/CONTRIBUTING.md).

## Citation

If you reference or build on this repository, please use the metadata in [CITATION.cff](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/LICENSE).
