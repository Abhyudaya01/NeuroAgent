# NeuroAgent Architecture

## Overview

NeuroAgent is organized as a modular agent pipeline rather than a monolithic prompt.

The execution flow is:

1. `planner`
2. `executor`
3. `critic`
4. `report`

The graph is defined in [graph.py](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/agent/graph.py).

## Core Nodes

### Planner

The planner:
- reads the research question
- determines whether local retrieval memory is available
- decomposes the question into 3 to 7 tasks
- assigns each task a tool

Valid tools currently include:
- `web_search`
- `document_retrieval`
- `synthesis`
- `code_interpreter`

### Executor

The executor:
- selects the task’s tool
- runs the appropriate tool implementation
- stores the result, sources, confidence, and status
- resolves research output into an explicit answer with source passages or an evidence gap
- blocks dependent work until its prerequisite facts are resolved

Important retry-aware behaviors:
- retrieval retries rewrite the query using the critic’s verbal feedback
- synthesis tasks are routed away from fake code execution
- real code tasks still use the code interpreter path

### Critic

The critic evaluates each task output on:
- factual grounding
- source reliability
- completeness
- specificity

If the result is weak and retry budget remains, the graph loops back to the executor.
An explicit evidence gap forces a zero score. A prerequisite can receive one bounded
recheck when dependent work discovers a missing input. Transitive dependent results
are invalidated, while unrelated completed work is preserved. A calculation can
search for an additional fact on its first own retry. The global transition budget
routes to a partial report when exhausted.

### Report

The report node assembles:
- question
- summary
- last critic feedback
- task findings
- full source list

## Tools

### Search

`agent/tools/search.py`

- uses Tavily
- filters weak domains
- prefers stronger academic sources
- returns structured evidence blocks
- can extract relevant passages from the top two source pages on a first retry
- retains short factual passages during that retry for evidence resolution

### Retrieval

`agent/tools/retrieval.py`

- queries the local FAISS index
- supports retry-time broader retrieval
- returns content and source metadata

### Synthesis

Synthesis uses prior findings, original question constraints, retry feedback, and
actual source identifiers to answer the current subtask. It requests evidenced
inputs and a formula for calculations, and an explicit missing fact when needed.
The original question supplies context without requiring every intermediate step
to produce its final answer.

### Code Interpreter

The code path exists for genuinely executable Python tasks and can still be useful for analysis or structured manipulation when code execution is appropriate.

## State Model

Shared graph state includes:
- research question
- session id
- current plan
- current task index
- tool history
- critic score and feedback
- final report
- W&B task rows

This state is defined in [state.py](/Users/abhyudayalohani/Documents/New%20project/NeuroAgent/agent/state.py).

## Evaluation View

The architecture was designed to support ablations:
- baseline behavior
- no-memory behavior
- full critique + memory behavior

That is one of the key reasons the system is split into clean nodes rather than flattened into one chained prompt.

Evidence-gate implementation details, validation results, and grading corrections are recorded in [EVIDENCE_REPAIR.md](EVIDENCE_REPAIR.md).
