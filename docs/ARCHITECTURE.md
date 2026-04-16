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
- returns structured evidence blocks rather than a synthesized paragraph

### Retrieval

`agent/tools/retrieval.py`

- queries the local FAISS index
- supports retry-time broader retrieval
- returns content and source metadata

### Synthesis

The synthesis path is model-driven rather than code-driven.

It takes prior task findings and their actual sources, then asks GPT-4o to:
- name approaches
- compare them concretely
- recommend a best answer with cited evidence

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
