# ReAct Agent Evals

A local MVP for executing, querying, visualizing, and manually reviewing the agent implemented in `../ReAct/main.py`. The dashboard calls the real `run_agent()` function; it does not maintain a copy of the agent loop.

## Features

- Single-prompt and repeatable batch evaluations
- Complete model/tool trajectory capture
- Deterministic tool, argument, observation, answer, iteration, and latency metrics
- Optional local Ollama judge, disabled by default
- SQLite history with manual pass/fail reviews
- Streamlit explorer and Plotly trend/latency charts
- Offline automated tests through injected model clients

## Setup

From this directory, create and activate a Python environment, then install dependencies:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Make sure Ollama is running and the default model is available:

    ollama pull llama3.2:3b

Launch the dashboard:

    streamlit run app.py

The SQLite database is created automatically at `data/evals.db` and is intentionally ignored by Git.

## Evaluation case format

Cases are stored in `data/eval_cases.json` using schema version 1. Each case supports:

- `id`: stable unique identifier
- `prompt`: input sent to the agent
- `expected_tool_calls`: tool names and exact normalized argument objects
- `expected_observation_terms`: case-insensitive substrings expected in tool observations
- `expected_answer_terms`: case-insensitive substrings expected in the final answer
- `forbidden_answer_terms`: substrings that mark an answer violation
- `tags`, `difficulty`, and optional `max_iterations`

Expected calls are matched without considering order. Missing calls reduce recall, unexpected calls reduce precision, and both affect F1. String arguments ignore case and repeated whitespace but do not use fuzzy edit-distance matching. When no calls are expected and none occur, tool precision, recall, and F1 are all 1. Metrics without expectations are stored as null.

## Optional judge

The judge receives the case, trajectory, tool observations, and final answer. It returns 0–1 scores for task success, grounding, and tool-use appropriateness. It is non-deterministic and supplementary: malformed output or an unavailable judge model is recorded without losing deterministic scores.

Environment overrides:

- `EVALS_AGENT_MODEL`
- `EVALS_MAX_ITERATIONS`
- `EVALS_JUDGE_MODEL`
- `EVALS_JUDGE_ENABLED`
- `EVALS_DATABASE_PATH`
- `EVALS_CASE_SUITE_PATH`

## Extending the platform

1. Add or change tools in `../ReAct/main.py` and expose their schemas and callables to `run_agent()`.
2. Add versioned cases to `data/eval_cases.json`.
3. Add new deterministic metrics in `evals_platform/metrics.py`.
4. Add fields as JSON first; normalize the SQLite schema only when query requirements justify it.

Stateful or side-effecting tools should eventually receive per-case setup, sandbox, and teardown hooks before being used in batch runs.

## Tests

Run from this directory:

    python -m pytest

Tests do not require a running Ollama server.
