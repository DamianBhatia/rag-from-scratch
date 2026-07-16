# Product Backlog

This backlog tracks proposed bug fixes, features, and engineering improvements. Items are grouped by repository area and are not commitments to a delivery date.

## Priority legend

- **P0** — Blocks safe or correct use
- **P1** — High-value capability or reliability fix
- **P2** — Important improvement
- **P3** — Nice-to-have or exploratory work

## Core ReAct agent (`ReAct/`)

### Features

- [ ] **P1 — Execute independent tool calls concurrently**
  - Detect multiple tool calls emitted in the same model turn and execute independent calls with a bounded worker pool or async task group.
  - Preserve deterministic observation ordering and associate each result with its tool-call ID.
  - Record individual latency plus total wall-clock latency, cancellation, timeout, and partial-failure details.
  - Add tests proving concurrency, ordering, timeout behavior, and isolation when one call fails.

- [ ] **P1 — Add a sandboxed Codex command-execution tool**
  - Create an ephemeral, network-restricted container for every execution session.
  - Mount only an explicitly approved workspace or disposable copy; do not expose host credentials, Docker sockets, or unrelated paths.
  - Apply CPU, memory, process, output-size, and wall-clock limits, and run as a non-root user with a read-only base filesystem.
  - Support command submission, streamed stdout/stderr, exit codes, cancellation, artifact collection, and guaranteed teardown.
  - Define an allow/deny policy and require explicit confirmation for writes or other side effects outside the disposable sandbox.
  - Add adversarial tests for path traversal, command injection, secret access, resource exhaustion, and cleanup after failure.

- [ ] **P1 — Support first-class tool-call IDs**
  - Preserve model-provided IDs in assistant messages, tool observations, events, and trajectory steps.
  - Correctly correlate results when the same tool is requested more than once in a turn.

- [ ] **P2 — Introduce tool metadata and execution policies**
  - Describe whether a tool is read-only, side-effecting, idempotent, parallel-safe, or requires approval.
  - Validate tool arguments against their JSON schemas before invoking Python callables.
  - Add per-tool timeout, retry, and concurrency settings.

- [ ] **P2 — Add reusable agent skills**
  - Load versioned skill instructions independently from tool schemas.
  - Record selected skill names and versions in each trajectory for reproducibility.

- [ ] **P2 — Add conversation persistence and resume support**
  - Save and reload sessions without coupling the agent loop to a specific storage backend.
  - Add message compaction or summarization when history approaches the model context limit.

- [ ] **P3 — Add planning and human-approval checkpoints**
  - Let the agent present a plan before executing multi-step or side-effecting work.
  - Support approve, reject, edit, and cancel decisions through a structured event protocol.

### Bug fixes and reliability

- [ ] **P1 — Handle streamed tool-call fragments correctly**
  - Merge partial tool-call names and argument fragments by tool-call index or ID instead of treating every chunk as a complete call.
  - Cover malformed, duplicated, and out-of-order chunks in tests.

- [ ] **P1 — Add model and tool timeouts with cancellation**
  - Prevent a stalled model stream or tool from blocking an agent run indefinitely.
  - Return a structured termination reason while retaining the partial trajectory.

- [ ] **P2 — Prevent duplicate execution during retries**
  - Add idempotency keys and explicit retry semantics for side-effecting tools.

- [ ] **P2 — Improve protocol compatibility**
  - Normalize Ollama response objects and dictionaries at one boundary.
  - Preserve provider-specific metadata without leaking it into evaluator contracts.

- [ ] **P2 — Validate message history before model calls**
  - Reject invalid roles, missing tool-call associations, and non-serializable content with actionable errors.

## Retrieval-augmented generation (`rag/`)

### Features

- [ ] **P1 — Integrate retrieval as an agent tool**
  - Expose retrieval through a typed tool schema while keeping the standalone RAG example usable.
  - Return source IDs, text, and similarity scores so answers can cite evidence.

- [ ] **P1 — Add grounded citations and source display**
  - Require generated claims to reference retrieved chunks.
  - Surface sources and scores in CLI and future UI responses.

- [ ] **P2 — Persist and incrementally update the vector index**
  - Cache embeddings on disk using content hashes and model/version metadata.
  - Re-embed only added or changed documents and remove stale entries.

- [ ] **P2 — Add configurable ingestion and chunking**
  - Support multiple text files, metadata, chunk size, overlap, and encoding selection.
  - Separate loading, chunking, embedding, retrieval, and generation into testable components.

- [ ] **P2 — Improve retrieval quality**
  - Add score thresholds, metadata filters, hybrid keyword/vector retrieval, and optional reranking.
  - Create a small retrieval evaluation suite for recall and ranking quality.

- [ ] **P3 — Add additional document formats**
  - Support Markdown, PDF, and HTML through optional loaders with clear dependency boundaries.

### Bug fixes and reliability

- [ ] **P1 — Resolve data paths relative to the module**
  - Make the example work regardless of the caller's current working directory.

- [ ] **P1 — Guard cosine similarity edge cases**
  - Validate vector dimensions and handle empty or zero-norm vectors without division-by-zero errors.

- [ ] **P2 — Prevent duplicate in-memory indexing**
  - Clear or deduplicate the current index when ingestion runs more than once in a process.

- [ ] **P2 — Add embedding and chat failure handling**
  - Report unavailable Ollama services or models clearly and add bounded retry behavior for transient failures.

- [ ] **P2 — Validate retrieval inputs**
  - Reject blank queries and invalid `top_n` values, and define behavior for an empty index.

## Evaluation platform (`evals-platform/`)

### Features

- [ ] **P1 — Evaluate concurrent tool execution**
  - Add cases and metrics for expected parallel groups, wall-clock savings, result correlation, partial failures, and deterministic ordering.

- [ ] **P1 — Add sandbox execution evaluations**
  - Test command success, artifacts, output truncation, timeouts, cancellation, policy denial, isolation, and cleanup.
  - Keep all fixtures disposable and ensure tests cannot mutate the developer's working tree.

- [ ] **P1 — Add run-to-run regression comparison**
  - Compare a candidate run against a baseline by case, metric, latency, token usage, and failure reason.
  - Highlight statistically meaningful regressions and export a machine-readable CI result.

- [ ] **P1 — Add model and prompt provenance**
  - Persist exact model identifiers, model parameters, tool-schema versions, skill versions, prompts, source revision, and environment metadata.

- [ ] **P2 — Add dataset and schema version migrations**
  - Introduce explicit SQLite migrations and case-suite upgrade tooling.
  - Preserve compatibility with existing local evaluation history.

- [ ] **P2 — Add richer metrics**
  - Track token usage, estimated cost, time to first token, per-tool latency percentiles, citation correctness, and retrieval recall.
  - Add configurable metric thresholds by tag and difficulty.

- [ ] **P2 — Add repeated and parameterized runs**
  - Execute cases multiple times across models, temperatures, prompts, and tool configurations.
  - Report variance and pass rates rather than relying only on one sample.

- [ ] **P2 — Add evaluation case authoring and import**
  - Create, validate, clone, tag, and edit cases from the dashboard.
  - Support JSONL import/export with validation previews.

- [ ] **P2 — Add CI-friendly headless execution**
  - Provide a CLI that selects cases, runs evaluations, exports JSON/JUnit, and exits non-zero when configured gates fail.

- [ ] **P3 — Add evaluator adapter plugins**
  - Register additional agent implementations behind the existing result contract and compare them in the same run.

### Bug fixes and reliability

- [ ] **P1 — Make interrupted runs recoverable**
  - Mark abandoned `running` records as interrupted on startup or resume them safely.
  - Persist enough progress to avoid losing completed case results.

- [ ] **P1 — Improve concurrent database access**
  - Add retry/backoff around transient SQLite lock errors and test simultaneous dashboard reads and evaluation writes.

- [ ] **P2 — Validate persisted JSON at storage boundaries**
  - Detect corrupt snapshots and return structured errors instead of failing an entire dashboard page.

- [ ] **P2 — Make batch cancellation explicit**
  - Allow users to stop pending cases, retain completed results, and finalize the run with a `cancelled` status.

- [ ] **P2 — Strengthen judge output handling**
  - Capture raw judge output, distinguish transport from validation errors, and support bounded retries for malformed responses.

## Interactive applications (`ReAct/` and `evals-platform/app.py`)

### Features

- [ ] **P1 — Create an interactive UI chatbot**
  - Add a Streamlit chat experience backed by the canonical `run_agent()` implementation.
  - Stream assistant tokens, tool calls, observations, elapsed time, and errors in expandable trace panels.
  - Preserve multi-turn sessions and provide new-chat, stop-generation, retry, and export controls.
  - Render citations and downloadable sandbox artifacts safely.
  - Add approval prompts before side-effecting tools execute.

- [ ] **P2 — Add runtime controls to the chatbot**
  - Select model, iteration limit, enabled tools, retrieval settings, and sandbox limits.
  - Validate settings and display model/service availability before a run begins.

- [ ] **P2 — Link chats to evaluation cases**
  - Convert a useful or failed conversation into a draft evaluation case.
  - Replay a stored chat against another model or agent configuration.

- [ ] **P2 — Improve accessibility and responsive layout**
  - Add keyboard navigation, clear status announcements, accessible colors, and usable narrow-screen layouts.

### Bug fixes and reliability

- [ ] **P1 — Isolate per-user UI state**
  - Ensure concurrent Streamlit users do not share conversation, progress, cancellation, or selected-run state.

- [ ] **P2 — Sanitize rendered model and tool output**
  - Treat generated HTML, links, terminal output, and filenames as untrusted content.

- [ ] **P2 — Avoid duplicate submissions on rerun**
  - Make chat and evaluation actions idempotent across Streamlit script reruns.

## Testing and quality (`evals-platform/tests/` and repository-wide)

- [ ] **P1 — Add dedicated tests for the RAG pipeline**
  - Use injected fake embedding and chat clients so tests remain offline and deterministic.

- [ ] **P1 — Add repository-wide linting and type checking**
  - Configure a formatter, linter, and static type checker with one documented command for local and CI use.

- [ ] **P1 — Add continuous integration**
  - Run unit tests, linting, type checks, dependency checks, and sandbox security tests on pull requests.

- [ ] **P2 — Add property-based protocol tests**
  - Generate malformed and fragmented model/tool payloads to verify the agent never crashes or executes unintended calls.

- [ ] **P2 — Add end-to-end smoke tests**
  - Cover CLI chat, interactive UI, evaluation persistence, and container teardown with model and container boundaries mocked where appropriate.

- [ ] **P2 — Track test coverage by area**
  - Publish separate coverage for the agent, RAG pipeline, evaluator, storage, and UI-supporting logic.

## Developer experience and documentation (repository root)

- [ ] **P1 — Add a root dependency and environment setup**
  - Provide a consistent Python version, dependency strategy, and one setup path for all repository examples.

- [ ] **P1 — Correct and verify entry-point documentation**
  - Ensure root documentation references the actual `rag/rag.py` and `rag/cat-facts.txt` paths and commands work from documented directories.

- [ ] **P2 — Add configuration templates**
  - Provide a safe example environment file with model names, timeouts, database paths, and sandbox limits; never include credentials.

- [ ] **P2 — Add structured logging and trace correlation**
  - Emit run, session, model-turn, and tool-call IDs consistently across the agent, chatbot, and evaluator.
  - Redact secrets and configurable sensitive fields before persistence or display.

- [ ] **P2 — Document threat models and trust boundaries**
  - Cover prompt injection, tool misuse, untrusted retrieved content, container escape, secret leakage, and unsafe artifact handling.

- [ ] **P2 — Add contribution guidance**
  - Document how to add tools, skills, RAG sources, metrics, migrations, tests, and backlog items.

- [ ] **P3 — Package reusable components**
  - Move importable code into a conventional source layout while preserving simple learning-oriented entry points.

## Suggested implementation order

1. Fix streamed tool-call assembly, add tool-call IDs, and introduce timeouts.
2. Add tool metadata, argument validation, and bounded concurrent execution.
3. Build the disposable command sandbox and its security test suite.
4. Create the interactive chatbot with approvals and cancellation.
5. Extend the evaluation platform for concurrency, sandboxing, provenance, and regression gates.
6. Refactor and harden the RAG pipeline, then expose retrieval as an agent tool.
7. Complete repository-wide CI, typing, structured logging, and security documentation.
