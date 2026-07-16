"""Interactive Streamlit dashboard for the local ReAct evaluation platform.

The application is the presentation layer over :mod:`evals_platform`. It loads
environment-aware configuration, initializes SQLite persistence, reads the JSON
case suite, and delegates all execution to :class:`~evals_platform.evaluator.Evaluator`.
It never reimplements the agent loop: evaluations reach the canonical
``ReAct.main.run_agent`` function through ``ReactAdapter``.

Three tabs cover the evaluation workflow: an aggregate overview with Plotly
quality and latency charts, a batch/ad-hoc execution form with live trajectory
events, and a filterable result explorer with manual pass/fail review. Streamlit
reruns this module after interactions, while durable state remains in SQLite and
small UI-only state remains in ``st.session_state``.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from evals_platform.config import load_config
from evals_platform.evaluator import Evaluator
from evals_platform.models import EvalCase, EvalSettings, ExpectedToolCall, load_case_suite
from evals_platform.storage import EvalStorage

st.set_page_config(
    page_title="ReAct Evals",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: rgba(115, 86, 255, 0.07); border: 1px solid rgba(115, 86, 255, 0.18); padding: 1rem; border-radius: 0.8rem;}
      .muted {color: #7a7a88; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
storage = EvalStorage(config.database_path)
evaluator = Evaluator(storage)

try:
    cases = load_case_suite(config.case_suite_path)
    suite_error = None
except Exception as exc:
    cases = []
    suite_error = str(exc)


def percent(value: float | None) -> str:
    """Format an optional ratio as a whole-number percentage for metric cards."""

    return "—" if value is None or pd.isna(value) else f"{value:.0%}"


def average_metric(rows: list[dict], key: str) -> float | None:
    """Average one flattened metric across result rows, ignoring missing values."""

    values = [row.get(f"metric_{key}") for row in rows]
    values = [float(value) for value in values if value is not None and not isinstance(value, bool)]
    return sum(values) / len(values) if values else None


def render_overview() -> None:
    """Render repository-wide run counts, quality trends, latency, and history."""

    runs = storage.list_runs()
    rows = storage.list_results(limit=5000)
    completed = sum(run["status"] == "completed" for run in runs)
    reviewed = [row for row in rows if row["manual_verdict"] != "unreviewed"]
    manual_pass = (
        sum(row["manual_verdict"] == "pass" for row in reviewed) / len(reviewed)
        if reviewed
        else None
    )
    latencies = [float(row["latency_total_ms"]) for row in rows]
    median_latency = pd.Series(latencies).median() if latencies else None

    columns = st.columns(6)
    columns[0].metric("Runs", len(runs))
    columns[1].metric("Completion", percent(completed / len(runs) if runs else None))
    columns[2].metric("Tool F1", percent(average_metric(rows, "tool_call_f1")))
    columns[3].metric("Answer recall", percent(average_metric(rows, "answer_term_recall")))
    columns[4].metric("Manual pass", percent(manual_pass))
    columns[5].metric(
        "Median latency",
        "—" if median_latency is None else f"{median_latency / 1000:.2f}s",
    )

    if not rows:
        st.info("No evaluation history yet. Run the seed suite to establish a baseline.")
        return

    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    frame["tool_f1"] = pd.to_numeric(frame.get("metric_tool_call_f1"), errors="coerce")
    frame["answer_recall"] = pd.to_numeric(
        frame.get("metric_answer_term_recall"), errors="coerce"
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            px.scatter(
                frame,
                x="created_at",
                y=["tool_f1", "answer_recall"],
                color="model",
                hover_data=["case_id", "status"],
                title="Quality over time",
                labels={"value": "Score", "created_at": "Executed"},
            ),
            use_container_width=True,
        )
    with chart_right:
        st.plotly_chart(
            px.box(
                frame,
                x="model",
                y="latency_total_ms",
                points="all",
                hover_data=["case_id"],
                title="Total latency by model",
                labels={"latency_total_ms": "Latency (ms)"},
            ),
            use_container_width=True,
        )

    st.subheader("Recent runs")
    run_frame = pd.DataFrame(runs)
    st.dataframe(
        run_frame[
            ["name", "mode", "model", "status", "result_count", "success_count", "started_at"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def parse_expected_calls(raw: str) -> tuple[ExpectedToolCall, ...]:
    """Parse ad-hoc expected tool calls from a JSON array entered in the UI.

    Blank input means no expected calls. Invalid JSON or a non-array top-level
    value raises an error that the caller displays next to the input form.
    """

    if not raw.strip():
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Expected tool calls must be a JSON array.")
    return tuple(ExpectedToolCall.from_dict(item) for item in payload)


def render_run_evaluations() -> None:
    """Render batch/single-case controls and execute selected evaluations.

    Progress callbacks translate evaluator and agent events into a progress bar
    and a compact live trace. Completed results are persisted by the evaluator
    before a summary table is shown.
    """

    st.caption("Runs use the exact `run_agent()` implementation from ReAct/main.py.")
    if suite_error:
        st.error(f"The seed suite could not be loaded: {suite_error}")

    mode = st.radio("Execution mode", ["Batch suite", "Single prompt"], horizontal=True)
    settings_left, settings_middle, settings_right = st.columns([2, 1, 1])
    model = settings_left.text_input("Agent model", config.default_model)
    max_iterations = settings_middle.number_input(
        "Maximum iterations", min_value=1, max_value=20, value=config.default_max_iterations
    )
    judge_enabled = settings_right.toggle("Use LLM judge", value=config.judge_enabled)
    judge_model = (
        st.text_input("Judge model", config.judge_model) if judge_enabled else config.judge_model
    )

    selected_cases: list[EvalCase] = []
    run_name = ""
    if mode == "Batch suite":
        labels = {
            f"{case.case_id} · {case.difficulty} · {', '.join(case.tags)}": case
            for case in cases
        }
        selected_labels = st.multiselect(
            "Evaluation cases", list(labels), default=list(labels)
        )
        selected_cases = [labels[label] for label in selected_labels]
        run_name = st.text_input(
            "Run name", f"Weather baseline · {datetime.now():%Y-%m-%d %H:%M}"
        )
    else:
        prompt = st.text_area("Prompt", placeholder="Ask the agent to complete a task...")
        with st.expander("Optional expectations"):
            expected_calls = st.text_area(
                "Expected tool calls (JSON)",
                value='[{"name": "get_current_weather", "arguments": {"location": "London"}}]',
            )
            expected_observations = st.text_input("Expected observation terms (comma-separated)")
            expected_answer = st.text_input("Expected answer terms (comma-separated)")
            forbidden_answer = st.text_input("Forbidden answer terms (comma-separated)")
        run_name = st.text_input("Run name", f"Ad hoc · {datetime.now():%Y-%m-%d %H:%M}")
        if prompt.strip():
            try:
                selected_cases = [
                    EvalCase(
                        case_id=f"adhoc-{datetime.now():%Y%m%d%H%M%S}",
                        prompt=prompt.strip(),
                        expected_tool_calls=parse_expected_calls(expected_calls),
                        expected_observation_terms=tuple(
                            item.strip() for item in expected_observations.split(",") if item.strip()
                        ),
                        expected_answer_terms=tuple(
                            item.strip() for item in expected_answer.split(",") if item.strip()
                        ),
                        forbidden_answer_terms=tuple(
                            item.strip() for item in forbidden_answer.split(",") if item.strip()
                        ),
                        tags=("ad-hoc",),
                        difficulty="unspecified",
                    )
                ]
            except Exception as exc:
                st.error(f"Invalid expectations: {exc}")

    execute = st.button(
        "Run evaluation",
        type="primary",
        disabled=not selected_cases or not model.strip(),
        use_container_width=True,
    )
    if execute:
        progress = st.progress(0, text="Starting evaluation...")
        status_box = st.empty()
        trace_box = st.empty()
        trace_lines: list[str] = []

        def show_progress(event: dict) -> None:
            if event["type"] == "case_started":
                progress.progress(
                    (event["index"] - 1) / event["total"],
                    text=f"Running {event['case_id']}...",
                )
            elif event["type"] == "case_completed":
                progress.progress(
                    event["index"] / event["total"],
                    text=f"Completed {event['case_id']} · {event['status']}",
                )
            elif event["type"] == "agent_event":
                agent_event = event["event"]
                if agent_event["type"] == "tool_complete":
                    trace_lines.append(
                        f"**{agent_event.get('tool_name')}**({agent_event.get('arguments')}) → "
                        f"{agent_event.get('observation')}"
                    )
                elif agent_event["type"] == "model_complete" and agent_event.get("content"):
                    trace_lines.append(f"Model: {agent_event['content']}")
                trace_box.markdown("\n\n".join(trace_lines[-8:]))

        try:
            run_id, results = evaluator.run_cases(
                selected_cases,
                EvalSettings(
                    model=model.strip(),
                    max_iterations=int(max_iterations),
                    judge_enabled=judge_enabled,
                    judge_model=judge_model,
                ),
                name=run_name or "Unnamed run",
                mode="batch" if mode == "Batch suite" else "single",
                progress_callback=show_progress,
            )
            st.session_state["last_run_id"] = run_id
            status_box.success(
                f"Saved run {run_id[:8]} with {len(results)} result(s)."
            )
            summary = [
                {
                    "case": result.case.case_id,
                    "status": result.status,
                    "tool_f1": result.metrics.get("tool_call_f1"),
                    "answer_recall": result.metrics.get("answer_term_recall"),
                    "latency_ms": result.metrics.get("total_latency_ms"),
                    "answer": result.agent_result.get("final_answer"),
                }
                for result in results
            ]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        except Exception as exc:
            progress.empty()
            status_box.error(f"Evaluation failed: {exc}")


def render_results_explorer() -> None:
    """Render result filters, trajectory details, metrics, and review controls."""

    all_runs = storage.list_runs()
    all_rows = storage.list_results(limit=5000)
    if not all_rows:
        st.info("No results are available yet.")
        return

    run_options = {"All runs": None, **{f"{run['name']} · {run['id'][:8]}": run["id"] for run in all_runs}}
    models = sorted({row["model"] for row in all_rows})
    tags = sorted({tag for row in all_rows for tag in row.get("tags", [])})
    first, second, third = st.columns(3)
    run_label = first.selectbox("Run", list(run_options))
    model = second.selectbox("Model", ["All", *models])
    status = third.selectbox("Status", ["All", "success", "partial", "failed"])
    fourth, fifth, sixth = st.columns(3)
    verdict = fourth.selectbox("Manual verdict", ["All", "unreviewed", "pass", "fail"])
    tag = fifth.selectbox("Tag", ["All", *tags])
    text = sixth.text_input("Search prompt, answer, or case ID")

    rows = storage.list_results(
        run_id=run_options[run_label],
        model=None if model == "All" else model,
        status=None if status == "All" else status,
        verdict=None if verdict == "All" else verdict,
        tag=None if tag == "All" else tag,
        text=text or None,
        limit=2000,
    )
    if not rows:
        st.warning("No results match the current filters.")
        return

    table = pd.DataFrame(
        [
            {
                "result_id": row["id"],
                "case": row["case_id"],
                "status": row["status"],
                "model": row["model"],
                "tool_f1": row.get("metric_tool_call_f1"),
                "answer_recall": row.get("metric_answer_term_recall"),
                "latency_ms": row["latency_total_ms"],
                "verdict": row["manual_verdict"],
                "created": row["created_at"],
            }
            for row in rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    labels = {f"{row['case_id']} · {row['id'][:8]} · {row['status']}": row["id"] for row in rows}
    selected_label = st.selectbox("Inspect result", list(labels))
    detail = storage.get_result(labels[selected_label])
    if not detail:
        return

    st.subheader(detail["case_id"])
    st.markdown(f"**Prompt**  \n{detail['prompt']}")
    expected_column, answer_column = st.columns(2)
    with expected_column:
        st.markdown("**Expected behavior**")
        st.json(detail["case_snapshot"], expanded=False)
    with answer_column:
        st.markdown("**Final answer**")
        st.write(detail["final_answer"] or "_No final answer produced._")

    metric_column, judge_column = st.columns(2)
    with metric_column:
        st.markdown("**Deterministic metrics**")
        st.json(detail["metrics"], expanded=True)
    with judge_column:
        st.markdown("**LLM judge**")
        st.json(detail.get("judge") or {"status": "not run"}, expanded=True)

    st.markdown("**Trajectory**")
    for index, step in enumerate(detail["steps"], start=1):
        title = f"{index}. {step.get('kind', 'step').title()} · iteration {step.get('iteration')}"
        if step.get("tool_name"):
            title += f" · {step['tool_name']}"
        with st.expander(title):
            st.json(step)

    if detail.get("error_message"):
        st.error(f"{detail.get('error_type')}: {detail['error_message']}")

    with st.form(f"review-{detail['id']}"):
        verdict_value = st.selectbox(
            "Manual verdict",
            ["unreviewed", "pass", "fail"],
            index=["unreviewed", "pass", "fail"].index(detail["manual_verdict"]),
        )
        notes = st.text_area("Review notes", value=detail["manual_notes"])
        if st.form_submit_button("Save review", type="primary"):
            storage.update_review(detail["id"], verdict_value, notes)
            st.success("Review saved.")
            st.rerun()


st.title("🧭 ReAct Agent Evals")
st.caption(
    "Execute, inspect, and compare the tool-calling behavior of the agent in ReAct/main.py."
)

overview_tab, run_tab, explorer_tab = st.tabs(
    ["Overview", "Run evaluations", "Results explorer"]
)
with overview_tab:
    render_overview()
with run_tab:
    render_run_evaluations()
with explorer_tab:
    render_results_explorer()
