"""Integration tests for SQLite persistence and evaluation orchestration.

The tests exercise schema initialization, JSON round trips, query filtering,
manual review updates, mixed-success batch execution, and run-level status
aggregation. A queued fake model keeps the evaluator path deterministic while
still passing through the real adapter and ReAct loop.
"""

from __future__ import annotations

from evals_platform.evaluator import Evaluator
from evals_platform.models import EvalCase, EvalSettings, EvaluationResult
from evals_platform.react_adapter import ReactAdapter
from evals_platform.storage import EvalStorage


class FakeChat:
    """Minimal queued chat double for evaluator integration tests."""

    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, **kwargs):
        """Yield the next fixed model response or raise its queued exception."""

        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return iter(response)


def test_storage_round_trip_filter_and_manual_review(tmp_path):
    """Stored JSON, flattened metrics, filters, and reviews round-trip correctly."""

    storage = EvalStorage(tmp_path / "evals.db")
    settings = EvalSettings(model="test-model", max_iterations=2)
    storage.create_run("run-1", "test", "single", settings)
    case = EvalCase(case_id="case-1", prompt="hello", tags=("control",))
    storage.append_result(
        EvaluationResult(
            result_id="result-1",
            run_id="run-1",
            case=case,
            status="success",
            agent_result={
                "termination_reason": "final_answer",
                "final_answer": "hello",
                "messages": [{"role": "assistant", "content": "hello"}],
                "steps": [{"kind": "model", "iteration": 1}],
                "total_latency_ms": 1.5,
                "error_type": None,
                "error_message": None,
            },
            metrics={"tool_call_f1": 1.0},
        )
    )
    storage.finalize_run("run-1", "completed")

    row = storage.list_results(tag="control", model="test-model")[0]
    assert row["steps"][0]["kind"] == "model"
    assert row["metric_tool_call_f1"] == 1

    storage.update_review("result-1", "pass", "Looks correct")
    reviewed = storage.get_result("result-1")
    assert reviewed["manual_verdict"] == "pass"
    assert reviewed["manual_notes"] == "Looks correct"


def test_evaluator_persists_mixed_batch_and_finalizes_partial(tmp_path):
    """A mixed-success batch persists both cases and finalizes as partial."""

    storage = EvalStorage(tmp_path / "evals.db")
    chat = FakeChat(
        [
            [{"message": {"content": "Hello!", "tool_calls": []}}],
            RuntimeError("model unavailable"),
        ]
    )
    evaluator = Evaluator(storage, adapter=ReactAdapter(chat_client=chat))
    cases = [
        EvalCase(case_id="ok", prompt="Say hello", expected_answer_terms=("hello",)),
        EvalCase(case_id="error", prompt="Fail now"),
    ]

    run_id, results = evaluator.run_cases(
        cases,
        EvalSettings(model="fake", max_iterations=2),
        name="mixed",
        mode="batch",
    )

    assert [result.status for result in results] == ["success", "failed"]
    assert storage.list_runs()[0]["status"] == "partial"
    assert len(storage.list_results(run_id=run_id)) == 2
