"""Single and batch orchestration for ReAct agent evaluations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .judge import OllamaJudge
from .metrics import calculate_metrics
from .models import EvalCase, EvalSettings, EvaluationResult
from .react_adapter import ReactAdapter
from .storage import EvalStorage

ProgressCallback = Callable[[dict[str, Any]], None]


class Evaluator:
    def __init__(
        self,
        storage: EvalStorage,
        adapter: ReactAdapter | None = None,
        judge_factory: Callable[[str], OllamaJudge] | None = None,
    ) -> None:
        self.storage = storage
        self.adapter = adapter or ReactAdapter()
        self.judge_factory = judge_factory or OllamaJudge

    def run_cases(
        self,
        cases: Iterable[EvalCase],
        settings: EvalSettings,
        *,
        name: str,
        mode: str,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[str, list[EvaluationResult]]:
        selected_cases = list(cases)
        if not selected_cases:
            raise ValueError("At least one evaluation case is required.")
        run_id = str(uuid.uuid4())
        self.storage.create_run(run_id, name, mode, settings)
        results: list[EvaluationResult] = []
        judge = (
            self.judge_factory(settings.judge_model or settings.model)
            if settings.judge_enabled
            else None
        )

        try:
            for index, case in enumerate(selected_cases, start=1):
                def agent_event(event: dict[str, Any]) -> None:
                    if progress_callback:
                        progress_callback(
                            {
                                "type": "agent_event",
                                "index": index,
                                "total": len(selected_cases),
                                "case_id": case.case_id,
                                "event": event,
                            }
                        )

                if progress_callback:
                    progress_callback(
                        {
                            "type": "case_started",
                            "index": index,
                            "total": len(selected_cases),
                            "case_id": case.case_id,
                        }
                    )
                try:
                    agent_result = self.adapter.run(
                        case.prompt,
                        model=settings.model,
                        max_iterations=case.max_iterations or settings.max_iterations,
                        event_callback=agent_event,
                    )
                    metrics = calculate_metrics(case, agent_result)
                    judge_result = judge.evaluate(case, agent_result) if judge else None
                    if agent_result.status == "completed":
                        result_status = "success"
                    elif agent_result.status == "completed_with_errors":
                        result_status = "partial"
                    else:
                        result_status = "failed"
                    agent_payload = agent_result.to_dict()
                except Exception as exc:
                    result_status = "failed"
                    metrics = {}
                    judge_result = None
                    agent_payload = {
                        "status": "failed",
                        "termination_reason": "evaluator_error",
                        "final_answer": "",
                        "messages": [],
                        "steps": [],
                        "iterations": 0,
                        "total_latency_ms": 0,
                        "model_latency_ms": 0,
                        "tool_latency_ms": 0,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }

                result = EvaluationResult(
                    result_id=str(uuid.uuid4()),
                    run_id=run_id,
                    case=case,
                    status=result_status,
                    agent_result=agent_payload,
                    metrics=metrics,
                    judge=judge_result,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                self.storage.append_result(result)
                results.append(result)
                if progress_callback:
                    progress_callback(
                        {
                            "type": "case_completed",
                            "index": index,
                            "total": len(selected_cases),
                            "case_id": case.case_id,
                            "status": result_status,
                        }
                    )

            statuses = {result.status for result in results}
            if statuses == {"success"}:
                run_status = "completed"
            elif statuses == {"failed"}:
                run_status = "failed"
            else:
                run_status = "partial"
            self.storage.finalize_run(run_id, run_status)
            return run_id, results
        except Exception as exc:
            self.storage.finalize_run(run_id, "failed", str(exc))
            raise
