"""Optional non-deterministic Ollama judge for agent evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable

import ollama

from .models import EvalCase


class OllamaJudge:
    def __init__(
        self, model: str, chat_client: Callable[..., Any] | None = None
    ) -> None:
        self.model = model
        self.chat_client = chat_client or ollama.chat

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message", {})
            return str(message.get("content", ""))
        message = getattr(response, "message", None)
        return str(getattr(message, "content", ""))

    @staticmethod
    def parse_response(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Judge did not return a JSON object.")
        payload = json.loads(cleaned[start : end + 1])
        for key in ("task_success", "groundedness", "tool_appropriateness"):
            score = float(payload[key])
            if not 0 <= score <= 1:
                raise ValueError(f"Judge score '{key}' must be between 0 and 1.")
            payload[key] = score
        payload["rationale"] = str(payload.get("rationale", ""))
        return payload

    def evaluate(self, case: EvalCase, agent_result: Any) -> dict[str, Any]:
        """Return judge scores, or a structured judge error without raising."""

        prompt = {
            "task": (
                "Evaluate the agent trajectory. Return only JSON with numeric scores from "
                "0 to 1 for task_success, groundedness, and tool_appropriateness, plus rationale."
            ),
            "case": case.to_dict(),
            "trajectory": [asdict(step) for step in agent_result.steps],
            "final_answer": agent_result.final_answer,
        }
        try:
            response = self.chat_client(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict agent evaluator."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                stream=False,
            )
            return {"status": "ok", **self.parse_response(self._content(response))}
        except Exception as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
