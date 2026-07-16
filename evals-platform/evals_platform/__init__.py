"""Public data-model interface for the local ReAct evaluation platform.

The package separates case definitions, agent adaptation, metric calculation,
optional LLM judging, orchestration, persistence, and UI concerns. This module
re-exports the records most callers need when constructing cases or consuming
evaluation results; execution services remain available from their dedicated
modules to keep imports lightweight and explicit.
"""

from .models import EvalCase, EvalSettings, EvaluationResult, ExpectedToolCall

__all__ = ["EvalCase", "EvalSettings", "EvaluationResult", "ExpectedToolCall"]
