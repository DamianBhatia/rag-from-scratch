"""Public interface for the repository's instrumented ReAct agent.

Importing from this package exposes the agent runner and its two structured
trajectory records without requiring callers to depend on the layout of
``ReAct.main``. The Streamlit evaluation platform uses the same implementation
through a thin adapter, ensuring dashboard runs exercise the production loop
rather than a test-specific copy.
"""

from .main import AgentRunResult, AgentStep, run_agent

__all__ = ["AgentRunResult", "AgentStep", "run_agent"]