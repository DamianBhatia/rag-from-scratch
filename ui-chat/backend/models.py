"""Request models for the local chat API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentMessage(BaseModel):
    """An Ollama-compatible message carried between completed turns."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "tool"]
    content: str = Field(max_length=100_000)
    tool_name: str | None = Field(default=None, max_length=200)
    tool_calls: list[Any] | None = None


class ChatRequest(BaseModel):
    """One user turn plus the opaque history from the prior completion."""

    prompt: str = Field(min_length=1, max_length=20_000)
    prior_messages: list[AgentMessage] = Field(default_factory=list, max_length=200)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value
