"""Base agent class providing shared LLM access and logging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.llm import LLMClient
from src.utils.logger import get_logger


class BaseAgent(ABC):
    """All agents share an LLM client and a logger."""

    name: str = "BaseAgent"

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def _system_prompt(self) -> str:
        """Override to set the agent's system prompt."""
        return f"You are the {self.name} of an AI video production house."
