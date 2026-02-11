from __future__ import annotations

from collections.abc import Callable

from .base import TradingAgent
from .manual_agent import ManualAgent
from .rsi_threshold_agent import RsiThresholdAgent
from .template_agent import TemplateAgent

AgentFactory = Callable[[], TradingAgent]

_AGENT_FACTORIES: dict[str, AgentFactory] = {
    ManualAgent.key: ManualAgent,
    RsiThresholdAgent.key: RsiThresholdAgent,
    TemplateAgent.key: TemplateAgent,
}


def available_agents() -> dict[str, str]:
    return {key: factory().display_name for key, factory in _AGENT_FACTORIES.items()}


def create_agent(key: str) -> TradingAgent:
    normalized = key.strip().lower()
    factory = _AGENT_FACTORIES.get(normalized)
    if factory is None:
        supported = ", ".join(sorted(_AGENT_FACTORIES))
        raise ValueError(f"Unknown agent '{key}'. Available: {supported}")
    return factory()
