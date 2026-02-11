from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AgentDecision, MarketSnapshot, PositionState


class TradingAgent(ABC):
    key = "base"
    display_name = "Base Agent"

    @abstractmethod
    def on_snapshot(self, snapshot: MarketSnapshot, position: PositionState) -> AgentDecision:
        """Return the next suggestion for the current market snapshot."""
