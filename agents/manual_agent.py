from __future__ import annotations

from .base import TradingAgent
from .models import AgentAction, AgentDecision, MarketSnapshot, PositionState


class ManualAgent(TradingAgent):
    key = "manual"
    display_name = "Manual (click-only)"

    def on_snapshot(self, snapshot: MarketSnapshot, position: PositionState) -> AgentDecision:
        _ = snapshot
        _ = position
        return AgentDecision(
            action=AgentAction.HOLD,
            reason="waiting for clicks",
        )
