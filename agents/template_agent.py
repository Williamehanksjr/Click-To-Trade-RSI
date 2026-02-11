from __future__ import annotations

from .base import TradingAgent
from .models import AgentAction, AgentDecision, MarketSnapshot, PositionState


class TemplateAgent(TradingAgent):
    """
    Copy this class when building a new strategy agent.

    Minimal contract:
      - Implement `on_snapshot(...)`
      - Return AgentDecision(action=..., reason="...")
    """

    key = "template"
    display_name = "Template (customize me)"

    def on_snapshot(self, snapshot: MarketSnapshot, position: PositionState) -> AgentDecision:
        # TODO: Replace this placeholder strategy with your own logic.
        # Example inputs available:
        #   - snapshot.symbol
        #   - snapshot.timestamp
        #   - snapshot.close
        #   - snapshot.rsi
        #   - position (RISK_OFF / LONG / SHORT)
        _ = snapshot
        _ = position

        # TODO: Change this action once you add your strategy conditions.
        return AgentDecision(
            action=AgentAction.HOLD,
            reason="template placeholder - add strategy rules",
        )
