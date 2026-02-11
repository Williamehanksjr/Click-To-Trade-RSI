from __future__ import annotations

from dataclasses import dataclass

from .base import TradingAgent
from .models import AgentAction, AgentDecision, MarketSnapshot, PositionState


@dataclass(frozen=True)
class RsiThresholdConfig:
    oversold: float = 30.0
    overbought: float = 70.0
    exit_level: float = 50.0


class RsiThresholdAgent(TradingAgent):
    key = "rsi-threshold"
    display_name = "RSI Threshold"

    def __init__(self, config: RsiThresholdConfig | None = None):
        self.config = config or RsiThresholdConfig()

    def on_snapshot(self, snapshot: MarketSnapshot, position: PositionState) -> AgentDecision:
        rsi = snapshot.rsi

        if position == PositionState.RISK_OFF:
            if rsi <= self.config.oversold:
                return AgentDecision(AgentAction.BUY, f"rsi <= {self.config.oversold:.0f}")
            if rsi >= self.config.overbought:
                return AgentDecision(AgentAction.SELL, f"rsi >= {self.config.overbought:.0f}")

        if position == PositionState.LONG and rsi >= self.config.exit_level:
            return AgentDecision(AgentAction.EXIT, f"rsi >= {self.config.exit_level:.0f}")

        if position == PositionState.SHORT and rsi <= self.config.exit_level:
            return AgentDecision(AgentAction.EXIT, f"rsi <= {self.config.exit_level:.0f}")

        return AgentDecision(AgentAction.HOLD, "no signal")
