from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PositionState(str, Enum):
    RISK_OFF = "RISK_OFF"
    LONG = "LONG"
    SHORT = "SHORT"


class AgentAction(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    timestamp: str
    close: float
    rsi: float


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction = AgentAction.HOLD
    reason: str = ""
