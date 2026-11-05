"""Simulation-only order tracking for AI signals."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import count
from typing import Any, Mapping


@dataclass
class PaperOrder:
    """An order that exists only in the local paper-trading ledger."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    predicted_direction: str
    signal_score: float | None
    status: str = "OPEN"
    exit_price: float | None = None
    actual_direction: str | None = None
    pnl: float | None = None
    prediction_correct: bool | None = None
    created_at: str = ""
    closed_at: str | None = None


class PaperTradingService:
    """Create and track simulated orders without any broker dependency."""

    def __init__(self) -> None:
        self._orders: dict[str, PaperOrder] = {}
        self._sequence = count(1)

    def place_order(self, signal: Mapping[str, Any], symbol: str, quantity: float, entry_price: float) -> dict[str, Any]:
        """Record an AI signal as a paper order; no external order is sent."""
        side = str(signal.get("side", signal.get("signal", ""))).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("signal must contain BUY or SELL direction")
        if quantity <= 0 or entry_price <= 0:
            raise ValueError("quantity and entry_price must be positive")
        order_id = f"PAPER-{next(self._sequence):06d}"
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            entry_price=float(entry_price),
            predicted_direction=side,
            signal_score=float(signal["score"]) if isinstance(signal.get("score"), (int, float)) else None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._orders[order_id] = order
        return asdict(order)

    def close_order(self, order_id: str, exit_price: float) -> dict[str, Any]:
        """Close a simulated order and compare its outcome to its prediction."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown paper order: {order_id}")
        if order.status != "OPEN":
            raise ValueError(f"paper order {order_id} is already closed")
        if exit_price <= 0:
            raise ValueError("exit_price must be positive")
        order.exit_price = float(exit_price)
        order.actual_direction = "BUY" if exit_price > order.entry_price else "SELL" if exit_price < order.entry_price else "FLAT"
        order.pnl = (exit_price - order.entry_price) * order.quantity if order.side == "BUY" else (order.entry_price - exit_price) * order.quantity
        order.prediction_correct = order.actual_direction == order.predicted_direction
        order.status = "CLOSED"
        order.closed_at = datetime.now(timezone.utc).isoformat()
        return asdict(order)

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown paper order: {order_id}")
        return asdict(order)

    def list_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        return [asdict(order) for order in self._orders.values() if status is None or order.status == status.upper()]