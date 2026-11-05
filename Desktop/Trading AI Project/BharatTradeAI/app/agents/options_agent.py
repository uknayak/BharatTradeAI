"""Options flow agent backed by the option-chain indicator engine."""

from collections.abc import Iterable
from typing import Any, Mapping

from app.indicators.options import OptionContract, analyze_option_chain


def analyze_options_flow(
    chain: Iterable[OptionContract | Mapping[str, Any]] | None = None,
    underlying_price: float | None = None,
) -> dict[str, Any]:
    if chain is not None or underlying_price is not None:
        if chain is None or underlying_price is None:
            raise ValueError("chain and underlying_price must be provided together")
        return {"agent": "options_agent", "status": "ready", "analysis": analyze_option_chain(chain, underlying_price)}
    return {"agent": "options_agent", "status": "ready"}
