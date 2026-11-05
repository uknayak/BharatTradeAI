"""Pure option-chain calculations used by agents and API adapters."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OptionContract:
    """The normalized fields required to analyze one option contract."""

    strike: float
    option_type: str
    open_interest: float = 0.0
    change_in_open_interest: float = 0.0
    volume: float = 0.0
    implied_volatility: float | None = None
    premium: float = 0.0

    def __post_init__(self) -> None:
        option_type = self.option_type.upper()
        if option_type not in {"CE", "PE", "CALL", "PUT"}:
            raise ValueError("option_type must be CE, PE, CALL, or PUT")
        if not isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be a positive finite number")
        object.__setattr__(self, "option_type", "CE" if option_type in {"CE", "CALL"} else "PE")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _contract(row: OptionContract | Mapping[str, Any]) -> OptionContract:
    if isinstance(row, OptionContract):
        return row
    option_type = str(row.get("option_type", row.get("type", row.get("side", ""))))
    return OptionContract(
        strike=_number(row.get("strike", row.get("strike_price"))),
        option_type=option_type,
        open_interest=_number(row.get("open_interest", row.get("oi"))),
        change_in_open_interest=_number(row.get("change_in_open_interest", row.get("change_oi", row.get("oi_change")))),
        volume=_number(row.get("volume")),
        implied_volatility=(
            _number(row["implied_volatility"], 0.0)
            if row.get("implied_volatility") is not None
            else (_number(row["iv"], 0.0) if row.get("iv") is not None else None)
        ),
        premium=_number(row.get("premium", row.get("last_price", row.get("ltp")))),
    )


def _weighted_average(rows: list[OptionContract], field: str) -> float | None:
    valid = [(getattr(row, field), row.open_interest) for row in rows if getattr(row, field) is not None]
    weight = sum(item[1] for item in valid)
    if not valid or weight <= 0:
        return None
    return sum(value * oi for value, oi in valid) / weight


def analyze_option_chain(
    chain: Iterable[OptionContract | Mapping[str, Any]],
    underlying_price: float,
) -> dict[str, Any]:
    """Return OI flow, pricing, moneyness, PCR, max pain, and concentration.

    ``underlying_price`` is required because moneyness and max pain cannot be
    inferred reliably from the option rows alone.
    """
    spot = _number(underlying_price)
    if spot <= 0:
        raise ValueError("underlying_price must be positive")
    contracts = [_contract(row) for row in chain]
    if not contracts:
        raise ValueError("chain must contain at least one contract")

    strikes = sorted({row.strike for row in contracts})
    atm_strike = min(strikes, key=lambda strike: (abs(strike - spot), strike))
    by_type = {"CE": [row for row in contracts if row.option_type == "CE"], "PE": [row for row in contracts if row.option_type == "PE"]}

    def totals(rows: list[OptionContract]) -> dict[str, float | None]:
        return {
            "open_interest": sum(row.open_interest for row in rows),
            "change_in_open_interest": sum(row.change_in_open_interest for row in rows),
            "volume": sum(row.volume for row in rows),
            "average_iv": _weighted_average(rows, "implied_volatility"),
            "premium": sum(row.premium for row in rows),
            "average_premium": _weighted_average(rows, "premium"),
        }

    call_oi = sum(row.open_interest for row in by_type["CE"])
    put_oi = sum(row.open_interest for row in by_type["PE"])
    call_volume = sum(row.volume for row in by_type["CE"])
    put_volume = sum(row.volume for row in by_type["PE"])
    concentration = []
    for strike in strikes:
        strike_rows = [row for row in contracts if row.strike == strike]
        oi = sum(row.open_interest for row in strike_rows)
        concentration.append({"strike": strike, "open_interest": oi, "share": oi / sum(row.open_interest for row in contracts) if sum(row.open_interest for row in contracts) else 0.0})
    concentration.sort(key=lambda item: (-item["open_interest"], item["strike"]))

    pain = {
        strike: sum(row.open_interest * max(strike - row.strike, 0) for row in by_type["CE"])
        + sum(row.open_interest * max(row.strike - strike, 0) for row in by_type["PE"])
        for strike in strikes
    }
    max_pain = min(pain, key=lambda strike: (pain[strike], strike))

    return {
        "underlying_price": spot,
        "atm_strike": atm_strike,
        "pcr": {"open_interest": put_oi / call_oi if call_oi else None, "volume": put_volume / call_volume if call_volume else None},
        "calls": totals(by_type["CE"]),
        "puts": totals(by_type["PE"]),
        "max_pain": {"strike": max_pain, "pain_by_strike": pain},
        "strike_concentration": concentration,
        "contracts": [
            {**row.__dict__, "moneyness": "ATM" if row.strike == atm_strike else ("ITM" if (row.option_type == "CE") == (row.strike < spot) else "OTM")}
            for row in contracts
        ],
    }


def compute_put_call_ratio(
    chain: Iterable[OptionContract | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute PCR, retaining the original placeholder's no-argument contract."""
    if chain is None:
        return {"indicator": "put_call_ratio", "status": "not_implemented"}
    contracts = [_contract(row) for row in chain]
    call_oi = sum(row.open_interest for row in contracts if row.option_type == "CE")
    put_oi = sum(row.open_interest for row in contracts if row.option_type == "PE")
    return {"indicator": "put_call_ratio", "open_interest": put_oi / call_oi if call_oi else None}
