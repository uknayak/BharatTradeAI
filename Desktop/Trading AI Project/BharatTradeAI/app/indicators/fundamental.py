"""Fundamental metrics and valuation calculations."""

from typing import Any, Mapping


def _value(data: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if data.get(name) is not None:
            try:
                return float(data[name])
            except (TypeError, ValueError):
                return None
    return None


def _ratio(numerator: float | None, denominator: float | None, scale: float = 1.0) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator * scale


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current / previous - 1) * 100


def analyze_fundamentals(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize company fundamentals and calculate derived ratios.

    Monetary values remain in the units supplied by the caller. Percentage
    outputs such as margins, returns, and ownership are expressed in percent.
    """
    revenue = _value(current, "revenue", "sales", "total_revenue")
    profit = _value(current, "profit", "net_profit", "pat")
    eps = _value(current, "eps", "earnings_per_share")
    equity = _value(current, "equity", "shareholders_equity", "net_worth")
    debt = _value(current, "debt", "total_debt", "borrowings")
    shares = _value(current, "shares_outstanding", "shares")
    price = _value(current, "price", "market_price", "ltp")
    book_value_per_share = _value(current, "book_value_per_share", "bvps")
    market_cap = _value(current, "market_cap")
    ebit = _value(current, "ebit", "operating_profit")
    capital_employed = _value(current, "capital_employed")
    if eps is None:
        eps = _ratio(profit, shares)
    if capital_employed is None and equity is not None and debt is not None:
        capital_employed = equity + debt - (_value(current, "cash") or 0)
    if ebit is None and profit is not None:
        ebit = profit + (_value(current, "interest_expense", "interest") or 0)
    if book_value_per_share is None:
        book_value_per_share = _ratio(equity, shares)
    book_value = _value(current, "book_value") or equity
    price_to_book = _ratio(price, book_value_per_share)
    if price_to_book is None:
        price_to_book = _ratio(market_cap, book_value)

    metrics: dict[str, Any] = {
        "revenue": revenue,
        "profit": profit,
        "eps": eps,
        "roe": _ratio(profit, equity, 100),
        "roce": _ratio(ebit, capital_employed, 100),
        "debt": debt,
        "debt_to_equity": _ratio(debt, equity),
        "margins": {
            "net": _ratio(profit, revenue, 100),
            "operating": _ratio(_value(current, "operating_profit", "ebit"), revenue, 100),
            "ebitda": _ratio(_value(current, "ebitda"), revenue, 100),
        },
        "valuation": {
            "pe": _ratio(price, eps),
            "pb": price_to_book,
        },
        "ownership": {
            "promoter": _value(current, "promoter", "promoter_holding", "promoter_holding_percent"),
            "fii": _value(current, "fii", "fii_holding", "fii_holding_percent"),
            "dii": _value(current, "dii", "dii_holding", "dii_holding_percent"),
        },
    }
    if previous is not None:
        metrics["growth"] = {
            "revenue": _growth(revenue, _value(previous, "revenue", "sales", "total_revenue")),
            "profit": _growth(profit, _value(previous, "profit", "net_profit", "pat")),
            "eps": _growth(eps, _value(previous, "eps", "earnings_per_share")),
        }
    return metrics