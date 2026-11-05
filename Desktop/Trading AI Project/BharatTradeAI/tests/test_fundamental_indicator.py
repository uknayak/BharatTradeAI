import pytest

from app.agents.fundamental_agent import analyze_fundamentals as analyze_agent
from app.indicators.fundamental import analyze_fundamentals


def test_fundamentals_calculate_profitability_valuation_and_ownership():
    result = analyze_fundamentals(
        {
            "revenue": 1000,
            "profit": 100,
            "equity": 500,
            "debt": 200,
            "ebit": 140,
            "shares_outstanding": 10,
            "price": 300,
            "operating_profit": 150,
            "promoter": 48.5,
            "fii": 22.0,
            "dii": 12.5,
        },
        {"revenue": 800, "profit": 80, "eps": 8},
    )

    assert result["revenue"] == 1000
    assert result["profit"] == 100
    assert result["eps"] == 10
    assert result["roe"] == pytest.approx(20)
    assert result["roce"] == pytest.approx(20)
    assert result["debt"] == 200
    assert result["margins"]["net"] == pytest.approx(10)
    assert result["valuation"]["pe"] == pytest.approx(30)
    assert result["valuation"]["pb"] == pytest.approx(6)
    assert result["ownership"] == {"promoter": 48.5, "fii": 22.0, "dii": 12.5}
    assert result["growth"] == {"revenue": pytest.approx(25), "profit": pytest.approx(25), "eps": pytest.approx(25)}


def test_fundamental_agent_and_zero_denominators_are_safe():
    assert analyze_agent() == {"agent": "fundamental_agent", "status": "ready"}
    result = analyze_agent({"revenue": 0, "profit": 10, "equity": 0, "price": 100})
    assert result["analysis"]["margins"]["net"] is None
    assert result["analysis"]["roe"] is None
    assert result["analysis"]["valuation"]["pe"] is None