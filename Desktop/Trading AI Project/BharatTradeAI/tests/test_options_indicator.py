import pytest

from app.indicators.options import OptionContract, analyze_option_chain, compute_put_call_ratio
from app.agents.options_agent import analyze_options_flow


def test_option_chain_calculates_flow_moneyness_and_max_pain():
    chain = [
        OptionContract(100, "CE", open_interest=100, change_in_open_interest=10, volume=50, implied_volatility=20, premium=8),
        OptionContract(110, "CE", open_interest=200, volume=80, implied_volatility=22, premium=3),
        OptionContract(100, "PE", open_interest=150, change_in_open_interest=-5, volume=75, implied_volatility=21, premium=7),
        OptionContract(110, "PE", open_interest=50, volume=20, implied_volatility=24, premium=12),
    ]

    result = analyze_option_chain(chain, underlying_price=105)

    assert result["atm_strike"] == 100
    assert result["pcr"]["open_interest"] == pytest.approx(200 / 300)
    assert result["calls"]["change_in_open_interest"] == 10
    assert result["puts"]["volume"] == 95
    assert result["max_pain"]["strike"] == 100
    assert [row["moneyness"] for row in result["contracts"]] == ["ATM", "OTM", "ATM", "ITM"]
    assert result["strike_concentration"][0]["strike"] == 100


def test_mapping_rows_and_legacy_pcr_helper_are_supported():
    assert compute_put_call_ratio() == {"indicator": "put_call_ratio", "status": "not_implemented"}
    result = compute_put_call_ratio([{"strike": 100, "type": "CE", "oi": 4}, {"strike": 100, "type": "PE", "oi": 6}])
    assert result["open_interest"] == pytest.approx(1.5)


def test_options_agent_delegates_to_chain_engine():
    result = analyze_options_flow(
        [{"strike": 100, "type": "CE", "oi": 10}, {"strike": 100, "type": "PE", "oi": 20}],
        100,
    )
    assert result["status"] == "ready"
    assert result["analysis"]["pcr"]["open_interest"] == 2