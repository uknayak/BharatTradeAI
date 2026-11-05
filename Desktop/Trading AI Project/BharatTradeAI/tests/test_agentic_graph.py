from app.agentic.graph import run_agentic_analysis


def test_agentic_pipeline_runs_in_required_order_and_keeps_numeric_outputs():
    result = run_agentic_analysis({
        "period": 3,
        "market_data": [100, 101, 102, 103, 104],
        "options_chain": [{"strike": 100, "type": "CE", "oi": 10}, {"strike": 100, "type": "PE", "oi": 20}],
        "underlying_price": 102,
        "candles": [{"high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 100} for i in range(5)],
        "fundamentals": {"revenue": 100, "profit": 10, "equity": 50, "debt": 10, "shares": 2, "price": 20},
        "news_records": [{"title": "TCS reports strong profit growth", "url": "news-1"}],
    })

    assert result["execution_trace"] == ["market", "options", "technical", "fundamental", "news", "ml", "risk", "validator", "explanation"]
    assert result["market"]["analysis"]["direction"] == "BULLISH"
    assert result["options"]["analysis"]["pcr"]["open_interest"] == 2
    assert result["fundamental"]["analysis"]["eps"] == 5
    assert result["validation"]["valid"] is True
    assert "Numerical analysis" in result["explanation"]


def test_explainer_receives_summary_but_cannot_replace_numeric_state():
    seen = {}

    def explainer(summary):
        seen.update(summary)
        return "Explanation only"

    result = run_agentic_analysis({"market_data": [100, 101, 102, 105], "period": 2, "llm_explainer": explainer})

    assert result["explanation"] == "Explanation only"
    assert seen["market"]["analysis"]["direction"] == "BULLISH"
    assert result["risk"]["approved"] is True