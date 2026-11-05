from app.agentic.scoring import DEFAULT_WEIGHTS, score_band, score_signal


def test_score_bands_match_iteration_11_thresholds():
    assert [score_band(value) for value in (0, 30, 31, 45, 46, 59, 60, 74, 75, 84, 85, 100)] == [
        "VERY WEAK", "VERY WEAK", "WEAK", "WEAK", "NEUTRAL", "NEUTRAL",
        "WATCH", "WATCH", "STRONG", "STRONG", "VERY STRONG", "VERY STRONG",
    ]


def test_score_signal_uses_defaults_and_renormalizes_available_components():
    result = score_signal({"market": {"analysis": {"direction": "BULLISH"}}})

    assert result["score"] == 100
    assert result["band"] == "VERY STRONG"
    assert result["weights"] == DEFAULT_WEIGHTS
    assert result["available_components"] == ["market_regime"]
    assert result["weight_source"] == "default_until_backtested"