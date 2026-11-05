from discord import Color

from app.alerts.discord import build_signal_embed


def test_signal_embed_reproduces_requested_component_and_status_emojis():
    embed = build_signal_embed({
        "score": 78,
        "band": "STRONG",
        "valid": True,
        "components": {
            "technical": 80,
            "options": 75,
            "ml": 82,
            "market_regime": 70,
            "news": 78,
            "fundamental": 65,
        },
    })

    payload = embed.to_dict()
    field_text = " ".join(field["name"] for field in payload["fields"])
    assert all(emoji in field_text for emoji in ("📊", "🔥", "🐋", "🚀", "⚡", "😴", "🟢"))
    assert payload["color"] == Color.green().value
    assert "STRONG" in payload["description"]