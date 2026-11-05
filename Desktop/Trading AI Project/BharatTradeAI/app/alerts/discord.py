"""Discord embeds and webhook delivery for scored trading signals."""

from collections.abc import Mapping
from typing import Any

import requests
from discord import Color, Embed


COMPONENTS = (
    ("technical", "📊", "Technical"),
    ("options", "🔥", "Options"),
    ("ml", "🐋", "ML"),
    ("market_regime", "🚀", "Market Regime"),
    ("news", "⚡", "News"),
    ("fundamental", "😴", "Fundamental"),
)


def _score_value(components: Mapping[str, Any], name: str) -> str:
    value = components.get(name)
    return f"{float(value):.1f}/100" if isinstance(value, (int, float)) else "N/A"


def build_signal_embed(signal: Mapping[str, Any]) -> Embed:
    """Build a Discord embed from the validator's scoring result."""
    score = signal.get("score")
    band = signal.get("band", "UNKNOWN")
    valid = signal.get("valid", True)
    components = signal.get("components", {})
    if not isinstance(components, Mapping):
        components = {}
    approved = bool(valid)
    embed = Embed(
        title="BharatTradeAI Signal",
        description=f"**{band}** | Score: **{float(score):.1f}/100**" if isinstance(score, (int, float)) else f"**{band}**",
        color=Color.green() if approved else Color.red(),
    )
    for name, emoji, label in COMPONENTS:
        embed.add_field(name=f"{emoji} {label}", value=_score_value(components, name), inline=True)
    embed.add_field(
        name="🟢 Signal approved" if approved else "🔴 Signal rejected",
        value="Risk checks passed" if approved else ", ".join(signal.get("errors", [])) or "Risk checks failed",
        inline=False,
    )
    embed.set_footer(text="Weights are provisional until calibrated with backtesting")
    return embed


def send_discord_alert(signal: Mapping[str, Any] | None = None, webhook_url: str | None = None) -> dict[str, Any]:
    """Send one signal embed to a Discord webhook when configured."""
    if signal is None or not webhook_url:
        return {"channel": "discord", "status": "not_configured"}
    response = requests.post(webhook_url, json={"embeds": [build_signal_embed(signal).to_dict()]}, timeout=10)
    response.raise_for_status()
    return {"channel": "discord", "status": "sent", "status_code": response.status_code}
