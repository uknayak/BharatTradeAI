"""Sequential LangGraph orchestration for numerical market analysis."""

from typing import Any, Callable, TypedDict

from app.agents.fundamental_agent import analyze_fundamentals
from app.agents.market_agent import analyze_market_regime
from app.agents.ml_agent import run_ml_prediction
from app.agents.news_agent import analyze_news
from app.agents.options_agent import analyze_options_flow
from app.agents.risk_agent import evaluate_risk
from app.agents.technical_agent import analyze_technicals
from app.agents.validator_agent import validate_signal


class AnalysisState(TypedDict, total=False):
    market_data: Any
    options_chain: Any
    underlying_price: float
    candles: Any
    fundamentals: dict[str, Any]
    previous_fundamentals: dict[str, Any]
    news_records: Any
    ml_model: dict[str, Any]
    ml_features: Any
    period: int
    signal_weights: dict[str, float]
    llm_explainer: Callable[[dict[str, Any]], str]
    market: dict[str, Any]
    options: dict[str, Any]
    technical: dict[str, Any]
    fundamental: dict[str, Any]
    news: dict[str, Any]
    ml: dict[str, Any]
    risk: dict[str, Any]
    validation: dict[str, Any]
    signal_score: dict[str, Any]
    explanation: str
    execution_trace: list[str]


def _trace(state: AnalysisState, name: str) -> list[str]:
    return [*state.get("execution_trace", []), name]


def market_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("market_data") is None:
        return {"market": {"agent": "market_agent", "status": "skipped"}, "execution_trace": _trace(state, "market")}
    return {"market": analyze_market_regime(state["market_data"], state.get("period", 14)), "execution_trace": _trace(state, "market")}


def options_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("options_chain") is None or state.get("underlying_price") is None:
        return {"options": {"agent": "options_agent", "status": "skipped"}, "execution_trace": _trace(state, "options")}
    return {"options": analyze_option_chain_safe(state), "execution_trace": _trace(state, "options")}


def analyze_option_chain_safe(state: AnalysisState) -> dict[str, Any]:
    return analyze_options_flow(state["options_chain"], state["underlying_price"])


def technical_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("candles") is None:
        return {"technical": {"agent": "technical_agent", "status": "skipped"}, "execution_trace": _trace(state, "technical")}
    return {"technical": analyze_technicals(state["candles"], state.get("period", 14)), "execution_trace": _trace(state, "technical")}


def fundamental_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("fundamentals") is None:
        return {"fundamental": {"agent": "fundamental_agent", "status": "skipped"}, "execution_trace": _trace(state, "fundamental")}
    return {"fundamental": analyze_fundamentals(state["fundamentals"], state.get("previous_fundamentals")), "execution_trace": _trace(state, "fundamental")}


def news_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("news_records") is None:
        return {"news": {"agent": "news_agent", "status": "skipped"}, "execution_trace": _trace(state, "news")}
    return {"news": analyze_news(state["news_records"]), "execution_trace": _trace(state, "news")}


def ml_node(state: AnalysisState) -> dict[str, Any]:
    if state.get("ml_model") is None or state.get("ml_features") is None:
        return {"ml": {"agent": "ml_agent", "status": "skipped"}, "execution_trace": _trace(state, "ml")}
    return {"ml": {"agent": "ml_agent", "status": "ready", "prediction": run_ml_prediction(state["ml_model"], state["ml_features"])}, "execution_trace": _trace(state, "ml")}


def risk_node(state: AnalysisState) -> dict[str, Any]:
    return {"risk": evaluate_risk(state), "execution_trace": _trace(state, "risk")}


def validator_node(state: AnalysisState) -> dict[str, Any]:
    return {"validation": validate_signal(state), "execution_trace": _trace(state, "validator")}


def explain_node(state: AnalysisState) -> dict[str, Any]:
    summary = {key: state.get(key) for key in ("market", "options", "technical", "fundamental", "news", "ml", "risk", "validation")}
    explainer = state.get("llm_explainer")
    explanation = explainer(summary) if explainer else "Numerical analysis completed; review validation and risk outputs before acting."
    return {"explanation": explanation, "execution_trace": _trace(state, "explanation")}


NODE_SEQUENCE = (market_node, options_node, technical_node, fundamental_node, news_node, ml_node, risk_node, validator_node, explain_node)


class _FallbackGraph:
    def invoke(self, state: AnalysisState) -> AnalysisState:
        current = dict(state)
        for node in NODE_SEQUENCE:
            current.update(node(current))
        return current


def build_graph() -> Any:
    """Build and compile LangGraph, with a local runner for optional installs."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _FallbackGraph()
    graph = StateGraph(AnalysisState)
    names = ("market", "options", "technical", "fundamental", "news", "ml", "risk", "validator", "explanation")
    for name, node in zip(names, NODE_SEQUENCE):
        graph.add_node(name, node)
    graph.add_edge(START, "market")
    for previous, current in zip(names, names[1:]):
        graph.add_edge(previous, current)
    graph.add_edge("explanation", END)
    return graph.compile()


def run_agentic_analysis(state: AnalysisState) -> AnalysisState:
    """Execute the fixed orchestration graph; LLM output is explanation only."""
    return build_graph().invoke(state)