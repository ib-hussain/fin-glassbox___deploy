#!/usr/bin/env python3
"""
code/deploy.py

Streamlit deployment interface for fin-glassbox.

This UI intentionally hides engineering controls from the end user. The repo root,
device, chunk and split are supplied through CLI defaults, while the user interacts
with ticker/date, inference mode, risk profile and transparency panels.

Run:
    streamlit run deploy.py -- --repo-root . --device cpu --default-chunk 3 --default-split test

Qwen narrator packaging:
    The app automatically looks for:
        outputs/models/Narrator/Qwen3-0.6B
    If that local model folder exists, Qwen is used as the human-readable narrator.
    If it is absent, the deterministic built-in narrator is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-chunk", type=int, default=3)
    parser.add_argument("--default-split", default="test")
    return parser.parse_known_args()[0]


ARGS = parse_args()
REPO_ROOT = Path(ARGS.repo_root).resolve()
CODE_DIR = REPO_ROOT 
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit("Streamlit is required. Install it with: pip install streamlit") from exc

from inference import (  # noqa: E402
    ExplanationNarrator,
    FinGlassboxInferenceEngine,
    InferenceRuntimeConfig,
    fusion_csv_path,
    json_safe,
    position_sizing_csv_path,
)


DEFAULT_CHUNK = int(ARGS.default_chunk)
DEFAULT_SPLIT = str(ARGS.default_split)
DEFAULT_DEVICE = str(ARGS.device)
QWEN_LOCAL_PATH = REPO_ROOT / "outputs" / "models" / "Narrator" / "Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine(repo_root: str, device: str, chunk: int, split: str, exposure_mode: str, horizon_mode: str, prefer_final: bool) -> FinGlassboxInferenceEngine:
    cfg = InferenceRuntimeConfig(
        repo_root=repo_root,
        device=device,
        chunk=int(chunk),
        split=str(split),
        exposure_mode=exposure_mode,
        horizon_mode=horizon_mode,
        prefer_final_model=prefer_final,
    )
    return FinGlassboxInferenceEngine(cfg)


@st.cache_resource(show_spinner=False)
def get_narrator(local_model_path: str, device: str) -> ExplanationNarrator:
    path = Path(local_model_path)
    if not path.exists():
        return ExplanationNarrator("", device=device, local_files_only=True, max_new_tokens=900)
    narrator = ExplanationNarrator(str(path), device=device, local_files_only=True, max_new_tokens=900)
    narrator.load()
    return narrator


@st.cache_data(show_spinner=False)
def load_picker_frame(repo_root: str, mode: str, chunk: int, split: str) -> pd.DataFrame:
    root = Path(repo_root)
    if mode == "Historical replay":
        path = fusion_csv_path(root, int(chunk), str(split))
    else:
        path = position_sizing_csv_path(root, int(chunk), str(split))

    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date"])

    df = pd.read_csv(path, usecols=["ticker", "date"], dtype={"ticker": str}, low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def fmt_num(value: Any, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def fmt_pct(value: Any, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}%"
    except Exception:
        return "n/a"


def small_df(mapping: Dict[str, Any], value_name: str = "value") -> pd.DataFrame:
    return pd.DataFrame([{"metric": k, value_name: safe_float(v, 0.0)} for k, v in mapping.items() if v is not None])


def render_progress(label: str, value: Any) -> None:
    v = max(0.0, min(1.0, safe_float(value, 0.0)))
    st.write(f"**{label}** — {v:.3f}")
    st.progress(v)


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.set_page_config(
        page_title="fin-glassbox",
        page_icon="🥀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🥀fin-glassbox")
    st.caption("Explainable multimodal financial risk inference — research demonstration and decision-support interface, not personalised financial advice.")


def choose_ticker_date(repo_root: str, mode: str, chunk: int, split: str) -> Tuple[str, Optional[str], pd.DataFrame]:
    picker_mode = "Historical replay" if mode == "Historical replay" else "Frozen-cached inference"
    picker = load_picker_frame(repo_root, picker_mode, int(chunk), str(split))
    if picker.empty:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        date = st.text_input("Date YYYY-MM-DD; leave blank for latest", value="").strip()
        return ticker, date or None, picker

    tickers = sorted(picker["ticker"].dropna().unique().tolist())
    default = "AAPL" if "AAPL" in tickers else "A" if "A" in tickers else tickers[0]
    ticker = st.selectbox("Ticker", tickers, index=tickers.index(default))

    dates = sorted(picker.loc[picker["ticker"] == ticker, "date"].dropna().unique().tolist())
    date = st.selectbox("Date", dates, index=max(0, len(dates) - 1)) if dates else None
    return str(ticker), str(date) if date else None, picker


def render_decision_cards(result: Dict[str, Any]) -> None:
    decision = result.get("decision", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommendation", str(decision.get("final_recommendation", "UNKNOWN")))
    c2.metric("Confidence", fmt_num(decision.get("final_fusion_confidence"), 3))
    c3.metric("Risk score", fmt_num(decision.get("final_fusion_risk_score"), 3))
    c4.metric("Position", fmt_pct(decision.get("final_position_pct"), 3))
    c5.metric("Top risk", str(decision.get("top_attention_risk_driver", "unknown")))


def render_explanation(result: Dict[str, Any]) -> None:
    st.subheader("Human-readable explanation")
    st.info(result.get("human_explanation", "No explanation available."))


def render_core_charts(result: Dict[str, Any]) -> None:
    row = result.get("full_row", {})
    decision = result.get("decision", {})

    st.subheader("Decision analytics")

    risk_scores = {
        "Volatility": row.get("volatility_risk_score", decision.get("volatility_risk_score")),
        "Drawdown": row.get("drawdown_risk_score", decision.get("drawdown_risk_score")),
        "VaR/CVaR": row.get("var_cvar_risk_score", decision.get("var_cvar_risk_score")),
        "Contagion": row.get("contagion_risk_score", decision.get("contagion_risk_score")),
        "Liquidity": row.get("liquidity_risk_score", decision.get("liquidity_risk_score")),
        "Regime": row.get("regime_risk_score", decision.get("regime_risk_score")),
    }
    attention = {
        "Volatility": decision.get("risk_attention_volatility", row.get("risk_attention_volatility")),
        "Drawdown": decision.get("risk_attention_drawdown", row.get("risk_attention_drawdown")),
        "VaR/CVaR": decision.get("risk_attention_var_cvar", row.get("risk_attention_var_cvar")),
        "Contagion": decision.get("risk_attention_contagion", row.get("risk_attention_contagion")),
        "Liquidity": decision.get("risk_attention_liquidity", row.get("risk_attention_liquidity")),
        "Regime": decision.get("risk_attention_regime", row.get("risk_attention_regime")),
    }
    probs = {
        "SELL": decision.get("learned_sell_prob", row.get("learned_sell_prob")),
        "HOLD": decision.get("learned_hold_prob", row.get("learned_hold_prob")),
        "BUY": decision.get("learned_buy_prob", row.get("learned_buy_prob")),
    }
    branch_weights = {
        "Quantitative": decision.get("learned_quantitative_weight", row.get("learned_quantitative_weight")),
        "Qualitative": decision.get("learned_qualitative_weight", row.get("learned_qualitative_weight")),
    }
    regime_probs = {
        "Calm": row.get("prob_calm"),
        "Volatile": row.get("prob_volatile"),
        "Crisis": row.get("prob_crisis"),
        "Rotation": row.get("prob_rotation"),
    }
    technical = {
        "Trend": row.get("trend_score"),
        "Momentum": row.get("momentum_score"),
        "Timing confidence": row.get("timing_confidence"),
        "Technical confidence": row.get("technical_confidence"),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Component risk scores")
        st.bar_chart(small_df(risk_scores, "score").set_index("metric"))
    with c2:
        st.markdown("#### Quantitative risk attention")
        st.bar_chart(small_df(attention, "attention").set_index("metric"))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Final action probabilities")
        st.bar_chart(small_df(probs, "probability").set_index("metric"))
    with c4:
        st.markdown("#### Branch fusion weights")
        st.bar_chart(small_df(branch_weights, "weight").set_index("metric"))

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Regime probabilities")
        st.bar_chart(small_df(regime_probs, "probability").set_index("metric"))
    with c6:
        st.markdown("#### Technical signal context")
        st.bar_chart(small_df(technical, "score").set_index("metric"))

    st.markdown("#### Position and rule-barrier context")
    pos_table = {
        "pre_rule_position_pct": safe_float(row.get("pre_rule_learned_position_fraction", 0.0)) * 100.0,
        "final_position_pct": row.get("final_position_pct"),
        "recommended_capital_pct": row.get("recommended_capital_pct"),
        "user_rule_cap_pct": safe_float(row.get("user_rule_cap_fraction", 0.0)) * 100.0,
    }
    st.bar_chart(small_df(pos_table, "percent").set_index("metric"))

    tail_cols = {
        "VaR 95": row.get("var_95"),
        "VaR 99": row.get("var_99"),
        "CVaR 95": row.get("cvar_95"),
        "CVaR 99": row.get("cvar_99"),
        "Expected drawdown 10d": row.get("expected_drawdown_10d"),
        "Expected drawdown 30d": row.get("expected_drawdown_30d"),
        "Volatility 10d": row.get("vol_10d"),
        "Volatility 30d": row.get("vol_30d"),
    }
    st.markdown("#### Tail-risk and path-risk figures")
    st.dataframe(pd.DataFrame([tail_cols]), use_container_width=True)


def render_model_transparency(result: Dict[str, Any]) -> None:
    modules = result.get("intermediate_outputs", {}).get("modules", {})
    if not modules:
        st.write("No module-level output dictionary was returned for this mode.")
        return

    summary_rows = []
    for name, row in modules.items():
        if not isinstance(row, dict):
            continue
        summary_rows.append({
            "module": name,
            "ticker": row.get("ticker", ""),
            "date": row.get("date", ""),
            "recommendation": row.get("final_recommendation", row.get("quantitative_recommendation", row.get("qualitative_recommendation", ""))),
            "risk_score": row.get("final_fusion_risk_score", row.get("quantitative_risk_score", row.get("drawdown_risk_score", row.get("volatility_risk_score", row.get("regime_risk_score", ""))))),
            "confidence": row.get("final_fusion_confidence", row.get("quantitative_confidence", row.get("qualitative_confidence", row.get("technical_confidence", row.get("regime_confidence", ""))))),
            "xai_summary": row.get("fusion_xai_summary", row.get("quantitative_xai_summary", row.get("qualitative_xai_summary", row.get("xai_summary", row.get("size_reduction_reasons", ""))))),
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for name, row in modules.items():
        with st.expander(f"{name}", expanded=False):
            st.json(json_safe(row))


def render_transparency_sections(result: Dict[str, Any]) -> None:
    with st.expander("Full system decision JSON", expanded=False):
        st.json(json_safe(result.get("decision", {})))

    with st.expander("Every available model/module output", expanded=False):
        render_model_transparency(result)

    with st.expander("Raw result object", expanded=False):
        st.json(json_safe(result))

    sources = result.get("runtime_sources") or {}
    if result.get("source_file") or sources:
        with st.expander("Runtime/source files", expanded=False):
            if result.get("source_file"):
                st.write({"source_file": result.get("source_file")})
            if sources:
                st.json(json_safe(sources))


def apply_qwen_narrator(result: Dict[str, Any], device: str) -> Dict[str, Any]:
    narrator = get_narrator(str(QWEN_LOCAL_PATH), device)
    out = dict(result)
    out["human_explanation"] = narrator.narrate(out)
    out["narrator"] = {
        "model_path": str(QWEN_LOCAL_PATH),
        "qwen_packaged": QWEN_LOCAL_PATH.exists(),
        "qwen_loaded": bool(narrator.available),
        "fallback_used": not bool(narrator.available),
        "load_error": getattr(narrator, "load_error", ""),
    }
    return out


def manual_payload_form(ticker: str, date: Optional[str], chunk: int, split: str) -> Dict[str, Any]:
    st.subheader("Manual what-if inputs")
    st.caption("These values are passed into the frozen QuantitativeAnalyst and FusionEngine path. Advanced missing fields are filled by deployment defaults.")

    left, right = st.columns(2)
    with left:
        trend_score = st.slider("trend_score", 0.0, 1.0, 0.50, 0.01)
        momentum_score = st.slider("momentum_score", 0.0, 1.0, 0.50, 0.01)
        timing_confidence = st.slider("timing_confidence", 0.0, 1.0, 0.50, 0.01)
        technical_confidence = st.slider("technical_confidence", 0.0, 1.0, 0.50, 0.01)
        recommended_capital_pct = st.slider("recommended_capital_pct", 0.0, 15.0, 5.0, 0.25)
    with right:
        volatility_risk = st.slider("volatility_risk_score", 0.0, 1.0, 0.35, 0.01)
        drawdown_risk = st.slider("drawdown_risk_score", 0.0, 1.0, 0.43, 0.01)
        var_cvar_risk = st.slider("var_cvar_risk_score", 0.0, 1.0, 0.43, 0.01)
        contagion_risk = st.slider("contagion_risk_score", 0.0, 1.0, 0.42, 0.01)
        liquidity_risk = st.slider("liquidity_risk_score", 0.0, 1.0, 0.21, 0.01)
        regime_risk = st.slider("regime_risk_score", 0.0, 1.0, 0.78, 0.01)

    q1, q2 = st.columns(2)
    with q1:
        qualitative_score = st.slider("qualitative_score", -1.0, 1.0, 0.0, 0.01)
        qualitative_risk = st.slider("qualitative_risk_score", 0.0, 1.0, 0.5, 0.01)
    with q2:
        qualitative_conf = st.slider("qualitative_confidence", 0.0, 1.0, 0.0, 0.01)
        text_available = st.checkbox("text_available", value=False)

    recommended_capital_fraction = recommended_capital_pct / 100.0
    position_fraction_of_max = min(max(recommended_capital_fraction / 0.10, 0.0), 1.0)
    combined_risk = (
        0.20 * volatility_risk
        + 0.15 * drawdown_risk
        + 0.15 * var_cvar_risk
        + 0.25 * contagion_risk
        + 0.15 * liquidity_risk
        + 0.10 * regime_risk
    )

    position_sizing = {
        "ticker": ticker, "date": date, "chunk": chunk, "split": split,
        "exposure_mode": "moderate", "horizon_mode": "short",
        "trend_score": trend_score, "momentum_score": momentum_score,
        "timing_confidence": timing_confidence, "technical_confidence": technical_confidence,
        "technical_direction_score_rule": (trend_score - 0.5) * 0.6 + (momentum_score - 0.5) * 0.4,
        "volatility_risk_score": volatility_risk, "drawdown_risk_score": drawdown_risk,
        "var_cvar_risk_score": var_cvar_risk, "contagion_risk_score": contagion_risk,
        "liquidity_risk_score": liquidity_risk, "regime_risk_score": regime_risk,
        "combined_risk_score": combined_risk,
        "position_fraction_of_max": position_fraction_of_max,
        "recommended_capital_fraction": recommended_capital_fraction,
        "recommended_capital_pct": recommended_capital_pct,
        "max_single_stock_exposure": 0.10,
        "regime_confidence": 0.50,
        "hard_cap_applied": 0.0,
        "pre_cap_position_fraction_of_max": position_fraction_of_max,
        "pre_cap_capital_fraction": recommended_capital_fraction,
        "risk_bucket_fraction": position_fraction_of_max,
        "size_bucket": "manual",
        "binding_cap_source": "manual",
        "regime_label": "manual",
        "tradable": 1.0,
    }

    qualitative = {
        "ticker": ticker, "date": date,
        "event_count": 1.0 if text_available else 0.0,
        "sentiment_event_count": 1.0 if text_available else 0.0,
        "news_event_count": 1.0 if text_available else 0.0,
        "qualitative_score": qualitative_score,
        "qualitative_risk_score": qualitative_risk,
        "qualitative_confidence": qualitative_conf,
        "qualitative_recommendation": "BUY" if qualitative_score > 0.2 else "SELL" if qualitative_score < -0.2 else "HOLD",
        "max_event_risk_score": qualitative_risk,
        "mean_event_risk_score": qualitative_risk,
        "mean_sentiment_score": qualitative_score,
        "mean_news_impact_score": qualitative_score,
        "mean_news_importance": qualitative_conf,
        "dominant_qualitative_driver": "manual_text_input" if text_available else "no_text_event",
        "qualitative_xai_summary": "Manual qualitative what-if input.",
    }

    return {"ticker": ticker, "date": date, "chunk": chunk, "split": split, "position_sizing": position_sizing, "qualitative": qualitative}


def run_mode(engine: FinGlassboxInferenceEngine, mode: str, ticker: str, date: Optional[str], manual_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if mode == "Historical replay":
        return engine.historical(ticker=ticker, date=date, chunk=DEFAULT_CHUNK, split=DEFAULT_SPLIT)
    if mode == "Frozen-cached inference":
        return engine.frozen_cached(ticker=ticker, date=date, chunk=DEFAULT_CHUNK, split=DEFAULT_SPLIT)
    if mode == "Manual what-if":
        if manual_payload is None:
            raise ValueError("Manual input payload missing.")
        return engine.manual(manual_payload, chunk=DEFAULT_CHUNK, split=DEFAULT_SPLIT)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    render_header()

    with st.sidebar:
        st.header("fin-glassbox controls")
        st.caption(f"Runtime: chunk{DEFAULT_CHUNK}_{DEFAULT_SPLIT} · device={DEFAULT_DEVICE}")
        mode = st.radio("Inference mode", ["Frozen-cached inference", "Historical replay", "Manual what-if"], index=0)
        exposure_mode = st.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
        horizon_mode = st.selectbox("Horizon", ["short", "long"], index=0)
        prefer_final = True

        st.header("Narrator")
        if QWEN_LOCAL_PATH.exists():
            st.success("Qwen3-0.6B packaged locally")
        else:
            st.warning("Qwen3-0.6B not packaged yet; using deterministic narrator fallback.")

        st.header("Transparency")
        st.caption("Full system JSON, model/module outputs, and raw result are always available below the charts.")

    engine = get_engine(str(REPO_ROOT), DEFAULT_DEVICE, DEFAULT_CHUNK, DEFAULT_SPLIT, exposure_mode, horizon_mode, prefer_final)

    st.subheader("Input selection")
    ticker, date, picker = choose_ticker_date(str(REPO_ROOT), mode, DEFAULT_CHUNK, DEFAULT_SPLIT)
    if not picker.empty:
        st.caption(f"Loaded {len(picker):,} ticker-date choices for chunk{DEFAULT_CHUNK}_{DEFAULT_SPLIT}.")

    manual_payload = None
    if mode == "Manual what-if":
        manual_payload = manual_payload_form(ticker, date, DEFAULT_CHUNK, DEFAULT_SPLIT)

    if not st.button("Run inference", type="primary", use_container_width=True):
        st.info("Select a ticker/date and run inference.")
        return

    with st.spinner("Running frozen inference and building explanations..."):
        try:
            result = run_mode(engine, mode, ticker, date, manual_payload)
            result = apply_qwen_narrator(result, DEFAULT_DEVICE)
        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)
            return

    render_decision_cards(result)
    render_explanation(result)
    render_core_charts(result)
    render_transparency_sections(result)

    st.download_button(
        "Download complete result JSON",
        data=json.dumps(json_safe(result), indent=2, default=str),
        file_name=f"fin_glassbox_{mode.lower().replace(' ', '_')}_{ticker}_{date or 'latest'}.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
