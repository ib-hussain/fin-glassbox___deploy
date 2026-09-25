#!/usr/bin/env python3
"""
code/inference.py

fin-glassbox inference orchestrator
===================================

This file is designed to work as both:
    1. an importable Python module for Streamlit / deployment code, and
    2. a standalone CLI script for local inference tests.

Implemented inference modes
---------------------------
1. historical
   Reads already-generated Fusion outputs:
       outputs/results/FusionEngine/fused_decisions_chunk{chunk}_{split}.csv

2. frozen-cached
   Uses cached project data for the chosen ticker/date, then runs frozen trained
   downstream models in memory:
       PositionSizing row + Qualitative daily row
           -> frozen QuantitativeAnalyst
           -> frozen FusionEngine + rule barrier

   This is not a replay of fused CSV rows. It reloads trained checkpoints and
   recomputes the Quantitative + Fusion layers without saving new prediction CSVs.

3. manual
   Same frozen path as frozen-cached, but the input row is supplied manually as JSON.
   This supports what-if experiments without fetching data from the internet.

Important deployment boundary
-----------------------------
The first frozen runtime version deliberately starts at cached prepared project data.
It does not fetch fresh market/text/macro data. Full raw-data inference can be added
later by wiring TemporalEncoder, FinBERT, Technical, Volatility, Drawdown, StemGNN,
MTGNN, VaR/CVaR, Liquidity, and PositionSizing upstream.

No fundamentals are used anywhere in this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

VALID_SPLITS = {"train", "val", "test"}
VALID_CHUNKS = {1, 2, 3}


class InferenceError(RuntimeError):
    """Raised when inference cannot be completed safely."""


def repo_root_from(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        return Path(".").resolve()
    return Path(value).resolve()


def ensure_repo_imports(repo_root: Path) -> None:
    """Add project module directories to sys.path without requiring package install."""
    paths = [
        repo_root / "code",
        repo_root / "code" / "fusion",
        repo_root / "code" / "analysts",
        repo_root / "code" / "riskEngine",
        repo_root / "code" / "gnn",
        repo_root / "code" / "encoders",
    ]
    for p in paths:
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


def normalise_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def normalise_date(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid date: {value!r}")
    return str(dt.date())


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return json_safe(obj.detach().cpu().item())
        return json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, default=str), encoding="utf-8")


def resolve_device(device: str) -> str:
    device = str(device or "cpu").strip().lower()
    if device.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            return "cpu"
        return device
    return "cpu"


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_header(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def row_dict_from_frame(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) != 1:
        raise InferenceError(f"Expected exactly one row, got {len(df)}")
    return json_safe(df.iloc[0].to_dict())


def find_row_in_csv(
    path: Path,
    ticker: str,
    date: Optional[str] = None,
    *,
    chunksize: int = 250_000,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Chunk-scan a large CSV and return one ticker/date row.

    If date is None, returns the latest available row for the ticker.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    ticker = normalise_ticker(ticker)
    date_norm = normalise_date(date) if date is not None else None

    header = read_header(path)
    if "ticker" not in header:
        raise InferenceError(f"{path} has no ticker column")
    if "date" not in header:
        raise InferenceError(f"{path} has no date column")

    final_usecols = None
    if usecols is not None:
        requested = list(dict.fromkeys(["ticker", "date", *usecols]))
        final_usecols = [c for c in requested if c in header]

    latest: Optional[pd.DataFrame] = None
    for chunk in pd.read_csv(path, dtype={"ticker": str}, usecols=final_usecols, chunksize=int(chunksize), low_memory=False):
        if "ticker" not in chunk.columns or "date" not in chunk.columns:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper().str.strip()
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        sub = chunk[chunk["ticker"] == ticker]
        if len(sub) == 0:
            continue
        if date_norm is not None:
            sub = sub[sub["date"] == date_norm]
            if len(sub):
                return sub.tail(1).reset_index(drop=True)
        else:
            latest = sub.tail(1).reset_index(drop=True)

    if latest is not None:
        return latest
    if date_norm is None:
        raise LookupError(f"No row found for ticker={ticker} in {path}")
    raise LookupError(f"No row found for ticker={ticker}, date={date_norm} in {path}")


def maybe_find_row(path: Path, ticker: str, date: Optional[str], usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    try:
        return find_row_in_csv(path, ticker, date, usecols=usecols)
    except Exception:
        return None


# =============================================================================
# PATHS AND CONFIG OBJECTS
# =============================================================================

@dataclass
class InferenceRuntimeConfig:
    repo_root: str = "."
    device: str = "cpu"
    chunk: int = 3
    split: str = "test"
    exposure_mode: str = "moderate"
    horizon_mode: str = "short"
    batch_size: int = 1024
    num_workers: int = 0
    prefer_final_model: bool = True

    def validate(self) -> "InferenceRuntimeConfig":
        self.repo_root = str(repo_root_from(self.repo_root))
        self.chunk = int(self.chunk)
        if self.chunk not in VALID_CHUNKS:
            raise ValueError("chunk must be 1, 2, or 3")
        self.split = str(self.split).strip().lower()
        if self.split not in VALID_SPLITS:
            raise ValueError("split must be train, val, or test")
        self.device = resolve_device(self.device)
        self.batch_size = max(1, int(self.batch_size))
        self.num_workers = max(0, int(self.num_workers))
        return self

    @property
    def root(self) -> Path:
        return Path(self.repo_root).resolve()


def fusion_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "FusionEngine" / f"fused_decisions_chunk{chunk}_{split}.csv"


def fusion_xai_json_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "FusionEngine" / "xai" / f"fused_decisions_chunk{chunk}_{split}_xai_summary.json"


def position_sizing_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "PositionSizing" / f"position_sizing_chunk{chunk}_{split}.csv"


def qualitative_daily_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "QualitativeAnalyst" / f"qualitative_daily_chunk{chunk}_{split}.csv"


def quantitative_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "QuantitativeAnalyst" / f"quantitative_analysis_chunk{chunk}_{split}.csv"


def technical_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "TechnicalAnalyst" / f"predictions_chunk{chunk}_{split}.csv"


def volatility_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "Volatility" / f"predictions_chunk{chunk}_{split}.csv"


def drawdown_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "Drawdown" / f"predictions_chunk{chunk}_{split}.csv"


def stemgnn_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "StemGNN" / f"contagion_scores_chunk{chunk}_{split}.csv"


def regime_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "MTGNNRegime" / f"predictions_chunk{chunk}_{split}.csv"


def var_cvar_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "risk" / "chunks" / f"var_cvar_chunk{chunk}_{split}.csv"


def liquidity_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "risk" / "chunks" / f"liquidity_chunk{chunk}_{split}.csv"


def sentiment_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "analysts" / "sentiment" / f"chunk{chunk}_{split}_predictions.csv"


def news_csv_path(root: Path, chunk: int, split: str) -> Path:
    return root / "outputs" / "results" / "analysts" / "news" / f"chunk{chunk}_{split}_news_predictions.csv"


# =============================================================================
# EXPLANATION NARRATOR
# =============================================================================

class ExplanationNarrator:
    """Optional lightweight LLM narrator, with deterministic fallback.

    The LLM never changes the model decision. It only converts structured model output
    into a more readable explanation.
    """

    def __init__(self, model_name: str = "", device: str = "cpu", local_files_only: bool = True, max_new_tokens: int = 700) -> None:
        self.model_name = str(model_name or "").strip()
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = None
        self.model = None
        self.available = False
        self.load_error = ""

    def load(self) -> bool:
        if not self.model_name:
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only, trust_remote_code=True)
            kwargs: Dict[str, Any] = {"local_files_only": self.local_files_only, "trust_remote_code": True}
            if torch is not None and self.device.startswith("cuda"):
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            if torch is not None and not self.device.startswith("cuda"):
                self.model.to("cpu")
            self.available = True
            return True
        except Exception as exc:
            self.available = False
            self.load_error = str(exc)
            return False

    @staticmethod
    def compact_context(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build a rich but bounded context for explanation."""
        if "decision" in payload:
            decision = dict(payload.get("decision", {}))
            full_row = dict(payload.get("full_row", {}))
            modules = dict(payload.get("intermediate_outputs", {}).get("modules", {}))
            intermediate = dict(payload.get("intermediate_outputs", {}))
        else:
            decision = dict(payload)
            full_row = dict(payload)
            modules = {}
            intermediate = {}

        key_fields = [
            "ticker", "date", "final_recommendation", "final_fusion_signal",
            "final_fusion_risk_score", "final_fusion_confidence",
            "final_position_pct", "learned_recommendation",
            "learned_sell_prob", "learned_hold_prob", "learned_buy_prob",
            "learned_quantitative_weight", "learned_qualitative_weight",
            "rule_barrier_reasons", "rule_changed_action", "branch_weight_dominance",
            "top_attention_risk_driver", "text_available",
            "quantitative_recommendation", "risk_adjusted_quantitative_signal",
            "quantitative_risk_score", "quantitative_confidence",
            "qualitative_recommendation", "qualitative_score",
            "qualitative_risk_score", "qualitative_confidence",
            "regime_label", "regime_confidence", "liquidity_score",
            "tradable", "combined_risk_score", "size_reduction_reasons",
            "binding_cap_source", "hard_cap_applied",
            "technical_direction_score", "trend_score", "momentum_score",
            "timing_confidence", "technical_confidence",
            "volatility_risk_score", "drawdown_risk_score",
            "var_cvar_risk_score", "contagion_risk_score",
            "liquidity_risk_score", "regime_risk_score",
            "vol_10d", "vol_30d", "expected_drawdown_10d",
            "expected_drawdown_30d", "var_95", "var_99",
            "cvar_95", "cvar_99", "contagion_5d", "contagion_20d",
            "contagion_60d", "prob_calm", "prob_volatile",
            "prob_crisis", "prob_rotation",
            "quantitative_xai_summary", "qualitative_xai_summary",
            "fusion_xai_summary",
        ]

        selected: Dict[str, Any] = {}
        for source in [full_row, decision]:
            for k in key_fields:
                if k in source and k not in selected:
                    selected[k] = source[k]

        module_summaries: Dict[str, Any] = {}
        for name, row in modules.items():
            if isinstance(row, dict):
                row_small: Dict[str, Any] = {}
                for k, v in row.items():
                    lk = str(k).lower()
                    if (
                        "xai" in lk or "summary" in lk or "score" in lk or "risk" in lk or
                        "confidence" in lk or "prob" in lk or "regime" in lk or
                        "recommendation" in lk or "position" in lk or "cap" in lk or
                        "driver" in lk or "trend" in lk or "momentum" in lk or
                        "vol_" in lk or "drawdown" in lk or "var" in lk or "cvar" in lk or
                        "contagion" in lk or k in {"ticker", "date", "tradable", "size_bucket"}
                    ):
                        row_small[k] = v
                    if len(row_small) >= 70:
                        break
                module_summaries[name] = row_small

        return {
            "decision": selected,
            "module_outputs": module_summaries,
            "recomputed_intermediate_outputs": {
                k: v for k, v in intermediate.items()
                if k in {"quantitative", "qualitative"}
            },
        }

    @staticmethod
    def deterministic(payload: Dict[str, Any]) -> str:
        context = ExplanationNarrator.compact_context(payload)
        summary = context.get("decision", {})
        modules = context.get("module_outputs", {})

        rec = summary.get("final_recommendation", "UNKNOWN")
        ticker = summary.get("ticker", "the selected ticker")
        date = summary.get("date", "the selected date")
        conf = summary.get("final_fusion_confidence", summary.get("confidence", None))
        risk = summary.get("final_fusion_risk_score", summary.get("risk_score", None))
        pos = summary.get("final_position_pct", summary.get("position_pct", None))
        top = summary.get("top_attention_risk_driver", summary.get("top_risk_driver", "unknown"))
        rules = summary.get("rule_barrier_reasons", "no_rule_override")
        q_weight = summary.get("learned_quantitative_weight", None)
        qual_weight = summary.get("learned_qualitative_weight", None)
        text_avail = summary.get("text_available", None)

        def fmt(x: Any, nd: int = 3) -> str:
            try:
                v = float(x)
                return f"{v:.{nd}f}"
            except Exception:
                return "n/a"

        risk_lines = []
        for label, key in [
            ("volatility", "volatility_risk_score"),
            ("drawdown", "drawdown_risk_score"),
            ("VaR/CVaR", "var_cvar_risk_score"),
            ("contagion", "contagion_risk_score"),
            ("liquidity", "liquidity_risk_score"),
            ("regime", "regime_risk_score"),
        ]:
            if key in summary:
                risk_lines.append(f"{label}={fmt(summary.get(key))}")

        module_note = ""
        if modules:
            module_note = f" The raw output includes cached module-level traces for: {', '.join(sorted(modules.keys()))}."

        return (
            f"For {ticker} on {date}, the system recommends {rec}. "
            f"The fused confidence is {fmt(conf)} and the fused risk score is {fmt(risk)}, "
            f"so the model is presenting this as a constrained, risk-aware decision rather than a high-certainty call. "
            f"The suggested position size is {fmt(pos, 3)}% of capital. "
            f"The dominant quantitative risk driver is {top}. Component risk context: {', '.join(risk_lines) if risk_lines else 'not available'}. "
            f"The learned fusion layer assigned weight {fmt(q_weight)} to the quantitative branch and {fmt(qual_weight)} to the qualitative branch. "
            f"Text evidence availability for this ticker-date is {text_avail}, so the qualitative branch may be neutral or sparse. "
            f"Rule-barrier trace: {rules}. "
            f"This is a model-generated research output for decision support and explainability, not personalised financial advice."
            f"{module_note}"
        )

    def narrate(self, payload: Dict[str, Any]) -> str:
        context = self.compact_context(payload)
        if not self.available:
            return self.deterministic(payload)
        prompt = (
            "You are the explanation narrator for an explainable financial risk-management system. "
            "The financial recommendation has already been produced by trained models. "
            "Do not change the recommendation. Do not invent missing data. Do not provide personalised financial advice. "
            "Write a comprehensive but readable explanation for a dashboard user. "
            "Structure it into: 1) final decision, 2) confidence and position size, 3) key risk drivers, "
            "4) module-by-module evidence, 5) rule-barrier/risk-control effect, 6) limitations. "
            "Use the supplied JSON only. Be specific and reference the modules by name.\n\n"
            f"FULL_MODEL_CONTEXT_JSON:\n{json.dumps(json_safe(context), indent=2)}\n\nEXPLANATION:"
        )
        try:
            assert self.tokenizer is not None and self.model is not None
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            if torch is not None and self.device.startswith("cuda"):
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if "EXPLANATION:" in text:
                text = text.split("EXPLANATION:", 1)[-1].strip()
            return text.strip() or self.deterministic(payload)
        except Exception:
            return self.deterministic(payload)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class FinGlassboxInferenceEngine:
    """Unified inference engine for historical, frozen-cached, and manual modes."""

    def __init__(self, config: InferenceRuntimeConfig) -> None:
        self.config = config.validate()
        self.repo_root = self.config.root
        ensure_repo_imports(self.repo_root)
        self._fusion_module = None
        self._quant_module = None

    # ------------------------------------------------------------------
    # Inspection and packaging
    # ------------------------------------------------------------------

    def inspect(self) -> Dict[str, Any]:
        root = self.repo_root
        rows: Dict[str, Any] = {}
        for chunk in sorted(VALID_CHUNKS):
            for split in ["val", "test"]:
                fpath = fusion_csv_path(root, chunk, split)
                xpath = fusion_xai_json_path(root, chunk, split)
                rows[f"fusion_chunk{chunk}_{split}"] = {
                    "csv": str(fpath),
                    "csv_exists": fpath.exists(),
                    "csv_rows": count_csv_rows(fpath) if fpath.exists() else 0,
                    "xai": str(xpath),
                    "xai_exists": xpath.exists(),
                    "xai_size_mb": round(xpath.stat().st_size / 1024 / 1024, 4) if xpath.exists() else 0.0,
                }
        for chunk in sorted(VALID_CHUNKS):
            rows[f"quant_model_chunk{chunk}"] = self._model_file_status("QuantitativeAnalyst", chunk)
            rows[f"fusion_model_chunk{chunk}"] = self._model_file_status("FusionEngine", chunk)
        return {"repo_root": str(root), "device": self.config.device, "files": rows}

    def _model_file_status(self, module: str, chunk: int) -> Dict[str, Any]:
        root = self.repo_root
        base = root / "outputs" / "models" / module / f"chunk{chunk}"
        files = {
            "best_model": base / "best_model.pt",
            "final_model": base / "final_model.pt",
            "scaler": base / "scaler.npz",
        }
        return {
            k: {"path": str(p), "exists": p.exists(), "size_mb": round(p.stat().st_size / 1024 / 1024, 4) if p.exists() else 0.0}
            for k, p in files.items()
        }

    def package_plan(self, mode: str = "frozen-cached", chunks: Optional[List[int]] = None, splits: Optional[List[str]] = None) -> Dict[str, Any]:
        mode = str(mode).strip().lower()
        chunks = chunks or [self.config.chunk]
        splits = splits or [self.config.split]
        files: List[str] = ["code/inference.py", "code/deploy.py"]
        if mode in {"historical", "all"}:
            for c in chunks:
                for s in splits:
                    files.append(f"outputs/results/FusionEngine/fused_decisions_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/FusionEngine/xai/fused_decisions_chunk{c}_{s}_xai_summary.json")
        if mode in {"frozen-cached", "manual", "all"}:
            files.extend([
                "code/fusion/fusion_layer.py",
                "code/analysts/quantitative_analyst.py",
                "code/analysts/qualitative_analyst.py",
            ])
            for c in chunks:
                files.append(f"outputs/models/QuantitativeAnalyst/chunk{c}/best_model.pt")
                files.append(f"outputs/models/QuantitativeAnalyst/chunk{c}/final_model.pt")
                files.append(f"outputs/models/QuantitativeAnalyst/chunk{c}/scaler.npz")
                files.append(f"outputs/models/FusionEngine/chunk{c}/best_model.pt")
                files.append(f"outputs/models/FusionEngine/chunk{c}/final_model.pt")
                files.append(f"outputs/models/FusionEngine/chunk{c}/scaler.npz")
                files.append("outputs/models/Narrator/Qwen3-0.6B")
                for s in splits:
                    files.append(f"outputs/results/PositionSizing/position_sizing_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/QualitativeAnalyst/qualitative_daily_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/TechnicalAnalyst/predictions_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/Volatility/predictions_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/Drawdown/predictions_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/StemGNN/contagion_scores_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/MTGNNRegime/predictions_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/risk/chunks/var_cvar_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/risk/chunks/liquidity_chunk{c}_{s}.csv")
                    files.append(f"outputs/results/analysts/sentiment/chunk{c}_{s}_predictions.csv")
                    files.append(f"outputs/results/analysts/news/chunk{c}_{s}_news_predictions.csv")
        unique = list(dict.fromkeys(files))
        return {
            "mode": mode,
            "chunks": chunks,
            "splits": splits,
            "files": unique,
            "existing": {f: (self.repo_root / f).exists() for f in unique},
            "note": "For first local demo, chunk3 val/test is enough. Full lower-level raw inference needs more code/data/model artefacts.",
        }

    # ------------------------------------------------------------------
    # Mode 1: historical replay
    # ------------------------------------------------------------------

    def historical(self, ticker: str, date: Optional[str] = None, chunk: Optional[int] = None, split: Optional[str] = None) -> Dict[str, Any]:
        chunk = int(chunk or self.config.chunk)
        split = str(split or self.config.split)
        path = fusion_csv_path(self.repo_root, chunk, split)
        row_df = find_row_in_csv(path, ticker, date)
        row = row_dict_from_frame(row_df)
        result = self._decision_payload(row, mode="historical", chunk=chunk, split=split)
        result["source_file"] = str(path)
        return result

    # ------------------------------------------------------------------
    # Cached module-output collection
    # ------------------------------------------------------------------

    def _safe_module_row(self, path: Path, ticker: str, date: str, *, mode: str = "ticker_date") -> Optional[Dict[str, Any]]:
        try:
            if mode == "ticker_date":
                df = maybe_find_row(path, ticker, date)
            elif mode == "ticker":
                if not path.exists():
                    return None
                chunks = []
                for chunk_df in pd.read_csv(path, chunksize=200000, dtype={"ticker": str}, low_memory=False):
                    if "ticker" not in chunk_df.columns:
                        return None
                    sub = chunk_df[chunk_df["ticker"].astype(str).str.upper().str.strip() == normalise_ticker(ticker)]
                    if len(sub):
                        chunks.append(sub.tail(1))
                        break
                if not chunks:
                    return None
                df = pd.concat(chunks, ignore_index=True)
            elif mode == "date":
                if not path.exists():
                    return None
                target_date = normalise_date(date)
                chunks = []
                for chunk_df in pd.read_csv(path, chunksize=200000, low_memory=False):
                    if "date" not in chunk_df.columns:
                        return None
                    dates = pd.to_datetime(chunk_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    sub = chunk_df[dates == target_date]
                    if len(sub):
                        chunks.append(sub.tail(1))
                        break
                if not chunks:
                    return None
                df = pd.concat(chunks, ignore_index=True)
            else:
                return None
            if df is None or len(df) == 0:
                return None
            return row_dict_from_frame(df)
        except Exception as exc:
            return {"_error": str(exc), "_path": str(path)}

    def collect_module_outputs(self, ticker: str, date: str, chunk: int, split: str) -> Dict[str, Any]:
        """Collect cached row-level outputs from all available modules."""
        ticker = normalise_ticker(ticker)
        date = normalise_date(date)
        paths = {
            "technical_analyst": (technical_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "volatility_model": (volatility_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "drawdown_model": (drawdown_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "var_cvar": (var_cvar_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "liquidity": (liquidity_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "stemgnn_contagion": (stemgnn_csv_path(self.repo_root, chunk, split), "ticker"),
            "mtgnn_regime": (regime_csv_path(self.repo_root, chunk, split), "date"),
            "position_sizing": (position_sizing_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "qualitative_daily": (qualitative_daily_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "quantitative_analyst_saved": (quantitative_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "fusion_saved": (fusion_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "sentiment_events": (sentiment_csv_path(self.repo_root, chunk, split), "ticker_date"),
            "news_events": (news_csv_path(self.repo_root, chunk, split), "ticker_date"),
        }
        out: Dict[str, Any] = {}
        for name, (path, lookup_mode) in paths.items():
            row = self._safe_module_row(path, ticker, date, mode=lookup_mode)
            if row is not None:
                out[name] = row
        return out

    # ------------------------------------------------------------------
    # Mode 2: frozen-cached inference
    # ------------------------------------------------------------------

    def frozen_cached(self, ticker: str, date: Optional[str] = None, chunk: Optional[int] = None, split: Optional[str] = None) -> Dict[str, Any]:
        chunk = int(chunk or self.config.chunk)
        split = str(split or self.config.split)
        ticker = normalise_ticker(ticker)
        date_norm = normalise_date(date) if date is not None else None

        pos_path = position_sizing_csv_path(self.repo_root, chunk, split)
        pos_df = find_row_in_csv(pos_path, ticker, date_norm)
        selected_date = str(pos_df.iloc[0]["date"])

        qual_path = qualitative_daily_csv_path(self.repo_root, chunk, split)
        qual_df = maybe_find_row(qual_path, ticker, selected_date)
        if qual_df is None or len(qual_df) == 0:
            qual_df = self._neutral_qualitative(ticker, selected_date)

        quant_df = self._run_quantitative_from_position(pos_df, chunk, split)
        fusion_df = self._run_fusion_from_branch_rows(quant_df, qual_df, chunk, split)
        row = row_dict_from_frame(fusion_df)
        payload = self._decision_payload(row, mode="frozen-cached", chunk=chunk, split=split)
        payload["runtime_sources"] = {
            "position_sizing_csv": str(pos_path),
            "qualitative_daily_csv": str(qual_path),
            "quantitative_model": str(self._chosen_model_path("QuantitativeAnalyst", chunk)),
            "fusion_model": str(self._chosen_model_path("FusionEngine", chunk)),
        }
        modules = self.collect_module_outputs(ticker, selected_date, chunk, split)
        modules["quantitative_recomputed"] = row_dict_from_frame(quant_df)
        modules["qualitative_runtime"] = row_dict_from_frame(qual_df)
        modules["fusion_recomputed"] = row
        payload["intermediate_outputs"] = {
            "quantitative": row_dict_from_frame(quant_df),
            "qualitative": row_dict_from_frame(qual_df),
            "modules": modules,
        }
        payload["human_explanation"] = ExplanationNarrator().narrate(payload)
        return payload

    # ------------------------------------------------------------------
    # Mode 3: manual / what-if inference
    # ------------------------------------------------------------------

    def manual(self, input_payload: Dict[str, Any], chunk: Optional[int] = None, split: Optional[str] = None) -> Dict[str, Any]:
        chunk = int(chunk or input_payload.get("chunk", self.config.chunk))
        split = str(split or input_payload.get("split", self.config.split))
        ticker = normalise_ticker(str(input_payload.get("ticker", "MANUAL")))
        date_norm = normalise_date(input_payload.get("date", pd.Timestamp.today().date()))

        if "position_sizing" in input_payload and isinstance(input_payload["position_sizing"], dict):
            pos_row = dict(input_payload["position_sizing"])
        else:
            pos_row = dict(input_payload)
        pos_row["ticker"] = ticker
        pos_row["date"] = date_norm
        pos_df = pd.DataFrame([pos_row])

        if "qualitative" in input_payload and isinstance(input_payload["qualitative"], dict):
            qrow = dict(input_payload["qualitative"])
            qrow["ticker"] = ticker
            qrow["date"] = date_norm
            qual_df = pd.DataFrame([qrow])
        else:
            qual_df = self._neutral_qualitative(ticker, date_norm)

        quant_df = self._run_quantitative_from_position(pos_df, chunk, split)
        fusion_df = self._run_fusion_from_branch_rows(quant_df, qual_df, chunk, split)
        row = row_dict_from_frame(fusion_df)
        payload = self._decision_payload(row, mode="manual", chunk=chunk, split=split)
        payload["manual_input"] = json_safe(input_payload)
        modules = {
            "position_sizing_manual": row_dict_from_frame(pos_df),
            "qualitative_manual": row_dict_from_frame(qual_df),
            "quantitative_recomputed": row_dict_from_frame(quant_df),
            "fusion_recomputed": row,
        }
        payload["intermediate_outputs"] = {
            "quantitative": row_dict_from_frame(quant_df),
            "qualitative": row_dict_from_frame(qual_df),
            "modules": modules,
        }
        payload["human_explanation"] = ExplanationNarrator().narrate(payload)
        return payload

    # ------------------------------------------------------------------
    # Frozen model internals
    # ------------------------------------------------------------------

    def _import_quant(self):
        if self._quant_module is None:
            import quantitative_analyst as qa  # type: ignore
            self._quant_module = qa
        return self._quant_module

    def _import_fusion(self):
        if self._fusion_module is None:
            import fusion_layer as fl  # type: ignore
            self._fusion_module = fl
        return self._fusion_module

    def _chosen_model_path(self, module: str, chunk: int) -> Path:
        base = self.repo_root / "outputs" / "models" / module / f"chunk{chunk}"
        final = base / "final_model.pt"
        best = base / "best_model.pt"
        if self.config.prefer_final_model and final.exists():
            return final
        if best.exists():
            return best
        if final.exists():
            return final
        raise FileNotFoundError(f"No checkpoint found for {module} chunk{chunk}: checked {final} and {best}")

    def _run_quantitative_from_position(self, pos_df: pd.DataFrame, chunk: int, split: str) -> pd.DataFrame:
        if torch is None:
            raise InferenceError("PyTorch is required for frozen-cached/manual inference")
        qa = self._import_quant()

        qcfg = qa.QuantitativeAnalystConfig(repo_root=str(self.repo_root), device=self.config.device)
        qcfg.batch_size = int(self.config.batch_size)
        qcfg.num_workers = int(self.config.num_workers)
        qcfg.resolve_paths()

        # qa.load_model uses best_model.pt. For deployment we prefer final_model.pt if requested.
        model_path = self._chosen_model_path("QuantitativeAnalyst", chunk)
        payload = torch.load(model_path, map_location=qcfg.device, weights_only=False)
        model_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        model = qa.QuantitativeRiskAttentionModel(
            n_risks=len(qa.RISK_COLUMNS),
            context_dim=len(qa.CONTEXT_COLUMNS),
            attention_dim=int(model_cfg.get("attention_dim", qcfg.attention_dim)),
            hidden_dim=int(model_cfg.get("hidden_dim", qcfg.hidden_dim)),
            n_layers=int(model_cfg.get("n_layers", qcfg.n_layers)),
            dropout=float(model_cfg.get("dropout", qcfg.dropout)),
        ).to(qcfg.device)
        state = payload.get("state_dict", payload.get("model_state_dict", payload))
        model.load_state_dict(state)
        model.eval()
        scaler = qa.ContextScaler.load(qa.scaler_path(qcfg, chunk))

        df = pos_df.copy()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise InferenceError("Position input contains invalid date")

        risk, context, _target, prepared = qa.prepare_arrays(df, qcfg)
        context_scaled = scaler.transform(context)
        pred = qa.predict_batches(model, risk, context_scaled, qcfg)
        outputs = pred["outputs"]
        attention = pred["attention"]
        pooled_risk = pred["pooled_risk"]

        out = prepared.copy()
        out["risk_adjusted_quantitative_signal"] = qa.clip11(outputs[:, 0])
        out["quantitative_risk_score"] = qa.clip01(outputs[:, 1])
        out["quantitative_confidence"] = qa.clip01(outputs[:, 2])
        out["attention_pooled_risk_score"] = qa.clip01(pooled_risk)
        for i, name in enumerate(qa.RISK_NAMES):
            out[f"risk_attention_{name}"] = attention[:, i]
        top_idx = np.argmax(attention, axis=1)
        out["top_attention_risk_driver"] = [qa.RISK_NAMES[int(i)] for i in top_idx]
        out["technical_direction_score"] = out["technical_direction_score_rule"]
        out["quantitative_risk_state"] = qa.classify_risk_state(out["quantitative_risk_score"])
        out["recommended_capital_fraction"] = qa.safe_numeric(out, "recommended_capital_fraction", 0.0)
        out["recommended_capital_pct"] = qa.safe_numeric(out, "recommended_capital_pct", 0.0)
        out["quantitative_recommendation"] = qa.classify_action(
            out["risk_adjusted_quantitative_signal"],
            out["quantitative_risk_score"],
            out["quantitative_confidence"],
            out["recommended_capital_fraction"],
            qcfg,
        )
        out["quantitative_action_strength"] = (
            np.abs(out["risk_adjusted_quantitative_signal"].values) * out["quantitative_confidence"].values
        ).astype(np.float32)
        out["xai_summary"] = out.apply(qa.build_row_xai, axis=1)
        out = out.rename(columns={"xai_summary": "quantitative_xai_summary"})
        out["chunk"] = int(chunk)
        out["split"] = str(split)
        return out.reset_index(drop=True)

    def _neutral_qualitative(self, ticker: str, date: str) -> pd.DataFrame:
        fl = self._import_fusion()
        base = pd.DataFrame([{"ticker": normalise_ticker(ticker), "date": pd.to_datetime(normalise_date(date))}])
        return fl.neutral_qualitative_frame(base).reset_index(drop=True)

    def _run_fusion_from_branch_rows(self, quant_df: pd.DataFrame, qual_df: pd.DataFrame, chunk: int, split: str) -> pd.DataFrame:
        if torch is None:
            raise InferenceError("PyTorch is required for frozen-cached/manual inference")
        fl = self._import_fusion()

        fcfg = fl.FusionConfig(repo_root=str(self.repo_root), device=self.config.device)
        fcfg.batch_size = int(self.config.batch_size)
        fcfg.num_workers = int(self.config.num_workers)
        fcfg.exposure_mode = self.config.exposure_mode
        fcfg.horizon_mode = self.config.horizon_mode
        fcfg.allow_missing_qualitative = True
        fcfg.resolve_paths()

        q = quant_df.copy()
        d = qual_df.copy()
        q["ticker"] = q["ticker"].astype(str).str.upper().str.strip()
        d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
        q["date"] = pd.to_datetime(q["date"], errors="coerce")
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        q = q.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        d = d.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

        merged = q.merge(d, on=["ticker", "date"], how="left", suffixes=("", "_qual"))
        neutral = fl.neutral_qualitative_frame(merged[["ticker", "date"]])
        for col in neutral.columns:
            if col in ["ticker", "date"]:
                continue
            if col not in merged.columns:
                merged[col] = neutral[col].values
            else:
                if merged[col].dtype == object:
                    merged[col] = merged[col].fillna(neutral[col])
                else:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(neutral[col])

        df_prepared = fl.prepare_fusion_dataframe(merged, fcfg)
        x, feature_names = fl.build_feature_matrix(df_prepared)

        model_path = self._chosen_model_path("FusionEngine", chunk)
        payload = torch.load(model_path, map_location=fcfg.device, weights_only=False)
        model_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        model = fl.HybridFusionModel(
            input_dim=int(payload.get("input_dim", len(payload.get("feature_names", feature_names)))),
            hidden_dim=int(model_cfg.get("hidden_dim", fcfg.hidden_dim)),
            n_layers=int(model_cfg.get("n_layers", fcfg.n_layers)),
            dropout=float(model_cfg.get("dropout", fcfg.dropout)),
        ).to(fcfg.device)
        state = payload.get("state_dict", payload.get("model_state_dict", payload))
        model.load_state_dict(state)
        model.eval()
        scaler = fl.FusionScaler.load(fl.scaler_path(fcfg, chunk))

        expected = list(payload.get("feature_names", scaler.feature_names))
        if expected and list(feature_names) != expected:
            raise InferenceError(
                "Frozen Fusion feature schema mismatch. "
                f"expected={len(expected)} generated={len(feature_names)}"
            )

        x_scaled = scaler.transform(x)
        aux = df_prepared[["risk_adjusted_quantitative_signal", "qualitative_score"]].values.astype(np.float32)
        pred = fl.predict_batches(model, x_scaled, aux, fcfg)
        outputs = pred["outputs"]
        logits = pred["action_logits"]
        branch_weights = pred["branch_weights"]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        pred_id = probs.argmax(axis=1)

        out = df_prepared.copy()
        out["learned_sell_prob"] = probs[:, fl.ACTION_TO_ID["SELL"]].astype(np.float32)
        out["learned_hold_prob"] = probs[:, fl.ACTION_TO_ID["HOLD"]].astype(np.float32)
        out["learned_buy_prob"] = probs[:, fl.ACTION_TO_ID["BUY"]].astype(np.float32)
        out["learned_recommendation"] = [fl.ID_TO_ACTION[int(i)] for i in pred_id]
        out["learned_fusion_signal"] = fl.clip11(outputs[:, 0]).astype(np.float32)
        out["learned_fusion_risk_score"] = fl.clip01(outputs[:, 1]).astype(np.float32)
        out["learned_fusion_confidence"] = fl.clip01(outputs[:, 2]).astype(np.float32)
        out["learned_position_multiplier"] = fl.clip01(outputs[:, 3]).astype(np.float32)
        out["learned_quantitative_weight"] = branch_weights[:, 0].astype(np.float32)
        out["learned_qualitative_weight"] = branch_weights[:, 1].astype(np.float32)

        out = fl.apply_user_rule_barrier(out, fcfg)
        out["final_fusion_signal"] = out["learned_fusion_signal"]
        out["final_fusion_risk_score"] = out["learned_fusion_risk_score"]
        out["final_fusion_confidence"] = out["learned_fusion_confidence"]
        out["branch_weight_dominance"] = np.where(
            out["learned_quantitative_weight"].values >= out["learned_qualitative_weight"].values,
            "quantitative",
            "qualitative",
        )
        out["fusion_xai_summary"] = [fl.build_fusion_xai_row(row) for _, row in out.iterrows()]
        out["chunk"] = int(chunk)
        out["split"] = str(split)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _decision_payload(self, row: Dict[str, Any], mode: str, chunk: int, split: str) -> Dict[str, Any]:
        important = [
            "ticker", "date", "final_recommendation", "final_fusion_signal", "final_fusion_risk_score",
            "final_fusion_confidence", "final_position_fraction", "final_position_pct",
            "learned_recommendation", "learned_sell_prob", "learned_hold_prob", "learned_buy_prob",
            "learned_quantitative_weight", "learned_qualitative_weight", "branch_weight_dominance",
            "rule_changed_action", "rule_barrier_reasons", "quantitative_recommendation",
            "risk_adjusted_quantitative_signal", "quantitative_risk_score", "quantitative_confidence",
            "qualitative_recommendation", "qualitative_score", "qualitative_risk_score", "qualitative_confidence",
            "text_available", "top_attention_risk_driver", "attention_pooled_risk_score",
            "risk_attention_volatility", "risk_attention_drawdown", "risk_attention_var_cvar",
            "risk_attention_contagion", "risk_attention_liquidity", "risk_attention_regime",
            "regime_label", "liquidity_score", "tradable", "drawdown_risk_score",
            "contagion_risk_score", "combined_risk_score", "fusion_xai_summary",
            "quantitative_xai_summary", "qualitative_xai_summary",
        ]
        summary = {k: row.get(k) for k in important if k in row}
        payload = {
            "mode": mode,
            "chunk": int(chunk),
            "split": str(split),
            "decision": summary,
            "full_row": row,
        }
        payload["human_explanation"] = ExplanationNarrator().narrate(payload)
        return payload


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="fin-glassbox inference orchestrator")
    p.add_argument("command", choices=["inspect", "historical", "frozen-cached", "manual", "package-plan", "narrate"])
    p.add_argument("--repo-root", default=".")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ticker", default="")
    p.add_argument("--date", default="")
    p.add_argument("--input-json", default="")
    p.add_argument("--output-json", default="")
    p.add_argument("--mode", default="frozen-cached", choices=["historical", "frozen-cached", "manual", "all"])
    p.add_argument("--chunks", default="", help="Comma-separated chunks for package-plan, e.g. 3 or 1,2,3")
    p.add_argument("--splits", default="", help="Comma-separated splits for package-plan, e.g. val,test")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--exposure-mode", default="moderate", choices=["conservative", "moderate", "aggressive"])
    p.add_argument("--horizon-mode", default="short", choices=["short", "long"])
    p.add_argument("--prefer-best-model", action="store_true", help="Use best_model.pt instead of final_model.pt when both exist")
    p.add_argument("--llm-model", default="", help="Optional local/HF path, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--llm-allow-download", action="store_true", help="Allow transformers to download the narrator model if not local")
    return p


def _parse_int_list(text: str, default: List[int]) -> List[int]:
    if not text.strip():
        return default
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str, default: List[str]) -> List[str]:
    if not text.strip():
        return default
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    args = build_parser().parse_args()
    cfg = InferenceRuntimeConfig(
        repo_root=args.repo_root,
        device=args.device,
        chunk=args.chunk,
        split=args.split,
        exposure_mode=args.exposure_mode,
        horizon_mode=args.horizon_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefer_final_model=not bool(args.prefer_best_model),
    ).validate()
    engine = FinGlassboxInferenceEngine(cfg)

    if args.command == "inspect":
        result = engine.inspect()
    elif args.command == "historical":
        if not args.ticker:
            raise SystemExit("--ticker is required")
        result = engine.historical(args.ticker, args.date or None)
    elif args.command == "frozen-cached":
        if not args.ticker:
            raise SystemExit("--ticker is required")
        result = engine.frozen_cached(args.ticker, args.date or None)
    elif args.command == "manual":
        if not args.input_json:
            raise SystemExit("--input-json is required")
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        result = engine.manual(payload)
    elif args.command == "package-plan":
        chunks = _parse_int_list(args.chunks, [args.chunk])
        splits = _parse_str_list(args.splits, [args.split])
        result = engine.package_plan(mode=args.mode, chunks=chunks, splits=splits)
    elif args.command == "narrate":
        if not args.input_json:
            raise SystemExit("--input-json is required")
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        narrator = ExplanationNarrator(args.llm_model, device=cfg.device, local_files_only=not args.llm_allow_download, max_new_tokens=900)
        narrator.load()
        result = {"human_explanation": narrator.narrate(payload), "llm_loaded": narrator.available, "llm_error": getattr(narrator, "load_error", "")}
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")

    if args.output_json:
        write_json(Path(args.output_json), result)
        print(f"WROTE {args.output_json}")
    print(json.dumps(json_safe(result), indent=2, default=str))


if __name__ == "__main__":
    main()
