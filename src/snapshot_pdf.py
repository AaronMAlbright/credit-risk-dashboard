"""
PDF snapshot export — one-click briefing document.

Generates a clean, printable PDF snapshot of the current dashboard state.
Uses only Python stdlib plus reportlab (pip install reportlab).

The PDF contains:
  Page 1: Header + current regime + key metrics table
  Page 2: Signal scores table + factor contributions
  Page 3: Backtest summary stats

Falls back gracefully (available=False) if reportlab is not installed.

Public API
----------
  generate_snapshot_pdf(df, output_path) -> dict   {path, success, error, n_pages}
  generate_snapshot_bytes(df) -> bytes | None       (for Streamlit download button)
"""

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_val(row, key, default=float("nan")):
    """Return row[key] if present and not NaN, else default."""
    val = row.get(key, default)
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return val  # string column — return as-is


def _fmt_pct(val, decimals=1):
    """Format a decimal fraction as percentage string, e.g. 0.65 -> '65.0%'."""
    try:
        f = float(val)
        if math.isnan(f):
            return "—"
        return f"{f * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_float(val, decimals=2):
    """Format a float to fixed decimals, returning '—' on nan/None."""
    try:
        f = float(val)
        if math.isnan(f):
            return "—"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if val not in (None, "") else "—"


def _fmt_score(val):
    """Format a 0-100 score to one decimal place."""
    return _fmt_float(val, 1)


# ---------------------------------------------------------------------------
# Color / style constants (defined after reportlab import in the function)
# ---------------------------------------------------------------------------

_HDR_BG   = "#1a2035"   # table header background
_ROW_A    = "#0e1117"   # alternating row A
_ROW_B    = "#151820"   # alternating row B
_BORDER   = "#2d3550"   # light-gray border
_WHITE    = "#ffffff"
_LIGHT    = "#c8cfe8"   # light text for data rows
_ACCENT   = "#4a90d9"   # section title accent


# ---------------------------------------------------------------------------
# generate_snapshot_pdf
# ---------------------------------------------------------------------------

def generate_snapshot_pdf(df: pd.DataFrame, output_path=None) -> dict:
    """Generate a PDF snapshot of the current dashboard state.

    Parameters
    ----------
    df : pd.DataFrame
        Full dashboard DataFrame. Latest values pulled from df.iloc[-1].
    output_path : str | Path | io.BytesIO | None
        Where to write the PDF. If None a BytesIO buffer is created and
        returned in the result dict under key 'buffer'.

    Returns
    -------
    dict with keys:
        success      : bool
        available    : bool
        path         : output_path (or None if BytesIO)
        buffer       : BytesIO (only when output_path is None / BytesIO)
        n_pages      : int
        error        : str | None
    """
    # --- guarded import ---
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return {
            "success": False,
            "available": False,
            "error": "reportlab not installed. Run: pip install reportlab",
        }

    # --- resolve output ---
    use_buffer = output_path is None or isinstance(output_path, io.BytesIO)
    if use_buffer:
        buf = output_path if isinstance(output_path, io.BytesIO) else io.BytesIO()
    else:
        buf = str(output_path)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # --- latest row ---
    row = df.iloc[-1] if len(df) > 0 else pd.Series(dtype=object)

    # --- colour helpers (after reportlab is imported) ---
    hdr_bg   = colors.HexColor(_HDR_BG)
    row_a    = colors.HexColor(_ROW_A)
    row_b    = colors.HexColor(_ROW_B)
    border_c = colors.HexColor(_BORDER)
    white_c  = colors.white
    light_c  = colors.HexColor(_LIGHT)
    accent_c = colors.HexColor(_ACCENT)

    # --- styles ---
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DashTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=white_c,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "DashSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=light_c,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=accent_c,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "DashBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=light_c,
        spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        textColor=colors.HexColor("#5a6080"),
        alignment=TA_CENTER,
    )

    # --- table styling factory ---
    def _table_style(n_data_rows: int) -> TableStyle:
        """Build alternating-row dark table style for n_data_rows of data (excl. header)."""
        cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0), hdr_bg),
            ("TEXTCOLOR",   (0, 0), (-1, 0), white_c),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUND", (0, 1), (-1, -1), [row_a, row_b]),
            ("TEXTCOLOR",   (0, 1), (-1, -1), light_c),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("GRID",        (0, 0), (-1, -1), 0.4, border_c),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]
        # Explicit alternating background per row
        for i in range(1, n_data_rows + 1):
            bg = row_a if i % 2 == 1 else row_b
            cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        return TableStyle(cmds)

    def _hr():
        return HRFlowable(
            width="100%", thickness=0.5, color=border_c, spaceAfter=6, spaceBefore=6,
        )

    def _footer():
        return Paragraph(
            f"Generated {today_str} · Macro Credit Risk Dashboard", footer_style
        )

    # -----------------------------------------------------------------------
    # Extract values from latest row
    # -----------------------------------------------------------------------
    final_decision  = str(row.get("final_decision", "—")) if "final_decision" in row.index else "—"
    composite_score = _safe_val(row, "composite_risk_score_smooth", float("nan"))
    composite_str   = _fmt_score(composite_score)

    hy_spread   = _fmt_float(_safe_val(row, "hy_spread"),   2)
    ig_spread   = _fmt_float(_safe_val(row, "ig_spread",
                              _safe_val(row, "bamlc0a0cm",
                              _safe_val(row, "ig_oas"))),    2)
    vix_val     = _fmt_float(_safe_val(row, "vix"),          1)
    sp500_val   = _fmt_float(_safe_val(row, "sp500"),        1)

    # Fed funds: try multiple column aliases
    fed_funds_val = _safe_val(row, "dff",
                     _safe_val(row, "fed_funds_rate",
                     _safe_val(row, "fed_funds")))
    fed_funds_str = _fmt_float(fed_funds_val, 2)

    # 10y - 3m yield spread
    t10y  = _safe_val(row, "t10y2y",  # fallback proxy
             _safe_val(row, "dgs10"))
    t3m   = _safe_val(row, "dtb3",
             _safe_val(row, "t3m"))
    if not math.isnan(float(t10y)) and not math.isnan(float(t3m)):
        spread_10y3m = _fmt_float(float(t10y) - float(t3m), 2)
    else:
        # Try direct column
        spread_10y3m = _fmt_float(_safe_val(row, "t10y3m"), 2)

    equity_weight = _safe_val(row, "equity_weight", float("nan"))
    equity_w_str  = _fmt_pct(equity_weight) if not (
        isinstance(equity_weight, float) and math.isnan(equity_weight)
    ) else "—"

    taylor_rate = _safe_val(row, "taylor_rate", float("nan"))
    policy_gap  = _safe_val(row, "policy_gap",  float("nan"))
    if not (isinstance(taylor_rate, float) and math.isnan(taylor_rate)):
        policy_stance_str = f"Taylor: {_fmt_float(taylor_rate, 2)}% (gap {_fmt_float(policy_gap, 2)}pp)"
    else:
        policy_stance_str = str(row.get("policy_stance", "—")) if "policy_stance" in row.index else "—"

    model_confidence = str(row.get("model_confidence", "—")) if "model_confidence" in row.index else "—"

    # Action / detail text
    final_detail = str(row.get("final_decision_detail", final_decision)) \
        if "final_decision_detail" in row.index else final_decision

    # Date from index or column
    try:
        record_date = str(df.index[-1])[:10] if len(df) > 0 else today_str
    except Exception:
        record_date = today_str

    # -----------------------------------------------------------------------
    # Signal scores
    # -----------------------------------------------------------------------
    _SIGNAL_COLS = [
        ("treasury_stress_score_smooth",         "Treasury Stress"),
        ("complacency_score_smooth",             "Complacency"),
        ("credit_market_risk_score_smooth",      "Credit Market Risk"),
        ("macro_risk_score_smooth",              "Macro Risk"),
        ("liquidity_regime_score_smooth",        "Liquidity"),
        ("enhanced_funding_stress_score_smooth", "Funding Stress"),
        ("fx_commodity_score_smooth",            "FX / Commodity"),
        ("banking_stress_score_smooth",          "Banking Stress"),
        ("mean_reversion_score_smooth",          "Mean Reversion"),
    ]
    # Raw counterparts (drop "_smooth" suffix)
    signal_rows = []
    for col_smooth, label in _SIGNAL_COLS:
        if col_smooth in row.index:
            smooth_val = _safe_val(row, col_smooth)
            raw_col = col_smooth.replace("_smooth", "")
            raw_val = _safe_val(row, raw_col, float("nan"))
            signal_rows.append([
                label,
                _fmt_score(raw_val),
                _fmt_score(smooth_val),
                _classify_score(smooth_val),
            ])

    # Signal contributions column
    contrib_rows = []
    if "signal_contributions" in row.index:
        contrib_raw = row["signal_contributions"]
        if isinstance(contrib_raw, dict):
            for sig, contrib in contrib_raw.items():
                contrib_rows.append([sig, _fmt_float(contrib, 3)])
        elif isinstance(contrib_raw, str):
            try:
                import json as _json
                contrib_dict = _json.loads(contrib_raw)
                for sig, contrib in contrib_dict.items():
                    contrib_rows.append([sig, _fmt_float(contrib, 3)])
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Backtest stats
    # -----------------------------------------------------------------------
    has_backtest = any(c in df.columns for c in ("cum_blended", "cum_net", "cum_sp500"))
    bt_rows = []
    if has_backtest:
        # Compute simple stats from cumulative return columns
        def _cum_to_stats(col):
            if col not in df.columns:
                return {"total": float("nan"), "vol": float("nan"), "mdd": float("nan"), "sharpe": float("nan")}
            series = df[col].dropna()
            if len(series) < 2:
                return {"total": float("nan"), "vol": float("nan"), "mdd": float("nan"), "sharpe": float("nan")}
            # total return from first to last
            total = float(series.iloc[-1]) / float(series.iloc[0]) - 1 if float(series.iloc[0]) != 0 else float("nan")
            # daily returns (from pct_change on cumulative)
            rets = series.pct_change().dropna()
            ann_vol  = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else float("nan")
            ann_ret  = float((1 + total) ** (252 / max(len(series), 1)) - 1) if not math.isnan(total) else float("nan")
            sharpe   = ann_ret / ann_vol if (not math.isnan(ann_vol) and ann_vol > 0) else float("nan")
            # max drawdown
            rolling_max = series.cummax()
            dd = (series - rolling_max) / rolling_max
            mdd = float(dd.min()) if len(dd) > 0 else float("nan")
            return {"total": total, "vol": ann_vol, "mdd": mdd, "sharpe": sharpe}

        strat_col = "cum_net" if "cum_net" in df.columns else "cum_blended"
        strat = _cum_to_stats(strat_col)
        bench = _cum_to_stats("cum_sp500")

        bt_rows = [
            ["Total Return",    _fmt_pct(strat["total"]),  _fmt_pct(bench["total"])],
            ["Ann. Volatility", _fmt_pct(strat["vol"]),    _fmt_pct(bench["vol"])],
            ["Max Drawdown",    _fmt_pct(strat["mdd"]),    _fmt_pct(bench["mdd"])],
            ["Sharpe Ratio",    _fmt_float(strat["sharpe"], 2), _fmt_float(bench["sharpe"], 2)],
            ["Current Eq. Wt", equity_w_str, "100%"],
        ]

    # -----------------------------------------------------------------------
    # Build story
    # -----------------------------------------------------------------------
    story = []
    PAGE_W = letter[0]
    M = 0.75 * inch
    usable_w = PAGE_W - 2 * M

    # ===== PAGE 1 =====
    story.append(Paragraph("Macro Credit Risk Dashboard", title_style))
    story.append(Paragraph(
        f"Regime: {final_decision} &nbsp;|&nbsp; Score: {composite_str}/100 &nbsp;|&nbsp; {record_date}",
        subtitle_style,
    ))
    story.append(_hr())

    # Key metrics table (3 col × 3 row data)
    story.append(Paragraph("Key Metrics", section_style))
    col_w = usable_w / 3
    metrics_data = [
        ["HY Spread",       "IG Spread",     "VIX"],
        [hy_spread,         ig_spread,        vix_val],
        ["S&P 500",         "Fed Funds",      "10y−3m Spread"],
        [sp500_val,         fed_funds_str,    spread_10y3m],
        ["Composite Score", "Equity Weight",  "Policy Stance"],
        [composite_str + "/100", equity_w_str, policy_stance_str],
    ]
    metrics_tbl = Table(metrics_data, colWidths=[col_w, col_w, col_w])
    # custom style: alternating by label/value pairs
    metrics_tbl.setStyle(TableStyle([
        # Header rows (0, 2, 4)
        ("BACKGROUND",   (0, 0), (-1, 0), hdr_bg),
        ("BACKGROUND",   (0, 2), (-1, 2), hdr_bg),
        ("BACKGROUND",   (0, 4), (-1, 4), hdr_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0), white_c),
        ("TEXTCOLOR",    (0, 2), (-1, 2), white_c),
        ("TEXTCOLOR",    (0, 4), (-1, 4), white_c),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTNAME",     (0, 4), (-1, 4), "Helvetica-Bold"),
        # Data rows (1, 3, 5)
        ("BACKGROUND",   (0, 1), (-1, 1), row_a),
        ("BACKGROUND",   (0, 3), (-1, 3), row_b),
        ("BACKGROUND",   (0, 5), (-1, 5), row_a),
        ("TEXTCOLOR",    (0, 1), (-1, 5), light_c),
        ("FONTNAME",     (0, 1), (-1, 5), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",         (0, 0), (-1, -1), 0.4, border_c),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_tbl)
    story.append(Spacer(1, 0.15 * inch))
    story.append(_hr())

    # Current Regime section
    story.append(Paragraph("Current Regime", section_style))
    regime_lines = [
        f"<b>Decision:</b> {final_detail}",
        f"<b>Confidence:</b> {model_confidence}",
        f"<b>Equity Allocation:</b> {equity_w_str}",
    ]
    for line in regime_lines:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_footer())

    # ===== PAGE BREAK → PAGE 2 =====
    from reportlab.platypus import PageBreak
    story.append(PageBreak())

    story.append(Paragraph("Signal Scores", title_style))
    story.append(Spacer(1, 0.08 * inch))

    if signal_rows:
        sig_header = ["Signal", "Raw Score", "Smoothed", "Label"]
        sig_col_w  = [usable_w * 0.40, usable_w * 0.18, usable_w * 0.18, usable_w * 0.24]
        sig_data = [sig_header] + signal_rows
        sig_tbl  = Table(sig_data, colWidths=sig_col_w)
        sig_tbl.setStyle(_table_style(len(signal_rows)))
        # left-align first column
        sig_tbl.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ]))
        story.append(sig_tbl)
    else:
        story.append(Paragraph("Signal score columns not found in DataFrame.", body_style))

    story.append(Spacer(1, 0.15 * inch))
    story.append(_hr())

    # Factor contributions
    story.append(Paragraph("Factor Contributions", section_style))
    if contrib_rows:
        contrib_header = ["Signal", "Contribution"]
        contrib_col_w  = [usable_w * 0.60, usable_w * 0.40]
        contrib_data   = [contrib_header] + contrib_rows
        contrib_tbl    = Table(contrib_data, colWidths=contrib_col_w)
        contrib_tbl.setStyle(_table_style(len(contrib_rows)))
        contrib_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (0, -1), "LEFT")]))
        story.append(contrib_tbl)
    else:
        story.append(Paragraph("Factor contribution data not available.", body_style))

    story.append(Spacer(1, 0.1 * inch))
    story.append(_footer())

    # ===== PAGE BREAK → PAGE 3 (backtest) =====
    if has_backtest and bt_rows:
        story.append(PageBreak())
        story.append(Paragraph("Backtest Performance", title_style))
        story.append(Spacer(1, 0.08 * inch))

        bt_header   = ["Metric", "Strategy", "S&P 500"]
        bt_col_w    = [usable_w * 0.40, usable_w * 0.30, usable_w * 0.30]
        bt_data     = [bt_header] + bt_rows
        bt_tbl      = Table(bt_data, colWidths=bt_col_w)
        bt_tbl.setStyle(_table_style(len(bt_rows)))
        bt_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (0, -1), "LEFT")]))
        story.append(bt_tbl)
        story.append(Spacer(1, 0.1 * inch))
        story.append(_footer())

    # -----------------------------------------------------------------------
    # Build PDF
    # -----------------------------------------------------------------------
    n_pages = 2 + (1 if (has_backtest and bt_rows) else 0)

    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=M, rightMargin=M,
        topMargin=M, bottomMargin=M,
        title="Macro Credit Risk Dashboard",
        author="Credit Risk Dashboard",
    )

    try:
        doc.build(story)
    except Exception as e:
        return {"success": False, "available": True, "error": str(e), "n_pages": 0}

    if use_buffer:
        buf.seek(0)
        return {
            "success": True,
            "available": True,
            "path": None,
            "buffer": buf,
            "n_pages": n_pages,
            "error": None,
        }
    else:
        return {
            "success": True,
            "available": True,
            "path": output_path,
            "buffer": None,
            "n_pages": n_pages,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Score label helper (used internally)
# ---------------------------------------------------------------------------

def _classify_score(score) -> str:
    """Map a 0-100 composite score to a human-readable label."""
    try:
        s = float(score)
        if math.isnan(s):
            return "—"
    except (TypeError, ValueError):
        return "—"
    if s >= 80:
        return "Very High"
    if s >= 65:
        return "High"
    if s >= 50:
        return "Elevated"
    if s >= 35:
        return "Moderate"
    if s >= 20:
        return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# Public: generate_snapshot_bytes
# ---------------------------------------------------------------------------

def generate_snapshot_bytes(df: pd.DataFrame) -> bytes | None:
    """Return PDF snapshot as raw bytes for a Streamlit download button.

    Returns None if PDF generation fails (e.g. reportlab not installed).
    """
    result = generate_snapshot_pdf(df, output_path=None)
    if result.get("success") and result.get("buffer") is not None:
        buf = result["buffer"]
        if isinstance(buf, io.BytesIO):
            return buf.getvalue()
    return None
