"""
tools/financial_tools.py

LangChain @tool definitions for the Business Health Agent.

All tools now read from the unified data loader (data/loader.py) which handles:
  - Kaggle CSV files  (data/kaggle/*.csv)   ← primary
  - Mock JSON         (data/mock.json)       ← fallback

No tool cares which source is active — they all receive the same normalised DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool

from data.loader import (
    get_available_periods,
    load_transactions,
    resolve_period,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fmt(amount: float) -> str:
    """Format a number as currency string."""
    return f"${amount:,.2f}"


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_profit_loss(period: str) -> dict:
    """
    Returns profit and loss summary for the given period.

    Args:
        period: 'this-month', 'last-month', 'two-months-ago', a year-month like
                '2023-06', or a year like '2023'.

    Returns revenue breakdown, cost of goods sold, gross profit, operating expenses,
    and net profit with margin percentages.
    Use for questions about profitability, revenue, margins, or expenses.
    """
    df = load_transactions()
    key = resolve_period(period, df)
    if key is None:
        return {"error": f"No data found. Available periods: {get_available_periods(df)}"}

    sub = df[df["year_month"] == key] if "year_month" in df.columns else df

    revenue_df = sub[sub["row_class"] == "revenue"]
    cogs_df    = sub[sub["row_class"] == "cogs"]
    opex_df    = sub[sub["row_class"] == "opex"]

    total_revenue = revenue_df["credit"].sum()
    total_cogs    = cogs_df["debit"].sum()
    gross_profit  = total_revenue - total_cogs
    total_opex    = opex_df["debit"].sum()
    op_profit     = gross_profit - total_opex
    # Approximate tax at 30% of operating profit
    tax           = max(op_profit * 0.30, 0)
    net_profit    = op_profit - tax

    # Revenue breakdown by description/category
    rev_breakdown = (
        revenue_df.groupby("description")["credit"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .to_dict()
    )

    # Opex breakdown
    opex_breakdown = (
        opex_df.groupby("description")["debit"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    return {
        "period": key,
        "data_rows_analysed": len(sub),
        "revenue": {
            "breakdown": {k: round(v, 2) for k, v in rev_breakdown.items()},
            "total": round(total_revenue, 2),
        },
        "cost_of_goods_sold": round(total_cogs, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_margin_pct": _pct(gross_profit, total_revenue),
        "operating_expenses": {
            "breakdown": {k: round(v, 2) for k, v in opex_breakdown.items()},
            "total": round(total_opex, 2),
        },
        "operating_profit": round(op_profit, 2),
        "estimated_tax": round(tax, 2),
        "net_profit": round(net_profit, 2),
        "net_margin_pct": _pct(net_profit, total_revenue),
        "is_profitable": net_profit > 0,
    }


@tool
def get_balance_sheet(date: Optional[str] = None) -> dict:
    """
    Returns a balance sheet snapshot — assets, liabilities, equity, and key ratios.

    Args:
        date: optional ISO date like '2023-06-30'. Defaults to most recent data.

    Use for questions about financial position, solvency, liquidity, debt, or cash balance.
    """
    df = load_transactions()

    # Filter to on-or-before date if provided
    if date and "date" in df.columns:
        import pandas as pd
        cutoff = pd.Timestamp(date)
        sub = df[df["date"] <= cutoff]
    else:
        sub = df

    # Cash = net of all credits minus debits across the full history
    cash = sub["credit"].sum() - sub["debit"].sum()
    cash = max(cash, 0)

    # Receivables = outstanding revenue rows (credit with no offsetting debit)
    receivable_rows = sub[sub["row_class"] == "revenue"]
    accounts_receivable = receivable_rows["credit"].sum() * 0.15  # ~15% still outstanding

    # Inventory proxy = COGS debit that hasn't been sold yet
    cogs_total = sub[sub["row_class"] == "cogs"]["debit"].sum()
    inventory  = cogs_total * 0.20

    # Assets
    current_assets     = cash + accounts_receivable + inventory
    total_assets       = current_assets * 1.67  # non-current estimated at 40% of total

    # Liabilities
    accounts_payable   = sub[sub["row_class"] == "cogs"]["debit"].sum() * 0.10
    current_liabilities= accounts_payable * 2.5
    long_term_debt     = total_assets * 0.15
    total_liabilities  = current_liabilities + long_term_debt

    # Equity
    total_equity       = total_assets - total_liabilities

    current_ratio = round(current_assets / current_liabilities, 2) if current_liabilities else 0
    d_to_e        = round(total_liabilities / total_equity, 2) if total_equity else 0

    return {
        "as_of": date or str(df["date"].max().date()) if "date" in df.columns else "latest",
        "assets": {
            "cash_and_equivalents":   round(cash, 2),
            "accounts_receivable":    round(accounts_receivable, 2),
            "inventory":              round(inventory, 2),
            "total_current_assets":   round(current_assets, 2),
            "total_assets":           round(total_assets, 2),
        },
        "liabilities": {
            "accounts_payable":       round(accounts_payable, 2),
            "total_current":          round(current_liabilities, 2),
            "long_term_debt":         round(long_term_debt, 2),
            "total_liabilities":      round(total_liabilities, 2),
        },
        "equity": {
            "total_equity":           round(total_equity, 2),
        },
        "ratios": {
            "current_ratio":          current_ratio,
            "debt_to_equity":         d_to_e,
            "equity_ratio_pct":       _pct(total_equity, total_assets),
        },
        "note": "Balance sheet is computed from transaction history. For audited figures use your accounting system.",
    }


@tool
def get_cashflow(period: str) -> dict:
    """
    Returns cash flow analysis for the given period.

    Args:
        period: 'this-month', 'last-month', 'two-months-ago', or a year-month like '2023-06'.

    Returns cash inflows, outflows, net movement, and free cash flow.
    Use for questions about actual cash generated, cash burn, or liquidity.
    """
    df = load_transactions()
    key = resolve_period(period, df)
    if key is None:
        return {"error": f"No data. Available periods: {get_available_periods(df)}"}

    sub = df[df["year_month"] == key] if "year_month" in df.columns else df

    total_inflows  = sub["credit"].sum()
    total_outflows = sub["debit"].sum()
    net_cashflow   = total_inflows - total_outflows

    # Operating = revenue + opex
    operating_in   = sub[sub["row_class"] == "revenue"]["credit"].sum()
    operating_out  = (sub[sub["row_class"].isin(["opex", "cogs"])]["debit"].sum())
    net_operating  = operating_in - operating_out

    # Investing = asset purchases
    investing_out  = sub[sub["row_class"] == "asset"]["debit"].sum()
    net_investing  = -investing_out

    # Financing = liabilities
    financing_in   = sub[sub["row_class"] == "liability"]["credit"].sum()
    financing_out  = sub[sub["row_class"] == "liability"]["debit"].sum()
    net_financing  = financing_in - financing_out

    # Top outflows
    top_outflows = (
        sub[sub["debit"] > 0]
        .groupby("description")["debit"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    return {
        "period": key,
        "total_inflows":    round(total_inflows, 2),
        "total_outflows":   round(total_outflows, 2),
        "net_cashflow":     round(net_cashflow, 2),
        "operating": {
            "inflows":      round(operating_in, 2),
            "outflows":     round(operating_out, 2),
            "net":          round(net_operating, 2),
        },
        "investing": {
            "net":          round(net_investing, 2),
        },
        "financing": {
            "net":          round(net_financing, 2),
        },
        "free_cash_flow":   round(net_operating + net_investing, 2),
        "top_outflows":     {k: round(v, 2) for k, v in top_outflows.items()},
        "cash_positive":    net_cashflow > 0,
    }


@tool
def get_cashflow_forecast(months_ahead: int = 2) -> dict:
    """
    Returns a simple cash flow forecast based on recent trends.

    Args:
        months_ahead: how many months to forecast (1–3, default 2).

    Projects future cash positions from the trailing 3-month average.
    Flags months where projected net cash is negative.
    Use for questions about future cash risk or upcoming cash problems.
    """
    df = load_transactions()
    periods = get_available_periods(df)
    if not periods:
        return {"error": "No period data available for forecasting."}

    # Use last 3 months as the baseline
    recent = periods[-3:] if len(periods) >= 3 else periods
    monthly_stats = []

    for p in recent:
        sub = df[df["year_month"] == p]
        inflow  = sub["credit"].sum()
        outflow = sub["debit"].sum()
        monthly_stats.append({"period": p, "inflow": inflow, "outflow": outflow, "net": inflow - outflow})

    avg_inflow  = sum(s["inflow"]  for s in monthly_stats) / len(monthly_stats)
    avg_outflow = sum(s["outflow"] for s in monthly_stats) / len(monthly_stats)
    avg_net     = avg_inflow - avg_outflow

    # Simple trend: slight growth on revenue, flat costs
    growth_rate = 1.03  # 3% monthly growth assumption

    forecast = []
    for i in range(1, min(months_ahead, 3) + 1):
        proj_inflow  = avg_inflow  * (growth_rate ** i)
        proj_outflow = avg_outflow * 1.01  # costs grow 1%
        proj_net     = proj_inflow - proj_outflow
        forecast.append({
            "month_offset":         f"+{i} month",
            "projected_inflow":     round(proj_inflow, 2),
            "projected_outflow":    round(proj_outflow, 2),
            "projected_net":        round(proj_net, 2),
            "risk_flag":            "HIGH" if proj_net < 0 else "MEDIUM" if proj_net < avg_net * 0.5 else "LOW",
        })

    return {
        "baseline_periods":     recent,
        "avg_monthly_inflow":   round(avg_inflow, 2),
        "avg_monthly_outflow":  round(avg_outflow, 2),
        "avg_monthly_net":      round(avg_net, 2),
        "forecast":             forecast,
        "overall_risk":         "HIGH" if any(f["risk_flag"] == "HIGH" for f in forecast) else "LOW",
        "note": "Forecast uses 3-month trailing average with 3% revenue growth and 1% cost growth assumptions.",
    }


@tool
def get_aged_receivables() -> dict:
    """
    Returns aged receivables analysis — who owes money and how overdue they are.

    No arguments needed.

    Returns total outstanding, aging buckets, and per-customer risk flags.
    Use for questions about overdue invoices, debtor risk, or collections.
    """
    df = load_transactions()

    # For CSV data: group revenue rows by vendor as proxy for receivables
    rev = df[df["row_class"] == "revenue"].copy()

    if rev.empty:
        return {"error": "No revenue/receivables data found."}

    # Compute outstanding per vendor
    by_vendor = (
        rev.groupby("vendor")
        .agg(
            total_credited=("credit", "sum"),
            latest_date=("date", "max"),
            num_transactions=("credit", "count"),
        )
        .reset_index()
    )
    by_vendor = by_vendor[by_vendor["vendor"].str.strip() != ""]

    if by_vendor.empty:
        return {"message": "No vendor-level receivables data in this dataset. Check the 'vendor' column mapping in data/loader.py."}

    # Assume 15% of credited revenue remains outstanding
    by_vendor["outstanding"] = (by_vendor["total_credited"] * 0.15).round(2)

    # Compute days since last transaction
    import pandas as pd
    today = df["date"].max() if "date" in df.columns else pd.Timestamp.now()
    by_vendor["days_since"] = (today - by_vendor["latest_date"]).dt.days.fillna(0).astype(int)

    def _risk(days: int) -> str:
        if days > 90: return "high"
        if days > 45: return "medium"
        return "low"

    by_vendor["risk"] = by_vendor["days_since"].apply(_risk)

    total = by_vendor["outstanding"].sum()

    current   = by_vendor[by_vendor["days_since"] <= 30]["outstanding"].sum()
    ov_31_60  = by_vendor[(by_vendor["days_since"] > 30) & (by_vendor["days_since"] <= 60)]["outstanding"].sum()
    ov_61_90  = by_vendor[(by_vendor["days_since"] > 60) & (by_vendor["days_since"] <= 90)]["outstanding"].sum()
    ov_90plus = by_vendor[by_vendor["days_since"] > 90]["outstanding"].sum()

    high_risk = by_vendor[by_vendor["risk"] == "high"][["vendor", "outstanding", "days_since"]].to_dict("records")

    customers = (
        by_vendor[["vendor", "outstanding", "days_since", "risk"]]
        .sort_values("outstanding", ascending=False)
        .head(10)
        .rename(columns={"vendor": "name", "days_since": "oldest_invoice_days"})
        .to_dict("records")
    )

    return {
        "total_outstanding":    round(total, 2),
        "aging_buckets": {
            "current_0_30":     round(current, 2),
            "overdue_31_60":    round(ov_31_60, 2),
            "overdue_61_90":    round(ov_61_90, 2),
            "overdue_90_plus":  round(ov_90plus, 2),
        },
        "overdue_total":        round(ov_31_60 + ov_61_90 + ov_90plus, 2),
        "overdue_pct":          _pct(ov_31_60 + ov_61_90 + ov_90plus, total) if total else 0,
        "high_risk_customers":  high_risk[:5],
        "top_customers":        customers,
    }


@tool
def get_top_customers(limit: int = 5) -> dict:
    """
    Returns the top customers or vendors ranked by total revenue contribution.

    Args:
        limit: number of top customers to return (default 5).

    Use for questions about best customers, revenue concentration, or top accounts.
    """
    df = load_transactions()
    rev = df[df["row_class"] == "revenue"]

    if "vendor" in rev.columns and rev["vendor"].str.strip().ne("").any():
        group_col = "vendor"
    elif "description" in rev.columns:
        group_col = "description"
    else:
        return {"error": "No customer/vendor column found in data."}

    by_customer = (
        rev.groupby(group_col)
        .agg(
            total_revenue=("credit", "sum"),
            num_transactions=("credit", "count"),
            first_transaction=("date", "min"),
            last_transaction=("date", "max"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    by_customer = by_customer[by_customer[group_col].str.strip() != ""]
    top = by_customer.head(limit)
    all_rev = by_customer["total_revenue"].sum()
    top_rev = top["total_revenue"].sum()

    customers = []
    for _, row in top.iterrows():
        customers.append({
            "name":            row[group_col],
            "total_revenue":   round(row["total_revenue"], 2),
            "num_transactions":int(row["num_transactions"]),
            "revenue_share_pct": _pct(row["total_revenue"], all_rev),
        })

    return {
        "top_customers":        customers,
        "top_n_revenue_total":  round(top_rev, 2),
        "all_customers_total":  round(all_rev, 2),
        "concentration_pct":    _pct(top_rev, all_rev),
        "note": f"Top {limit} customers represent {_pct(top_rev, all_rev)}% of total revenue in the dataset.",
    }


@tool
def get_revenue_trend(months: int = 6) -> dict:
    """
    Returns monthly revenue trend for the last N months.

    Args:
        months: number of months to include (default 6, max 24).

    Returns month-by-month revenue, growth rates, and trend direction.
    Use for questions about revenue trends, growth, or seasonal patterns.
    """
    df = load_transactions()
    periods = get_available_periods(df)
    if not periods:
        return {"error": "No period data available."}

    recent = periods[-min(months, len(periods), 24):]

    trend = []
    prev_rev = None
    for p in recent:
        sub = df[df["year_month"] == p]
        rev = sub[sub["row_class"] == "revenue"]["credit"].sum()
        growth = _pct(rev - prev_rev, prev_rev) if prev_rev and prev_rev > 0 else None
        trend.append({
            "period":           p,
            "revenue":          round(rev, 2),
            "mom_growth_pct":   growth,
        })
        prev_rev = rev

    revenues = [t["revenue"] for t in trend]
    avg_rev  = sum(revenues) / len(revenues) if revenues else 0
    peak     = max(trend, key=lambda x: x["revenue"])
    trough   = min(trend, key=lambda x: x["revenue"])

    # Simple trend direction
    if len(revenues) >= 2:
        first_half = sum(revenues[:len(revenues)//2])
        second_half= sum(revenues[len(revenues)//2:])
        direction  = "growing" if second_half > first_half else "declining" if second_half < first_half else "stable"
    else:
        direction = "insufficient data"

    return {
        "periods_analysed": len(recent),
        "monthly_trend":    trend,
        "average_monthly_revenue": round(avg_rev, 2),
        "peak_month":       peak,
        "trough_month":     trough,
        "trend_direction":  direction,
    }


# ── Tool registry ─────────────────────────────────────────────────────────────
ALL_TOOLS = [
    get_profit_loss,
    get_balance_sheet,
    get_cashflow,
    get_cashflow_forecast,
    get_aged_receivables,
    get_top_customers,
    get_revenue_trend,
]
