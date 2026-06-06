"""
data/loader.py

Unified data loader for the Business Health Agent.

Priority order:
  1. Kaggle CSV files  →  data/kaggle/*.csv   (real dataset)
  2. Mock JSON         →  data/mock.json       (fallback for demo without Kaggle)

The loader normalises both sources into a standard DataFrame schema so every
financial tool works identically regardless of which source is active.

Expected Kaggle CSV columns (Small Business Financial Dataset 2022-2023):
  date, account, category, subcategory, description, debit, credit, balance, vendor, employee

If your CSV has different column names, edit COLUMN_MAP below — nothing else needs to change.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# ── Column name mapping ───────────────────────────────────────────────────────
# Maps your CSV column names → internal standard names.
# Edit the RIGHT side to match your actual CSV headers.
COLUMN_MAP = {
    "date":        ["date", "Date", "transaction_date", "TransactionDate", "DATE"],
    "account":     ["account", "Account", "account_name", "AccountName", "account_type"],
    "category":    ["category", "Category", "type", "Type", "transaction_type"],
    "subcategory": ["subcategory", "Subcategory", "sub_category", "SubCategory", "description_type"],
    "description": ["description", "Description", "memo", "Memo", "notes", "Notes", "details"],
    "debit":       ["debit", "Debit", "amount_debit", "expense", "Expense", "amount_out"],
    "credit":      ["credit", "Credit", "amount_credit", "income", "Income", "amount_in"],
    "balance":     ["balance", "Balance", "running_balance", "RunningBalance"],
    "vendor":      ["vendor", "Vendor", "supplier", "Supplier", "payee", "Payee", "customer", "Customer"],
    "employee":    ["employee", "Employee", "staff", "Staff", "employee_name"],
}

# ── Account / category classification ────────────────────────────────────────
REVENUE_KEYWORDS    = ["sales", "revenue", "income", "receipt", "payment received",
                       "service fee", "consulting", "product sale"]
COGS_KEYWORDS       = ["cogs", "cost of goods", "raw material", "direct labour",
                       "direct labor", "inventory purchase", "stock purchase"]
OPEX_KEYWORDS       = ["salary", "wage", "payroll", "rent", "utility", "utilities",
                       "marketing", "advertising", "insurance", "depreciation",
                       "software", "subscription", "travel", "professional fee",
                       "accounting", "legal", "office"]
ASSET_KEYWORDS      = ["equipment", "vehicle", "property", "furniture", "computer"]
LIABILITY_KEYWORDS  = ["loan", "debt", "mortgage", "credit line", "payable"]


def _find_col(df: pd.DataFrame, standard_name: str) -> str | None:
    """Return the actual column name in df that matches the standard name."""
    candidates = COLUMN_MAP.get(standard_name, [standard_name])
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to internal standard names and coerce types."""
    rename = {}
    for std, candidates in COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns and std not in rename.values():
                rename[c] = std
                break

    df = df.rename(columns=rename)

    # Ensure required columns exist
    for col in ["debit", "credit", "balance"]:
        if col not in df.columns:
            df[col] = 0.0

    df["debit"]  = pd.to_numeric(df.get("debit",  0), errors="coerce").fillna(0.0)
    df["credit"] = pd.to_numeric(df.get("credit", 0), errors="coerce").fillna(0.0)
    df["balance"]= pd.to_numeric(df.get("balance",0), errors="coerce").fillna(0.0)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        df["year"]       = df["date"].dt.year

    for col in ["account", "category", "subcategory", "description", "vendor", "employee"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Derive amount = credit - debit (positive = money in, negative = money out)
    df["amount"] = df["credit"] - df["debit"]

    # Classify rows
    df["row_class"] = df.apply(_classify_row, axis=1)

    return df


def _classify_row(row) -> str:
    """Classify a transaction row into a financial category."""
    text = " ".join([
        str(row.get("account", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
        str(row.get("description", "")),
    ]).lower()

    if any(k in text for k in REVENUE_KEYWORDS):    return "revenue"
    if any(k in text for k in COGS_KEYWORDS):        return "cogs"
    if any(k in text for k in OPEX_KEYWORDS):        return "opex"
    if any(k in text for k in ASSET_KEYWORDS):       return "asset"
    if any(k in text for k in LIABILITY_KEYWORDS):   return "liability"

    # Fallback: if credit > 0 and debit == 0 → revenue; else expense
    if row.get("credit", 0) > 0 and row.get("debit", 0) == 0:
        return "revenue"
    if row.get("debit", 0) > 0 and row.get("credit", 0) == 0:
        return "expense"
    return "other"


# ── CSV loader ────────────────────────────────────────────────────────────────
def _load_csv() -> pd.DataFrame | None:
    kaggle_dir = ROOT / "kaggle"
    csv_files  = sorted(kaggle_dir.glob("*.csv"))
    if not csv_files:
        return None

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["_source_file"] = f.name
            frames.append(df)
            print(f"  ✓ Loaded {f.name}  ({len(df):,} rows, columns: {list(df.columns)})")
        except Exception as e:
            print(f"  ⚠ Could not read {f.name}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    return _coerce(combined)


# ── Mock JSON loader ──────────────────────────────────────────────────────────
def _load_mock_json() -> pd.DataFrame:
    """
    Convert mock.json into the same normalised DataFrame so tools work identically.
    """
    mock_path = ROOT / "mock.json"
    with open(mock_path) as f:
        data = json.load(f)

    rows = []
    for month, pl in data.get("profit_loss", {}).items():
        base = {"year_month": month, "date": pd.Timestamp(month + "-01")}

        # Revenue rows
        for k, v in pl.get("revenue", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "revenue", "description": k,
                         "credit": v, "debit": 0, "amount": v, "row_class": "revenue",
                         "vendor": "", "employee": "", "account": "Sales"})

        # COGS rows
        for k, v in pl.get("cost_of_goods_sold", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "cogs", "description": k,
                         "debit": v, "credit": 0, "amount": -v, "row_class": "cogs",
                         "vendor": "", "employee": "", "account": "COGS"})

        # Opex rows
        for k, v in pl.get("operating_expenses", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "opex", "description": k,
                         "debit": v, "credit": 0, "amount": -v, "row_class": "opex",
                         "vendor": "", "employee": k if "salary" in k or "wage" in k else "",
                         "account": "Operating Expenses"})

    # Receivables
    for cust in data.get("aged_receivables", {}).get("customers", []):
        rows.append({
            "date": pd.Timestamp(data["aged_receivables"]["as_of"]),
            "year_month": data["aged_receivables"]["as_of"][:7],
            "category": "receivable", "description": "Outstanding invoice",
            "credit": 0, "debit": 0, "amount": cust["outstanding"],
            "row_class": "revenue", "vendor": cust["name"],
            "employee": "", "account": "Accounts Receivable",
            "_overdue_days": cust["oldest_invoice_days"],
            "_risk": cust["risk"],
        })

    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    print("  ✓ Loaded mock.json fallback data")
    return df


# ── Public API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    """
    Load transactions from CSV (Kaggle) if available, else fall back to mock JSON.
    Cached after first load.
    """
    csv_df = _load_csv()
    if csv_df is not None and len(csv_df) > 0:
        print(f"\n📊 Data source: Kaggle CSV  ({len(csv_df):,} rows total)\n")
        return csv_df

    print("\n📊 Data source: mock JSON (no CSV found in data/kaggle/)\n")
    return _load_mock_json()


def get_available_periods(df: pd.DataFrame) -> list[str]:
    """Return sorted list of year_month strings in the dataset."""
    if "year_month" not in df.columns:
        return []
    return sorted(df["year_month"].dropna().unique().tolist())


def resolve_period(period: str, df: pd.DataFrame) -> str | None:
    """Map human period strings to year_month keys available in df."""
    periods = get_available_periods(df)
    if not periods:
        return None

    latest  = periods[-1]
    prev    = periods[-2] if len(periods) > 1 else periods[-1]
    two_ago = periods[-3] if len(periods) > 2 else periods[-1]

    aliases = {
        "this-month": latest,  "this month": latest,
        "last-month": prev,    "last month": prev,
        "previous-month": prev,"previous month": prev,
        "two-months-ago": two_ago,
        "latest": latest,
    }
    key = period.lower().strip()
    if key in aliases:
        return aliases[key]
    # Direct match e.g. "2023-06"
    if key in periods:
        return key
    # Partial match e.g. "2023"
    matches = [p for p in periods if p.startswith(key)]
    return matches[-1] if matches else latest
