from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# ── Column name mapping ───────────────────────────────────────────────────────
COLUMN_MAP = {
    "date":        ["date", "Date", "transaction_date", "TransactionDate", "DATE", "pay_date"],
    "account":     ["account", "Account", "account_name", "AccountName", "account_type"],
    "category":    ["category", "Category", "transaction_type"],
    "type":        ["type", "Type"],
    "subcategory": ["subcategory", "Subcategory", "sub_category", "SubCategory"],
    "description": ["description", "Description", "memo", "Memo", "notes", "Notes"],
    "debit":       ["debit", "Debit", "amount_debit", "amount_out"],
    "credit":      ["credit", "Credit", "amount_credit", "amount_in"],
    "balance":     ["balance", "Balance", "running_balance"],
    "vendor":      ["vendor", "Vendor", "supplier", "Supplier", "payee", "Payee", "customer", "Customer"],
    "employee":    ["employee", "Employee", "employee_name", "staff", "Staff"],
}

# ── Classification keywords ───────────────────────────────────────────────────
REVENUE_KEYWORDS   = ["sales", "revenue", "income", "receipt", "payment received",
                      "service fee", "consulting", "product sale", "sales revenue",
                      "daily sales deposit"]
COGS_KEYWORDS      = ["cogs", "cost of goods", "raw material", "direct labour",
                      "direct labor", "inventory purchase", "stock purchase", "supplies"]
OPEX_KEYWORDS      = ["salary", "wage", "payroll", "rent", "utility", "utilities",
                      "marketing", "advertising", "insurance", "depreciation",
                      "software", "subscription", "travel", "professional fee",
                      "accounting", "legal", "office", "operating expense",
                      "contractor pay", "employee pay", "maintenance"]
ASSET_KEYWORDS     = ["equipment", "vehicle", "property", "furniture", "computer"]
LIABILITY_KEYWORDS = ["loan", "debt", "mortgage", "credit line", "payable"]


def _clean_currency(series: pd.Series) -> pd.Series:
    """Strip $ and commas then coerce to float. Handles '$1,059.00' -> 1059.0"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    ).fillna(0.0)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and coerce types."""
    # Rename columns
    rename = {}
    for std, candidates in COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns and std not in rename.values():
                rename[c] = std
                break
    df = df.rename(columns=rename)

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        df["year"] = df["date"].dt.year

    # Handle single amount + type column (your Kaggle CSVs)
    # e.g. amount="$802.00", type="Credit" or "Debit"
    if "amount" in df.columns and "type" in df.columns:
        amt = _clean_currency(df["amount"]).abs()
        is_credit = df["type"].astype(str).str.strip().str.lower() == "credit"
        df["credit"] = amt.where(is_credit, 0.0)
        df["debit"]  = amt.where(~is_credit, 0.0)
    else:
        # Separate debit/credit columns
        for col in ["debit", "credit"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = _clean_currency(df[col])

    # Balance column
    if "balance" in df.columns:
        df["balance"] = _clean_currency(df["balance"])
    else:
        df["balance"] = 0.0

    # Net amount
    df["amount"] = df["credit"] - df["debit"]

    # String columns
    for col in ["account", "category", "subcategory", "description", "vendor", "employee"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Classify each row
    df["row_class"] = df.apply(_classify_row, axis=1)

    return df


def _classify_row(row) -> str:
    """Classify a transaction into a financial category."""
    text = " ".join([
        str(row.get("account", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
        str(row.get("description", "")),
    ]).lower()

    if any(k in text for k in REVENUE_KEYWORDS):   return "revenue"
    if any(k in text for k in COGS_KEYWORDS):       return "cogs"
    if any(k in text for k in OPEX_KEYWORDS):       return "opex"
    if any(k in text for k in ASSET_KEYWORDS):      return "asset"
    if any(k in text for k in LIABILITY_KEYWORDS):  return "liability"

    # Fallback on credit/debit values
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

    return _coerce(pd.concat(frames, ignore_index=True))


# ── Mock JSON loader ──────────────────────────────────────────────────────────
def _load_mock_json() -> pd.DataFrame:
    mock_path = ROOT / "mock.json"
    with open(mock_path) as f:
        data = json.load(f)

    rows = []
    for month, pl in data.get("profit_loss", {}).items():
        base = {"year_month": month, "date": pd.Timestamp(month + "-01")}
        for k, v in pl.get("revenue", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "revenue", "description": k,
                         "credit": v, "debit": 0, "amount": v, "row_class": "revenue",
                         "vendor": "", "employee": "", "account": "Sales"})
        for k, v in pl.get("cost_of_goods_sold", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "cogs", "description": k,
                         "debit": v, "credit": 0, "amount": -v, "row_class": "cogs",
                         "vendor": "", "employee": "", "account": "COGS"})
        for k, v in pl.get("operating_expenses", {}).items():
            if k == "total": continue
            rows.append({**base, "category": "opex", "description": k,
                         "debit": v, "credit": 0, "amount": -v, "row_class": "opex",
                         "vendor": "", "employee": "", "account": "Operating Expenses"})
    for cust in data.get("aged_receivables", {}).get("customers", []):
        rows.append({
            "date": pd.Timestamp(data["aged_receivables"]["as_of"]),
            "year_month": data["aged_receivables"]["as_of"][:7],
            "category": "receivable", "description": "Outstanding invoice",
            "credit": 0, "debit": 0, "amount": cust["outstanding"],
            "row_class": "revenue", "vendor": cust["name"],
            "employee": "", "account": "Accounts Receivable",
        })

    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    print("  ✓ Loaded mock.json fallback data")
    return df


# ── Public API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    """Load CSV if available, else fall back to mock JSON. Cached after first load."""
    csv_df = _load_csv()
    if csv_df is not None and len(csv_df) > 0:
        print(f"\n📊 Data source: Kaggle CSV  ({len(csv_df):,} rows total)\n")
        return csv_df
    print("\n📊 Data source: mock JSON (no CSV found in data/kaggle/)\n")
    return _load_mock_json()


def get_available_periods(df: pd.DataFrame) -> list[str]:
    if "year_month" not in df.columns:
        return []
    return sorted(df["year_month"].dropna().unique().tolist())


def resolve_period(period: str, df: pd.DataFrame) -> str | None:
    periods = get_available_periods(df)
    if not periods:
        return None
    latest  = periods[-1]
    prev    = periods[-2] if len(periods) > 1 else periods[-1]
    two_ago = periods[-3] if len(periods) > 2 else periods[-1]
    aliases = {
        "this-month": latest,   "this month": latest,
        "last-month": prev,     "last month": prev,
        "previous-month": prev, "previous month": prev,
        "two-months-ago": two_ago,
        "latest": latest,
    }
    key = period.lower().strip()
    if key in aliases:
        return aliases[key]
    if key in periods:
        return key
    matches = [p for p in periods if p.startswith(key)]
    return matches[-1] if matches else latest
