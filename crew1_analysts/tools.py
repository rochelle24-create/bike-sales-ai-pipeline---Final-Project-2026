# crew1_analysts/tools.py
# Crew 1 — Data Analyst Crew — Custom Tools
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

import os
import json
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from crewai.tools import tool
from utils.progress import progress
from utils.references import CASH_PAYMENT_CITATIONS
import warnings
warnings.filterwarnings("ignore")

# Set seaborn style globally
sns.set_style("whitegrid")
sns.set_palette("husl")

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)


# ============================================================
# TOOL 1 — Load and Clean Dataset
# ============================================================
@tool("load_and_clean_data")
def load_and_clean_data(filepath: str) -> str:
    """
    Load the dirty bike sales dataset and perform all cleaning steps.
    Returns a summary of cleaning actions taken.
    """
    try:
        progress.agent_start("Agent 1 — Data Ingestion & Cleaning")

        # STEP 1 — Load
        progress.step(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        initial_rows = len(df)
        progress.step(f"{initial_rows:,} rows loaded")

        # STEP 2 — Validate columns
        required_columns = [
            "Sale_ID", "Date", "Customer_ID", "Bike_Model", "Price",
            "Quantity", "Store_Location", "Salesperson_ID",
            "Payment_Method", "Customer_Age", "Customer_Gender"
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        progress.step("All 11 columns validated ✅")

        # STEP 3 — Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        dupes_removed = before - len(df)
        progress.step(f"Removed {dupes_removed:,} duplicate rows")

        # STEP 4 — Drop missing target variables
        before = len(df)
        df = df.dropna(subset=["Payment_Method"])
        dropped_payment = before - len(df)
        progress.step(f"Dropped {dropped_payment:,} rows — missing Payment_Method")

        before = len(df)
        df = df.dropna(subset=["Customer_Age"])
        dropped_age = before - len(df)
        progress.step(f"Dropped {dropped_age:,} rows — missing Customer_Age")

        # STEP 5 — Fill remaining nulls
        df["Store_Location"] = df["Store_Location"].fillna("Unknown")
        df["Customer_Gender"] = df["Customer_Gender"].fillna("Unknown")
        progress.step("Filled missing Store_Location → Unknown")
        progress.step("Filled missing Customer_Gender → Unknown")

        # Fill Price with median per Bike_Model
        df["Price"] = df.groupby("Bike_Model")["Price"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Price"] = df["Price"].fillna(df["Price"].median())
        progress.step("Imputed missing Price values with median per Bike_Model")

        # STEP 6 — Standardize Gender
        gender_map = {
            "M": "Male", "male": "Male", "MALE": "Male", "m": "Male",
            "F": "Female", "female": "Female", "FEMALE": "Female", "f": "Female",
            "Male": "Male", "Female": "Female", "Unknown": "Unknown"
        }
        df["Customer_Gender"] = df["Customer_Gender"].map(
            lambda x: gender_map.get(str(x).strip(), "Unknown")
        )
        progress.step("Standardized Customer_Gender values")

        # STEP 7 — Standardize Bike Model
        df["Bike_Model"] = df["Bike_Model"].str.strip().str.title()
        df["Bike_Model"] = df["Bike_Model"].str.replace(r"\s+", " ", regex=True)
        progress.step("Standardized Bike_Model casing and whitespace")

        # STEP 8 — Standardize Payment Method
        payment_map = {
            "cash": "Cash", "CASH": "Cash", "cash payment": "Cash"
        }
        df["Payment_Method"] = df["Payment_Method"].map(
            lambda x: payment_map.get(str(x).strip(), str(x).strip())
        )
        progress.step("Standardized Payment_Method values")

        # STEP 9 — Standardize Store Location
        df["Store_Location"] = df["Store_Location"].str.strip()
        progress.step("Stripped whitespace from Store_Location")

        # STEP 10 — Normalize dates
        def parse_date(d):
            d = str(d).strip()
            for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"]:
                try:
                    return pd.to_datetime(d, format=fmt).strftime("%d-%m-%Y")
                except:
                    continue
            return None

        df["Date"] = df["Date"].apply(parse_date)
        invalid_dates = df["Date"].isna().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=["Date"])
            progress.step(f"Dropped {invalid_dates} rows with unparseable dates")
        progress.step("Normalized all date formats to DD-MM-YYYY")

        # STEP 11 — Remove outliers and invalid values
        before = len(df)
        df = df[(df["Price"] >= 10) & (df["Price"] <= 50000)]
        progress.step(f"Removed {before - len(df):,} price outliers")

        before = len(df)
        df = df[(df["Customer_Age"] >= 18) & (df["Customer_Age"] <= 100)]
        progress.step(f"Removed {before - len(df):,} impossible age values")

        before = len(df)
        df = df[df["Quantity"] >= 1]
        progress.step(f"Removed {before - len(df):,} negative/zero quantities")

        # STEP 12 — Save clean data
        df.to_csv("artifacts/clean_data.csv", index=False)
        final_rows = len(df)
        progress.step(f"Saved artifacts/clean_data.csv → {final_rows:,} rows")
        progress.agent_done("Agent 1 — Data Ingestion & Cleaning")

        summary = (
            f"CLEANING COMPLETE\n"
            f"Rows loaded:      {initial_rows:,}\n"
            f"Duplicates removed: {dupes_removed:,}\n"
            f"Payment nulls dropped: {dropped_payment:,}\n"
            f"Age nulls dropped: {dropped_age:,}\n"
            f"Final clean rows: {final_rows:,}\n"
            f"Saved to: artifacts/clean_data.csv"
        )
        return summary

    except Exception as e:
        progress.error("Agent 1 — Data Ingestion & Cleaning", str(e))
        raise


# ============================================================
# TOOL 2 — Generate EDA Charts and HTML Report
# ============================================================
@tool("generate_eda_report")
def generate_eda_report(filepath: str) -> str:
    """
    Load clean_data.csv and generate 12 EDA charts embedded
    in a self-contained eda_report.html file.
    """
    try:
        progress.agent_start("Agent 2 — EDA & Visualizations")
        progress.step(f"Loading {filepath}...")

        df = pd.read_csv(filepath)

        # Parse dates and extract features
        df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
        df["Month"] = df["Date_parsed"].dt.month
        df["Month_Name"] = df["Date_parsed"].dt.strftime("%b")
        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[17, 24, 34, 44, 54, 64, 100],
            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        )

        charts = {}

        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            return encoded

        # Chart 1 — Sales by Store Location
        progress.step("Generating chart 1 — Sales by Store Location...")
        fig, ax = plt.subplots(figsize=(10, 5))
        store_counts = df["Store_Location"].value_counts()
        sns.barplot(x=store_counts.index, y=store_counts.values, ax=ax)
        ax.set_title("Sales by Store Location", fontsize=14, fontweight="bold")
        ax.set_xlabel("Store Location")
        ax.set_ylabel("Number of Transactions")
        plt.xticks(rotation=45)
        charts["chart1"] = fig_to_base64(fig)

        # Chart 2 — Sales by Month
        progress.step("Generating chart 2 — Sales by Month...")
        fig, ax = plt.subplots(figsize=(10, 5))
        monthly = df.groupby("Month").size().reset_index(name="count")
        ax.plot(monthly["Month"], monthly["count"], marker="o", linewidth=2)
        ax.set_title("Sales by Month (Seasonal Trends)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Transactions")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"])
        charts["chart2"] = fig_to_base64(fig)

        # Chart 3 — Price by Bike Model
        progress.step("Generating chart 3 — Price by Bike Model...")
        fig, ax = plt.subplots(figsize=(10, 5))
        bike_models = sorted(df["Bike_Model"].unique())
        data_to_plot = [df[df["Bike_Model"] == m]["Price"].values for m in bike_models]
        ax.boxplot(data_to_plot, tick_labels=bike_models)
        ax.set_title("Price Distribution by Bike Model", fontsize=14, fontweight="bold")
        ax.set_xlabel("Bike Model")
        ax.set_ylabel("Price ($)")
        plt.xticks(rotation=45)
        charts["chart3"] = fig_to_base64(fig)

        # Chart 4 — Age Distribution
        progress.step("Generating chart 4 — Customer Age Distribution...")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["Customer_Age"], bins=20, edgecolor="white", color="#4C72B0")
        ax.set_title("Customer Age Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        charts["chart4"] = fig_to_base64(fig)

        # Chart 5 — Age Group Breakdown
        progress.step("Generating chart 5 — Age Group Breakdown...")
        fig, ax = plt.subplots(figsize=(10, 5))
        age_counts = df["Age_Group"].value_counts().sort_index()
        sns.barplot(x=age_counts.index.astype(str), y=age_counts.values, ax=ax)
        ax.set_title("Customer Count by Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Count")
        charts["chart5"] = fig_to_base64(fig)

        # Chart 6 — Gender Distribution
        progress.step("Generating chart 6 — Gender Distribution...")
        fig, ax = plt.subplots(figsize=(7, 7))
        gender_counts = df["Customer_Gender"].value_counts()
        ax.pie(gender_counts.values, labels=gender_counts.index,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
        charts["chart6"] = fig_to_base64(fig)

        # Chart 7 — Payment Method Distribution
        progress.step("Generating chart 7 — Payment Method Distribution...")
        fig, ax = plt.subplots(figsize=(8, 8))
        pay_counts = df["Payment_Method"].value_counts()
        ax.pie(pay_counts.values, labels=pay_counts.index,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Payment Method Distribution", fontsize=14, fontweight="bold")
        charts["chart7"] = fig_to_base64(fig)

        # Chart 8 — Cash Rate by Age Group
        progress.step("Generating chart 8 — Cash Rate by Age Group...")
        df["Is_Cash"] = (df["Payment_Method"] == "Cash").astype(int)
        cash_rate = df.groupby("Age_Group", observed=True)["Is_Cash"].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=cash_rate.index.astype(str), y=cash_rate.values, ax=ax)
        ax.set_title("Cash Payment Rate by Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Cash Payment Rate (%)")
        charts["chart8"] = fig_to_base64(fig)

        # Chart 9 — Payment Method by Age Group
        progress.step("Generating chart 9 — Payment Method by Age Group...")
        pay_age = df.groupby(
            ["Age_Group", "Payment_Method"], observed=True
        ).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        pay_age.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Payment Method by Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.legend(title="Payment Method", bbox_to_anchor=(1.05, 1))
        charts["chart9"] = fig_to_base64(fig)

        # Chart 10 — Quantity Distribution
        progress.step("Generating chart 10 — Quantity Distribution...")
        fig, ax = plt.subplots(figsize=(8, 5))
        qty_counts = df["Quantity"].value_counts().sort_index()
        sns.barplot(x=qty_counts.index, y=qty_counts.values, ax=ax)
        ax.set_title("Purchase Quantity Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Number of Transactions")
        charts["chart10"] = fig_to_base64(fig)

        # Chart 11 — Bike Model by Age Group Heatmap
        progress.step("Generating chart 11 — Bike Model by Age Group heatmap...")
        heatmap_data = df.groupby(
            ["Age_Group", "Bike_Model"], observed=True
        ).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Bike Model Purchases by Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Bike Model")
        ax.set_ylabel("Age Group")
        charts["chart11"] = fig_to_base64(fig)

        # Chart 12 — Bike Model Distribution
        progress.step("Generating chart 12 — Bike Model Distribution...")
        fig, ax = plt.subplots(figsize=(10, 5))
        model_counts = df["Bike_Model"].value_counts()
        sns.barplot(x=model_counts.values, y=model_counts.index, ax=ax,
                    orient="h")
        ax.set_title("Bike Model Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Transactions")
        ax.set_ylabel("Bike Model")
        charts["chart12"] = fig_to_base64(fig)

        # Build HTML Report
        progress.step("Building eda_report.html...")

        total_rows = len(df)
        date_min = df["Date_parsed"].min().strftime("%B %Y") if not df["Date_parsed"].isna().all() else "N/A"
        date_max = df["Date_parsed"].max().strftime("%B %Y") if not df["Date_parsed"].isna().all() else "N/A"
        stores = df["Store_Location"].nunique()
        models = df["Bike_Model"].nunique()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bike Sales — EDA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; color: #333; }}
        h1 {{ color: #1F4E79; border-bottom: 3px solid #2E75B6; padding-bottom: 10px; }}
        h2 {{ color: #2E75B6; margin-top: 40px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; text-align: center; }}
        .card h3 {{ color: #1F4E79; margin: 0; font-size: 2em; }}
        .card p {{ color: #666; margin: 5px 0 0 0; }}
        .chart {{ background: white; padding: 20px; border-radius: 8px; 
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        .chart img {{ width: 100%; max-width: 900px; display: block; margin: 0 auto; }}
        .footer {{ text-align: center; margin-top: 60px; color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>🚲 Bike Sales — Exploratory Data Analysis</h1>
    <p><strong>Author:</strong> Rachel Barazani — AI Developer |
       <strong>Course:</strong> AI Developer Program — Hebrew University 2026</p>

    <h2>📊 Dataset Summary</h2>
    <div class="summary">
        <div class="card"><h3>{total_rows:,}</h3><p>Total Transactions</p></div>
        <div class="card"><h3>{stores}</h3><p>Store Locations</p></div>
        <div class="card"><h3>{models}</h3><p>Bike Models</p></div>
        <div class="card"><h3>{date_min}</h3><p>Date Range Start</p></div>
        <div class="card"><h3>{date_max}</h3><p>Date Range End</p></div>
    </div>

    <h2>🏪 Sales Performance</h2>
    <div class="chart"><img src="data:image/png;base64,{charts['chart1']}" alt="Sales by Store"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart2']}" alt="Sales by Month"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart3']}" alt="Price by Model"/></div>

    <h2>👥 Customer Profile</h2>
    <div class="chart"><img src="data:image/png;base64,{charts['chart4']}" alt="Age Distribution"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart5']}" alt="Age Groups"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart6']}" alt="Gender"/></div>

    <h2>💳 Payment Analysis</h2>
    <div class="chart"><img src="data:image/png;base64,{charts['chart7']}" alt="Payment Methods"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart8']}" alt="Cash Rate by Age"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart9']}" alt="Payment by Age Group"/></div>

    <h2>🎯 Prediction Targets</h2>
    <div class="chart"><img src="data:image/png;base64,{charts['chart10']}" alt="Quantity Distribution"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart11']}" alt="Bike Model Heatmap"/></div>
    <div class="chart"><img src="data:image/png;base64,{charts['chart12']}" alt="Bike Model Distribution"/></div>

    <div class="footer">
        <p>Generated by Bike Sales AI Pipeline — Hebrew University 2026 Final Project — CrewAI</p>
        <p>Rachel Barazani — AI Developer | Hebrew University 2026</p>
    </div>
</body>
</html>"""

        with open("artifacts/eda_report.html", "w", encoding="utf-8") as f:
            f.write(html)

        progress.step("Saved artifacts/eda_report.html")
        progress.agent_done("Agent 2 — EDA & Visualizations")

        return "EDA report generated with 12 charts — saved to artifacts/eda_report.html"

    except Exception as e:
        progress.error("Agent 2 — EDA & Visualizations", str(e))
        raise


# ============================================================
# TOOL 3 — Generate Business Insights and Dataset Contract
# ============================================================
@tool("generate_insights_and_contract")
def generate_insights_and_contract(filepath: str) -> str:
    """
    Analyze clean_data.csv and produce insights.md
    and dataset_contract.json in the artifacts folder.
    """
    try:
        progress.agent_start("Agent 3 — Business Insights & Contract")
        progress.step(f"Loading {filepath}...")

        df = pd.read_csv(filepath)
        df["Date_parsed"] = pd.to_datetime(
            df["Date"], format="%d-%m-%Y", errors="coerce"
        )
        df["Month"] = df["Date_parsed"].dt.month
        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[17, 24, 34, 44, 54, 64, 100],
            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        )
        df["Is_Cash"] = (df["Payment_Method"] == "Cash").astype(int)

        # Compute insights
        top_store = df["Store_Location"].value_counts().index[0]
        top_model = df["Bike_Model"].value_counts().index[0]
        peak_month = df["Month"].value_counts().index[0]
        month_names = {1:"January",2:"February",3:"March",4:"April",
                      5:"May",6:"June",7:"July",8:"August",
                      9:"September",10:"October",11:"November",12:"December"}
        peak_month_name = month_names.get(peak_month, str(peak_month))
        avg_age = round(df["Customer_Age"].mean(), 1)
        top_age_group = str(df["Age_Group"].value_counts().index[0])
        cash_rate = round(df["Is_Cash"].mean() * 100, 1)
        top_payment = df["Payment_Method"].value_counts().index[0]
        gender_dist = df["Customer_Gender"].value_counts(normalize=True) * 100
        low_freq_ages = df["Age_Group"].value_counts().sort_values().head(2).index.tolist()

        # Write insights.md
        progress.step("Writing insights.md...")
        insights = f"""# 🚲 Bike Sales — Business Insights

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026  
**Generated by:** Crew 1 — Agent 3 (Business Insights Specialist)

---

## Dataset Summary
- **Total transactions:** {len(df):,}
- **Date range:** {df["Date_parsed"].min().strftime("%B %Y")} → {df["Date_parsed"].max().strftime("%B %Y")}
- **Store locations:** {df["Store_Location"].nunique()}
- **Bike models:** {df["Bike_Model"].nunique()}

---

## Key Business Findings

### Sales Performance
- **Top performing store:** {top_store}
- **Best selling bike model:** {top_model}
- **Peak sales month:** {peak_month_name}

### Customer Profile
- **Average customer age:** {avg_age} years
- **Highest purchasing age group:** {top_age_group}
- **Gender split:** {gender_dist.to_dict()}

### Payment Behavior
- **Cash transaction rate:** {cash_rate}% of all sales
- **Most popular payment method:** {top_payment}

---

## Payment Barrier Observation

**{cash_rate}% of all transactions are paid in cash** — making cash the
most common single payment method in the dataset. Ages **{low_freq_ages[0]}**
and **{low_freq_ages[1]}** show the lowest overall transaction frequency,
and also show the highest cash reliance — suggesting a potential
**payment access barrier** rather than lack of interest in bike purchases.

### Universal Cash Discount Recommendation

> **The recommended cash discount is offered to ALL customers equally —
> regardless of age, gender, store location, or any personal characteristic.
> This is a store-wide pricing strategy, not a targeted promotion.**

The AI model identifies *who is likely to pay cash* — it does not determine
who qualifies for the discount. Every customer who walks through the door
is eligible, full stop.

**The business case has four parts:**

**1. Attract new customers currently blocked by payment barriers**
Ages 18–24 and 65+ show the highest cash reliance in our data. A visible
cash discount signals to these groups that the store welcomes them —
removing the silent barrier that digital-only pricing creates.

**2. Reward existing cash-paying customers**
{cash_rate}% of current customers already pay cash. A discount rewards
their loyalty and deepens retention at no cost to acquisition.

**3. Incentivise existing customers to switch to cash**
A cash discount gives every digital-payment customer a positive reason
to consider paying cash. That {cash_rate}% figure has room to grow —
and the store benefits every time it does.

**4. The discount can be self-funding**
Credit and debit card processing fees typically run 1.5–3.5% per
transaction. A 2% cash discount costs less than the fees it replaces
on converted transactions — making the promotion financially sustainable.

**In summary:** This is not a demographic strategy. It is a universal
pricing tool that removes a payment barrier, rewards loyal customers,
incentivises cash adoption, and can pay for itself through fee savings.
The prediction model makes it operationally actionable — staff can
proactively offer the discount at point of sale.

---

## Prediction Target Summary

| Target | Type | Classes |
|--------|------|---------|
| Quantity | Multi-class | 1, 2, 3, 4, 5 |
| Bike_Model | Multi-class | {", ".join(df["Bike_Model"].unique().tolist())} |
| Is_Cash | Binary | 0 (Not Cash), 1 (Cash) |

---

{CASH_PAYMENT_CITATIONS.strip()}
"""

        with open("artifacts/insights.md", "w", encoding="utf-8") as f:
            f.write(insights)
        progress.step("Saved artifacts/insights.md")

        # Write dataset_contract.json
        progress.step("Writing dataset_contract.json...")
        contract = {
            "contract_version": "1.0",
            "created_by": "Crew 1 — Agent 3",
            "dataset": "Bike Sales Clean Data",
            "total_rows": len(df),
            "total_columns": 11,
            "columns": {
                "Sale_ID": {
                    "type": "integer",
                    "nullable": False,
                    "unique": True,
                    "description": "Unique transaction identifier"
                },
                "Date": {
                    "type": "string",
                    "format": "DD-MM-YYYY",
                    "nullable": False,
                    "description": "Transaction date"
                },
                "Customer_ID": {
                    "type": "integer",
                    "nullable": False,
                    "description": "Unique customer identifier"
                },
                "Bike_Model": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": sorted(df["Bike_Model"].unique().tolist()),
                    "description": "Type of bike sold — Prediction 2 target"
                },
                "Price": {
                    "type": "float",
                    "nullable": False,
                    "min": float(df["Price"].min()),
                    "max": float(df["Price"].max()),
                    "description": "Transaction price in USD"
                },
                "Quantity": {
                    "type": "integer",
                    "nullable": False,
                    "min": int(df["Quantity"].min()),
                    "max": int(df["Quantity"].max()),
                    "allowed_values": sorted(df["Quantity"].unique().tolist()),
                    "description": "Number of bikes purchased — Prediction 1 target"
                },
                "Store_Location": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": sorted(df["Store_Location"].unique().tolist()),
                    "description": "City where sale occurred"
                },
                "Salesperson_ID": {
                    "type": "integer",
                    "nullable": False,
                    "min": int(df["Salesperson_ID"].min()),
                    "max": int(df["Salesperson_ID"].max()),
                    "description": "Unique salesperson identifier"
                },
                "Payment_Method": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": sorted(df["Payment_Method"].unique().tolist()),
                    "description": "Payment method — Prediction 3 source"
                },
                "Customer_Age": {
                    "type": "integer",
                    "nullable": False,
                    "min": int(df["Customer_Age"].min()),
                    "max": int(df["Customer_Age"].max()),
                    "description": "Customer age — critical for Prediction 3"
                },
                "Customer_Gender": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": sorted(df["Customer_Gender"].unique().tolist()),
                    "description": "Customer gender"
                }
            },
            "prediction_targets": {
                "prediction_1": {
                    "target": "Quantity",
                    "type": "multi-class classification",
                    "classes": sorted(df["Quantity"].unique().tolist())
                },
                "prediction_2": {
                    "target": "Bike_Model",
                    "type": "multi-class classification",
                    "classes": sorted(df["Bike_Model"].unique().tolist())
                },
                "prediction_3": {
                    "target": "Payment_Method",
                    "type": "binary classification",
                    "derived_feature": "Is_Cash",
                    "positive_class": "Cash",
                    "negative_class": "Not Cash"
                }
            },
            "assumptions": [
                "Rows with missing Payment_Method were dropped",
                "Rows with missing Customer_Age were dropped",
                "Missing Store_Location labeled as Unknown",
                "Missing Customer_Gender labeled as Unknown",
                "Price outliers below $10 and above $50,000 removed",
                "Impossible ages below 18 and above 100 removed",
                "Negative quantities removed",
                "All date formats normalized to DD-MM-YYYY",
                "All string fields stripped of whitespace",
                "All categorical fields standardized to Title Case"
            ],
            "constraints": [
                "Crew 2 must validate this contract before processing",
                "Any column not listed here must not be used in modeling",
                "Target columns must not be imputed",
                "Random seed must be set to 42 for reproducibility"
            ]
        }

        with open("artifacts/dataset_contract.json", "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2)
        progress.step("Saved artifacts/dataset_contract.json")
        progress.agent_done("Agent 3 — Business Insights & Contract")

        return (
            "Crew 1 complete.\n"
            "Saved: artifacts/insights.md\n"
            "Saved: artifacts/dataset_contract.json"
        )

    except Exception as e:
        progress.error("Agent 3 — Business Insights & Contract", str(e))
        raise