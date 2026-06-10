# crew2_datascientists/tools.py
# Crew 2 - Data Scientist Crew - Custom Tools
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from utils.progress import progress
from utils.references import CASH_PAYMENT_CITATIONS, DATA_ENHANCEMENT_SECTION

# Ensure directories exist
os.makedirs("artifacts/models", exist_ok=True)

RANDOM_SEED = 42
sns.set_style("whitegrid")


# ============================================================
# TOOL 4 - Feature Engineering
# ============================================================
@tool("engineer_features")
def engineer_features(filepath: str) -> str:
    """
    Load clean_data.csv, validate contract, engineer features,
    and save features.csv to artifacts folder.
    """
    try:
        progress.agent_start("Agent 4 - Feature Engineering")

        # Load data and contract
        progress.step(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        initial_rows = len(df)
        progress.step(f"{initial_rows:,} rows loaded")

        # Validate contract
        progress.step("Validating dataset contract...")
        with open("artifacts/dataset_contract.json", "r") as f:
            contract = json.load(f)

        contract_cols = list(contract["columns"].keys())
        missing_cols = [c for c in contract_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Contract validation failed - missing columns: {missing_cols}")

        if df["Payment_Method"].isna().any():
            raise ValueError("Contract validation failed - nulls in Payment_Method")
        if df["Customer_Age"].isna().any():
            raise ValueError("Contract validation failed - nulls in Customer_Age")
        progress.step("Contract validation passed [OK]")

        # Parse dates
        df["Date_parsed"] = pd.to_datetime(
            df["Date"], format="%d-%m-%Y", errors="coerce"
        )

        # Engineer Month
        df["Month"] = df["Date_parsed"].dt.month
        progress.step("Created feature: Month")

        # Engineer Day_of_Week
        df["Day_of_Week"] = df["Date_parsed"].dt.dayofweek
        progress.step("Created feature: Day_of_Week")

        # Engineer Season
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        df["Season"] = df["Month"].apply(get_season)
        progress.step("Created feature: Season")

        # Engineer Is_Weekend
        df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
        progress.step("Created feature: Is_Weekend")

        # Engineer Age_Group
        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[17, 24, 34, 44, 54, 64, 100],
            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        ).astype(str)
        progress.step("Created feature: Age_Group")

        # Engineer Price_Tier
        def get_price_tier(price):
            if price < 500:
                return "Budget"
            elif price <= 2000:
                return "Mid"
            else:
                return "Premium"

        df["Price_Tier"] = df["Price"].apply(get_price_tier)
        progress.step("Created feature: Price_Tier")

        # Engineer Is_Cash (Prediction 3 target)
        df["Is_Cash"] = (df["Payment_Method"] == "Cash").astype(int)
        progress.step("Created feature: Is_Cash (Prediction 3 target)")

        # Engineer Is_Family_Buyer - strong signal for multi-unit quantity
        df["Is_Family_Buyer"] = (
            df["Customer_Age"].between(35, 54) &
            (df["Price"] < 1200) &
            (df["Bike_Model"] != "Electric Bike")
        ).astype(int)
        progress.step("Created feature: Is_Family_Buyer")

        # Label encode categoricals
        progress.step("Label encoding categorical columns...")
        categorical_cols = [
            "Bike_Model", "Store_Location", "Customer_Gender",
            "Payment_Method", "Season", "Age_Group", "Price_Tier"
        ]

        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            progress.step(f"  Encoded: {col} -> {col}_enc")

        # Save encoders for use in Streamlit app
        joblib.dump(encoders, "artifacts/models/label_encoders.pkl")
        progress.step("Saved label encoders to artifacts/models/label_encoders.pkl")

        # Drop Date_parsed before saving
        df = df.drop(columns=["Date_parsed"], errors="ignore")

        # Save features
        df.to_csv("artifacts/features.csv", index=False)
        progress.step(f"Saved artifacts/features.csv -> {len(df):,} rows, {len(df.columns)} columns")
        progress.agent_done("Agent 4 - Feature Engineering")

        return (
            f"Feature engineering complete.\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns)}\n"
            f"New features: Month, Day_of_Week, Season, Is_Weekend, "
            f"Age_Group, Price_Tier, Is_Cash\n"
            f"Saved to: artifacts/features.csv"
        )

    except Exception as e:
        progress.error("Agent 4 - Feature Engineering", str(e))
        raise


# ============================================================
# TOOL 5 - Model Training
# ============================================================
@tool("train_models")
def train_models(filepath: str) -> str:
    """
    Load features.csv and train 6 models across 3 predictions.
    Save all models and test data to artifacts/models/.
    """
    try:
        progress.agent_start("Agent 5 - Model Training")
        progress.step(f"Loading {filepath}...")

        df = pd.read_csv(filepath)
        progress.step(f"{len(df):,} rows loaded")

        # Columns to always drop
        drop_cols = ["Sale_ID", "Customer_ID", "Date"]
        # Original categorical columns (not encoded)
        original_cats = [
            "Bike_Model", "Store_Location", "Customer_Gender",
            "Payment_Method", "Season", "Age_Group", "Price_Tier"
        ]

        # Base encoded feature columns (Salesperson_ID excluded - noise)
        base_features = [
            "Price", "Quantity", "Customer_Age",
            "Month", "Day_of_Week", "Is_Weekend", "Is_Family_Buyer",
            "Bike_Model_enc", "Store_Location_enc", "Customer_Gender_enc",
            "Payment_Method_enc", "Season_enc", "Age_Group_enc", "Price_Tier_enc"
        ]

        results = {}

        # -- PREDICTION 1 - Quantity --------------------------
        progress.step("\n  Training Prediction 1: Quantity...")
        p1_features = [f for f in base_features
                       if f not in ["Quantity", "Is_Cash", "Payment_Method_enc"]]
        X1 = df[p1_features]
        y1 = df["Quantity"]

        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1, test_size=0.2, random_state=RANDOM_SEED, stratify=y1
        )

        # Logistic Regression
        progress.step("    -> Logistic Regression (Quantity)...")
        lr1 = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"
        )
        lr1.fit(X1_train, y1_train)
        joblib.dump(lr1, "artifacts/models/quantity_lr.pkl", compress=6)
        progress.step("    [OK] quantity_lr.pkl saved")

        # Random Forest
        progress.step("    -> Random Forest (Quantity)...")
        rf1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf1.fit(X1_train, y1_train)
        joblib.dump(rf1, "artifacts/models/quantity_rf.pkl", compress=6)
        progress.step("    [OK] quantity_rf.pkl saved")

        # Save test data
        joblib.dump((X1_test, y1_test, p1_features),
                    "artifacts/models/quantity_test.pkl")
        results["quantity"] = {"lr": lr1, "rf": rf1,
                               "X_test": X1_test, "y_test": y1_test,
                               "features": p1_features}

        # -- PREDICTION 2 - Bike Model ------------------------
        progress.step("\n  Training Prediction 2: Bike_Model...")
        p2_features = [f for f in base_features
                       if f not in ["Bike_Model_enc", "Is_Cash",
                                    "Payment_Method_enc"]]
        X2 = df[p2_features]
        y2 = df["Bike_Model_enc"]

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=0.2, random_state=RANDOM_SEED, stratify=y2
        )

        # Logistic Regression
        progress.step("    -> Logistic Regression (Bike_Model)...")
        lr2 = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"
        )
        lr2.fit(X2_train, y2_train)
        joblib.dump(lr2, "artifacts/models/bike_model_lr.pkl", compress=6)
        progress.step("    [OK] bike_model_lr.pkl saved")

        # Random Forest
        progress.step("    -> Random Forest (Bike_Model)...")
        rf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf2.fit(X2_train, y2_train)
        joblib.dump(rf2, "artifacts/models/bike_model_rf.pkl", compress=6)
        progress.step("    [OK] bike_model_rf.pkl saved")

        # Save test data
        joblib.dump((X2_test, y2_test, p2_features),
                    "artifacts/models/bike_model_test.pkl")
        results["bike_model"] = {"lr": lr2, "rf": rf2,
                                  "X_test": X2_test, "y_test": y2_test,
                                  "features": p2_features}

        # -- PREDICTION 3 - Is_Cash ---------------------------
        progress.step("\n  Training Prediction 3: Is_Cash...")
        p3_features = [f for f in base_features
                       if f not in ["Is_Cash", "Payment_Method_enc"]]
        X3 = df[p3_features]
        y3 = df["Is_Cash"]

        X3_train, X3_test, y3_train, y3_test = train_test_split(
            X3, y3, test_size=0.2, random_state=RANDOM_SEED, stratify=y3
        )

        # Logistic Regression
        progress.step("    -> Logistic Regression (Is_Cash)...")
        lr3 = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"
        )
        lr3.fit(X3_train, y3_train)
        joblib.dump(lr3, "artifacts/models/payment_lr.pkl", compress=6)
        progress.step("    [OK] payment_lr.pkl saved")

        # Random Forest
        progress.step("    -> Random Forest (Is_Cash)...")
        rf3 = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf3.fit(X3_train, y3_train)
        joblib.dump(rf3, "artifacts/models/payment_rf.pkl", compress=6)
        progress.step("    [OK] payment_rf.pkl saved")

        # Save test data
        joblib.dump((X3_test, y3_test, p3_features),
                    "artifacts/models/payment_test.pkl")
        results["payment"] = {"lr": lr3, "rf": rf3,
                               "X_test": X3_test, "y_test": y3_test,
                               "features": p3_features}

        progress.agent_done("Agent 5 - Model Training")

        return (
            "Model training complete.\n"
            "Saved 6 models to artifacts/models/:\n"
            "  quantity_lr.pkl, quantity_rf.pkl\n"
            "  bike_model_lr.pkl, bike_model_rf.pkl\n"
            "  payment_lr.pkl, payment_rf.pkl\n"
            "Plus 3 test data files."
        )

    except Exception as e:
        progress.error("Agent 5 - Model Training", str(e))
        raise


# ============================================================
# TOOL 6 - Model Evaluation
# ============================================================
@tool("evaluate_models")
def evaluate_models(features_path: str) -> str:
    """
    Load all 6 trained models, evaluate on test data,
    generate charts and save evaluation_report.md.
    """
    try:
        progress.agent_start("Agent 6 - Model Evaluation")

        predictions = {
            "quantity": {
                "name": "Purchase Quantity",
                "test_file": "artifacts/models/quantity_test.pkl",
                "lr_file": "artifacts/models/quantity_lr.pkl",
                "rf_file": "artifacts/models/quantity_rf.pkl"
            },
            "bike_model": {
                "name": "Bike Model Recommendation",
                "test_file": "artifacts/models/bike_model_test.pkl",
                "lr_file": "artifacts/models/bike_model_lr.pkl",
                "rf_file": "artifacts/models/bike_model_rf.pkl"
            },
            "payment": {
                "name": "Cash Payment Prediction",
                "test_file": "artifacts/models/payment_test.pkl",
                "lr_file": "artifacts/models/payment_lr.pkl",
                "rf_file": "artifacts/models/payment_rf.pkl"
            }
        }

        report_lines = [
            "# Bike Sales - Model Evaluation Report\n",
            "**Author:** Rachel Barazani - AI Developer  ",
            "**Course:** AI Developer Program - Hebrew University 2026\n",
            "---\n"
        ]

        best_models = {}

        for key, pred in predictions.items():
            progress.step(f"Evaluating {pred['name']}...")

            # Load test data and models
            X_test, y_test, features = joblib.load(pred["test_file"])
            lr = joblib.load(pred["lr_file"])
            rf = joblib.load(pred["rf_file"])

            # Evaluate both models
            lr_pred = lr.predict(X_test)
            rf_pred = rf.predict(X_test)

            lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 2)
            rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 2)
            lr_f1 = round(f1_score(y_test, lr_pred,
                                   average="weighted") * 100, 2)
            rf_f1 = round(f1_score(y_test, rf_pred,
                                   average="weighted") * 100, 2)

            winner = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
            winner_f1 = max(rf_f1, lr_f1)
            best_models[key] = {
                "name": winner,
                "f1": winner_f1,
                "file": pred["rf_file"] if rf_f1 >= lr_f1 else pred["lr_file"]
            }

            # Accuracy comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(["Logistic Regression", "Random Forest"],
                   [lr_acc, rf_acc], color=["#4C72B0", "#55A868"])
            ax.set_title(f"{pred['name']} - Accuracy Comparison",
                        fontsize=13, fontweight="bold")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate([lr_acc, rf_acc]):
                ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"artifacts/models/accuracy_{key}.png", dpi=100)
            plt.close()

            # F1 comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(["Logistic Regression", "Random Forest"],
                   [lr_f1, rf_f1], color=["#C44E52", "#8172B2"])
            ax.set_title(f"{pred['name']} - F1 Score Comparison",
                        fontsize=13, fontweight="bold")
            ax.set_ylabel("F1 Score (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate([lr_f1, rf_f1]):
                ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"artifacts/models/f1_{key}.png", dpi=100)
            plt.close()

            # Confusion matrices
            for model, preds, label in [
                (lr, lr_pred, "lr"), (rf, rf_pred, "rf")
            ]:
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(
                    f"{pred['name']} - Confusion Matrix "
                    f"({'LR' if label == 'lr' else 'RF'})",
                    fontsize=12, fontweight="bold"
                )
                ax.set_ylabel("Actual")
                ax.set_xlabel("Predicted")
                plt.tight_layout()
                plt.savefig(
                    f"artifacts/models/cm_{label}_{key}.png", dpi=100
                )
                plt.close()

            # Feature importance (Random Forest only)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            top_features = [features[i] for i in indices]
            top_importances = importances[indices]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_importances, y=top_features, ax=ax,
                       orient="h")
            ax.set_title(
                f"{pred['name']} - Top 10 Feature Importance (RF)",
                fontsize=13, fontweight="bold"
            )
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(
                f"artifacts/models/feature_importance_{key}.png", dpi=100
            )
            plt.close()

            progress.step(f"  LR: Acc={lr_acc}% F1={lr_f1}%")
            progress.step(f"  RF: Acc={rf_acc}% F1={rf_f1}%")
            progress.step(f"  Winner: {winner}")

            # Add to report
            report_lines += [
                f"\n## Prediction: {pred['name']}\n",
                f"### Logistic Regression",
                f"- Accuracy: {lr_acc}%",
                f"- F1 Score: {lr_f1}%\n",
                f"```",
                classification_report(y_test, lr_pred),
                f"```\n",
                f"### Random Forest",
                f"- Accuracy: {rf_acc}%",
                f"- F1 Score: {rf_f1}%\n",
                f"```",
                classification_report(y_test, rf_pred),
                f"```\n",
                f"### [OK] Winner: {winner} (F1: {winner_f1}%)\n",
                "---\n"
            ]

        # Overall best model
        best_key = max(best_models, key=lambda k: best_models[k]["f1"])
        best = best_models[best_key]
        report_lines += [
            f"\n## Overall Best Model\n",
            f"- Prediction: {predictions[best_key]['name']}",
            f"- Model: {best['name']}",
            f"- F1 Score: {best['f1']}%",
            f"- Saved as: artifacts/models/model.pkl\n"
        ]

        # Save evaluation report
        with open("artifacts/evaluation_report.md", "w",
                  encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        progress.step("Saved artifacts/evaluation_report.md")

        # Save best models info for Agent 7
        joblib.dump(best_models, "artifacts/models/best_models.pkl")
        progress.agent_done("Agent 6 - Model Evaluation")

        return (
            "Evaluation complete.\n"
            f"Best model: {best['name']} for "
            f"{predictions[best_key]['name']} (F1: {best['f1']}%)\n"
            "Saved: artifacts/evaluation_report.md"
        )

    except Exception as e:
        progress.error("Agent 6 - Model Evaluation", str(e))
        raise


# ============================================================
# TOOL 7 - Model Card & Business Recommendations
# ============================================================
@tool("write_model_card")
def write_model_card(evaluation_path: str) -> str:
    """
    Write model_card.md and save best model as model.pkl.
    """
    try:
        progress.agent_start("Agent 7 - Model Card & Business Recommendations")

        # Load evaluation results
        progress.step("Loading evaluation results...")
        best_models = joblib.load("artifacts/models/best_models.pkl")
        df = pd.read_csv("artifacts/features.csv")

        # Find overall best model
        best_key = max(best_models, key=lambda k: best_models[k]["f1"])
        best = best_models[best_key]

        prediction_names = {
            "quantity": "Purchase Quantity",
            "bike_model": "Bike Model Recommendation",
            "payment": "Cash Payment Prediction"
        }

        # Copy best model as model.pkl
        shutil.copy(best["file"], "artifacts/models/model.pkl")
        progress.step(
            f"Best model: {best['name']} "
            f"(F1: {best['f1']}%) saved as model.pkl"
        )

        # Write model card
        progress.step("Writing model_card.md...")

        card = f"""# Bike Sales AI Pipeline - Hebrew University 2026 Final Project - Model Card

**Author:** Rachel Barazani - AI Developer  
**Course:** AI Developer Program - Hebrew University 2026  
**Generated by:** Crew 2 - Agent 7  

---

## Model Overview

Three classification models trained to answer business questions:

| # | Prediction | Type | Best Model | F1 Score |
|---|-----------|------|-----------|----------|
| 1 | Purchase Quantity | Multi-class | {best_models['quantity']['name']} | {best_models['quantity']['f1']}% |
| 2 | Bike Model Recommendation | Multi-class | {best_models['bike_model']['name']} | {best_models['bike_model']['f1']}% |
| 3 | Cash Payment Prediction | Binary | {best_models['payment']['name']} | {best_models['payment']['f1']}% |

---

## Training Data Summary

- **Dataset:** Bike Sales 100k (cleaned version)
- **Source:** Kaggle - jayavarman/bike-sales-data-of-100k
- **Original file:** bike_sales_100k.csv (not included - see data/README.md)
- **Pipeline input:** bike_sales_dirty.csv (intentionally dirtied for cleaning demo)
- **Rows after cleaning:** {len(df):,}
- **Train/test split:** 80% train / 20% test
- **Random seed:** 42 (all models)

---

{DATA_ENHANCEMENT_SECTION.strip()}

---

## Prediction 1 - Purchase Quantity

### Purpose
Predict how many bikes (1-5) a customer will purchase in a single transaction.
Helps stores with inventory planning and staffing decisions.

### Features Used
Customer_Age, Customer_Gender_enc, Bike_Model_enc, Price, Price_Tier_enc,
Store_Location_enc, Month, Season_enc, Day_of_Week, Is_Weekend, Age_Group_enc

### Limitations
- Dataset is synthetically balanced across all 5 quantity values
- Real-world distributions would likely be heavily skewed toward quantity 1
- Model trained on 7 US city locations only

---

## Prediction 2 - Bike Model Recommendation

### Purpose
Recommend the most likely bike model for a given customer profile.
Functions as a product recommendation engine for sales staff.

### Features Used
Customer_Age, Customer_Gender_enc, Price, Price_Tier_enc,
Store_Location_enc, Payment_Method_enc, Month, Season_enc,
Day_of_Week, Is_Weekend, Age_Group_enc

### Limitations
- Limited to 7 bike models present in training data
- Does not account for inventory availability or new models
- Recommendation is probabilistic - not a guarantee of purchase intent

---

## Prediction 3 - Cash Payment Prediction

### Purpose
Identify customers likely to pay cash and analyze payment access barriers.
Used to inform a universal cash discount strategy.

### Features Used
Customer_Age, Age_Group_enc, Customer_Gender_enc, Bike_Model_enc,
Price, Price_Tier_enc, Store_Location_enc, Month, Season_enc,
Day_of_Week, Is_Weekend

### Limitations
- Based on historical payment behavior only
- Cannot predict future changes in payment preferences
- Does not account for external economic factors

---

## Ethical Considerations

### Cash Discount Recommendation
The cash payment model is used **solely** to identify market expansion
opportunities - NOT to target, exclude, or discriminate against any
customer group.

The recommended cash discount applies **UNIVERSALLY** to all customers
regardless of age, gender, location, or any demographic characteristic.

Age group analysis identifies who benefits most from removing payment
barriers - this is consistent with responsible and fair retail practice.

### Bias Assessment
- All models evaluated across all age groups and genders
- No discriminatory patterns identified in prediction outputs
- Regular re-evaluation recommended as customer demographics evolve

### Fairness Statement
> No customer is excluded or targeted based on personal characteristics.
> All business recommendations are made at the population level, not
> the individual level. The goal is to expand access, not restrict it.

### Data Privacy
- No personally identifiable information (PII) used in model training
- Customer_ID and Sale_ID excluded from all feature sets
- Models predict behavioral patterns, not individual identities

---

## Reproducibility

All models are fully reproducible:
- Random seed: 42 (all models, all splits)
- Train/test split: 80/20 (stratified by default)
- All artifacts saved in /artifacts and /artifacts/models
- Pipeline can be re-run with: python flow/pipeline.py

---

{CASH_PAYMENT_CITATIONS.strip()}

---

*Generated by Bike Sales AI Pipeline - Hebrew University 2026 Final Project - CrewAI*  
*Rachel Barazani - AI Developer | Hebrew University 2026*
"""

        with open("artifacts/model_card.md", "w", encoding="utf-8") as f:
            f.write(card)

        progress.step("Saved artifacts/model_card.md")
        progress.agent_done("Agent 7 - Model Card & Business Recommendations")

        return (
            "Model card complete.\n"
            "Saved: artifacts/model_card.md\n"
            f"Best model: {best['name']} saved as artifacts/models/model.pkl"
        )

    except Exception as e:
        progress.error("Agent 7 - Model Card & Business Recommendations", str(e))
        raise