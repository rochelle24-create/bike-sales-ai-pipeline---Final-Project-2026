# crew2_datascientists/tasks.py
# Crew 2 — Data Scientist Crew — Task Definitions
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

from crewai import Task
from crew2_datascientists.agents import (
    feature_engineer,
    model_trainer,
    model_evaluator,
    model_card_writer
)


# ============================================================
# TASK 4 — Feature Engineering
# ============================================================
task_feature_engineering = Task(
    description="""
    Read artifacts/clean_data.csv and artifacts/dataset_contract.json.
    Validate the contract then engineer new features.

    STEP 1 — VALIDATE CONTRACT:
    - Load dataset_contract.json
    - Confirm all contract columns exist in clean_data.csv
    - Confirm column data types match contract
    - Confirm no nulls in Payment_Method or Customer_Age
    - If any check fails raise a ValueError with clear message

    STEP 2 — ENGINEER FEATURES:
    Create these new columns:

    From Date:
    - Month: integer 1-12 extracted from Date
    - Day_of_Week: integer 0-6 (Monday=0, Sunday=6)
    - Season: string Winter/Spring/Summer/Fall
      (Winter: Dec,Jan,Feb | Spring: Mar,Apr,May |
       Summer: Jun,Jul,Aug | Fall: Sep,Oct,Nov)
    - Is_Weekend: integer 1 if Saturday or Sunday else 0

    From Customer_Age:
    - Age_Group: string category
      (18-24, 25-34, 35-44, 45-54, 55-64, 65+)

    From Price:
    - Price_Tier: string category
      (Budget: <500, Mid: 500-2000, Premium: >2000)

    From Payment_Method:
    - Is_Cash: integer 1 if Cash else 0
      (THIS IS THE TARGET FOR PREDICTION 3)

    STEP 3 — ENCODE CATEGORICALS:
    Label encode these columns for modeling:
    - Bike_Model
    - Store_Location
    - Customer_Gender
    - Payment_Method
    - Season
    - Age_Group
    - Price_Tier

    Keep both original and encoded versions:
    - Original: Bike_Model, Store_Location, etc.
    - Encoded: Bike_Model_enc, Store_Location_enc, etc.

    STEP 4 — SAVE:
    - Save to artifacts/features.csv
    - Log all features created and final row/column count

    FAIL GRACEFULLY:
    - Wrap in try/except
    - Log clear error messages
    - Raise exceptions that stop the pipeline
    """,
    expected_output="""
    A features CSV saved to artifacts/features.csv with:
    - All original 11 columns preserved
    - 7 new engineered features added
    - Label encoded versions of all categorical columns
    - Same row count as clean_data.csv
    - Zero null values
    - A feature engineering summary log
    """,
    agent=feature_engineer
)


# ============================================================
# TASK 5 — Model Training
# ============================================================
task_model_training = Task(
    description="""
    Load artifacts/features.csv and train 6 models across
    3 business predictions.

    SETUP:
    - Random seed = 42 for ALL operations
    - Train/test split = 80/20 for ALL predictions
    - Drop these columns from ALL feature sets:
      Sale_ID, Customer_ID, Date, Date_parsed (if exists)

    PREDICTION 1 — Quantity (multi-class classification):
    Target: Quantity (classes 1, 2, 3, 4, 5)
    Features: all encoded columns EXCEPT Quantity and Is_Cash

    Train Model 1a: Logistic Regression
    - max_iter=1000, random_state=42
    - Save as artifacts/models/quantity_lr.pkl

    Train Model 1b: Random Forest Classifier
    - n_estimators=100, random_state=42
    - Save as artifacts/models/quantity_rf.pkl

    PREDICTION 2 — Bike Model (multi-class classification):
    Target: Bike_Model_enc
    Features: all encoded columns EXCEPT Bike_Model,
              Bike_Model_enc, and Is_Cash

    Train Model 2a: Logistic Regression
    - max_iter=1000, random_state=42
    - Save as artifacts/models/bike_model_lr.pkl

    Train Model 2b: Random Forest Classifier
    - n_estimators=100, random_state=42
    - Save as artifacts/models/bike_model_rf.pkl

    PREDICTION 3 — Is_Cash (binary classification):
    Target: Is_Cash
    Features: all encoded columns EXCEPT Payment_Method,
              Payment_Method_enc, and Is_Cash

    Train Model 3a: Logistic Regression
    - max_iter=1000, random_state=42
    - Save as artifacts/models/payment_lr.pkl

    Train Model 3b: Random Forest Classifier
    - n_estimators=100, random_state=42
    - Save as artifacts/models/payment_rf.pkl

    SAVE TEST DATA:
    - Save X_test and y_test for each prediction as:
      artifacts/models/quantity_test.pkl
      artifacts/models/bike_model_test.pkl
      artifacts/models/payment_test.pkl

    FAIL GRACEFULLY:
    - Wrap each model in try/except
    - Log training completion and model file sizes
    - Raise exceptions that stop the pipeline
    """,
    expected_output="""
    6 trained model files saved to artifacts/models/:
    - quantity_lr.pkl, quantity_rf.pkl
    - bike_model_lr.pkl, bike_model_rf.pkl
    - payment_lr.pkl, payment_rf.pkl
    Plus 3 test data files for evaluation.
    All models trained with random_state=42.
    """,
    agent=model_trainer,
    context=[task_feature_engineering]
)


# ============================================================
# TASK 6 — Model Evaluation
# ============================================================
task_model_evaluation = Task(
    description="""
    Load all 6 trained models and evaluate them on test data.
    Generate a comprehensive evaluation report.

    FOR EACH PREDICTION (Quantity, Bike_Model, Is_Cash):

    STEP 1 — LOAD:
    - Load both models (.pkl files)
    - Load corresponding test data

    STEP 2 — EVALUATE BOTH MODELS:
    Compute for each model:
    - Accuracy score
    - F1 Score (weighted average)
    - Classification Report (precision, recall, f1 per class)
    - Confusion Matrix

    STEP 3 — GENERATE CHARTS:
    Save these charts to artifacts/models/:

    For each prediction:
    - accuracy_comparison_[prediction].png
      (bar chart: LR vs RF accuracy)
    - f1_comparison_[prediction].png
      (bar chart: LR vs RF F1 score)
    - confusion_matrix_lr_[prediction].png
    - confusion_matrix_rf_[prediction].png
    - feature_importance_rf_[prediction].png
      (top 10 features from Random Forest)

    STEP 4 — WRITE evaluation_report.md:

    Structure:
    # Bike Sales — Model Evaluation Report

    ## Prediction 1: Purchase Quantity
    ### Logistic Regression Results
    - Accuracy: X%
    - F1 Score: X%
    - Classification Report table

    ### Random Forest Results
    - Accuracy: X%
    - F1 Score: X%
    - Classification Report table

    ### Winner: [model name] with [metric]

    ## Prediction 2: Bike Model Recommendation
    [same structure]

    ## Prediction 3: Cash Payment Prediction
    [same structure]

    ## Overall Best Model
    - Best model across all predictions by F1 score
    - Saved as artifacts/models/model.pkl

    Save to artifacts/evaluation_report.md

    FAIL GRACEFULLY:
    - Wrap in try/except
    - Log all metrics clearly
    """,
    expected_output="""
    artifacts/evaluation_report.md with full metrics for all 6 models.
    15 chart files saved to artifacts/models/.
    Clear winner identified for each prediction.
    """,
    agent=model_evaluator,
    context=[task_feature_engineering, task_model_training]
)


# ============================================================
# TASK 7 — Model Card & Business Recommendations
# ============================================================
task_model_card = Task(
    description="""
    Write a comprehensive model card and business recommendation.

    STEP 1 — READ EVALUATION RESULTS:
    - Load artifacts/evaluation_report.md
    - Identify best model per prediction by F1 score
    - Copy best overall model to artifacts/models/model.pkl

    STEP 2 — WRITE artifacts/model_card.md:

    # Bike Sales AI Pipeline — Hebrew University 2026 Final Project — Model Card
    **Author:** Rachel Barazani — AI Developer
    **Course:** AI Developer Program — Hebrew University 2026

    ## Model Overview
    Three classification models predicting:
    1. Purchase Quantity (how many bikes)
    2. Bike Model Recommendation (which bike)
    3. Cash Payment Prediction (payment barrier analysis)

    ## Training Data Summary
    - Dataset: Bike Sales 100k (cleaned)
    - Source: Kaggle — jayavarman/bike-sales-data-of-100k
    - Rows used: [actual count]
    - Train/test split: 80/20
    - Random seed: 42

    ## Model Performance Summary
    | Prediction | Best Model | Accuracy | F1 Score |
    |------------|-----------|----------|----------|
    | Quantity   | [model]   | [score]  | [score]  |
    | Bike Model | [model]   | [score]  | [score]  |
    | Is_Cash    | [model]   | [score]  | [score]  |

    ## Prediction 1 — Purchase Quantity
    ### Purpose
    Predict how many bikes a customer will purchase in one transaction.
    ### Features Used
    [list top features by importance]
    ### Limitations
    - Dataset is synthetically balanced — real distributions may differ
    - Model trained on 7 US city locations only

    ## Prediction 2 — Bike Model Recommendation
    ### Purpose
    Recommend the most likely bike model for a customer profile.
    ### Features Used
    [list top features by importance]
    ### Limitations
    - Limited to 7 bike models in training data
    - Does not account for inventory availability

    ## Prediction 3 — Cash Payment Prediction
    ### Purpose
    Identify customers likely to pay cash and analyze payment barriers.
    ### Features Used
    [list top features by importance]
    ### Limitations
    - Model trained on historical payment behavior only
    - Cannot predict future payment method changes

    ## Ethical Considerations

    ### Cash Discount Recommendation
    The cash payment model is used to identify market expansion
    opportunities — NOT to target, exclude, or discriminate
    against any customer group.

    The recommended cash discount applies UNIVERSALLY to all
    customers regardless of age, gender, location, or any
    demographic characteristic.

    Age group analysis identifies who benefits most from removing
    payment barriers — consistent with responsible retail practice.

    ### Bias Assessment
    - Model evaluated across all age groups and genders
    - No discriminatory patterns identified in predictions
    - Regular re-evaluation recommended as customer base evolves

    ### Fairness Statement
    No customer is excluded or targeted based on personal
    characteristics. All recommendations are made at the
    business level, not the individual level.

    STEP 3 — SAVE best model:
    - Load evaluation results
    - Find model with highest weighted F1 score
    - Copy that model file to artifacts/models/model.pkl
    - Log: "Best model: [name] F1=[score] saved as model.pkl"

    FAIL GRACEFULLY:
    - Wrap in try/except
    - Log completion of each section
    """,
    expected_output="""
    artifacts/model_card.md — comprehensive model documentation
    artifacts/models/model.pkl — best performing model saved
    Both files required for project submission.
    """,
    agent=model_card_writer,
    context=[task_feature_engineering, task_model_training, task_model_evaluation]
)