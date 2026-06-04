# crew1_analysts/tasks.py
# Crew 1 — Data Analyst Crew — Task Definitions
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

from crewai import Task
from crew1_analysts.agents import data_ingestion_agent, eda_agent, contract_agent


# ============================================================
# TASK 1 — Data Ingestion, Validation & Cleaning
# ============================================================
task_ingest_clean = Task(
    description="""
    Load and clean the raw bike sales dataset from data/bike_sales_dirty.csv.
    
    Follow these exact steps in order:
    
    STEP 1 — LOAD & VALIDATE:
    - Load data/bike_sales_dirty.csv using pandas
    - Verify all 11 columns exist: Sale_ID, Date, Customer_ID, Bike_Model, 
      Price, Quantity, Store_Location, Salesperson_ID, Payment_Method, 
      Customer_Age, Customer_Gender
    - If any column is missing raise a ValueError with a clear message
    - Log: total rows loaded
    
    STEP 2 — REMOVE DUPLICATES:
    - Drop all duplicate rows
    - Log: number of duplicates removed
    
    STEP 3 — HANDLE MISSING VALUES:
    - DROP rows where Payment_Method is null (target variable — never impute)
    - DROP rows where Customer_Age is null (critical feature for Prediction 3)
    - FILL Store_Location nulls with "Unknown"
    - FILL Customer_Gender nulls with "Unknown"
    - FILL Price nulls with median price per Bike_Model group
    - Log: rows dropped and values filled
    
    STEP 4 — STANDARDIZE GENDER:
    - Map all variants to standard values:
      M, male, MALE, m → Male
      F, female, FEMALE, f → Female
      Keep Unknown as Unknown
    
    STEP 5 — STANDARDIZE BIKE MODEL:
    - Strip whitespace from all Bike_Model values
    - Convert to Title Case
    - Valid values: BMX, Road Bike, Cruiser, Folding Bike, 
      Hybrid Bike, Electric Bike, Mountain Bike
    
    STEP 6 — STANDARDIZE PAYMENT METHOD:
    - Map all variants to standard values:
      cash, CASH, cash payment → Cash
      Keep all other values as-is but strip whitespace
    
    STEP 7 — NORMALIZE DATES:
    - Parse all date formats (DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY)
    - Save all dates as DD-MM-YYYY string format
    
    STEP 8 — REMOVE OUTLIERS & INVALID VALUES:
    - Remove rows where Price < 10 or Price > 50000
    - Remove rows where Customer_Age < 18 or Customer_Age > 100
    - Remove rows where Quantity < 1
    - Strip whitespace from Store_Location
    
    STEP 9 — SAVE OUTPUT:
    - Save cleaned dataframe to artifacts/clean_data.csv
    - Log final row count and column list
    
    FAIL GRACEFULLY:
    - Wrap all steps in try/except
    - Log clear error messages
    - Raise exceptions that stop the pipeline
    """,
    expected_output="""
    A cleaned CSV file saved to artifacts/clean_data.csv with:
    - All 11 original columns present
    - Zero null values in Payment_Method and Customer_Age
    - Standardized Gender values: Male, Female, Unknown only
    - Standardized Bike_Model values: Title Case, no extra spaces
    - Standardized Payment_Method values
    - Normalized dates in DD-MM-YYYY format
    - No price outliers, impossible ages, or negative quantities
    - A cleaning summary log showing rows before and after
    """,
    agent=data_ingestion_agent
)


# ============================================================
# TASK 2 — EDA & Visualizations
# ============================================================
task_eda = Task(
    description="""
    Perform exploratory data analysis on artifacts/clean_data.csv
    and produce a self-contained HTML report with 12 charts.
    
    Generate exactly these 12 charts using matplotlib and seaborn:
    
    SALES PERFORMANCE:
    1. Bar chart — Sales count by Store Location
    2. Line chart — Sales count by Month (seasonal trends)
    3. Box plot — Price distribution by Bike Model
    
    CUSTOMER PROFILE:
    4. Histogram — Customer Age distribution
    5. Bar chart — Customer count by Age Group 
       (18-24, 25-34, 35-44, 45-54, 55-64, 65-70)
    6. Pie chart — Gender distribution (Male/Female/Unknown)
    
    PAYMENT ANALYSIS (feeds Prediction 3):
    7. Pie chart — Payment Method distribution
    8. Bar chart — Cash rate by Age Group (% cash per age group)
    9. Grouped bar chart — Payment Method by Age Group
    
    PREDICTION TARGETS:
    10. Bar chart — Quantity distribution (count per quantity 1-5)
    11. Heatmap — Bike Model by Age Group (count heatmap)
    12. Bar chart — Bike Model distribution (count per model)
    
    For each chart:
    - Add a clear title and axis labels
    - Use seaborn style for clean visuals
    - Save as base64 encoded PNG
    
    BUILD HTML REPORT:
    - Create a single self-contained eda_report.html
    - Embed all 12 charts as base64 images (no external files)
    - Organize into 4 sections:
      Section 1: Dataset Summary (row count, date range, stores, models)
      Section 2: Sales Performance (charts 1-3)
      Section 3: Customer Profile (charts 4-6)
      Section 4: Payment Analysis (charts 7-9)
      Section 5: Prediction Targets (charts 10-12)
    - Save to artifacts/eda_report.html
    """,
    expected_output="""
    A self-contained HTML file saved to artifacts/eda_report.html containing:
    - 12 charts embedded as base64 images
    - 5 organized sections with headers
    - Dataset summary statistics
    - Clean professional styling
    - No external dependencies — opens in any browser
    """,
    agent=eda_agent,
    context=[task_ingest_clean]
)


# ============================================================
# TASK 3 — Business Insights & Dataset Contract
# ============================================================
task_contract = Task(
    description="""
    Analyze artifacts/clean_data.csv and produce two output files.
    
    OUTPUT 1 — artifacts/insights.md:
    
    Write a business insights markdown file with these sections:
    
    ## Dataset Summary
    - Total transactions, date range, number of stores, bike models
    
    ## Key Business Findings
    
    ### Sales Performance
    - Top performing store by transaction count
    - Best selling bike model
    - Peak sales month and season
    
    ### Customer Profile
    - Average customer age
    - Age group with highest purchases
    - Gender split percentages
    
    ### Payment Behavior
    - Cash transaction rate (% of total)
    - Most popular payment method
    - Payment method breakdown
    
    ### Payment Barrier Observation
    Write this specific insight:
    - Which age groups have the lowest overall purchase frequency
    - Cross-reference with cash payment usage in those groups
    - State that a universal cash discount may reduce payment barriers
    - Emphasize: discount applies to ALL customers equally
    - No demographic is targeted or excluded
    
    ### Prediction Target Summary
    - Quantity distribution summary
    - Bike Model distribution summary  
    - Cash vs non-cash split
    
    OUTPUT 2 — artifacts/dataset_contract.json:
    
    Create a JSON contract with this exact structure:
    {{
      "contract_version": "1.0",
      "created_by": "Crew 1 — Agent 3",
      "dataset": "Bike Sales Clean Data",
      "total_rows": <actual row count>,
      "total_columns": 11,
      "columns": {{
        "<column_name>": {{
          "type": "<python type>",
          "nullable": false,
          "description": "<description>",
          "allowed_values": [<list if categorical>],
          "min": <if numeric>,
          "max": <if numeric>
        }}
      }},
      "prediction_targets": {{
        "prediction_1": {{
          "target": "Quantity",
          "type": "multi-class classification",
          "classes": [1, 2, 3, 4, 5]
        }},
        "prediction_2": {{
          "target": "Bike_Model", 
          "type": "multi-class classification",
          "classes": ["BMX", "Road Bike", "Cruiser", "Folding Bike", 
                      "Hybrid Bike", "Electric Bike", "Mountain Bike"]
        }},
        "prediction_3": {{
          "target": "Payment_Method",
          "type": "binary classification",
          "derived_feature": "Is_Cash",
          "positive_class": "Cash",
          "negative_class": "Not Cash"
        }}
      }},
      "assumptions": [<list of cleaning decisions made>],
      "constraints": [
        "Crew 2 must validate this contract before processing",
        "Target columns must not be imputed",
        "Random seed must be set to 42 for reproducibility"
      ]
    }}
    """,
    expected_output="""
    Two files saved to artifacts/:
    1. insights.md — business narrative with all sections complete
    2. dataset_contract.json — valid JSON schema contract
    Both files must be present for the CrewAI Flow validation to pass.
    """,
    agent=contract_agent,
    context=[task_ingest_clean, task_eda]
)