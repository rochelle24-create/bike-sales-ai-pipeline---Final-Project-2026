### API Key Protection

Three layers:
1. .gitignore — .env never committed to GitHub
2. .env.example — shows key name, never real value
3. python-dotenv — reads key from local .env at runtime

### Actual Cost of Final Run

Estimated: under $0.10 because direct tool calls mean the LLM
is only invoked for Agent 3 and Agent 7 narrative writing.
All data processing, ML training, and chart generation runs
in Python with zero LLM calls.

---

## Streamlit App Design

### 5 Pages

**Page 1 — Business Overview**
- 4 metric cards
- Embedded eda_report.html (iframe)
- Rendered insights.md

**Page 2 — Prediction 1: Quantity**
- Live prediction form
- Probability bars for classes 1-5
- Model comparison charts
- Confusion matrices
- Feature importance

**Page 3 — Prediction 2: Bike Model**
- Live recommendation form
- Probability bars for all 7 models
- Model comparison charts
- Confusion matrices
- Feature importance

**Page 4 — Prediction 3: Cash Payment**
- Purchase frequency by age group chart
- Cash rate by age group chart
- Live cash probability prediction with gauge
- Model comparison charts
- Universal cash discount recommendation box

**Page 5 — Model Documentation**
- Rendered model_card.md
- Rendered evaluation_report.md
- Rendered dataset_contract.json
- Download center for all artifacts

### Feature Mismatch Fix

Initial deployment had a prediction error because the Streamlit
input form included Payment_Method_enc in the Quantity prediction
features — but the model was trained without it.

Fix: Used model.feature_names_in_ to dynamically build the
feature vector from exactly the columns the model was trained on.
This prevents any future feature mismatch regardless of model changes.

---

## Deployment

### Streamlit Cloud

- URL: https://bikesalesfinalprojectrachelbarazani.streamlit.app
- Free tier — public GitHub repo
- Auto-deploys on every push to main
- Requirements file: streamlit_requirements.txt (app-only dependencies)
- Full requirements.txt not used for deployment (crewai not needed)

### What Gets Deployed

The Streamlit app reads pre-generated artifacts from the artifacts/
folder which are committed to GitHub. The pipeline itself does not
run on Streamlit Cloud — only the web app.

This means:
- No API keys needed on Streamlit Cloud
- No heavy ML dependencies needed for deployment
- Fast cold start — just loads CSV files and PKL models

---

## Research Sources

### Seasonal Bike Sales

**PeopleForBikes — Tracking Seasonality and International Sales
Trends of Kids' Bikes (August 2024)**  
URL: https://www.peopleforbikes.org/news/kids-bikes-sales-trends  

Key facts used:
- 42% of adult bicycles sold March through June
- December accounts for 25% of annual BMX unit sales
- June and December are strongest months for adult bikes

### Payment Behavior by Age

**Federal Reserve Diary of Consumer Payment Choice 2024**  
URL: https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice  

Key facts used:
- Consumers under 55 used cash for 12% of payments in 2023
- Consumers 55+ used cash for 22% of payments in 2023

**Federal Reserve Diary of Consumer Payment Choice 2025**  
URL: https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice  

Key facts used:
- Adults 65+ carry almost 3x more cash than adults 18-24
- Households earning under $25,000 and adults 55+ rely most on cash
- Adults 18-24 use mobile phones for 45% of all payments

**Federal Reserve Bank of Richmond — Speaking of the Economy (2025)**  
URL: https://www.richmondfed.org/podcasts/speaking_of_the_economy/2025/speaking_2025_09_17_payments_trends  

Key facts used:
- Cash declined from 31% of payments (2016) to 14% (2024)
- Adults 18-24 most likely to pay with mobile phone

---

## Lessons Learned

### What worked well

- Injecting real-world correlations before dirtying — transformed
  meaningless predictions into explainable business insights
- Direct tool calls instead of LLM tool invocation — 100% reliable
- Separating streamlit_requirements.txt from requirements.txt —
  clean deployment without unnecessary dependencies
- Model compression (n_estimators=50, max_depth=15, compress=3) —
  reduced 600MB files to under 7MB with minimal F1 impact
- Baking citations into tools.py constants — survive every pipeline
  re-run automatically

### What was challenging

- CrewAI tool invocation through LLM was unreliable — agents would
  reason about tasks without executing tools
- Random Forest model sizes were unexpectedly large — required
  parameter tuning and joblib compression
- Seaborn/matplotlib compatibility issue with CrewAI's patched
  warnings module — required switching Chart 3 to pure matplotlib
- Feature mismatch in Streamlit — resolved with feature_names_in_

### What to improve next

- Add a time series forecast using Prophet or ARIMA on the seasonal
  sales data
- Add a customer segmentation model (KMeans clustering)
- Build a REST API with Flask for programmatic access to predictions
- Add automated re-training when new data is available
- Add unit tests for all cleaning and feature engineering functions