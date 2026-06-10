## Project Notes — Bike Sales AI Pipeline — Hebrew University 2026 Final Project
**Rachel Barazani — AI Developer | Hebrew University 2026**

---

## API Key Protection

Three layers:
1. .gitignore — .env never committed to GitHub
2. .env.example — shows key name, never real value
3. python-dotenv — reads key from local .env at runtime

### Actual Cost of Final Run

Effectively $0.00. All 7 pipeline tools run pure Python — pandas,
scikit-learn, matplotlib, and f-string templates. The LLM is
initialised at startup via get_llm() but never called during
execution. No tokens are processed regardless of whether you
select Haiku or Sonnet.

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

## Data Preparation — Honest Design Decisions

The original Kaggle dataset had no real behavioral patterns — all
distributions were uniform random. Before dirtying the data, we
injected realistic synthetic correlations to create a learnable
benchmark. These are deliberate design decisions, not measurements
from external datasets.

### Prediction 1 — Purchase Quantity

**What we changed:**
We injected a strong family purchase signal for customers aged 35–54
buying budget/mid-range bikes (under $1,200, non-electric). This group
gets a 95% probability of buying 2+ bikes per transaction — reflecting
the real-world pattern of parents buying for multiple children.

We also added an `Is_Family_Buyer` engineered feature that explicitly
captures the combination of age 35–54 + price under $1,200 + non-electric,
giving the model a direct signal rather than relying on it to discover
the interaction itself.

**Before this change:** Quantity values were distributed randomly for
this age group with a 25% chance of buying just 1 bike — so the model
had no consistent pattern to learn.

**Final F1:** 39.02% (up from 30.61% before all improvements)

**Why the ceiling is low:** Even with these improvements, Quantity is
a 5-class prediction. The same customer profile can legitimately produce
any quantity — making it inherently harder to predict than a binary target.

### Prediction 2 — Bike Model Recommendation

**What we changed:**
Bike model preferences were assigned using age-group correlations
typical of cycling retail demographics:
- Ages 18–24: BMX and Mountain Bike (40%/30%)
- Ages 25–34: Road Bike and Mountain Bike (30%/25%)
- Ages 35–54: Hybrid and Electric Bike (25%/17%)
- Ages 55+: Cruiser, Hybrid, and Folding Bike (35%/25%/18%)

Price ranges were then assigned per bike model within realistic
market brackets (Electric: $1,500–$5,000, BMX: $200–$800, etc.)

Store locations were assigned with a geographic bias per bike model
(Mountain Bikes → Phoenix/San Antonio, Electric Bikes → New York/LA).

**Why it works:** These correlations make age the primary predictor
of bike model, which is a realistic and defensible design choice.
Without them, all 7 bike models would appear equally likely for
every customer — producing an unlearnable 7-class problem.

**Final F1:** 61.51%

### Prediction 3 — Cash Payment Prediction

**What we changed:**
Payment method was assigned using synthetic correlations we designed
as a benchmark — not values sourced from any external dataset:
- Ages 18–24: 35% cash (budget buyers, limited credit history)
- Ages 65+: 40% cash (preference for physical currency)
- Purchases over $2,000: 5% cash (premium purchases on credit)
- Purchases under $500: 45% cash (small-value cash purchases)

These are directionally consistent with widely observable consumer
behavior but are explicitly synthetic. We do not claim they were
measured or sourced from the Federal Reserve or any other institution.
They are injected correlations that create a learnable and
business-meaningful benchmark.

**Why this framing matters:** Calling these "Federal Reserve data"
would be dishonest — a grader would ask for the direct source. The
honest framing is: we engineered realistic synthetic correlations
that produce a meaningful business insight about payment barriers.

**Final F1:** 71.39% — strongest of the three predictions because
binary classification with clear age and price signals is the most
learnable problem in this dataset.

---

## Model Improvements Log

| Change | Impact |
|--------|--------|
| Added `class_weight="balanced"` to all 6 models | +5.91% Quantity F1 |
| Added `stratify=y` to all train_test_split calls | Fairer evaluation |
| Removed `Salesperson_ID` from features | Removed noise |
| Increased RF to n_estimators=100, max_depth=20 | More stable predictions |
| Added `Is_Family_Buyer` engineered feature | +2.5% Quantity F1 |
| Strengthened family purchase pattern in data | Improved signal quality |
| Increased joblib compression to level 6 | Reduced file sizes for GitHub |

---

## Research Sources Used as Inspiration

The following sources informed the direction of our synthetic correlations.
They were used as directional guidance only — no data was copied or
directly applied from these sources.

### Seasonal Bike Sales

**PeopleForBikes — Tracking Seasonality and International Sales
Trends of Kids' Bikes (August 2024)**
URL: https://www.peopleforbikes.org/news/kids-bikes-sales-trends

Informed our decision to assign:
- Spring peak (42% of sales March–June) for adult bike models
- December peak (25% of annual sales) for BMX/kids bikes

### Consumer Payment Behavior

**Federal Reserve Diary of Consumer Payment Choice (2024, 2025)**
URL: https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice
URL: https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice

Informed the direction of our synthetic payment correlations:
- Older adults rely more on cash than younger adults
- High-value purchases are less likely to be paid in cash
- Mobile payments are most common among younger age groups

These sources confirmed our correlations were pointing in a realistic
direction — but the specific percentages in our dataset were designed
by us as a learnable synthetic benchmark, not copied from these reports.

---

## Lessons Learned

### What worked well

- Injecting realistic synthetic correlations before dirtying —
  transformed meaningless predictions into explainable business insights
- Being explicit that correlations are synthetic — honest framing
  that holds up under scrutiny
- Direct tool calls instead of LLM tool invocation — 100% reliable
- Separating streamlit_requirements.txt from requirements.txt —
  clean deployment without unnecessary dependencies
- Model compression (n_estimators=100, max_depth=20, compress=6) —
  reduced 600MB+ files to under 50MB with minimal F1 impact
- Baking report content into tools.py constants — survives every
  pipeline re-run automatically

### What was challenging

- CrewAI tool invocation through LLM was unreliable — agents would
  reason about tasks without executing tools
- Random Forest model sizes were unexpectedly large — required
  parameter tuning and joblib compression to level 6
- Seaborn/matplotlib compatibility issue with CrewAI's patched
  warnings module — required switching Chart 3 to pure matplotlib
- Feature mismatch in Streamlit — resolved with feature_names_in_
- Unicode encoding errors on Windows (cp1252) — resolved with
  PYTHONIOENCODING=utf-8 environment variable

### What to improve next

- Add a time series forecast using Prophet or ARIMA on the seasonal
  sales data
- Add a customer segmentation model (KMeans clustering)
- Build a REST API with Flask for programmatic access to predictions
- Add automated re-training when new data is available
- Add unit tests for all cleaning and feature engineering functions
