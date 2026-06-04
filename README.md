# Bike Sales AI Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![License](https://img.shields.io/badge/License-MIT-green)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bikesalesfinalprojectrachelbarazani.streamlit.app)

---

## Author

**Rachel Barazani** — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---

## Business Problem

A retail bike chain operates 7 stores across the US. They want to grow
their customer base but don't know who they're missing.

The discovery: Certain customer groups are underrepresented in sales
data — not because they don't want bikes, but because they face payment
barriers. Digital payment dominance is creating an invisible wall between
the store and potential customers who prefer or rely on cash.

The solution: An automated AI pipeline that goes from raw dirty data
to business insights and predictive models — answering three questions:

1. How many bikes will a customer buy?
2. Which bike model should we recommend?
3. Would a universal cash discount attract new customers blocked by payment barriers?

---

## Three ML Predictions

| # | Target | Type | Best Model | F1 Score |
|---|--------|------|-----------|----------|
| 1 | Purchase Quantity | Multi-class | Random Forest | 34.37% |
| 2 | Bike Model Recommendation | Multi-class | Random Forest | 60.51% |
| 3 | Cash Payment Prediction | Binary | Random Forest | 69.74% |

Prediction 3 is the core business insight — identifying payment barriers
and recommending a universal cash discount strategy backed by Federal
Reserve payment data.

---

## Pipeline Architecture

The file got cut off at Pipeline Architecture. The rest didn't paste.
In Cursor place your cursor at the very end of the file after ## Pipeline Architecture and paste this:

bike_sales_dirty.csv  (100,200 rows — deliberately dirtied)
|
CREW 1 — Data Analyst Crew (3 agents)
Agent 1 — Ingest, validate, clean — 91,918 rows
Agent 2 — EDA + 12 visualizations — eda_report.html
Agent 3 — Business insights + dataset contract
|
CrewAI Flow — Validation Checkpoint
Schema validation
Contract match
Required features check
|
CREW 2 — Data Scientist Crew (4 agents)
Agent 4 — Feature engineering (25 features)
Agent 5 — Train 6 models across 3 predictions
Agent 6 — Evaluate + compare models
Agent 7 — Model card + business recommendation
|
Streamlit App — 5 pages (live deployment)

---

## AI Backend Options

This pipeline supports two AI backends selected at startup:

| Option | Backend | Cost | Use Case |
|--------|---------|------|----------|
| 1 | Ollama (local) | Free | Development + testing |
| 2 | Anthropic Haiku | ~$0.02/run | Budget cloud option |
| 3 | Anthropic Sonnet | ~$0.08/run | Final submission run |

> **:warning: Your API key is _never_ stored in GitHub — it is always read from your local `.env` file only! :warning:**

---

## Quick Start

### Option 1 — Ollama (Free)

```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3.1

git clone https://github.com/rochelle24-create/bike-sales-ai-pipeline---Final-Project-2026
cd bike-sales-ai-pipeline---Final-Project-2026

conda create -n bike-sales python=3.12 -y
conda activate bike-sales
pip install -r requirements.txt

python flow/pipeline.py
# Select: 1 (Ollama) then enter llama3.1

streamlit run app/streamlit_app.py
```

### Option 2 — Anthropic API

```bash
cp .env.example .env
# Add your key: ANTHROPIC_API_KEY=sk-ant-...

python flow/pipeline.py
# Select: 2 or 3 (Anthropic)
```

---

## Project Structure
bike-sales-ai-pipeline---Final-Project-2026/
|-- data/
|   |-- bike_sales_dirty.csv        pipeline input (dirtied dataset)
|   |-- README.md                   data documentation + Kaggle source
|-- artifacts/                      all pipeline outputs
|   |-- clean_data.csv
|   |-- eda_report.html
|   |-- insights.md
|   |-- dataset_contract.json
|   |-- features.csv
|   |-- evaluation_report.md
|   |-- model_card.md
|   |-- models/                     6 trained models + evaluation charts
|-- crew1_analysts/                 3 analyst agents
|-- crew2_datascientists/           4 data scientist agents
|-- flow/                           CrewAI Flow + validation checkpoints
|-- app/                            Streamlit web app (5 pages)
|-- utils/                          progress logger + LLM selector
|-- create_realistic_data.py        data preparation script
|-- requirements.txt                full pipeline dependencies
|-- streamlit_requirements.txt      deployment dependencies

---

## Pipeline Outputs

| Output | Description |
|--------|-------------|
| clean_data.csv | 91,918 rows after cleaning 10 data quality issues |
| eda_report.html | 12 embedded charts — self-contained HTML |
| insights.md | Business narrative + payment barrier observation |
| dataset_contract.json | Schema contract validated before Crew 2 |
| features.csv | 25 engineered features including seasonal + demographic |
| model.pkl | Best performing model (Random Forest) |
| evaluation_report.md | Full metrics for all 6 models |
| model_card.md | Model documentation + ethical considerations |

---

## Data Preparation

### Step 1 — Real-World Correlations Injected

The original Kaggle dataset had perfectly uniform random distributions
which do not reflect real retail behavior. Before dirtying the data,
evidence-based correlations were injected from published research.

**Seasonal Sales Patterns**

Based on PeopleForBikes (August 2024):
- 42% of adult bikes sold March through June (spring peak)
- December accounts for 25% of BMX annual sales (Christmas gifts)
- July through August summer family vacation bump

Source: https://www.peopleforbikes.org/news/kids-bikes-sales-trends

**Payment Behavior by Age Group**

Based on Federal Reserve Diary of Consumer Payment Choice (2024-2025):
- Ages 18-24: 12% cash, 45% mobile payments
- Ages 55-64: 22% cash
- Ages 65+: 35% cash (3x higher than ages 18-24)
- High value purchases over $2,000: 2% cash

Source: https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice
Source: https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice

**Bike Model Preferences by Age**

Based on general retail cycling industry knowledge:
- Ages 18-24: BMX and Mountain Bike
- Ages 25-34: Road Bike and Mountain Bike
- Ages 35-54: Hybrid and Electric Bike
- Ages 55+: Cruiser, Hybrid and Folding Bike

### Step 2 — 10 Data Quality Issues Injected

| Issue | Count | Cleaned By |
|-------|-------|-----------|
| Missing values (5 columns) | ~14,500 | Agent 1 — drop or label Unknown |
| Duplicate rows | 200 | Agent 1 — removed |
| Gender variants (M/male/MALE) | 1,600 | Agent 1 — standardized |
| Bike model casing issues | 500 | Agent 1 — Title Case |
| Mixed date formats | 3,000 | Agent 1 — normalized |
| Price outliers below $10 or above $50,000 | 150 | Agent 1 — removed |
| Impossible ages below 18 or above 100 | 60 | Agent 1 — removed |
| Negative quantities | 80 | Agent 1 — removed |
| Payment method variants | 300 | Agent 1 — standardized |
| Whitespace in location | 500 | Agent 1 — stripped |

---

## Core Business Insight

Our AI pipeline identified a payment barrier affecting two underserved
age groups — and recommends a universal cash discount strategy backed
by Federal Reserve payment data to bring them into our stores.

Ages 18-24 and 65+ show the lowest purchase frequency and the highest
cash payment reliance. A universal cash discount — available to all
customers equally — removes the payment barrier without targeting or
excluding any demographic.

---

## Ethical Considerations

The cash payment model is used solely to identify market expansion
opportunities. The recommended discount applies universally to all
customers regardless of age, gender, location, or any personal
characteristic. Full ethical statement available in artifacts/model_card.md.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| CrewAI | Multi-agent pipeline orchestration |
| Python 3.12 | All logic |
| Pandas | Data cleaning + feature engineering |
| Scikit-Learn | 6 ML models across 3 predictions |
| Matplotlib / Seaborn | 12 EDA charts + evaluation plots |
| Streamlit | 5 page web application |
| Ollama | Local LLM — free development |
| Anthropic API | Cloud LLM — final run |
| Streamlit Cloud | Live deployment |

---

## License

MIT License — free to use and modify.