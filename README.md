# Bike Sales AI Pipeline — Hebrew University 2026 Final Project

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

## Demo Video

A walkthrough of the Streamlit app including live predictions, model results, business insights, and the cash payment recommendation.

[Watch on Loom](https://www.loom.com/share/835b8fcdaf0148d7a3eb858b34b2b278)

## Live Pipeline Demo

A real-time recording of the full two-crew CrewAI pipeline running end to end, including live progress updates.

[Watch on Loom](https://www.loom.com/share/2e5dbb74ddba4ffebf0587408f12947c)


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
| 1 | Purchase Quantity | Multi-class | Random Forest | 39.02% |
| 2 | Bike Model Recommendation | Multi-class | Random Forest | 61.51% |
| 3 | Cash Payment Prediction | Binary | Random Forest | 71.39% |

Prediction 3 is the core business insight — identifying payment barriers
and recommending a universal cash discount strategy to bring underserved
customer groups into the stores.

---

## Pipeline Architecture

```
bike_sales_dirty.csv  (100,200 rows — deliberately dirtied)
        |
CREW 1 — Data Analyst Crew (3 agents)
  Agent 1 — Ingest, validate, clean → 91,948 rows
  Agent 2 — EDA + 12 visualizations → eda_report.html
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
```

---

## AI Backend Options

This pipeline supports two AI backends selected at startup:

| Option | Backend | Cost | Use Case |
|--------|---------|------|----------|
| 1 | Ollama (local) | Free | Development + testing |
| 2 | Anthropic Haiku | ~$0.00/run | Budget cloud option |
| 3 | Anthropic Sonnet | ~$0.00/run | Final submission run |

> **Note:** Cost is effectively $0.00 because all 7 pipeline tools run pure
> Python (pandas, scikit-learn, matplotlib). The LLM is initialised at
> startup but never invoked — no tokens are processed.

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

```
bike-sales-ai-pipeline---Final-Project-2026/
├── data/
│   ├── bike_sales_dirty.csv        pipeline input (dirtied dataset)
│   └── bike_sales_100k.csv         original Kaggle source (gitignored — download separately)
├── artifacts/                      all pipeline outputs
│   ├── clean_data.csv
│   ├── eda_report.html
│   ├── insights.md
│   ├── dataset_contract.json
│   ├── features.csv
│   ├── evaluation_report.md
│   ├── model_card.md
│   └── models/                     6 trained models + evaluation charts
├── crew1_analysts/                 3 analyst agents
├── crew2_datascientists/           4 data scientist agents
├── flow/                           CrewAI Flow + validation checkpoints
├── app/                            Streamlit web app (5 pages)
├── utils/                          progress logger + LLM selector
├── create_realistic_data.py        data preparation script
├── requirements.txt                full pipeline dependencies
└── streamlit_requirements.txt      deployment dependencies
```

---

## Pipeline Outputs

| Output | Description |
|--------|-------------|
| clean_data.csv | 91,948 rows after cleaning 10 data quality issues |
| eda_report.html | 12 embedded charts — self-contained HTML |
| insights.md | Business narrative + payment barrier observation |
| dataset_contract.json | Schema contract validated before Crew 2 |
| features.csv | 25 engineered features including seasonal + demographic |
| model.pkl | Best performing model (Random Forest) |
| evaluation_report.md | Full metrics for all 6 models |
| model_card.md | Model documentation + ethical considerations |

---

## Data Preparation

The original Kaggle dataset contained perfectly uniform random distributions
with no real-world behavioral patterns. Before dirtying the data, we
injected realistic synthetic correlations to create a learnable benchmark.
These are not claims about real customer data — they are deliberate design
decisions that make each prediction meaningful and explainable.

### Prediction 1 — Purchase Quantity

**What we changed:** Family purchase behavior was injected for customers
aged 35–54 buying budget or mid-range bikes (under $1,200, non-electric).
This group was assigned a 95% probability of buying 2 or more bikes per
transaction, reflecting the real-world pattern of parents buying bikes
for multiple children. We also added an `Is_Family_Buyer` interaction
feature that explicitly captures this combination (age 35–54 + price
under $1,200 + non-electric bike).

**Why it reflects reality:** A 40-year-old buying a $400 BMX is far more
likely to be buying for a child — or buying two — than a 25-year-old
making a solo purchase. Without this signal, the model sees random
quantity values with no learnable pattern.

### Prediction 2 — Bike Model Recommendation

**What we changed:** Bike model preferences were assigned based on
age-group correlations typical of the cycling retail industry:
- Ages 18–24: BMX and Mountain Bike (active/youth)
- Ages 25–34: Road Bike and Mountain Bike (fitness-focused)
- Ages 35–54: Hybrid and Electric Bike (commuting/leisure)
- Ages 55+: Cruiser, Hybrid, and Folding Bike (comfort/ease)

Price was then assigned per bike model within realistic market ranges
(e.g. Electric Bikes $1,500–$5,000, BMX $200–$800).

**Why it reflects reality:** These age-to-model correlations are
observable patterns in cycling retail. Without them, the dataset would
have seniors buying BMX bikes and teenagers buying Electric Bikes at
equal rates — producing a model that learns nothing useful.

### Prediction 3 — Cash Payment Prediction

**What we changed:** Payment method was assigned based on age and price
correlations we designed ourselves as a synthetic benchmark:
- Ages 18–24: 35% cash (limited credit access, budget purchases)
- Ages 65+: 40% cash (stronger preference for physical currency)
- Purchases over $2,000: 5% cash (high-value items go on credit)
- Purchases under $500: 45% cash (small-value items paid in cash)

**Why it reflects reality:** These are directionally consistent with
widely observed consumer payment behavior — younger and older customers
rely more on cash, premium purchases go on credit — but they are
synthetic correlations we injected, not measured values from any
external dataset. The framing is honest: we built a learnable benchmark
that produces a meaningful business insight about payment barriers.

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
age groups — ages 18–24 and 65+ — who show the lowest purchase frequency
and the highest cash payment reliance in our synthetic model.

A universal cash discount — available to all customers equally — removes
the payment barrier without targeting or excluding any demographic.
The recommendation applies universally regardless of age, gender,
location, or any personal characteristic.

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
