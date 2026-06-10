# utils/references.py
# External research citations for payment behavior analysis
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

DATA_ENHANCEMENT_SECTION = """
## Data Preparation - Synthetic Correlations Injected

The original Kaggle dataset contained uniform random distributions with
no real-world behavioral patterns. Before dirtying the data, we injected
realistic synthetic correlations to create a learnable benchmark.

These are deliberate design decisions - not measurements from external
datasets. The specific percentages and thresholds were chosen by us to
reflect directionally realistic behavior. They are documented and
reproducible.

### Seasonal Patterns
- Applied: BMX bikes assigned December-peak seasonality (25% of annual
  sales) reflecting holiday gift-buying patterns for children's bikes.
  All other models assigned spring peak (42% March-June) reflecting
  adult cycling season.
- Informed by: PeopleForBikes - Tracking Seasonality and International
  Sales Trends of Kids' Bikes (August 2024)
- URL: https://www.peopleforbikes.org/news/kids-bikes-sales-trends

### Payment Behavior by Age
- Applied: Older customers (65+) assigned higher cash rates (40%).
  Younger customers (18-24) assigned moderate cash rates (35%).
  High-value purchases (>$2,000) assigned very low cash rates (5%).
  Budget purchases (<$500) assigned higher cash rates (45%).
- These are synthetic correlations we designed as a benchmark -
  not values sourced from any external dataset.
- Directionally informed by: Federal Reserve Diary of Consumer Payment
  Choice research (2024-2025), which confirms older adults carry more
  cash and younger adults favour mobile payments. The specific
  percentages in our dataset are our own design choices.

### Family Purchase Behavior
- Applied: Customers aged 35-54 buying budget/mid-range non-electric
  bikes (under $1,200) assigned 95% probability of buying 2+ units,
  reflecting the pattern of parents buying bikes for multiple children.
- An Is_Family_Buyer engineered feature was added to give the model a
  direct signal for this pattern.

### Why This Matters
These injected correlations transform a random synthetic dataset into
one with learnable, explainable business patterns. Without them, all
three predictions would be close to random guessing.
"""

CASH_PAYMENT_CITATIONS = """## References - Sources Used as Directional Guidance

The following sources informed the direction of our synthetic payment
correlations. They were used as guidance only - no data was directly
copied or applied from these reports. The specific values in our dataset
are synthetic design choices.

1. **PeopleForBikes** - *Tracking Seasonality and International Sales Trends of Kids' Bikes (August 2024)*
   https://www.peopleforbikes.org/news/kids-bikes-sales-trends

2. **Federal Reserve Financial Services** - *2024 Findings from the Diary of Consumer Payment Choice*
   https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice

3. **Federal Reserve Financial Services** - *2025 Diary of Consumer Payment Choice*
   https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice

4. **Federal Reserve Bank of Richmond** - *2025 Speaking of the Economy - Payment Trends*
   https://www.richmondfed.org/podcasts/speaking_of_the_economy/2025/speaking_2025_09_17_payments_trends
"""
