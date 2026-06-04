# utils/references.py
# External research citations for payment behavior analysis
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

DATA_ENHANCEMENT_SECTION = """
## Data Enhancement — Real-World Research Sources

The synthetic dataset was enhanced with evidence-based correlations
before model training to improve real-world validity:

### Seasonal Patterns
- Source: PeopleForBikes — Tracking Seasonality and International
  Sales Trends of Kids' Bikes (August 2024)
- URL: https://www.peopleforbikes.org/news/kids-bikes-sales-trends
- Applied: BMX bikes assigned December-peak seasonality (25% of
  annual sales). All other models assigned spring peak (42%
  March–June) per adult bike industry data.

### Payment Behavior by Age
- Source: Federal Reserve Diary of Consumer Payment Choice 2024
- URL: https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice
- Source: Federal Reserve Diary of Consumer Payment Choice 2025
- URL: https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice
- Applied:
  - Ages 18–24: 12% cash, 45% mobile payments
  - Ages 55–64: 22% cash
  - Ages 65+: 35% cash (3x higher than 18–24)
  - High price (>$2,000): 2% cash

### Why This Matters
These enhancements ensure the model learns from realistic behavioral
patterns rather than purely random synthetic distributions. All
correlations are documented, sourced, and reproducible.
"""

CASH_PAYMENT_CITATIONS = """## References — Payment Behavior Research

External research supporting the cash payment analysis and universal discount recommendation:

1. **Federal Reserve Financial Services** — *2024 Findings from the Diary of Consumer Payment Choice*  
   https://www.frbservices.org/news/research/2024-findings-from-the-diary-of-consumer-payment-choice

2. **Federal Reserve Bank of Richmond** — *2025 Speaking of the Economy Podcast — Payment Trends*  
   https://www.richmondfed.org/podcasts/speaking_of_the_economy/2025/speaking_2025_09_17_payments_trends

3. **Federal Reserve Financial Services** — *2025 Diary of Consumer Payment Choice*  
   https://www.frbservices.org/news/fed360/issues/060325/cash-2025-findings-diary-consumer-payment-choice
"""
