# create_realistic_data.py
# Inject realistic correlations then dirty the dataset
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================
# LOAD ORIGINAL CLEAN DATA
# ============================================================
print("Loading original dataset...")
df = pd.read_csv("data/bike_sales_100k.csv")
print(f"Loaded {len(df):,} rows")

# ============================================================
# STEP 1 - INJECT REALISTIC CORRELATIONS
# ============================================================
print("\nInjecting realistic correlations...")

# --- Bike Model based on Age ---
def assign_bike_model(row):
    age = row["Customer_Age"]
    r = np.random.random()

    if age <= 24:
        # Young: BMX and Mountain Bike
        if r < 0.40:   return "BMX"
        elif r < 0.70: return "Mountain Bike"
        elif r < 0.85: return "Road Bike"
        elif r < 0.93: return "Hybrid Bike"
        elif r < 0.97: return "Folding Bike"
        elif r < 0.99: return "Cruiser"
        else:          return "Electric Bike"

    elif age <= 34:
        # Young adult: Road Bike and Mountain Bike
        if r < 0.30:   return "Road Bike"
        elif r < 0.55: return "Mountain Bike"
        elif r < 0.70: return "Hybrid Bike"
        elif r < 0.82: return "BMX"
        elif r < 0.91: return "Electric Bike"
        elif r < 0.96: return "Folding Bike"
        else:          return "Cruiser"

    elif age <= 44:
        # Mid age: balanced but Electric growing
        if r < 0.25:   return "Hybrid Bike"
        elif r < 0.45: return "Road Bike"
        elif r < 0.62: return "Electric Bike"
        elif r < 0.75: return "Mountain Bike"
        elif r < 0.85: return "Cruiser"
        elif r < 0.93: return "Folding Bike"
        else:          return "BMX"

    elif age <= 54:
        # Older adult: Hybrid and Electric
        if r < 0.30:   return "Electric Bike"
        elif r < 0.55: return "Hybrid Bike"
        elif r < 0.70: return "Cruiser"
        elif r < 0.82: return "Folding Bike"
        elif r < 0.91: return "Road Bike"
        elif r < 0.97: return "Mountain Bike"
        else:          return "BMX"

    else:
        # Senior 55+: Cruiser, Hybrid, Folding
        if r < 0.35:   return "Cruiser"
        elif r < 0.60: return "Hybrid Bike"
        elif r < 0.78: return "Folding Bike"
        elif r < 0.88: return "Electric Bike"
        elif r < 0.94: return "Road Bike"
        elif r < 0.98: return "Mountain Bike"
        else:          return "BMX"

print("  Assigning bike models based on age...")
df["Bike_Model"] = df.apply(assign_bike_model, axis=1)

# --- Price based on Bike Model ---
price_ranges = {
    "BMX":          (200,  800),
    "Mountain Bike":(400, 1800),
    "Road Bike":    (600, 2500),
    "Hybrid Bike":  (500, 2000),
    "Folding Bike": (300, 1200),
    "Cruiser":      (300, 1000),
    "Electric Bike":(1500, 5000),
}

print("  Assigning prices based on bike model...")
def assign_price(row):
    low, high = price_ranges[row["Bike_Model"]]
    return round(np.random.uniform(low, high), 2)

df["Price"] = df.apply(assign_price, axis=1)

# --- Quantity based on Age + Price ---
print("  Assigning quantity based on age and price...")
def assign_quantity(row):
    age = row["Customer_Age"]
    price = row["Price"]
    model = row["Bike_Model"]
    r = np.random.random()

    # Electric bikes almost always qty 1
    if model == "Electric Bike":
        if r < 0.85: return 1
        elif r < 0.97: return 2
        else: return 3

    # High price = lower quantity
    if price > 2000:
        if r < 0.70: return 1
        elif r < 0.90: return 2
        elif r < 0.98: return 3
        else: return 4

    # Young buyers tend to buy 1
    if age <= 24:
        if r < 0.60: return 1
        elif r < 0.80: return 2
        elif r < 0.92: return 3
        elif r < 0.98: return 4
        else: return 5

    # Strong family purchase: mid-age + budget non-electric
    if 35 <= age <= 54 and price < 1200 and model != "Electric Bike":
        if r < 0.05: return 1
        elif r < 0.20: return 2
        elif r < 0.55: return 3
        elif r < 0.80: return 4
        else: return 5

    # Mid-age buying expensive or electric bikes - solo purchase
    if 35 <= age <= 54:
        if r < 0.35: return 1
        elif r < 0.60: return 2
        elif r < 0.80: return 3
        elif r < 0.93: return 4
        else: return 5

    # Budget bikes bought in bulk
    if price < 500:
        if r < 0.20: return 1
        elif r < 0.40: return 2
        elif r < 0.60: return 3
        elif r < 0.80: return 4
        else: return 5

    # Default distribution
    if r < 0.30: return 1
    elif r < 0.55: return 2
    elif r < 0.75: return 3
    elif r < 0.90: return 4
    else: return 5

df["Quantity"] = df.apply(assign_quantity, axis=1)

# --- Payment Method based on Age + Price ---
print("  Assigning payment methods based on age and price...")
def assign_payment(row):
    age = row["Customer_Age"]
    price = row["Price"]
    r = np.random.random()

    # Young (18-24): more cash, less credit
    if age <= 24:
        if r < 0.35:   return "Cash"
        elif r < 0.55: return "Debit Card"
        elif r < 0.70: return "Credit Card"
        elif r < 0.82: return "Apple Pay"
        elif r < 0.92: return "Google Pay"
        else:          return "PayPal"

    # Senior (65+): more cash
    elif age >= 65:
        if r < 0.40:   return "Cash"
        elif r < 0.65: return "Debit Card"
        elif r < 0.80: return "Credit Card"
        elif r < 0.88: return "Apple Pay"
        elif r < 0.94: return "Google Pay"
        else:          return "PayPal"

    # High price: credit card or financing
    elif price > 2000:
        if r < 0.05:   return "Cash"
        elif r < 0.15: return "Debit Card"
        elif r < 0.55: return "Credit Card"
        elif r < 0.72: return "Apple Pay"
        elif r < 0.86: return "Google Pay"
        else:          return "PayPal"

    # Budget purchase: more cash
    elif price < 500:
        if r < 0.45:   return "Cash"
        elif r < 0.65: return "Debit Card"
        elif r < 0.78: return "Credit Card"
        elif r < 0.87: return "Apple Pay"
        elif r < 0.94: return "Google Pay"
        else:          return "PayPal"

    # Default mid range
    else:
        if r < 0.18:   return "Cash"
        elif r < 0.38: return "Debit Card"
        elif r < 0.60: return "Credit Card"
        elif r < 0.75: return "Apple Pay"
        elif r < 0.88: return "Google Pay"
        else:          return "PayPal"

df["Payment_Method"] = df.apply(assign_payment, axis=1)

# --- Store Location based on Bike Model ---
print("  Assigning store locations with realistic bias...")
store_bike_bias = {
    "Electric Bike":  ["New York", "Los Angeles", "Chicago"],
    "Road Bike":      ["Chicago", "Houston", "Phoenix"],
    "Mountain Bike":  ["Phoenix", "San Antonio", "Houston"],
    "BMX":            ["Philadelphia", "San Antonio", "Houston"],
    "Cruiser":        ["Los Angeles", "Philadelphia", "New York"],
    "Hybrid Bike":    ["New York", "Chicago", "Los Angeles"],
    "Folding Bike":   ["New York", "Los Angeles", "Philadelphia"],
}
all_stores = ["Philadelphia", "Chicago", "San Antonio",
              "Los Angeles", "Houston", "New York", "Phoenix"]

def assign_store(row):
    model = row["Bike_Model"]
    r = np.random.random()
    preferred = store_bike_bias.get(model, all_stores)

    if r < 0.70:
        return np.random.choice(preferred)
    return np.random.choice(all_stores)

df["Store_Location"] = df.apply(assign_store, axis=1)

# --- Realistic Seasonal Date Distribution ---
print("  Assigning realistic seasonal dates...")

import calendar

adult_monthly_weights = {
    1: 0.04,
    2: 0.05,
    3: 0.09,
    4: 0.11,
    5: 0.12,
    6: 0.10,
    7: 0.09,
    8: 0.08,
    9: 0.06,
    10: 0.05,
    11: 0.05,
    12: 0.16,
}

kids_monthly_weights = {
    1: 0.03,
    2: 0.04,
    3: 0.07,
    4: 0.09,
    5: 0.10,
    6: 0.08,
    7: 0.09,
    8: 0.08,
    9: 0.05,
    10: 0.04,
    11: 0.08,
    12: 0.25,
}

def assign_date(row):
    model = row["Bike_Model"]
    if model == "BMX":
        weights = kids_monthly_weights
    else:
        weights = adult_monthly_weights

    months = list(weights.keys())
    probs = list(weights.values())
    month = np.random.choice(months, p=probs)
    year = np.random.choice([2020, 2021, 2022, 2023, 2024])
    max_day = calendar.monthrange(year, month)[1]
    day = np.random.randint(1, max_day + 1)
    return f"{day:02d}-{month:02d}-{year}"

df["Date"] = df.apply(assign_date, axis=1)
print("  Dates assigned with realistic seasonal distribution")

# ============================================================
# STEP 2 - INJECT REALISTIC DIRT
# ============================================================
print("\nInjecting data quality issues...")
dirty = df.copy()

# 1. Missing values
for col, pct in [
    ("Customer_Age", 0.03),
    ("Price", 0.04),
    ("Customer_Gender", 0.025),
    ("Store_Location", 0.02),
    ("Payment_Method", 0.03)
]:
    idx = np.random.choice(dirty.index, size=int(len(dirty)*pct), replace=False)
    dirty.loc[idx, col] = np.nan

# 2. Gender variants
idx = dirty[dirty["Customer_Gender"]=="Male"].dropna().index
idx = np.random.choice(idx, size=800, replace=False)
dirty.loc[idx, "Customer_Gender"] = np.random.choice(
    ["M", "male", "MALE", "m"], size=800)
idx2 = dirty[dirty["Customer_Gender"]=="Female"].dropna().index
idx2 = np.random.choice(idx2, size=800, replace=False)
dirty.loc[idx2, "Customer_Gender"] = np.random.choice(
    ["F", "female", "FEMALE", "f"], size=800)

# 3. Bike model casing
idx = dirty[dirty["Bike_Model"]=="Mountain Bike"].index
idx = np.random.choice(idx, size=500, replace=False)
dirty.loc[idx, "Bike_Model"] = np.random.choice(
    ["mountain bike", "MOUNTAIN BIKE", "Mountain  Bike"], size=500)

# 4. Price outliers
idx = np.random.choice(dirty.index, size=150, replace=False)
dirty.loc[idx[:75], "Price"] = np.random.uniform(0.01, 5.00, size=75)
dirty.loc[idx[75:], "Price"] = np.random.uniform(50000, 99999, size=75)

# 5. Impossible ages
idx = np.random.choice(dirty.index, size=60, replace=False)
dirty.loc[idx, "Customer_Age"] = np.random.choice(
    [-1, 0, 5, 10, 150, 200], size=60)

# 6. Mixed date formats
idx = np.random.choice(dirty.index, size=2000, replace=False)
def mangle_date(d):
    try:
        parts = str(d).split("-")
        if len(parts) == 3:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
    except:
        pass
    return d
dirty.loc[idx, "Date"] = dirty.loc[idx, "Date"].apply(mangle_date)
idx2 = np.random.choice(dirty.index, size=1000, replace=False)
dirty.loc[idx2, "Date"] = dirty.loc[idx2, "Date"].apply(
    lambda d: str(d).replace("-", "."))

# 7. Duplicate rows
dupes = dirty.sample(n=200, random_state=42)
dirty = pd.concat([dirty, dupes], ignore_index=True)

# 8. Whitespace in store location
idx = np.random.choice(dirty.index, size=500, replace=False)
dirty.loc[idx, "Store_Location"] = dirty.loc[idx, "Store_Location"].apply(
    lambda x: f"  {x}  " if isinstance(x, str) else x)

# 9. Payment method variants
cash_idx = dirty[dirty["Payment_Method"]=="Cash"].dropna().index
idx = np.random.choice(cash_idx, size=min(300, len(cash_idx)), replace=False)
dirty.loc[idx, "Payment_Method"] = np.random.choice(
    ["cash", "CASH", "cash payment"], size=len(idx))

# 10. Negative quantities
idx = np.random.choice(dirty.index, size=80, replace=False)
dirty.loc[idx, "Quantity"] = -1

# ============================================================
# STEP 3 - SAVE
# ============================================================
dirty.to_csv("data/bike_sales_dirty.csv", index=False)

print(f"\n[OK] Done!")
print(f"Total rows: {len(dirty):,}")
print(f"Saved to: data/bike_sales_dirty.csv")
print(f"\nDirt injected:")
print(f"  Missing values:    ~14,500")
print(f"  Duplicate rows:    200")
print(f"  Gender variants:   1,600")
print(f"  Bike model casing: 500")
print(f"  Price outliers:    150")
print(f"  Impossible ages:   60")
print(f"  Mixed dates:       3,000")
print(f"  Whitespace:        500")
print(f"  Payment variants:  300")
print(f"  Negative qty:      80")