Real Estate Price Predictor — Gurgaon Market Analysis

An end-to-end real estate analytics project covering exploratory data analysis on 14,223 property listings across Gurgaon, followed by a machine learning Automated Valuation Model (AVM) deployed as a live Streamlit web app.

** Live App:** https://realestate-eda-xpm9nejc9anm7ouvvjzgyk.streamlit.app/  
** Dataset:** 14,223 Gurgaon property listings · 12 features  
** Stack:** Python · Pandas · NumPy · Scikit-learn · Streamlit

<img width="1908" height="938" alt="image" src="https://github.com/user-attachments/assets/68e0f124-7e9b-4cd9-afb3-dfff7235cb73" />



## Problem

Real estate pricing in Gurgaon is:
- Inconsistent across sectors and builders
- Non-transparent — same sector shows 10x price variance
- Hard to evaluate without local market knowledge

This makes it difficult for buyers and investors to make informed decisions without spending weeks researching listings manually.



## Solution

Built an Automated Valuation Model (AVM) that:
- Predicts property prices based on area, location, BHK, builder and status
- Calculates price per sqft for fair comparison across listings
- Classifies properties as Budget / Mid / Premium
- Provides a simple web interface for instant price estimates


## Exploratory Data Analysis

### Dataset Overview

| Metric | Value |

Total listings - 14,223 |
 Price range - ₹0.01Cr – ₹122.63Cr |
 Median price - ₹2.62Cr |
 Mean price - ₹4.00Cr |
 Median area - 2,015 sq ft |
 Median rate per sqft - ₹12,380 |
 Localities covered - 50+ Gurgaon sectors |
 Builders covered - 100+ including M3M, DLF, Godrej, Sobha |



### Finding 1 — Property type is heavily skewed toward apartments

| Type | Count | Share |

 Apartment - 10,894 - 76.6% |
 Plot - 2,166 - 15.2% |
 Floor - 881 - 6.2% |
 Villa - 205 - 1.4% |
 Penthouse - 32 - 0.2% |

**Insight:** Apartments dominate the Gurgaon market at 76.6% of all listings. Villas and penthouses together make up under 2% — this is a mid-to-premium apartment market, not a villa market.

---

### Finding 2 — 3 BHK and 4 BHK together represent 70% of all listings

| BHK | Count | Avg Price |
|---|---|---|
| 1 BHK | 251 | ₹1.29Cr |
| 2 BHK | 2,337 | ₹1.57Cr |
| 3 BHK | 6,150 | ₹3.08Cr |
| 4 BHK | 3,723 | ₹5.82Cr |
| 5 BHK | 587 | ₹12.17Cr |
| 6 BHK | 89 | ₹16.91Cr |

**Insight:** 3 BHK is the dominant unit type (43.2% of all listings). The jump from 4 BHK (₹5.82Cr avg) to 5 BHK (₹12.17Cr avg) is a 109% price increase — the luxury segment starts abruptly at 5 BHK. The dataset also contains dirty BHK values (99 BHK, 114 BHK) which were identified during EDA as data entry errors and handled in cleaning.

---

### Finding 3 — Sector 42 commands the highest rate per sqft by a large margin

| Locality | Avg Rate/sqft | Listings |
|---|---|---|
| Sector 42 | ₹55,989 | 60 |
| Sector 113 | ₹52,404 | 454 |
| Sector 53 | ₹29,203 | 133 |
| Sector 54 | ₹28,109 | 287 |
| Sector 57 | ₹23,206 | 202 |
| Sector 62 | ₹19,255 | 221 |

**Insight:** Sector 42 and Sector 113 command a 2–3x rate premium over other sectors. Sector 113 is especially notable — 454 listings at ₹52,404/sqft indicates a high-volume premium zone, not just a handful of outliers.

---

### Finding 4 — Ready-to-move properties command a price premium over under-construction

| Status | Listings | Avg Price |
|---|---|---|
| Ready to move | 7,823 (55.0%) | ₹4.15Cr |
| Under construction | 5,269 (37.0%) | ₹3.96Cr |
| New launch | 834 (5.9%) | ₹2.43Cr |
| Resale | 297 (2.1%) | ₹4.94Cr |

**Insight:** Resale properties (₹4.94Cr avg) are the most expensive segment — buyers pay a premium for proven, occupied properties. New launches (₹2.43Cr) are priced 41% below ready-to-move to compensate for construction risk and waiting time.

---

### Finding 5 — RERA approval does NOT command a price premium (counter-intuitive)

| RERA Status | Listings | Avg Price |
|---|---|---|
| RERA Approved | 5,449 (38.3%) | ₹3.86Cr |
| Not RERA Approved | 8,774 (61.7%) | ₹4.08Cr |

**Insight:** RERA-approved properties are actually priced 5.4% *lower* on average than non-RERA properties. This is counter-intuitive — RERA certification signals consumer protection and builder accountability, yet the market doesn't price it in as a premium. This likely reflects that premium luxury builders in Gurgaon operate under direct buyer trust, bypassing RERA registration. This finding directly challenges the assumption that regulatory compliance drives valuation.

---

### Finding 6 — Top builders dominate listing volume

| Builder | Listings |
|---|---|
| M3M | 1,845 |
| Godrej | 790 |
| DLF | 710 |
| Signature Global | 633 |
| Krisumi | 386 |
| Ireo | 380 |
| Sobha | 356 |
| BPTP | 344 |

**Insight:** M3M alone accounts for 13% of all listings — indicating heavy market concentration. The top 8 builders represent ~45% of all listings. This means Gurgaon's real estate market is dominated by a handful of large developers, not fragmented individual sellers.

---

### Data Quality Issues Found During EDA

| Issue | Details | Fix Applied |
|---|---|---|
 Dirty BHK values | 99, 114, 132 BHK entries found — impossible values, likely data entry errors | Flagged and excluded from BHK analysis |
 Extreme area outliers | Max area = 9,58,320 sq ft — clearly a plot/commercial entry, not residential | Filtered using IQR-based outlier detection |
 Price outliers | Min price = ₹0.01Cr — likely missing/erroneous entries | Removed records below ₹5L threshold |
 Rate/sqft outliers | Max ₹3,10,000/sqft — extreme outlier vs median ₹12,380 | Capped at 99th percentile for modelling |
 61.7% non-RERA | Majority of listings lack RERA data — affects regulatory analysis | Retained as a categorical feature |

---

## Machine Learning Model

### Approach
Built a **Random Forest Regressor** to predict property price from:
- Area (sq ft)
- Locality (sector)
- BHK count
- Builder / company name
- Property status (ready / under construction / resale)
- RERA approval status
- Flat type (apartment / villa / plot)

### Model Performance

| Metric | Value |
|---|---|
 R² Score | 0.85 |
 Interpretation | Model explains 85% of price variance |

### Feature Engineering
- Log-transformed price (target) to handle right-skewed distribution
- Label-encoded categorical features (locality, builder, status)
- Rate per sqft derived as `price / area`
- Property classification (Budget <₹1.5Cr / Mid ₹1.5–5Cr / Premium >₹5Cr)

---

## Streamlit Web App

**Features:**
- Real-time price prediction from user inputs
- Price per sqft calculation
- Budget / Mid / Premium classification
- Clean, interactive UI deployable on any device

**Live:** https://realestate-eda-xpm9nejc9anm7ouvvjzgyk.streamlit.app/

---

## How to Run Locally

```bash
git clone https://github.com/godshelwarrohit/Real_Estate-EDA
cd Real_Estate-EDA
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Takeaways

| # | Finding |
|---|---|
| 1 | Sector 42 and 113 command 2–3x the rate per sqft of other Gurgaon sectors |
| 2 | Resale properties are priced 19% above ready-to-move and 103% above new launches |
| 3 | RERA approval does not command a price premium — counter to common assumption |
| 4 | 3 BHK dominates at 43% of all listings — the core Gurgaon buyer segment |
| 5 | M3M alone holds 13% of all listings — market is builder-concentrated not fragmented |
| 6 | 5 BHK+ listings show a 109% price jump from 4 BHK — luxury starts abruptly |

---

## Author

**Rohit Godshelwar**  
 Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-learn · Streamlit  
