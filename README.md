# ☕ Dirty Café Sales — Data Cleaning & Visualization Project

This project focuses on transforming a messy café sales dataset into a clean, analyzable format and uncovering insights using **Python** 🐍.  
It demonstrates data cleaning, handling missing values, detecting invalid entries, and creating meaningful visualizations.

---

## 🚀 Project Overview

The **Dirty Café Sales** dataset initially contained:
- Missing values ❌  
- Invalid entries like `"ERROR"` and `"UNKNOWN"` ⚠️  
- Mixed data types and unformatted numbers 💥  

This project showcases how to:
- Clean and preprocess real-world messy data
- Visualize business trends
- Extract insights about sales, payment methods, and customer behavior

---

## 🧹 Data Cleaning Steps

| Step | Description |
|------|--------------|
| 🧱 **Loading** | Imported dataset using `pandas` |
| 🚫 **Invalid Values** | Replaced `"ERROR"` and `"UNKNOWN"` with `NaN` |
| 🧠 **Imputation** | Filled or dropped missing values using thresholds |
| 🔢 **Type Conversion** | Converted strings to numeric where needed |
| 💰 **Feature Engineering** | Created `Total Spent = Quantity × Price Per Unit` |
| 🧽 **Final Cleaning** | Removed duplicates and formatted columns |

---

## 📊 Exploratory Data Analysis (EDA)

Used **matplotlib** and **seaborn** to create visualizations:

- 🧁 **Top Selling Items**
- 💳 **Most Common Payment Methods**
- 📅 **Sales Trends by Date**
- 📍 **Location-based Performance**
- 💸 **Revenue Distribution**

These visuals help café management understand what sells, where, and when ☕💡

---

## 📈 Key Insights

- The café’s **top-selling items** drive majority of revenue  
- Certain **locations** underperform compared to others  
- **Payment method preferences** vary by location  
- Some products show **inconsistent pricing** — worth auditing  
- After cleaning, the dataset was **100% ready for analysis** and **machine learning**

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Libraries | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| IDE | Google Colab / Jupyter Notebook |

---

## 💻 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/dirty-cafe-sales.git
   cd dirty-cafe-sales
