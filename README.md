# 🚀 Multi-Source Alpha Research
### Momentum × Sentiment × Options Conditioning

---

## 📌 Project Overview

This repository contains a **systematic equity research project** focused on building and validating a **multi-source conditional alpha** using U.S. equities.

The goal is to study how:

- 📈 **Price momentum**
- 🧠 **Expectations-based sentiment**
- 📉 **Options-implied market information**

interact to produce **robust, risk-adjusted return signals**.

Rather than treating individual factors as standalone predictors, this project emphasizes **conditional factor behavior**, motivated by empirical findings that raw momentum exhibits **non-linear (U-shaped)** return profiles across the cross-section.

---

## 🎯 Research Motivation

Classic cross-sectional momentum is well-documented but suffers from:

- ❌ Non-monotonic decile returns  
- ⚠️ Regime-dependent crashes  
- 📊 Sensitivity to crowded trades  

This project explores whether momentum performance can be improved by **conditioning on expectation shifts**, captured via:

- 🧮 Earnings-based sentiment (**Tier 1**)  
- 🗂️ Management disclosure tone (**Tier 2**)  
- 🧾 Options-implied skew (**planned**)  

> **Guiding hypothesis:**  
> Expectation alignment determines whether momentum persists or mean-reverts.

---

## 🧠 Methodology

- Cross-sectional factor construction  
- Forward return alignment (no look-ahead bias)  
- Decile and Information Coefficient (IC) tests  
- Emphasis on robustness, interpretability, and low overfitting  

---

## 🗂️ Project Structure

```text
multi_source_alpha/
├─ signals/
│  ├─ momentum/
│  ├─ sentiment/
│  └─ options/
├─ research/
├─ backtests/
├─ utils/
└─ README.md 