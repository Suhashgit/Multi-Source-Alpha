Multi-Source Alpha Research



Momentum × Sentiment × Options Conditioning



Project Overview



This repository contains a systematic equity research project focused on building and validating a multi-source conditional alpha using U.S. equities. The goal is to study how price momentum, expectations-based sentiment, and options-implied market information interact to produce robust, risk-adjusted return signals.



Rather than treating individual factors as standalone predictors, this project emphasizes conditional factor behavior, motivated by empirical findings that raw momentum exhibits non-linear (U-shaped) return profiles across the cross-section.



Research Motivation



Classic cross-sectional momentum is well-documented but suffers from:



non-monotonic decile returns



regime-dependent crashes



sensitivity to crowded trades



This project explores whether momentum performance can be improved by conditioning on expectation shifts, captured via:



earnings-based sentiment (Tier 1)



management disclosure tone (Tier 2)



options-implied skew (planned)



The guiding hypothesis is that expectation alignment determines whether momentum persists or mean-reverts.



Current Components

1\. Price Momentum



6–12 month cross-sectional momentum (skip-month)



Daily z-scoring across S\&P 500 constituents



Validated using:



decile return tests



information coefficient (IC) analysis



Observed U-shaped return profile consistent with real-world equity data



2\. Earnings-Based Sentiment (Tier 1 – in progress)



Expectation-based sentiment using earnings surprises



Relative analysis (change vs prior expectations)



Designed as a slow-moving conditioning signal



3\. Filing Tone Sentiment (Tier 2 – planned)



NLP-based tone change from SEC filings (10-Q / 10-K)



Relative to firm-specific historical baselines



Used as confirmation / veto rather than a standalone predictor



4\. Options-Implied Metrics (planned)



Implied volatility skew and variance risk premium



Market-implied tail risk and crowding signals



Used for risk-filtering and position sizing



Methodology



Cross-sectional factor construction



Forward return alignment (no look-ahead bias)



Decile analysis and IC testing



Emphasis on interpretability, robustness, and low overfitting



Modular signal design to enable controlled experimentation



Project Structure

multi\_source\_alpha/

├─ signals/

│  ├─ momentum/

│  ├─ sentiment/

│  └─ options/

├─ research/

├─ backtests/

├─ utils/

└─ README.md



Status



This project is under active development.

Current focus is on implementing and validating expectation-based sentiment signals and integrating them into a conditional momentum framework.



Disclaimer



This repository is for research and educational purposes only.

It does not constitute investment advice or a trading recommendation.

