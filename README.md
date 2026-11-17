# 4-descriptive-statistics-probability
Foundations of descriptive statistics and probability for machine learning and data science

---

# Telecom Customer Churn Analysis: Descriptive Statistics & Probability Modeling

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

## Executive Summary

This repository contains an exploratory data analysis (EDA) and statistical inference pipeline designed to diagnose customer attrition (churn) drivers using a synthetic telecom dataset (`n = 1,000`). Rather than relying on brute-force outbound retention campaigns (such as randomly calling 200 customers), this workflow applies **descriptive statistics**, **distribution profiling**, and **conditional probability** to isolate high-risk customer segments and maximize retention ROI.

---

## Tech Stack & Dependencies

The analysis relies on core scientific computing and visualization libraries in Python:

* **NumPy (`np.random.Generator`):** Used for reproducible synthetic data generation via PCG64 pseudo-random number generators (`default_rng`), vectorized conditional logic (`np.where`), and sampling from theoretical probability distributions (Normal, Gamma, and Poisson).
* **Pandas (`pd.DataFrame`):** Used for tabular data manipulation, group-by aggregations, boolean masking, and computing summary statistics.
* **Matplotlib (`matplotlib.pyplot`):** Used for building custom, publication-grade visualization wrappers (`show_hist`, `compare_hist`, `show_bars`, `show_box`) with clean styling (despined axes and baseline reference annotations).

---

## Synthetic Data Generation Architecture

The dataset simulates an Airtel-style subscriber base (`1,000` records, `5` features) governed by realistic telecom domain heuristics:

| Feature | Data Type | Generative Distribution / Logic | Domain Interpretation |
| :--- | :--- | :--- | :--- |
| `tenure_months` | `int64` | Bimodal mixture: Discrete Uniform distribution between `1` and `12` months for new cohorts; Normal distribution (mean `54`, standard deviation `9`) clipped between `24` and `72` months for tenured cohorts. | Captures early-lifecycle churn risk versus long-term subscriber loyalty. |
| `monthly_charges` | `float64` | Contract-dependent mixture: Shifted Gamma distribution (`20` base + shape `2.0`, scale `22`) for Month-to-month plans; Normal distributions with mean `64` (std `9`) for 1-year plans and mean `78` (std `11`) for 2-year plans. | Models right-skewed overage and add-on billing in flexible plans versus predictable flat-rate billing in long-term contracts. |
| `contract` | `object` | Categorical multinomial draw with probabilities `0.55` (`Month-to-month`), `0.21` (`One year`), and `0.24` (`Two year`). | Primary structural lock-in mechanism. |
| `support_calls` | `int64` | Poisson distribution with an average arrival rate (`lambda`) of `1.3` calls per subscriber. | Discrete count data representing customer friction and service tickets. |
| `churned` | `int64` | Bernoulli trial where churn probability scales with contract type, capped support calls (`min(calls, 4)`), and early tenure (less than `12` months). | Binary target variable (`1` = Churned, `0` = Retained). |

---

## Core Concepts Demonstrated

### 1. Descriptive Statistics & Distributional Geometry
* **Measures of Central Tendency & Skewness:**
  * **Mean vs. Median:** Demonstrated how extreme values pull the arithmetic mean away from the center of mass. For `tenure_months`, the mean (`34.53` months) is substantially lower than the median (`45.0` months), revealing a **left-skewed (negatively skewed) bimodal distribution** driven by a large cohort of short-tenure subscribers.
  * **Mode:** Identified the most frequent contract category (`Month-to-month`, representing `55.2%` of the user base) to establish baseline exposure.
* **Measures of Dispersion (Spread):**
  * **Variance & Standard Deviation:** Contrasted `Month-to-month` versus `One year` subscribers. While both groups exhibit nearly identical average monthly billing (`$63.21` vs. `$63.57`), their volatility differs drastically (standard deviation of `30.05` vs. `9.10`). Relying solely on average metrics masks critical pricing variance across cohorts.
* **Non-Parametric Quartile Analysis (Box Plots):**
  * Utilized **Interquartile Range (`IQR = 75th percentile - 25th percentile`)** and whisker boundaries to compare distribution symmetry, median shifts, and upper-tail outliers across contract tiers without assuming a normal distribution.
* **Z-Score Standardization & Anomaly Detection:**
  * Applied standard scaling to transform raw billing amounts into unitless standard deviations from the mean:
    ```text
    Z-Score = (Observed Value - Mean) / Standard Deviation
    ```
  * Flagged records with an absolute Z-score of `2.0` or higher (`40` customers), isolating extreme billing outliers that warrant targeted retention monitoring or billing audits.

### 2. Probability Theory & Risk Stratification
* **Empirical Marginal & Complement Probabilities:**
  * Computed the baseline churn probability (`0.28`, or `28%` of total subscribers) and its complement, the retention probability (`1 - 0.28 = 0.72`, or `72%`).
  * *Engineering Takeaway:* A naive random sample of `200` outbound retention calls yields an expected value of `56` actual at-risk customers (`200 * 0.28`), wasting `72%` of outreach capacity on customers who would have stayed anyway.
* **Joint Probability & The Addition Rule (Overlapping Risk Events):**
  * Evaluated overlapping risk factors—subscribers on Month-to-month plans (`Event A`) and high-friction subscribers with `3` or more support calls (`Event B`):
    ```text
    P(A or B) = P(A) + P(B) - P(A and B)
    ```
  * Subtracting the joint overlap (`P(A and B) = 0.077`) prevents double-counting high-risk users when sizing the total addressable retention cohort (`P(A or B) = 0.632`, or `63.2%` of the base).
* **Conditional Probability & Lift Analysis:**
  * Quantified risk concentration by conditioning churn rates on contract tier:
    ```text
    P(Churn given Contract) = P(Churn and Contract) / P(Contract)
    ```
  * Stratifying by contract tier exposes massive divergence from the `0.280` global base rate:
    * **Month-to-month:** `0.404` churn probability (`1.44x` lift over baseline)
    * **One year:** `0.205` churn probability
    * **Two year:** `0.056` churn probability (`5x` reduction in churn risk compared to baseline)

---

## Code Annotations & Engineering Notes

If you are extending this notebook into a production machine learning pipeline, pay close attention to the following implementation details:

1. **Vectorized Simulation (`np.where` & `default_rng`):**
   * The `load_airtel` helper avoids slow Python `for`-loops by using nested `np.where` broadcasting and NumPy's modern `Generator` API (`np.random.default_rng(seed=7)`), ensuring deterministic, thread-safe dataset generation.
2. **Modular Visualization Wrappers (`_axes`, `show_hist`, `show_bars`):**
   * Plotting boilerplate (spine removal, grid ordering, dynamic bar annotations, baseline horizontal reference lines) is abstracted into reusable functions. This enforces visual consistency across exploratory cells and keeps analytical cells declarative.
3. **Sample vs. Population Standard Deviation (`ddof`):**
   * Note that `pd.Series.std()` defaults to Bessel's correction (`ddof=1`, sample standard deviation), whereas `np.std()` defaults to `ddof=0` (population standard deviation). For `n = 1,000`, the difference is negligible (roughly `0.05%`), though consistency should be maintained when building custom Scikit-Learn transformers (`StandardScaler` uses `ddof=0`).
4. **Tail Behavior in Z-Score Thresholding:**
   * Because `Month-to-month` charges follow a right-skewed Gamma distribution rather than a symmetric bell curve, the `2.0` standard deviation threshold exclusively captures upper-tail anomalies (`Z >= +2.0`) rather than symmetric two-tailed outliers. For heavily skewed features in production, consider **Modified Z-Scores (using Median Absolute Deviation)** or **IQR fencing (`75th percentile + 1.5 * IQR`)**.

---

## Quickstart & Execution

1. **Clone the repository and install dependencies:**
   ```bash
   pip install numpy pandas matplotlib jupyter
