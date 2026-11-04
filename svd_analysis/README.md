# DS701 Midterm Project: Master SVD & Clustering Pipeline

## Overview
This module contains the unified, end-to-end pipeline implementing **Singular Value Decomposition (SVD)** and **Unsupervised Clustering (Dendrogram / Ward Linkage and KMeans)** on the Bank Marketing dataset (`bank-additional-full.csv`). 

The pipeline strictly adheres to the course's honest validation protocol:
1. **Time-Based Frozen Partitioning:** Trained and cross-validated exclusively on rows 1–24,712 (development set); rows 24,713–41,188 remain frozen.
2. **Leakage Prevention:** Call `duration` and the campaign target `y` are strictly excluded from clustering and dimensionality reduction.
3. **No Supervised Black-Box Classifiers:** Uses strictly unsupervised SVD subspace projection and geometric clustering with **Prevalence-Lift (Affinity Ratio)** assignment to resolve minority mode collapse.
4. **Cost-Benefit Analysis (CBA):** Models real-world calling budget economics ($5.00/call, $100.00/conversion).

---

## Core Artifacts
- **[`master_pipeline.py`](master_pipeline.py):** Consolidated executable running SVD sweep, KMeans sweep, Dendrogram sweep, 5-fold CV evaluation, CBA, and visualization generation.
- **[`master_pipeline_dashboard.png`](master_pipeline_dashboard.png):** 4-panel publication-ready dashboard:
  - **Panel A:** SVD Dimensionality Regularization & Cophenetic Tree Correlation ($k=5$ sweet spot).
  - **Panel B:** Multi-Tier Clustering Generalization & Recall Curves across $K \in [4 \dots 20]$.
  - **Panel C:** Cost-Benefit Analysis Bar Chart ($5/call, $100/conversion) comparing Random vs. Dendrogram ($K=10, 15, 20$).
  - **Panel D:** Cumulative Net Profit vs. Contact Volume Budget Curves.
- **[`master_cba_summary.csv`](master_cba_summary.csv):** Detailed financial metrics across models.
- **[`master_clustering_sweep.csv`](master_clustering_sweep.csv):** 5-fold cross-validation metrics across $K$ and algorithms.
- **[`master_svd_sweep.csv`](master_svd_sweep.csv):** Frobenius reconstruction error, variance explained, and cophenetic correlation across $k$.
- **`archive/`:** Archived exploratory scripts and early prototype plots.

---

## Key Results Summary

### 1. SVD Regularization ($k=5$)
- Captures **61.4%** of data variance while reducing dimensionality from 58 to 5 dimensions.
- Increases Cophenetic Tree Fidelity from **0.487** (raw noisy one-hot space) to **0.625** (**+28.6% gain**).

### 2. Multi-Tier Generalization via Prevalence-Lift
- Predicts all **4 socioeconomic tiers** without mode collapse:
  - **Life-Stage (Students & Retirees):** **50.8% to 62.6% Recall**.
  - **Independent/Biz (Entrepreneurs & Self-Employed):** **17.3% to 22.6% Recall**.
  - **Macro F1:** Boosted from **0.1312** (majority baseline) to **0.2381–0.2489** (**+81% to +90% lift**).

### 3. Cost-Benefit Analysis ($5/call, $100/conversion)
At a conversion payout of $100, the required break-even conversion rate is $5 / $100 = 5.00\%$. Because the population base rate is only $4.81\%$, **random calling is unprofitable**:
- **Random Calling (150 calls):** 7.21 conversions $\to$ **-$28.89 Net Loss** (-3.9% ROI).
- **Dendrogram Prioritization ($K=15$):** 40 conversions $\to$ **+$3,250.00 Net Profit** (+433% ROI, **+$3,279 benefit over random**).
- **Dendrogram Prioritization ($K=20$):** 50 conversions $\to$ **+$4,250.00 Net Profit** (+567% ROI, **+$4,279 benefit over random**).
