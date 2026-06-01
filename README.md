# SCT_DS_2 - K-Means Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **SkillCraft Technology - Data Science Internship | Task 02**

---

## Task Description

Create a **K-Means Clustering algorithm** to group customers of a retail store based on their purchase history using the Mall Customers Dataset.

---

## Project Structure

```
SCT_DS_2/
│
├── Task2_KMeans.py              # Main Python script
├── mall_customers.csv           # Dataset
├── Task2_KMeans_Clustering.png  # Output visualization
└── README.md                    # Project documentation
```

---

## Dataset

**Mall Customer Segmentation Dataset**

| Column | Description |
|--------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Score assigned by the mall (1=low, 100=high) |

- **Rows:** 200 customers
- **Missing Values:** None
- **Source:** [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## Methodology

### 1. Data Preprocessing
- Loaded and explored the dataset
- Selected features: `Annual Income` and `Spending Score`
- Applied **StandardScaler** for feature normalization

### 2. Finding Optimal K
- **Elbow Method** - plotted WCSS (inertia) vs K (2-10)
- **Silhouette Score** - measured cluster separation quality
- Optimal **K = 5** identified from both methods

### 3. K-Means Model
- Algorithm: `KMeans` with `k-means++` initialization
- `n_init = 10` for stable results
- `random_state = 42` for reproducibility

---

## Results

**Silhouette Score: 0.56** (good cluster separation)

| Cluster | Count | Avg Age | Avg Income | Avg Score | Profile |
|---------|-------|---------|-----------|-----------|---------|
| C1 | 81 | 42.7 | $55k | 50 | Average Customers |
| C2 | 39 | 32.7 | $87k | 83 | VIP - High Income, High Spend |
| C3 | 22 | 25.3 | $26k | 79 | Impulsive Buyers |
| C4 | 35 | 41.1 | $88k | 16 | Savers - High Income, Low Spend |
| C5 | 23 | 45.2 | $26k | 21 | Budget-Conscious |

### Visualization Dashboard
The output PNG includes:
- Elbow Method plot
- Silhouette Score plot
- Gender Distribution pie chart
- Main Cluster Scatter (Income vs Score)
- Age Distribution by Cluster
- Cluster Sizes bar chart
- Avg Income & Score per Cluster
- Age vs Spending Score by Cluster

---

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Script
```bash
python Task2_KMeans.py
```

### 3. Output
- Console: dataset summary, cluster centroids, business interpretation
- File: `Task2_KMeans_Clustering.png` - full visualization dashboard

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | KMeans, StandardScaler, Silhouette Score |

---

## Key Learnings

- How to determine optimal K using Elbow Method and Silhouette Analysis
- Importance of feature scaling before clustering
- Business interpretation of customer segments for targeted marketing
- Using `k-means++` initialization for faster convergence

---
