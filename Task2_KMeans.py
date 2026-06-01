"""
Task 02 - K-Means Clustering on Mall Customer Dataset
SkillCraft Technology Internship - Data Science Track
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────
df = pd.read_csv('mall_customers.csv')
print("=" * 60)
print("TASK 02 - K-Means Customer Segmentation")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────
# 2. Feature Selection
# ─────────────────────────────────────────
# Use Annual Income & Spending Score (classic 2D clustering)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 3. Find Optimal K using Elbow Method + Silhouette
# ─────────────────────────────────────────
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# ─────────────────────────────────────────
# 4. Final Model with K=5
# ─────────────────────────────────────────
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster labels (business interpretation)
cluster_names = {
    0: 'High Income\nLow Spenders',
    1: 'Low Income\nLow Spenders',
    2: 'Average Income\nAverage Spenders',
    3: 'Low Income\nHigh Spenders',
    4: 'High Income\nHigh Spenders'
}

# Map cluster numbers based on centroid analysis
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids_original, columns=['Income', 'Score'])
centroid_df['Cluster'] = range(optimal_k)

print("\n\nCluster Centroids (Original Scale):")
print(centroid_df.to_string(index=False))

# ─────────────────────────────────────────
# 5. Visualizations — all in one figure
# ─────────────────────────────────────────
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#1a1a2e')

# --- Plot 1: Elbow Method ---
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_facecolor('#16213e')
ax1.plot(K_range, inertia, 'o-', color='#E74C3C', linewidth=2.5, markersize=8)
ax1.axvline(x=5, color='#F39C12', linestyle='--', linewidth=2, label='Optimal K=5')
ax1.set_title('Elbow Method', color='white', fontsize=13, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)', color='#aaa')
ax1.set_ylabel('Inertia (WCSS)', color='#aaa')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#16213e', labelcolor='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 2: Silhouette Score ---
ax2 = fig.add_subplot(3, 3, 2)
ax2.set_facecolor('#16213e')
ax2.plot(K_range, silhouette_scores, 's-', color='#2ECC71', linewidth=2.5, markersize=8)
ax2.axvline(x=5, color='#F39C12', linestyle='--', linewidth=2, label='Optimal K=5')
ax2.set_title('Silhouette Score', color='white', fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)', color='#aaa')
ax2.set_ylabel('Score', color='#aaa')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#16213e', labelcolor='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 3: Gender Distribution ---
ax3 = fig.add_subplot(3, 3, 3)
ax3.set_facecolor('#16213e')
gender_counts = df['Gender'].value_counts()
wedges, texts, autotexts = ax3.pie(
    gender_counts, labels=gender_counts.index,
    autopct='%1.1f%%', colors=['#9B59B6', '#3498DB'],
    textprops={'color': 'white', 'fontsize': 11}
)
ax3.set_title('Gender Distribution', color='white', fontsize=13, fontweight='bold')

# --- Plot 4: Main Cluster Scatter (Income vs Score) ---
ax4 = fig.add_subplot(3, 3, (4, 5))
ax4.set_facecolor('#16213e')
for i in range(optimal_k):
    mask = df['Cluster'] == i
    ax4.scatter(
        df.loc[mask, 'Annual Income (k$)'],
        df.loc[mask, 'Spending Score (1-100)'],
        c=colors[i], s=80, alpha=0.85,
        label=f'Cluster {i+1}', edgecolors='white', linewidths=0.4
    )
# Plot centroids
ax4.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    c='white', s=250, marker='*', zorder=5, label='Centroids'
)
ax4.set_title('K-Means Clustering: Annual Income vs Spending Score',
              color='white', fontsize=13, fontweight='bold')
ax4.set_xlabel('Annual Income (k$)', color='#aaa', fontsize=11)
ax4.set_ylabel('Spending Score (1-100)', color='#aaa', fontsize=11)
ax4.tick_params(colors='white')
ax4.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
for spine in ax4.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 5: Age Distribution by Cluster ---
ax5 = fig.add_subplot(3, 3, 6)
ax5.set_facecolor('#16213e')
for i in range(optimal_k):
    mask = df['Cluster'] == i
    ax5.hist(df.loc[mask, 'Age'], bins=10, color=colors[i],
             alpha=0.7, label=f'C{i+1}')
ax5.set_title('Age Distribution by Cluster', color='white', fontsize=13, fontweight='bold')
ax5.set_xlabel('Age', color='#aaa')
ax5.set_ylabel('Frequency', color='#aaa')
ax5.tick_params(colors='white')
ax5.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
for spine in ax5.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 6: Cluster Size Bar Chart ---
ax6 = fig.add_subplot(3, 3, 7)
ax6.set_facecolor('#16213e')
cluster_sizes = df['Cluster'].value_counts().sort_index()
bars = ax6.bar([f'C{i+1}' for i in cluster_sizes.index],
               cluster_sizes.values, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, cluster_sizes.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', va='bottom', color='white', fontsize=10)
ax6.set_title('Cluster Sizes', color='white', fontsize=13, fontweight='bold')
ax6.set_xlabel('Cluster', color='#aaa')
ax6.set_ylabel('Count', color='#aaa')
ax6.tick_params(colors='white')
for spine in ax6.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 7: Avg Income & Score per Cluster ---
ax7 = fig.add_subplot(3, 3, 8)
ax7.set_facecolor('#16213e')
cluster_means = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
x = np.arange(optimal_k)
width = 0.35
b1 = ax7.bar(x - width/2, cluster_means['Annual Income (k$)'],
             width, label='Avg Income', color='#3498DB', alpha=0.85)
b2 = ax7.bar(x + width/2, cluster_means['Spending Score (1-100)'],
             width, label='Avg Score', color='#E74C3C', alpha=0.85)
ax7.set_title('Avg Income & Score per Cluster', color='white', fontsize=13, fontweight='bold')
ax7.set_xlabel('Cluster', color='#aaa')
ax7.set_xticks(x)
ax7.set_xticklabels([f'C{i+1}' for i in range(optimal_k)], color='white')
ax7.tick_params(colors='white')
ax7.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
for spine in ax7.spines.values():
    spine.set_edgecolor('#444')

# --- Plot 8: Scatter Age vs Score colored by Cluster ---
ax8 = fig.add_subplot(3, 3, 9)
ax8.set_facecolor('#16213e')
for i in range(optimal_k):
    mask = df['Cluster'] == i
    ax8.scatter(df.loc[mask, 'Age'],
                df.loc[mask, 'Spending Score (1-100)'],
                c=colors[i], s=60, alpha=0.8,
                label=f'C{i+1}', edgecolors='white', linewidths=0.3)
ax8.set_title('Age vs Spending Score by Cluster', color='white', fontsize=13, fontweight='bold')
ax8.set_xlabel('Age', color='#aaa')
ax8.set_ylabel('Spending Score', color='#aaa')
ax8.tick_params(colors='white')
ax8.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
for spine in ax8.spines.values():
    spine.set_edgecolor('#444')

plt.suptitle('K-Means Customer Segmentation — Mall Customers Dataset',
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Task2_KMeans_Clustering.png',
            dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print("\nVisualization saved!")

# ─────────────────────────────────────────
# 6. Print Cluster Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)
summary = df.groupby('Cluster').agg(
    Count=('CustomerID', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income=('Annual Income (k$)', 'mean'),
    Avg_Score=('Spending Score (1-100)', 'mean')
).round(1)
print(summary)

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)
interps = {
    0: "High Income, Low Spenders → Cautious / Savers",
    1: "Low Income, Low Spenders → Budget-Conscious",
    2: "Average Income, Average Spenders → Typical Customers",
    3: "Low Income, High Spenders → Impulsive Buyers",
    4: "High Income, High Spenders → Target VIP Customers",
}
for k, v in interps.items():
    print(f"  Cluster {k+1}: {v}")

print(f"\nSilhouette Score (K=5): {silhouette_score(X_scaled, df['Cluster']):.4f}")
print("\n✅ Task 02 Complete!")
