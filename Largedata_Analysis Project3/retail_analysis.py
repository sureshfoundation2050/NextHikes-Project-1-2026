"""""
Retail Dataset — Univariate, Bivariate & Multivariate Analysis
================================================================
Dataset: retail_large_dataset.csv (100,000 orders)

Run this top to bottom. Each section is independent, so you can
run just the part you're studying.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style('whitegrid')
CHART_DIR = 'charts'

# ------------------------------------------------------------------
# STEP 0: Load & inspect
# ------------------------------------------------------------------
data = pd.read_csv('retail_large_dataset.csv')
print("Shape:", data.shape)
print(data.head())
print(data.describe())

# ==================================================================
# STEP 1: UNIVARIATE ANALYSIS  (one variable at a time)
# ==================================================================

# 1A. Histogram — Customer Age distribution (numerical)
plt.figure(figsize=(8, 5))
sns.histplot(data['age'], bins=30, kde=True, color='#B85042')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/1a_age_histogram.png', dpi=150)
plt.close()

# 1B. Histogram — Final Price distribution (numerical, likely skewed)
plt.figure(figsize=(8, 5))
sns.histplot(data['final_price'], bins=40, kde=True, color='#A7BEAE')
plt.title('Distribution of Final Order Price')
plt.xlabel('Final Price (₹)')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/1b_price_histogram.png', dpi=150)
plt.close()

# 1C. Count plot — Product Category (categorical)
plt.figure(figsize=(9, 5))
order = data['product_category'].value_counts().index
sns.countplot(y='product_category', data=data, order=order, color='#B85042')
plt.title('Number of Orders by Product Category')
plt.xlabel('Number of Orders')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/1c_category_countplot.png', dpi=150)
plt.close()

# 1D. Pie chart — Customer Segment share (categorical proportions)
plt.figure(figsize=(6, 6))
seg = data['customer_segment'].value_counts()
plt.pie(seg.values, labels=seg.index, autopct='%1.1f%%',
        colors=['#C4ABEA', '#A7BEAE', '#E7E8D1'])
plt.title('Customer Segment Share')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/1d_segment_piechart.png', dpi=150)
plt.close()

# ==================================================================
# STEP 2: BIVARIATE ANALYSIS  (two variables together)
# ==================================================================

# 2A. Categorical vs Numerical — Avg final price by product category
plt.figure(figsize=(9, 5))
avg_price = data.groupby('product_category')['final_price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price.values, y=avg_price.index, color='#B85042')
plt.title('Average Order Value by Product Category')
plt.xlabel('Average Final Price (₹)')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/2a_category_vs_price.png', dpi=150)
plt.close()

# 2B. Numerical vs Numerical — Discount % vs Final Price
plt.figure(figsize=(8, 5))
sample = data.sample(3000, random_state=42)  # sample for a readable scatter
sns.scatterplot(x='discount_percentage', y='final_price', data=sample,
                 alpha=0.4, color='#065A82')
plt.title('Discount % vs Final Price (sample of 3,000 orders)')
plt.xlabel('Discount (%)')
plt.ylabel('Final Price (₹)')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/2b_discount_vs_price.png', dpi=150)
plt.close()

# 2C. Categorical vs Categorical — Payment Method vs Return Status
plt.figure(figsize=(9, 5))
sns.countplot(x='payment_method', hue='return_status', data=data,
              palette=['#A7BEAE', '#B85042'])
plt.title('Returns by Payment Method')
plt.xlabel('Payment Method')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/2c_payment_vs_return.png', dpi=150)
plt.close()

# ==================================================================
# STEP 3: MULTIVARIATE ANALYSIS  (many variables at once)
# ==================================================================

# 3A. Correlation Heatmap — all numeric columns
plt.figure(figsize=(7, 5))
numeric_cols = ['age', 'product_price', 'quantity', 'discount_percentage',
                 'final_price', 'delivery_days']
corr = data[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap — Numeric Features')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/3a_correlation_heatmap.png', dpi=150)
plt.close()

# 3B. PCA — reduce numeric features to 2D, colored by customer_segment
scaled = StandardScaler().fit_transform(data[numeric_cols])
pca = PCA(n_components=2)
components = pca.fit_transform(scaled)
pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
pca_df['segment'] = data['customer_segment'].values
pca_sample = pca_df.sample(3000, random_state=42)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='segment', data=pca_sample,
                 palette=['#B85042', '#A7BEAE', '#065A82'], alpha=0.6)
plt.title(f'PCA of Numeric Features (explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/3b_pca_scatter.png', dpi=150)
plt.close()

print("\nAll charts saved to the 'charts/' folder.")
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
