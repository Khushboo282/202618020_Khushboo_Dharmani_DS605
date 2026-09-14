"""Task 1: EDA + Cleaning + Feature Engineering"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('/home/claude/airbnb-price-prediction/data/AB_NYC_2019.csv')

# ---------- FIG 1: Raw price distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['price'], bins=100, ax=axes[0], color='#2b6cb0')
axes[0].set_title('Raw Price Distribution (right-skewed)')
axes[0].set_xlim(0, 1000)
sns.histplot(np.log1p(df['price']), bins=60, ax=axes[1], color='#2f855a')
axes[1].set_title('log1p(Price) Distribution (near-normal)')
plt.tight_layout()
plt.savefig('/home/claude/airbnb-price-prediction/assets/fig1_price_distribution.png', bbox_inches='tight')
plt.close()

# ---------- FIG 2: Price by room_type and neighbourhood_group ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=df[df['price']<400], x='room_type', y='price', ax=axes[0], palette='Set2', hue='room_type', legend=False)
axes[0].set_title('Price by Room Type (capped at $400)')
order = df.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df[df['price']<400], x='neighbourhood_group', y='price', order=order, ax=axes[1], palette='Set3', hue='neighbourhood_group', legend=False)
axes[1].set_title('Price by Borough (capped at $400)')
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('/home/claude/airbnb-price-prediction/assets/fig2_price_by_category.png', bbox_inches='tight')
plt.close()

# ---------- FIG 3: Geographic scatter colored by price ----------
fig, ax = plt.subplots(figsize=(7, 7))
sample = df[df['price']<500].sample(15000, random_state=42)
sc = ax.scatter(sample['longitude'], sample['latitude'], c=sample['price'], cmap='viridis', s=4, alpha=0.6)
plt.colorbar(sc, label='Price ($)')
ax.set_title('Listing Locations Colored by Price')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('/home/claude/airbnb-price-prediction/assets/fig3_geo_price.png', bbox_inches='tight')
plt.close()

# ---------- FIG 4: Correlation heatmap (numeric, post feature engineering will redo) ----------
num_cols = ['latitude','longitude','minimum_nights','number_of_reviews','reviews_per_month',
            'calculated_host_listings_count','availability_365','price']
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Heatmap (Raw Numeric Features)')
plt.tight_layout()
plt.savefig('/home/claude/airbnb-price-prediction/assets/fig4_correlation_raw.png', bbox_inches='tight')
plt.close()

print("EDA figures saved.")
print(df.isnull().sum())
