# Script to generate EDA visualization figures

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from preprocess import load_and_clean_data, NYC_MIDTOWN_LAT, NYC_MIDTOWN_LON, haversine_distance

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

FIGURES_DIR = os.path.join('reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_eda_figures():
    print("Loading raw dataset for EDA figures...")
    raw_df = pd.read_csv('AB_NYC_2019.csv')
    df = load_and_clean_data(filter_outliers=True)

    # 1. Price Distribution: Raw vs Log-transformed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(raw_df[raw_df['price'] <= 1000]['price'], kde=True, ax=axes[0], color='#2b5c8f', bins=50)
    axes[0].set_title('Raw Price Distribution (<= $1,000)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Price ($ / night)', fontsize=11)
    axes[0].set_ylabel('Listing Count', fontsize=11)

    sns.histplot(df['log_price'], kde=True, ax=axes[1], color='#e06666', bins=50)
    axes[1].set_title('Log-Transformed Price Distribution log(1 + Price)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('log(1 + Price)', fontsize=11)
    axes[1].set_ylabel('Listing Count', fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'price_distribution.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # 2. Room Type vs Price Boxplot
    plt.figure(figsize=(9, 6))
    order = ['Entire home/apt', 'Private room', 'Shared room']
    palette = {'Entire home/apt': '#3498db', 'Private room': '#2ecc71', 'Shared room': '#e67e22'}
    sns.boxplot(x='room_type', y='price', data=df, order=order, palette=palette, showfliers=False)
    plt.title('Nightly Price by Room Type (Excluding Outliers for Readability)', fontsize=13, fontweight='bold')
    plt.xlabel('Room Type', fontsize=11)
    plt.ylabel('Price ($)', fontsize=11)
    # Add median annotations
    medians = df.groupby('room_type')['price'].median()
    for i, room in enumerate(order):
        med = medians[room]
        plt.text(i, med + 5, f"Median: ${med:.0f}", horizontalalignment='center', fontweight='bold', color='#111111')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'room_type_prices.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # 3. Price by Neighbourhood Group (Borough)
    plt.figure(figsize=(10, 6))
    borough_stats = df.groupby('neighbourhood_group')['price'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    sns.barplot(x=borough_stats.index, y=borough_stats['mean'], palette='Blues_r')
    plt.title('Mean Price by Borough (Neighbourhood Group)', fontsize=13, fontweight='bold')
    plt.xlabel('Borough', fontsize=11)
    plt.ylabel('Average Price ($ / night)', fontsize=11)
    for i, (borough, row) in enumerate(borough_stats.iterrows()):
        plt.text(i, row['mean'] + 3, f"${row['mean']:.1f}\n(n={int(row['count'])})",
                 horizontalalignment='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'borough_prices.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # 4. Correlation Matrix Heatmap
    df_corr = df[['price', 'log_price', 'latitude', 'longitude', 'minimum_nights', 
                  'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']].copy()
    df_corr['dist_to_midtown'] = haversine_distance(df['latitude'], df['longitude'], NYC_MIDTOWN_LAT, NYC_MIDTOWN_LON)
    
    corr = df_corr.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='vlag', vmin=-0.5, vmax=0.5, square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'correlation_matrix.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # 5. Geospatial Map Overlay with New_York_City_.png
    if os.path.exists('New_York_City_.png'):
        print("Generating geospatial map overlay on NYC map image...")
        try:
            nyc_img = Image.open('New_York_City_.png')
            fig, ax = plt.subplots(figsize=(12, 10))
            # NYC bounding box coordinates matching AB_NYC_2019 data and Kaggle standard map
            extent = [-74.255, -73.685, 40.490, 40.915]
            ax.imshow(nyc_img, zorder=0, extent=extent)
            
            # Subsample for clear map visualization
            sample_df = df[df['price'] <= 500].sample(min(12000, len(df)), random_state=42)
            scatter = ax.scatter(sample_df['longitude'], sample_df['latitude'], c=sample_df['price'],
                                 cmap='plasma', alpha=0.4, s=12, zorder=1)
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
            cbar.set_label('Price ($)', fontsize=11, fontweight='bold')
            ax.set_title('NYC Airbnb Listings Map Colored by Price ($)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=11)
            ax.set_ylabel('Latitude', fontsize=11)
            plt.tight_layout()
            fig_path = os.path.join(FIGURES_DIR, 'nyc_map_price.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"Saved: {fig_path}")
        except Exception as e:
            print(f"Error creating map overlay: {e}")

if __name__ == '__main__':
    generate_eda_figures()
    print("All EDA figures generated successfully.")
