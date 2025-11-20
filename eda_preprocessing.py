"""
Online Retail Dataset - EDA and Preprocessing
This script performs exploratory data analysis and preprocessing on the Online Retail dataset
and creates features for customer segmentation using RFM (Recency, Frequency, Monetary) analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load the Online Retail dataset"""
    print("Loading Online Retail dataset...")
    df = pd.read_excel('data/Online Retail.xlsx')
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nUnique values:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    # Remove rows with missing CustomerID
    print(f"\nRows before removing missing CustomerID: {len(df)}")
    df = df.dropna(subset=['CustomerID'])
    print(f"Rows after removing missing CustomerID: {len(df)}")
    
    # Remove rows with missing Description
    df = df.dropna(subset=['Description'])
    
    # Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove negative quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Create TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    print(f"\nFinal dataset shape: {df.shape}")
    
    return df

def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features for customer segmentation
    """
    print("\n" + "="*50)
    print("CREATING RFM FEATURES")
    print("="*50)
    
    # Get the most recent date in the dataset
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"\nSnapshot date for recency calculation: {snapshot_date}")
    
    # Calculate RFM metrics for each customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    print(f"\nRFM Data shape: {rfm.shape}")
    print("\nRFM Statistics:")
    print(rfm.describe())
    
    # Add some additional features
    rfm['AvgPurchaseValue'] = rfm['Monetary'] / rfm['Frequency']
    
    print("\nRFM DataFrame sample:")
    print(rfm.head(10))
    
    return rfm

def visualize_rfm(rfm):
    """Create visualizations for RFM analysis"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Recency distribution
    axes[0, 0].hist(rfm['Recency'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Recency (days)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Recency')
    
    # Frequency distribution
    axes[0, 1].hist(rfm['Frequency'], bins=50, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Frequency (# of purchases)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Frequency')
    
    # Monetary distribution
    axes[1, 0].hist(rfm['Monetary'], bins=50, color='salmon', edgecolor='black')
    axes[1, 0].set_xlabel('Monetary Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Monetary Value')
    
    # Correlation heatmap
    correlation = rfm.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('RFM Features Correlation')
    
    plt.tight_layout()
    plt.savefig('static/rfm_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'static/rfm_analysis.png'")
    plt.close()

def save_processed_data(rfm):
    """Save the processed RFM data"""
    rfm.to_csv('data/rfm_data.csv')
    print("\n" + "="*50)
    print("Processed RFM data saved to 'data/rfm_data.csv'")
    print("="*50)

def main():
    """Main function to run the entire preprocessing pipeline"""
    print("\n" + "="*50)
    print("ONLINE RETAIL DATASET - EDA & PREPROCESSING")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Clean data
    df = clean_data(df)
    
    # Create RFM features
    rfm = create_rfm_features(df)
    
    # Visualize RFM
    visualize_rfm(rfm)
    
    # Save processed data
    save_processed_data(rfm)
    
    print("\n✓ Preprocessing completed successfully!")
    print(f"✓ Total customers: {len(rfm)}")
    print(f"✓ Features created: {rfm.columns.tolist()}")

if __name__ == "__main__":
    main()
