"""
Customer Segmentation Model Training
This script trains a K-Means clustering model for customer segmentation
using RFM (Recency, Frequency, Monetary) features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_rfm_data():
    """Load the preprocessed RFM data"""
    print("Loading RFM data...")
    rfm = pd.read_csv('data/rfm_data.csv', index_col='CustomerID')
    print(f"Data shape: {rfm.shape}")
    print(f"\nFeatures: {rfm.columns.tolist()}")
    return rfm

def normalize_features(rfm):
    """Normalize the RFM features using StandardScaler"""
    print("\n" + "="*50)
    print("FEATURE NORMALIZATION")
    print("="*50)
    
    # Select features for clustering
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features]
    
    print(f"\nFeatures for clustering: {features}")
    print("\nOriginal data statistics:")
    print(X.describe())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=rfm.index)
    
    print("\nScaled data statistics:")
    print(X_scaled.describe())
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n✓ Scaler saved to 'models/scaler.pkl'")
    
    return X_scaled, scaler

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print("\n" + "="*50)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*50)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    print("\nTesting different numbers of clusters...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")
    
    # Plot elbow curve and silhouette scores
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal K')
    axes[0].grid(True)
    
    # Silhouette scores
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score for Different K')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('static/optimal_clusters.png', dpi=300, bbox_inches='tight')
    print("\n✓ Cluster analysis plot saved to 'static/optimal_clusters.png'")
    plt.close()
    
    # Find optimal K based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal number of clusters (based on silhouette score): {optimal_k}")
    
    return optimal_k

def train_kmeans_model(X_scaled, n_clusters):
    """Train K-Means clustering model"""
    print("\n" + "="*50)
    print("TRAINING K-MEANS MODEL")
    print("="*50)
    
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    print(f"✓ Model trained successfully!")
    print(f"✓ Inertia: {kmeans.inertia_:.2f}")
    print(f"✓ Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.3f}")
    
    # Save the model
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    print("\n✓ Model saved to 'models/kmeans_model.pkl'")
    
    return kmeans

def analyze_clusters(rfm, X_scaled, kmeans):
    """Analyze the characteristics of each cluster"""
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    # Add cluster labels to the original data
    rfm['Cluster'] = kmeans.labels_
    
    print(f"\nCluster distribution:")
    print(rfm['Cluster'].value_counts().sort_index())
    
    # Calculate cluster statistics
    cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\n Cluster Statistics (Mean values):")
    print(cluster_stats)
    
    # Assign customer segments based on RFM characteristics
    segment_names = []
    for idx in cluster_stats.index:
        recency = cluster_stats.loc[idx, 'Recency']
        frequency = cluster_stats.loc[idx, 'Frequency']
        monetary = cluster_stats.loc[idx, 'Monetary']
        
        if recency < 50 and frequency > 5 and monetary > 2000:
            segment_names.append('Champions')
        elif recency < 100 and frequency > 3 and monetary > 1000:
            segment_names.append('Loyal Customers')
        elif recency < 100 and monetary > 1500:
            segment_names.append('Big Spenders')
        elif recency > 200:
            segment_names.append('At Risk')
        elif frequency == 1:
            segment_names.append('New Customers')
        else:
            segment_names.append('Regular Customers')
    
    cluster_stats['Segment'] = segment_names
    print("\nCustomer Segments:")
    print(cluster_stats)
    
    # Save segment mapping
    cluster_stats.to_csv('data/cluster_segments.csv')
    print("\n✓ Cluster segments saved to 'data/cluster_segments.csv'")
    
    return rfm, cluster_stats

def visualize_clusters(rfm, X_scaled):
    """Create visualizations for the clusters"""
    print("\n" + "="*50)
    print("CREATING CLUSTER VISUALIZATIONS")
    print("="*50)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], 
                         c=rfm['Cluster'], cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('Recency')
    ax1.set_ylabel('Frequency')
    ax1.set_zlabel('Monetary')
    ax1.set_title('3D Cluster Visualization')
    plt.colorbar(scatter, ax=ax1)
    
    # Recency vs Frequency
    ax2 = fig.add_subplot(2, 3, 2)
    sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster', 
                    palette='viridis', s=50, alpha=0.6, ax=ax2)
    ax2.set_title('Recency vs Frequency')
    
    # Recency vs Monetary
    ax3 = fig.add_subplot(2, 3, 3)
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', 
                    palette='viridis', s=50, alpha=0.6, ax=ax3)
    ax3.set_title('Recency vs Monetary')
    
    # Frequency vs Monetary
    ax4 = fig.add_subplot(2, 3, 4)
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', 
                    palette='viridis', s=50, alpha=0.6, ax=ax4)
    ax4.set_title('Frequency vs Monetary')
    
    # Cluster size distribution
    ax5 = fig.add_subplot(2, 3, 5)
    rfm['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax5, color='skyblue')
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Number of Customers')
    ax5.set_title('Cluster Size Distribution')
    ax5.tick_params(axis='x', rotation=0)
    
    # Box plot for clusters
    ax6 = fig.add_subplot(2, 3, 6)
    rfm.boxplot(column='Monetary', by='Cluster', ax=ax6)
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Monetary Value')
    ax6.set_title('Monetary Value by Cluster')
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig('static/cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Cluster visualization saved to 'static/cluster_visualization.png'")
    plt.close()

def save_customer_segments(rfm):
    """Save the customer segments data"""
    rfm.to_csv('data/customer_segments.csv')
    print("\n✓ Customer segments saved to 'data/customer_segments.csv'")

def main():
    """Main function to run the model training pipeline"""
    print("\n" + "="*50)
    print("CUSTOMER SEGMENTATION - MODEL TRAINING")
    print("="*50)
    
    # Load RFM data
    rfm = load_rfm_data()
    
    # Normalize features
    X_scaled, scaler = normalize_features(rfm)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled, max_clusters=8)
    
    # Train K-Means model
    kmeans = train_kmeans_model(X_scaled, n_clusters=optimal_k)
    
    # Analyze clusters
    rfm, cluster_stats = analyze_clusters(rfm, X_scaled, kmeans)
    
    # Visualize clusters
    visualize_clusters(rfm, X_scaled)
    
    # Save customer segments
    save_customer_segments(rfm)
    
    print("\n" + "="*50)
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nModel artifacts saved:")
    print("  - models/kmeans_model.pkl")
    print("  - models/scaler.pkl")
    print("  - data/cluster_segments.csv")
    print("  - data/customer_segments.csv")
    print(f"\nTotal customers segmented: {len(rfm)}")
    print(f"Number of segments: {optimal_k}")

if __name__ == "__main__":
    main()
