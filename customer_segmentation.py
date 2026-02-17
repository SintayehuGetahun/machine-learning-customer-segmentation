# =========================================================
# CUSTOMER SEGMENTATION SYSTEM
# Using K-Means Clustering (Unsupervised Learning)
# =========================================================

# =========================
# 1️⃣ IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import sys  # detect environment

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (8,6)

print("Libraries Loaded Successfully ✅")

# =========================
# 2️⃣ LOAD DATASET
# =========================
def load_data(path="Mall_Customers.csv"):
    df = pd.read_csv(path)
    print("\nDataset Loaded Successfully")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    return df

# =========================
# 3️⃣ DATA CLEANING
# =========================
def clean_data(df):
    print("\nChecking Missing Values:")
    print(df.isnull().sum())

    df.drop_duplicates(inplace=True)
    print("\nData after removing duplicates:", df.shape)
    return df

# =========================
# 4️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# =========================
def eda(df):
    print("\nStatistical Summary:")
    print(df.describe())

    sns.countplot(x='Gender', data=df)
    plt.title("Gender Distribution")
    plt.show()

    sns.histplot(df['Age'], kde=True)
    plt.title("Age Distribution")
    plt.show()

    sns.histplot(df['Annual Income (k$)'], kde=True)
    plt.title("Annual Income Distribution")
    plt.show()

    sns.histplot(df['Spending Score (1-100)'], kde=True)
    plt.title("Spending Score Distribution")
    plt.show()

    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()

# =========================
# 5️⃣ FEATURE SELECTION & SCALING
# =========================
def preprocess_data(df):
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nData Scaling Completed ✅")
    return X_scaled, features

# =========================
# 6️⃣ FIND OPTIMAL CLUSTERS
# =========================
def find_optimal_clusters(X_scaled, max_k=10):
    wcss = []
    silhouette_scores = []

    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.plot(range(2, max_k+1), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.show()

# =========================
# 7️⃣ APPLY K-MEANS CLUSTERING
# =========================
def apply_kmeans(X_scaled, df, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print(f"\nK-Means Applied Successfully ✅ ({k} Clusters)")

    # -----------------
    # Assign business-friendly labels
    # -----------------
    cluster_labels = {
        0: "Older, low-income, low spenders",
        1: "Young, rich, big spenders",
        2: "Young, low-income, medium spenders",
        3: "Middle-age, moderate spenders",
        4: "Mixed, targeted promotions"
    }
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

    # Print first 10 customers with cluster & label
    print("\nFirst 10 Customers with Cluster Assignment:")
    print(df[['CustomerID','Age','Annual Income (k$)',
              'Spending Score (1-100)','Cluster','Cluster_Label']].head(10))

    return df, kmeans

# =========================
# 8️⃣ VISUALIZATIONS
# =========================
def plot_2d(df):
    plt.scatter(df['Annual Income (k$)'],
                df['Spending Score (1-100)'],
                c=df['Cluster'], cmap='rainbow', s=50, alpha=0.7)
    plt.title("Customer Segments (2D)")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.show()

def plot_3d(df):
    # Automatically choose renderer based on environment
    if "ipykernel" in sys.modules:
        pio.renderers.default = "notebook_connected"  # Jupyter Notebook
    else:
        pio.renderers.default = "browser"  # Terminal / VS Code

    fig = px.scatter_3d(df,
                        x='Age',
                        y='Annual Income (k$)',
                        z='Spending Score (1-100)',
                        color='Cluster_Label',
                        title='Customer Segments 3D')
    fig.show()

# =========================
# 9️⃣ CLUSTER PROFILING & BUSINESS INSIGHTS
# =========================
def cluster_summary(df, features):
    summary = df.groupby('Cluster_Label')[features].mean()
    print("\nCluster Profile Summary:")
    print(summary)

    # Optional: detailed stats per cluster
    for label in summary.index:
        print(f"\nCluster '{label}' Detailed Stats:")
        print(df[df['Cluster_Label']==label][features].describe())

    # -----------------
    # Business action table
    # -----------------
    business_info = {
        "Cluster_Label": [
            "Young, rich, big spenders",
            "Middle-age, moderate spenders",
            "Young, low-income, medium spenders",
            "Older, low-income, low spenders",
            "Mixed, targeted promotions"
        ],
        "Description": [
            "High income, high spending",
            "Medium income & spend",
            "Low income, medium spending",
            "Low income, low spending",
            "Miscellaneous segment"
        ],
        "Suggested_Action": [
            "Premium products, loyalty rewards",
            "Loyalty programs, newsletters",
            "Discounts, entry-level products",
            "Keep engaged via newsletters",
            "Targeted campaigns"
        ]
    }

    business_table = pd.DataFrame(business_info)
    print("\n📊 Cluster Business Insights:")
    print(business_table)

    return summary

# =========================
# 1️⃣0️⃣ SAVE RESULTS
# =========================
def save_results(df, path="Customer_Segmentation_Labeled.csv"):
    df.to_csv(path, index=False)
    print(f"\nResults Saved as '{path}' ✅")

# =========================
# MAIN FUNCTION
# =========================
def main():
    df = load_data()
    df = clean_data(df)
    eda(df)
    X_scaled, features = preprocess_data(df)
    find_optimal_clusters(X_scaled, max_k=10)

    optimal_k = 5
    df, kmeans = apply_kmeans(X_scaled, df, k=optimal_k)

    # Display 2D and 3D plots
    plot_2d(df)
    plot_3d(df)

    cluster_summary(df, features)
    save_results(df)

# =========================
# RUN MAIN
# =========================
if __name__ == "__main__":
    main()
