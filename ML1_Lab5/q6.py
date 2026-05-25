import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
# the function evaluate_kmeans does  k-means clustering for multiple k values and plot evaluation metrics
def evaluate_optimal_kmeans():
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv")  # load dataset
    X = df.drop(df.columns[2], axis=1).values  # features excluding target
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # split data
    k_values = range(2, 11)  # k values to test
    sil_scores = []  # silhouette scores
    ch_scores = []  # calinski-harabasz scores
    db_scores = []  # davies-bouldin indices
    for i in k_values:
        kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(X_train)  # fit k-means
        labels = kmeans.labels_  # get cluster labels
        sil_scores.append(silhouette_score(X_train, labels))  # silhouette score
        ch_scores.append(calinski_harabasz_score(X_train, labels))  # calinski-harabasz score
        db_scores.append(davies_bouldin_score(X_train, labels))  # davies-bouldin index
    plt.figure(figsize=(12, 4))  # prepare plot
    plt.subplot(1, 3, 1)
    plt.plot(k_values, sil_scores, marker='o')
    plt.title('Silhouette Score vs k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Silhouette Score')
    plt.subplot(1, 3, 2)
    plt.plot(k_values, ch_scores, marker='o', color='green')
    plt.title('Calinski-Harabasz Score vs k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Calinski-Harabasz Score')
    plt.subplot(1, 3, 3)
    plt.plot(k_values, db_scores, marker='o', color='red')
    plt.title('Davies-Bouldin Index vs k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Davies-Bouldin Index')
    plt.tight_layout()
    plt.show()
evaluate_optimal_kmeans()
