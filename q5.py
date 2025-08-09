import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  
# the function perform_kmeans_with_evaluation_scores clusters the data (excluding target) using KMeans and returns labels, centers, and evaluation scores(Silhouette Score,Calinski-Harabasz Score,Davies-Bouldin Index)
def perform_kmeans_with_evaluation_scores():  
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv")  
    X = df.drop(df.columns[2], axis=1).values  # use all attributes except target
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # split into train/test
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)  # fit kmeans
    labels = kmeans.labels_  # cluster labels
    centers = kmeans.cluster_centers_  # cluster centers
    sil_score = silhouette_score(X_train, labels)  # silhouette score
    ch_score = calinski_harabasz_score(X_train, labels)  # calinski-harabasz score
    db_index = davies_bouldin_score(X_train, labels)  # davies-bouldin index
    return labels, centers, sil_score, ch_score, db_index  
labels, centers, sil_score, ch_score, db_index = perform_kmeans_with_evaluation_scores()  
print("Cluster Labels:")  
print(labels)  
print("Cluster Centers:")  
print(centers)  
print("Silhouette Score:", sil_score)  
print("Calinski-Harabasz Score:", ch_score)  
print("Davies-Bouldin Index:", db_index)  
