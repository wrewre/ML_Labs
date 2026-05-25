import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.model_selection import train_test_split  
# the function perform_kmeans clusters the data (excluding target) using KMeans and returns labels and centers
def perform_kmeans():  
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv") 
    X = df.drop(df.columns[2], axis=1).values  # use all attributes except target
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # split into train/test
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)  # fit kmeans
    labels = kmeans.labels_  # cluster labels
    centers = kmeans.cluster_centers_  # cluster centers
    return labels, centers  
labels, centers = perform_kmeans()  
print("Cluster Labels:")  
print(labels)  
print("\nCluster Centers:")  
print(centers)  
