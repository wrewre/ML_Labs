import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
# the function perform_kmeans_elbow_plot computes value for k=2 to 19 and plots the elbow curve
def perform_kmeans_elbow_plot():  
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv")  
    X = df.drop(df.columns[2], axis=1).values  # use all attributes except target
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # split into train/test
    distortions = []  # list to hold ] values for each k
    for k in range(2, 20):  
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_train)  
        distortions.append(kmeans.inertia_)  
    plt.plot(range(2, 20), distortions, marker='o')  
    plt.xlabel('Number of clusters k')  
    plt.ylabel('Distortion (Inertia)')  
    plt.title('Elbow Method for Optimal k')  
    plt.grid(True)  
    plt.show()  
perform_kmeans_elbow_plot()
