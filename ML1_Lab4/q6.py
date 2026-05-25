import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
def plotting(X, Y, colors):
    plt.scatter(X, Y, c=colors) #scatterplot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Dataset-Based Colored Scatter Plot")
    plt.show()
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv").head(20)  #loading the sheet
X = df['0'] #first column
Y = df['1'] #second column
class_labels = df['LABEL']  
colors=[]
for i in class_labels: # Class division
    if i == 1:
        colors.append("blue")
    else:
        colors.append("red") 
plotting(X, Y, colors)


def knn_visualize_from_excel(k_values=[3, 4, 5]):
    data = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv")
    features = data.iloc[:20, [0, 1]].values
    labels = data.iloc[:20, -1].values
    test_x = np.arange(0, 10.1, 0.1)
    test_y = np.arange(0, 10.1, 0.1)
    test_grid = np.array(np.meshgrid(test_x, test_y)).T.reshape(-1, 2)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(features, labels)
        predictions = knn.predict(test_grid)
        plt.figure(figsize=(7, 6))
        plt.scatter(test_grid[:, 0], test_grid[:, 1], c=predictions, cmap='bwr', s=1, alpha=0.5)
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='bwr', edgecolor='black', s=60, label="Training Data")
        plt.title(f"kNN Classification (k={k})")
        plt.xlabel("Feature 1 (Column 0)")
        plt.ylabel("Feature 2 (Column 1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
knn_visualize_from_excel()

