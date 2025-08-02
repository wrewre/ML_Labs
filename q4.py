import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def plot_knn():
    X_train = np.random.uniform(1, 10, 20)  #random X values
    Y_train = np.random.uniform(1, 10, 20)  #random Y values
    classes = np.random.randint(0, 2, 20)   
    train_points = np.column_stack((X_train, Y_train))  #Making of (x, y) pairs
    test_x = np.arange(0, 10.1, 0.1)  #Test X values from 0 to 10
    test_y = np.arange(0, 10.1, 0.1)  #Test Y values from 0 to 10
    test_grid = np.array(np.meshgrid(test_x, test_y)).T.reshape(-1, 2)  #Creating 10,000 test points

    knn = KNeighborsClassifier(n_neighbors=3)  #Create kNN model with k=3
    knn.fit(train_points, classes)  

    predictions = knn.predict(test_grid)  
    colors=[]
    for i in predictions:
        if i == 1:
            colors.append("blue")
        else:
            colors.append("red") 
    plt.scatter(test_grid[:, 0], test_grid[:, 1], c=colors, s=1) 
    plt.title("kNN (k=3) Classification of Test Data")  
    plt.xlabel("X")  
    plt.ylabel("Y") 
    plt.show()  
plot_knn()