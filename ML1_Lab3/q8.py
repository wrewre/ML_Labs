import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def load_and_prepare_data_and_knn(filepath):
    data = pd.read_csv(filepath)
    target_classes = [1, 2]
    filtered_data = data[data['LABEL'].isin(target_classes)]
    features = filtered_data.drop(columns='LABEL')
    labels = filtered_data['LABEL']
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,random_state=42) # splitting
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,y_train)
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X_train,y_train)
    prediction1=neigh.predict(X_test)
    prediction2=nn.predict(X_test)
    k = list(range(1, 12))
    accuracies = [] # stores the classifier acccuracy
    for k1 in k:
        knn = KNeighborsClassifier(n_neighbors=k1)
        knn.fit(X_train, y_train) # model fitting
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
    plt.plot(k, accuracies, marker='o')
    plt.title('kNN Classifier k=(1,11)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(k)
    plt.grid(True)
    plt.show()
    return prediction1,prediction2
path=r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
prediction1,prediction2=load_and_prepare_data_and_knn(path)
print("The prediction of the KNN Classifier(k=3) is:",prediction1)
print("The prediction of the KNN Classifier(k=1) is:",prediction2)


