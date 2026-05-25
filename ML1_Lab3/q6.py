import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def load_and_prepare_data_and_knn(filepath):
    data = pd.read_csv(filepath)
    print("Available classes:", data['LABEL'].unique()) # unique available classes
    target_classes = [1, 2]
    filtered_data = data[data['LABEL'].isin(target_classes)]
    features = filtered_data.drop(columns='LABEL')
    labels = filtered_data['LABEL']
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,random_state=42) # splitting
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,y_train)
    accuracy=neigh.score(X_test, y_test)
    return accuracy
path=r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
accuracy=load_and_prepare_data_and_knn(path)
print("The Accuracy of the KNN Classifier is:",accuracy)
print("The reported accuracy is 0.9827586206896551 which is between 0 and 1.")