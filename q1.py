import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def load_and_prepare_data_and_knn(filepath):
    data = pd.read_csv(filepath)
    target_classes = [1, 2]
    filtered_data = data[data['LABEL'].isin(target_classes)]
    features = filtered_data.drop(columns='LABEL')
    labels = filtered_data['LABEL']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42) # splitting
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    pred_y_test = neigh.predict(X_test)
    pred_y_train = neigh.predict(X_train)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred_y_test))  #confusion matrix of tested set
    print(classification_report(y_test, pred_y_test)) #calculation of precision,recall,f1score
    print("Confusion Matrix")
    print(confusion_matrix(y_train, pred_y_train)) #confusion matrix of tested set
    print(classification_report(y_train, pred_y_train)) #calculation of precision,recall,f1score
    return pred_y_test

path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
y_pred_test = load_and_prepare_data_and_knn(path)
print(y_pred_test)
print("The model is regular fit as the predicted train accuracy and the test accuracy are similar.")