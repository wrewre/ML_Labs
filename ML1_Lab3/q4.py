import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    
    target_classes = [1, 2]
    filtered_data = data[data['LABEL'].isin(target_classes)]
    features = filtered_data.drop(columns='LABEL')
    labels = filtered_data['LABEL']
    print("Available classes:", data['LABEL'].unique()) # uniquely available classes
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,random_state=42) # splitting
    return X_train,X_test,y_train,y_test

path=r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
X_train,X_test,y_train,y_test=load_and_prepare_data(path)
print(f"X_train:",X_train.shape[0])
print(f"X_test:",X_test.shape[0])
print(f"Y_train:",y_train.shape[0])
print(f"Y_test:",y_test.shape[0])
