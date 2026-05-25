import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_evaluate_mlp(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['LABEL']).values
    y = df['LABEL'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    return train_acc, test_acc, class_report, conf_matrix

csv_file_path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv"
train_accuracy, test_accuracy, classification_report_str, confusion_matrix_array = train_evaluate_mlp(csv_file_path)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", confusion_matrix_array)
