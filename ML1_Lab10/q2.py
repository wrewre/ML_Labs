import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\College\3rd year\5th sem\machine_learning\lab6\DCT_mal.csv").sample(1000, random_state=42)
X = df.drop(columns=['LABEL'])
y = df['LABEL']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Number of components retained:", X_pca.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
