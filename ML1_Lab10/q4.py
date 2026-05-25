from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\College\3rd year\5th sem\machine_learning\lab6\DCT_mal.csv").sample(1000, random_state=42)
X = df.drop(columns=['LABEL'])
y = df['LABEL']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500, n_jobs=-1)
sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward', n_jobs=-1)
sfs.fit(X_scaled, y)

selected = sfs.get_support(indices=True)
print("Selected feature indices:", selected)
