import shap
import lime
import lime.lime_tabular
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\College\3rd year\5th sem\machine_learning\lab6\DCT_mal.csv").sample(500, random_state=42)
X = df.drop(columns=['LABEL'])
y = df['LABEL']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- SHAP ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100])

# ---- LIME ----
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=df.drop(columns=['label']).columns,
    class_names=[str(c) for c in np.unique(y)],
    mode='classification'
)
i = 0
exp = explainer_lime.explain_instance(X_test[i], model.predict_proba, num_features=5)
exp.show_in_notebook()
