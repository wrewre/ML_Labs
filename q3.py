import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

def run_pipeline_with_lime(dataset_path, sample_size=5000):
    data = pd.read_csv(dataset_path)
    if sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("gb", GradientBoostingClassifier(random_state=42))
    ]

    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=3,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", stacking_clf)
    ])

    pipeline.fit(X_train, y_train)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[str(cls) for cls in set(y)],
        discretize_continuous=True
    )

    i = 0
    exp = explainer.explain_instance(
        X_test[i],
        pipeline.predict_proba,
        num_features=10
    )

    print("Prediction probabilities:", pipeline.predict_proba([X_test[i]])[0])
    exp.show_in_notebook(show_table=True)

run_pipeline_with_lime(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv", sample_size=5000)
