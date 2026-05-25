import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_pipeline(dataset_path, sample_size=5000):
    data = pd.read_csv(dataset_path)
    if sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=-1)

    print("\nPipeline with Stacking Classifier")
    print("Accuracy on Test Set:", acc)
    print("CV Accuracy:", cv_scores.mean(), "std-", cv_scores.std())

run_pipeline(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv", sample_size=5000)
