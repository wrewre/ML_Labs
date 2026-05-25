import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_stacking_classifier(dataset_path, sample_size=5000):
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
    meta_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    results = {}
    for name, meta_model in meta_models.items():
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=3,
            n_jobs=-1,
            passthrough=False
        )
        stacking_clf.fit(X_train, y_train)
        y_pred = stacking_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(stacking_clf, X, y, cv=3, n_jobs=-1)
        results[name] = {
            "test_accuracy": acc,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
    return results

results = run_stacking_classifier(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv", sample_size=5000)
for model, metrics in results.items():
    print(f"\n Stacking with ",model,"as Meta Model ")
    print(f"Accuracy on Test Set: ",metrics['test_accuracy'])
    print(f"CV Accuracy: ",metrics['cv_mean_accuracy'],"std-",metrics['cv_std'])
