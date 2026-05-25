from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd

def hypertuning():
    data = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv")
    x = data.iloc[:, [0, 1]].values
    y = data.iloc[:, -1].values
    param_grid = {'n_neighbors': list(range(1, 11))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(x, y)
    rsearch = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=5, cv=5, random_state=42)
    rsearch.fit(x, y)
    print("GridSearch best k:", grid.best_params_['n_neighbors'])
    print("RandomSearch best k:", rsearch.best_params_['n_neighbors'])



hypertuning()