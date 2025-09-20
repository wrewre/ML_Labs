from sklearn.neural_network import MLPClassifier
import numpy as np

def train_mlp(X, y, hidden_layer_sizes=(2,), activation='logistic', 
              solver='adam', learning_rate_init=0.05, max_iter=10000, tol=0.002, random_state=1):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                        activation=activation, solver=solver,
                        learning_rate_init=learning_rate_init, max_iter=max_iter, 
                        tol=tol, random_state=random_state)
    clf.fit(X, y)
    converged = clf.n_iter_ < clf.max_iter
    preds = clf.predict(X)
    return converged, preds

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

converged_and, preds_and = train_mlp(X_and, y_and)
converged_xor, preds_xor = train_mlp(X_xor, y_xor)

print('AND Gate Converged:', converged_and)
print('AND Gate Predictions:', preds_and)
print('XOR Gate Converged:', converged_xor)
print('XOR Gate Predictions:', preds_xor)
