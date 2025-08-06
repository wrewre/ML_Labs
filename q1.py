import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r"C:\Users\year3\Desktop\ml\DCT_mal.csv")
X = df.iloc[:, 1].values.reshape(-1, 1) 
y = df.iloc[:, 2].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error on training data: {mse}")
y_test_pred = reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error on test data: {test_mse}")
