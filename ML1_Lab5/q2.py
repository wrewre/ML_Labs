import pandas as pd  
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np  
# the function regression_test_and_train trains a regression model and compares the prediction on the test data and compare the metric values between train and test set.
 
def regression_test_and_train(): 
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv")  
    X = df.iloc[:, 1].values.reshape(-1, 1)  # extracts the first attribute and makes it into a 2-d array
    y = df.iloc[:, 2].values # extracts the target attribute
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #  split into train/test
    reg = LinearRegression().fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)  # Predict train
    y_test_pred = reg.predict(X_test)  # Predict test
    mse_train = mean_squared_error(y_train, y_train_pred)  
    rmse_train = np.sqrt(mse_train)  # RMSE train
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100  # MAPE train
    r2_train = r2_score(y_train, y_train_pred)  # R2 train
    mse_test = mean_squared_error(y_test, y_test_pred)  
    rmse_test = np.sqrt(mse_test)  
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100  
    r2_test = r2_score(y_test, y_test_pred)  
    return mse_train, rmse_train, mape_train, r2_train, mse_test, rmse_test, mape_test, r2_test 
mse_train, rmse_train, mape_train, r2_train, mse_test, rmse_test, mape_test, r2_test = regression_test_and_train()  
print("Training Metrics:")  
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train, "%")
print("R2:", r2_train)
print("\nTest Metrics:") 
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test, "%")
print("R2:", r2_test)
