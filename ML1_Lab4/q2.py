import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_model(df):
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Price'], inplace=True)
    df = df[::-1].reset_index(drop=True) #make oldest first
    df['Prev_Price'] = df['Price'].shift(1)
    df.dropna(inplace=True)
    X = df[['Prev_Price']] #previous features
    y = df['Price'] #previous features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #Splitting the data 
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mape, r2

df = pd.read_excel(r"Copy of Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
mse, rmse, mape, r2 = evaluate_model(df)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R² Score:", r2)
