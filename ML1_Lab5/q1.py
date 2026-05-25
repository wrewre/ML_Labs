import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# the above function regression loads the  2 attributes i.e. x_train is an input feature and  y_train is the target. The function builds a regression model and predicts based on the training data.   
def regression():
    df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab5\DCT_mal.csv")
    X = df.iloc[:, 1].values.reshape(-1, 1)  # extracts the first attribute and makes it into a 2-d array
    y = df.iloc[:, 2].values  # extracts the target attribute
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #  split into train/test
    reg = LinearRegression().fit(X_train, y_train) 
    y_train_pred = reg.predict(X_train) # prediction
    return y_train_pred
pred=regression()
print(f"Prediction on the training data: {pred}")

