# SGD-Regressor-for-Multivariate-Linear-Regression

## Name: Joel Masilamani 
## Reg no: 212224220043

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load data, select features/targets, and split into train/test sets.

2. Scale features and targets using StandardScaler.

3. Train SGDRegressor with MultiOutputRegressor on training data.

4. Predict, inverse scale, and compute MSE.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
*/

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/734cd897-cb9d-4c95-8923-d699a4c281b2)

``` python
X = df.drop(columns=['AveOccup', 'HousingPrice'])  # Independent variables
X.info()
```
![image](https://github.com/user-attachments/assets/6ac33759-9faa-4315-87e1-acd6794895d4)

```python
Y = df[['AveOccup', 'HousingPrice']]  # Dependent variables (Multi-output)
Y.info()
```
![image](https://github.com/user-attachments/assets/b42af6f8-4414-4578-99b9-206ab4938f52)

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)

Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

```
## Output:
![image](https://github.com/user-attachments/assets/98c77681-3616-404f-8c56-a8dca3be6fc4)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
