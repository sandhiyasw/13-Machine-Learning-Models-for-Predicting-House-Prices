import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


import lightgbm as lgb


from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


# read dataset 
data=pd.read_csv(r"C:\Users\sandh\OneDrive\Desktop\python\july1st_housing regrssor_13 algorithims_fronand back end\13_algo_regressor_deploy\USA_Housing.csv")
data.head(5)

#conda update anaconda
#conda install spyder=5.5.1
# preproessing
X=data.drop(['Price','Address'],axis=1)
y=data['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# define models

models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
        'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    
    'KNN': KNeighborsRegressor(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
   
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
   
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
   
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")


#'LGBM': lgb.LGBMRegressor(),
#'XGBoost': xgb.XGBRegressor(),
#'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),

