import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

df = df = pd.read_csv('datasets\modified_human_age_prediction.csv')

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['age_(years)']
y_test = df_test['age_(years)']
del df_full_train['age_(years)']

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=True)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

d_full_train = xgb.DMatrix(
    X_full_train, label=y_full_train, feature_names=list(dv.get_feature_names_out()))

d_test = xgb.DMatrix(X_test, feature_names=list(dv.get_feature_names_out()))

xgb_params = {
    'eta': 0.1,
    'max_depth': 7,
    'min_child_weight': 10,
    'early_stopping_round': 50,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 0,

    'objective': 'reg:squarederror',
    'nthread': -1,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, d_full_train, num_boost_round=200)

y_pred = model.predict(d_test)

rms = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rms)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

output_file = f"model_xgb_eta={xgb_params['eta']}_score={rms.round(3)}.bin"

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")
