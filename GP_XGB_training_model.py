
import math
from datetime import timedelta
from forecast_function import forecast_item
from df_functions import normalize, add_date_information, plot_actual_vs_predicted, target_columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
# Necessary imports

from xgboost import XGBRegressor
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

data = pd.read_csv('synthetic_5_years_data.csv', parse_dates=['Date'])
data.columns = data.columns.str.strip()

data.set_index('Date', inplace=True)

data = normalize(data)
data = add_date_information(data)


target_columns = target_columns()  


for col in target_columns:
    data[f'{col}_lag1'] = data[col].shift(1)
    data[f'{col}_ma7'] = data[col].rolling(window=7).mean()
    
tscv = TimeSeriesSplit(n_splits=5)
    
models = {}
predictions = {}
scalers = {}
feature_lists = {}
r2_summary = {}

for col in target_columns:
    
    
    r2_scores = []
    mse_scores = []
    
    y = data[col].values
 # Drop ALL target columns + current col to prevent leakage
    drop_cols = target_columns + [col]
    X = data.drop(columns=drop_cols)
    
    excluded_features = []
    for colname in target_columns:
        excluded_features.extend([colname, f'{colname}_lag1', f'{colname}_ma7'])
    X = data.drop(columns=excluded_features)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        feature_lists[col] = X_train.columns.tolist()

        
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        models[col] = model
        scalers[col] = scaler
        
        
        
                # --- Get feature importances ---
        importances = model.feature_importances_
        feature_names = X.columns
        
        # --- Filter out the target column and any of its lag/rolling features ---
        # Also exclude their lag/rolling versions
        excluded_keywords = []
        for colname in target_columns:
            excluded_keywords.extend([colname, f'{colname}_lag1', f'{colname}_ma7'])
        
        # Get raw importances
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Filter out unwanted features
        filtered = [(name, score) for name, score in zip(feature_names, importances)
                    if name not in excluded_keywords]
        
        # Sort and get top 5
        top_filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
        top_features, top_importances = zip(*top_filtered)
        
        # Plot
        # plt.figure(figsize=(8, 4))
        # sns.barplot(x=top_importances, y=top_features, palette='mako')
        # plt.title(f"Top 5 Feature Importances for {col} (Cleaned)", fontsize=14)
        # plt.xlabel("Importance", fontsize=12)
        # plt.ylabel("Feature", fontsize=12)
        # plt.tight_layout()
        #plt.show()

        y_pred = model.predict(X_test_scaled)
        
        y_test_df = pd.DataFrame(y_test)
        y_pred_df = pd.DataFrame(y_pred)
        
        plot_actual_vs_predicted(y_test_df, y_pred_df, y_test_df.columns.tolist(), col)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        r2_scores.append(r2)
        mse_scores.append(mse)
        
        
        print(print(y_pred[0:3], y_test[0:3]))
        
        print(f"\nFold {fold+1} — {col}")
        print(f"R2 Score: {r2:.4f}, MSE: {mse:.4f}")
    r2_summary[col] = np.mean(r2_scores)
        
    print(f"\n--- {col.upper()} AVERAGE RESULTS ---")
    print(f"Avg R2: {np.mean(r2_scores):.4f}")
    print(f"Avg MSE: {np.mean(mse_scores):.4f}")
# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=list(r2_summary.keys()), y=list(r2_summary.values()))
plt.xticks(rotation=45)
plt.ylabel("Average R² Score")
plt.title("Average R² Score per Food Item (Cross-Validated)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
    

combined = pd.DataFrame()

for item in target_columns:
    try:
        forecast_df = forecast_item(
            feature_lists = feature_lists,
            item=item,
            days_ahead=5,
            data=data,
            models=models,
            scalers=scalers,
            target_columns=target_columns,
            plot=False
        )
        combined[item] = forecast_df[f'{item}_forecast']
    except Exception as e:
        print(f"Error with {item}: {e}")

# Save combined file
combined.to_csv("combined_forecasts.csv")


# import matplotlib.pyplot as plt

# future_dates = pd.to_datetime(future_dates)  # <- Enforce datetime type
# future_df = pd.DataFrame(index=future_dates)

# plt.figure(figsize=(10, 5))
# plt.plot(data.index[-30:], data['oakcakes'].iloc[-30:], label="Historical")
# plt.plot(forecast_df.index, forecast_df['oakcakes_forecast'], label="Forecast", linestyle="--")
# plt.title("30-Day Forecast for Oakcakes")
# plt.xlabel("Date")
# plt.ylabel("Normalized Sales")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# Show the forecast
print(forecast_df.head())

#forecast_item(days_ahead = 5, models = models, scalers = scalers, target_columns = target_columns)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Train and test set are converted to DMatrix objects,
    # # as it is required by learning API.
    # train_dmatrix = xg.DMatrix(data = X_train_scaled, label = y_train)
    # test_dmatrix = xg.DMatrix(data = X_test_scaled, label = y_test)
    
    # # Parameter dictionary specifying base learner
    # param = {'max_depth': 3, 'eta': 1, 'objective': 'reg:squarederror', 'subsample':0.8,
    #         'colsample_bytree':0.8,}
    # param['nthread'] = 5
    # param['eval_metric'] = ['auc', 'ams@0']
    
    # #evallist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # xgb_r = xg.train(params = param, dtrain = train_dmatrix, num_boost_round = 30)
    # pred = xgb_r.predict(test_dmatrix)

    
    # mse = mean_squared_error(y_test, pred)
    # r2 = r2_score(y_test, pred)

    # print(f"\n--- {col.upper()} ---")
    # print(f"R2 Score: {r2:.4f}")
    # print(f"MSE: {mse:.4f}")

    # models[col] = xgb_r
    # predictions[col] = pred
    # actual[col] = y_test
    # scores[col] = {'r2': r2, 'mse': mse}
    
    # # RMSE Computation
    # rmse = np.sqrt(MSE(y_test, pred))
    # print('RMSE :',rmse)
    # print(pred[0:3], y_test[0:3])