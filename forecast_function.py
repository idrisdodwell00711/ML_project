import pandas as pd

from datetime import timedelta
#forecast_item(days_ahead = 5, models = models, scalers = scalers, target_columns = target_columns)

def forecast_item(
    feature_lists:list,
    item: str,
    days_ahead: int,
    data: pd.DataFrame,
    models: dict,
    scalers: dict,
    target_columns: list,
    plot=True, 
    
):
    """
    Forecast future values for a given item (e.g. 'oakcakes').

    Parameters:
        item: str — name of the target column to forecast
        days_ahead: int — number of future days to predict
        data: pd.DataFrame — the original full historical DataFrame
        models: dict — trained models by target column
        scalers: dict — fitted scalers by target column
        target_columns: list — all target/support columns to exclude from features
        plot: bool — whether to show the forecast plot

    Returns:
        pd.DataFrame with future dates and predictions
    """

    # Ensure the model and scaler exist
    if item not in models or item not in scalers:
        raise ValueError(f"No trained model or scaler found for '{item}'")

    model = models[item]
    scaler = scalers[item]

    # Step 1: Generate future date index
    last_date = pd.to_datetime(data.index[-1], dayfirst=True)

    #future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
    
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
    future_dates = pd.to_datetime(future_dates)
    future_df = pd.DataFrame(index=future_dates)

    # Step 2: Add time features
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    future_df['quarter'] = future_df.index.quarter
    future_df['weekofyear'] = future_df.index.isocalendar().week.astype(int)

    # Step 3: Add lag/rolling features from latest real data
    latest_lag1 = data[item].iloc[-1]
    latest_ma7 = data[item].iloc[-7:].mean()
    future_df[f'{item}_lag1'] = latest_lag1
    future_df[f'{item}_ma7'] = latest_ma7

    # Step 4: Drop target/support columns and their lag features
    excluded_features = []
    for col in target_columns:
        excluded_features.extend([col, f'{col}_lag1', f'{col}_ma7'])

    feature_cols = feature_lists[item]
    # Enforce correct column order and fill any missing with 0s
    X_future = future_df.reindex(columns=feature_lists[item], fill_value=0)


    # Step 5: Scale and predict
    X_future_scaled = scaler.transform(X_future)
   
    future_df[f'{item}_forecast'] = model.predict(X_future_scaled)
    
    preds = model.predict(X_future_scaled)
    future_df[f'{item}_forecast'] = preds  
    future_df[f'{item}_forecast'] = preds
    
    # Confirm forecast exists
    print( f'{item}_forecast' in future_df.columns)
    

    return future_df[[f'{item}_forecast']]