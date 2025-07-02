import pandas as pd
import matplotlib.pyplot as plt

def normalize(data):
    #normalizes the food, drink, and temp columns of the data frame
    arg_max_Temp = data['Temp'].iloc[data['Temp'].argmax()]
    arg_max_food = data['total_food_sales'].iloc[data['total_food_sales'].argmax()]
    arg_max_drink = data['total_drink_sales'].iloc[data['total_drink_sales'].argmax()]

    data['Temp'] = data['Temp']/arg_max_Temp
    data['total_food_sales'] = data['total_food_sales']/arg_max_food
    data['total_drink_sales'] = data['total_drink_sales']/arg_max_drink
    
    return data

def add_date_information(data):
    

    data['month'] = pd.DatetimeIndex(data.index).month           # 1 = Jan, 12 = Dec
    data['year'] = pd.DatetimeIndex(data.index).year
    data['dayofweek'] = pd.DatetimeIndex(data.index).dayofweek      # 0 = Monday, 6 = Sunday 

    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    data['quarter'] = pd.DatetimeIndex(data.index).quarter
    data['weekofyear'] = pd.DatetimeIndex(data.index).isocalendar().week.astype(int)

    return data

import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_pred, target_columns, coll):
    """
    Plots actual vs predicted values for all target columns.
    
    Parameters:
        y_test (pd.DataFrame): Actual target values
        y_pred (np.ndarray): Predicted values from model
        target_columns (list): List of target column names (same order as y_pred)
    """
    for i, col in enumerate(target_columns):
        y_actual = pd.Series(y_test[col].values, index=y_test.index)
        y_forecast = pd.Series(y_pred[i], index=y_test.index)

        plt.figure(figsize=(12, 5))
        plt.plot(y_actual, label="Actual", linewidth=2)
        plt.plot(y_forecast, label="Predicted", linestyle="--", linewidth=2)
        plt.title(f"Prediction vs Actual: {coll}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Sales (scaled)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def target_columns():
    return ['coffee_sales', 'special','doorman_b','oakcakes', 'soup', 'salad', 'burrito', 'pig_goat', 'shatsu', 'delamere', 'food_waste', 'staff_ratio', 'TA', 'GPIO', 'coffee_sales.1', 'blue_roll', 'total_food_sales', 'total_drink_sales']