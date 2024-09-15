#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn_keras_wrapper import LSTMModelWrapper
from sklearn.model_selection import train_test_split

# The absolute path to the current Python file
python_file_path = os.path.abspath(__file__)

def prepare_train_data(csv_path: str) -> tuple:
    """
    Prepare the training data from a CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the features (X) and the target (y).
    """
    # Load the dataset
    df = pd.read_csv(csv_path, index_col=0)

    # Convert date string column to datetime column
    df['date'] = pd.to_datetime(df['date'])
    # Make date to index 
    df.set_index('date', inplace=True)
    df = df.asfreq('d')

    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['year'] = df.index.year

    target_column_name = 'calls'  # 'sby_need'
    feature_columns = ["day_of_week", "day_of_month", "day_of_year", "year"]

    # Prepare features and target
    X = df[feature_columns]
    y = df[target_column_name]

    # No test size used for production model. Full dataset is used after evaluation in notebook.
    # Evaluation happens in notebook.

    return X, y

def train_and_save_lstm(X_train: pd.DataFrame, y_train: pd.Series, regularization_for_underestimation: int = 1, epochs: int = 100, save_to: str = None):
    """
    Train an LSTM model and save it.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target values.
        regularization_for_underestimation (int): Regularization parameter to avoid underestimation.
        epochs (int): Number of training epochs.
        save_to (str): Path to save the model.
    """
    lstm_model = LSTMModelWrapper(historical_data_for_sequence=X_train[-30:], regularization_for_underestimation=regularization_for_underestimation)
    lstm_model.fit(X_train, y_train, epochs=epochs)

    if save_to is not None and save_to != "":
        lstm_model.save(save_to)

def main():
    """
    Main function that controls the training process.
    """
    data_file_path = os.path.join(os.path.dirname(os.path.dirname(python_file_path)), "data/raw/sickness_table.csv")

    # Transforms the raw data into the extended data required by the models and does a split features and target
    X_train, y_train = prepare_train_data(data_file_path)

    model_path = "models"

    model_dir = os.path.join(os.path.dirname(python_file_path), model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Counts from 1 to 50 in steps of 5. E.g. 1, 5, 10, ... 50
    for i in range(1, 52, 5):
        if i != 1:
            regularization_for_underestimation = i - 1
        else:
            regularization_for_underestimation = i

        model_save_path = os.path.join(model_dir, f"lstm_ru_{regularization_for_underestimation}")

        print("Train Model with underestimation regularization: ", regularization_for_underestimation)
        train_and_save_lstm(X_train, y_train, regularization_for_underestimation=regularization_for_underestimation, epochs=100, save_to=model_save_path)
        print("Saved model to: ", model_save_path + "*")

if __name__ == '__main__':
    main()
