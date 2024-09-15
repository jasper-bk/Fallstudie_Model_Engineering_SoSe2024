#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
from datetime import datetime, timedelta
import math
from sklearn_keras_wrapper import BaseModelWrapper, LSTMModelWrapper, GRUModelWrapper, CustomLoss  

python_file_path = os.path.abspath(__file__)
models_path = model_path = os.path.join(os.path.dirname(python_file_path), "models")
model_suffix = "_model.keras"

app = Flask(__name__)

def get_available_models() -> list:
    """
    Get a list of available models in the models directory.

    Returns:
        list: A list of model names.
    """
    models_dir = models_path
    models = set()
    for file_name in os.listdir(models_dir):
        if file_name.endswith(model_suffix):
            model_name = file_name.replace(model_suffix, "")
            models.add(model_name)
    return list(models)


def load_model_wrapper(model_name: str) -> BaseModelWrapper:
    """
    Load the model and scaler using the BaseModelWrapper superclass.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        BaseModelWrapper: The loaded model wrapper.
    """

    model_path = os.path.join(models_path, model_name)

    print("Try to load model from:", model_path)

    if not os.path.exists(model_path + model_suffix):
        raise FileNotFoundError(f"Model '{model_name}' not found.")
    return BaseModelWrapper.load(model_path)

def extract_features(date_str: str) -> dict:
    """
    Extract date features from a given date string.

    Args:
        date_str (str): The date string in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing date features.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "day_of_week": date.weekday(),
        "day_of_month": date.day,
        "day_of_year": date.timetuple().tm_yday,
        "year": date.year
    }

def create_historic_sequences(earliest_date: datetime, seq_length: int) -> pd.DataFrame:
    """
    Create historical sequences needed to avoid skipping the first sequence in prediction.

    Args:
        earliest_date (datetime): The earliest date in the dataset.
        seq_length (int): The sequence length of the model.

    Returns:
        pd.DataFrame: A DataFrame containing historical date features.
    """
    historic_dates = [(earliest_date - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, seq_length + 1)]
    historical_data_for_sequence = [extract_features(date) for date in historic_dates]
    historical_data_for_sequence_df = pd.DataFrame(historical_data_for_sequence)
    historical_data_for_sequence_df["date"] = historic_dates
    historical_data_for_sequence_df['date'] = pd.to_datetime(historical_data_for_sequence_df['date'])
    historical_data_for_sequence_df.set_index('date', inplace=True)
    historical_data_for_sequence_df = historical_data_for_sequence_df.asfreq('d')
    return historical_data_for_sequence_df

def predict_calls(dates: list, model_wrapper: BaseModelWrapper) -> dict:
    """
    Predict calls for given dates using the specified model wrapper.

    Args:
        dates (list): A list of date strings in YYYY-MM-DD format.
        model_wrapper (BaseModelWrapper): The model wrapper to use for prediction.

    Returns:
        dict: A dictionary with dates as keys and predicted calls as values.
    """
    features = [extract_features(date) for date in dates]
    df_features = pd.DataFrame(features)
    df_features["date"] = dates
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_features.set_index('date', inplace=True)
    df_features = df_features.asfreq('d')
    model_wrapper.historical_data_for_sequence = create_historic_sequences(df_features.index.min(), model_wrapper.seq_length)
    prediction = model_wrapper.predict(df_features)
    prediction_dict = {dates[i]: prediction[i] for i in range(len(prediction))}
    return prediction_dict

@app.route('/<model_name>/predict-calls', methods=['POST'])
def route_predict_calls(model_name: str):
    """
    Route to predict calls using a specified model.

    Args:
        model_name (str): The name of the model to use for prediction.

    Returns:
        Response: A JSON response containing the call predictions.
    """
    try:
        data = request.json
        dates = data['dates']
        model_wrapper = load_model_wrapper(model_name)
        prediction = predict_calls(dates, model_wrapper)
        return jsonify({'calls-prediction': prediction})
    except FileNotFoundError as e:
        return jsonify({'error': str(e), "usable models" : get_available_models()}), 404
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/<model_name>/predict-sby-need', methods=['POST'])
def route_predict_sby_need(model_name: str):
    """
    Route to predict standby need using a specified model.

    Args:
        model_name (str): The name of the model to use for prediction.

    Returns:
        Response: A JSON response containing the standby need predictions.
    """
    try:
        data = request.json
        date_list = list(data.keys())
        model_wrapper = load_model_wrapper(model_name)
        prediction = predict_calls(date_list, model_wrapper)
        result = {}
        calls_divisor = 4.82
        for key in prediction:
            if key in data:
                result[key] = max(0, math.floor(prediction[key] / calls_divisor) - data[key]["n_duty"])
        return jsonify({'sby-need-prediction': result})
    except FileNotFoundError as e:
        return jsonify({'error': str(e), "usable models" : get_available_models()}), 404
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.'}), 500


@app.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors by providing suggestions for available endpoints and models.

    Args:
        e (Exception): The exception object for the 404 error.

    Returns:
        Response: A JSON response containing the error message, suggestions for endpoints, and available models.
    """
    endpoints = [
    {'method': 'POST', 'url': '/<model_name>/predict-sby-need'},
    {'method': 'POST', 'url': '/<model_name>/predict-calls'}
    ]

    suggestions = [{'method': "POST", 'url': request.host_url.strip('/') + ep['url']} for ep in endpoints]
    return jsonify({
        'error': 'URL not found',
        'suggestions': suggestions,
        'usable models' : get_available_models()
    }), 404

if __name__ == '__main__':
    app.run(debug=False)
