#!/usr/bin/env python3

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

from sklearn_keras_wrapper import BaseModelWrapper, LSTMModelWrapper, GRUModelWrapper, CustomLoss  # Importiere die benutzerdefinierte Verlustfunktion und den Wrapper

# Lade das Modell und den Scaler
model_wrapper = BaseModelWrapper.load('./app/models/lstm')  # Lade das gespeicherte Modell und den Wrapper

def extract_features(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "day_of_week": date.weekday(),
        "day_of_month": date.day,
        "day_of_year": date.timetuple().tm_yday,
        "year": date.year
    }

# Needed to avoid skipping the first sequence in prediction
def create_historic_sequences(earlist_date):
    historic_dates = [(earlist_date - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, model_wrapper.seq_length + 1)]
    historical_data_for_sequence = [extract_features(date) for date in historic_dates]
    historical_data_for_sequence_df = pd.DataFrame(historical_data_for_sequence)
    historical_data_for_sequence_df["date"] = historic_dates
    historical_data_for_sequence_df['date'] = pd.to_datetime(historical_data_for_sequence_df['date'])
    historical_data_for_sequence_df.set_index('date', inplace=True)
    historical_data_for_sequence_df = historical_data_for_sequence_df.asfreq('d')

    return historical_data_for_sequence_df


# Funktion zur Vorhersage neuer Daten
def predict_calls(dates):
    # Extrahiere die Features aus den Datumsangaben
    features = [extract_features(date) for date in dates]
    df_features = pd.DataFrame(features)
    df_features["date"] = dates

    df_features['date'] = pd.to_datetime(df_features['date'])
    df_features.set_index('date', inplace=True)
    df_features = df_features.asfreq('d')

    

    # Erstelle historische Sequenz für Modell
    model_wrapper.historical_data_for_sequence = create_historic_sequences(df_features.index.min())
    
    # Führe die Vorhersage mit dem Modell durch
    prediction = model_wrapper.predict(df_features)

    prediction_dict = {dates[i]: prediction[i] for i in range(len(prediction))}
    
    return prediction_dict

app = Flask(__name__)

@app.route('/predict-calls', methods=['POST'])
def route_predict_calls():
    data = request.json
    dates = data['dates']  # Erwarte eine Liste von Datumsangaben im YYYY-MM-DD-Format
    
    # Berechne die Vorhersage
    prediction = predict_calls(dates)
    
    # Nur die Vorhersage für den letzten Zeitpunkt zurückgeben
    return jsonify({'calls-prediction': prediction})

@app.route('/predict-sby-need', methods=['POST'])
def route_predict_sby_need():
    data = request.json
    
    date_list = list(data.keys())
    
    # Berechne die Vorhersage
    prediction = predict_calls(date_list)

    result = {}

    calls_divisor = 4.7

    for key in prediction:
        if key in data:
            result[key] = max(0, math.floor(prediction[key] / calls_divisor) - data[key]["n_duty"])

    
    # Nur die Vorhersage für den letzten Zeitpunkt zurückgeben
    return jsonify({'sby-need-prediction': result})

if __name__ == '__main__':
    app.run(debug=True)