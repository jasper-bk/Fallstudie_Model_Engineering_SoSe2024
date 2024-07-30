#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import requests


# Funktion zur Vorhersage neuer Daten
def predict_readiness(calls, day_of_week, month):
    model = joblib.load('app/models/readiness_model.pkl')
    scaler = joblib.load('app/models/scaler.pkl')


    # Achtung bei mehr Features im Training-Set müssen diese hier auch vorhanden sein.
    # To-Do: Ggf. reevaluieren, ob man mehr oder andere Werte hinzunimmt und ob man alle Werte immer hat
    new_data = np.array([[calls, day_of_week, month]])

    # Neue Daten standardisieren
    # Not working
    #new_data_scaled = scaler.transform(new_data)
    
    # Vorhersage
    prediction = model.predict(new_data)
    return prediction[0]


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    n_sick = data['n_sick']
    calls = data['calls']
    day_of_week = data['day_of_week']
    month = data['month']
    
    prediction = predict_readiness(n_sick, calls, day_of_week, month)
    return jsonify({'prediction': prediction})


def test():
    # Beispielvorhersage
    new_calls = 50
    new_day_of_week = 2  # Mittwoch
    new_month = 7  # Juli

    prediction = predict_readiness(new_calls, new_day_of_week, new_month)
    print(f'Vorhergesagte Anzahl der Krankmeldungen: {prediction}')


if __name__ == '__main__':
    
    test()
    
    #app.run(debug=True)

    # To-Do: Einfache Web-UI bauen, um die API aufzurufen


