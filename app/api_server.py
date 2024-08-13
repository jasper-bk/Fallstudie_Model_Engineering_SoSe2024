#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import requests


# Funktion zur Vorhersage neuer Daten
def predict_readiness(data):
    model = joblib.load('app/models/readiness_model.pkl')
    #scaler = joblib.load('app/models/scaler.pkl')


    # Achtung bei mehr Features im Training-Set müssen diese hier auch vorhanden sein.
    # To-Do: Ggf. reevaluieren, ob man mehr oder andere Werte hinzunimmt und ob man alle Werte immer hat
    new_data_df = pd.DataFrame([data])

    # Neue Daten standardisieren
    # Not working
    #new_data_scaled = scaler.transform(new_data_df)
    
    # Vorhersage
    prediction = model.predict(new_data_df)
    return prediction[0]


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        prediction = predict_readiness(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def test():
    # Beispielvorhersage
    data = {
        'month' : 7, 
        'n_duty' : 1500, 
        'calls' : 3500, 
        'day_of_week' : 2
    }

    prediction = predict_readiness(data)
    print(f'Vorhergesagte Anzahl der Krankmeldungen: {prediction}')

    app.run(debug=True)


if __name__ == '__main__':
    
    test()
    
    #app.run(debug=True)

    # To-Do: Einfache Web-UI bauen, um die API aufzurufen


