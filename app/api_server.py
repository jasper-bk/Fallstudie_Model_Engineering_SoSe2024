import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf

# Asymmetrische Verlustfunktion definieren
def custom_loss(y_true, y_pred):
    regularization_for_underestimation = 50
    
    loss = tf.where(tf.greater(y_true, y_pred), 
                    tf.square(y_true - y_pred) * regularization_for_underestimation,  # Straffaktor für Unterschätzungen
                    tf.square(y_true - y_pred))
    return tf.reduce_mean(loss)

# Lade das Modell und den Scaler
model = load_model('./app/models/gru_model.keras', custom_objects={'custom_loss': custom_loss})  # Lade das gespeicherte GRU-Modell
scaler = joblib.load('./app/models/scaler.pkl')  # Lade den gespeicherten Scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # Zielwert ist die erste Spalte
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sequenzlänge definieren (muss mit dem Training übereinstimmen)
seq_length = 30

# Funktion zur Vorhersage neuer Daten
def predict_readiness(unix_timestamps):
    # Erstelle die Sequenzen aus den Zeitstempeln
    sequences = np.array([unix_timestamps[i:i+seq_length] for i in range(len(unix_timestamps) - seq_length)])
    
    # Skaliere die Zeitsequenzen
    scaled_sequences = scaler.transform(sequences)
    
    # Führe die Vorhersage mit dem Modell durch
    prediction = model.predict(scaled_sequences)
    
    # Skaliere die Vorhersagen zurück in den ursprünglichen Bereich
    prediction_rescaled = scaler.inverse_transform(
        np.concatenate((prediction, np.zeros((prediction.shape[0], scaled_sequences.shape[2] - 1))), axis=1)
    )[:, 0]  # Nur die erste Spalte (Zielvariable)
    
    return prediction_rescaled

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    unix_timestamps = data['unix_timestamps']  # Erwarte eine Liste von Unix-Zeitstempeln
    
    # Berechne die Vorhersage
    prediction = predict_readiness(unix_timestamps)
    
    # Nur die Vorhersage für den letzten Zeitpunkt zurückgeben
    return jsonify({'prediction': prediction[-1]})

if __name__ == '__main__':

    predict_readiness([
        1672531199, 1672617599, 1672703999, 1672790399, 1672876799,
        1672963199, 1673049599, 1673135999, 1673222399, 1673308799,
        1673395199, 1673481599, 1673567999, 1673654399, 1673740799,
        1673827199, 1673913599, 1673999999, 1674086399, 1674172799,
        1674259199, 1674345599, 1674431999, 1674518399, 1674604799,
        1674691199, 1674777599, 1674863999, 1674950399, 1675036799
    ])

    #app.run(debug=True)
