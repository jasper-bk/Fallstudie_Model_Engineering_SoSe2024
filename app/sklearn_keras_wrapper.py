#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten, GRU
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
import joblib
from keras.saving import register_keras_serializable
from keras.callbacks import EarlyStopping

@register_keras_serializable()
class CustomLoss:
    def __init__(self, regularization_for_underestimation):
        self.regularization_for_underestimation = regularization_for_underestimation

    def __call__(self, y_true, y_pred):
        loss = tf.where(tf.greater(y_true, y_pred), 
                        tf.square(y_true - y_pred) * self.regularization_for_underestimation, 
                        tf.square(y_true - y_pred))
        return tf.reduce_mean(loss)

    def get_config(self):
        return {'regularization_for_underestimation': self.regularization_for_underestimation}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BaseModelWrapper:
    def __init__(self, seq_length=30, units=100, dropout_rate=0.2, learning_rate=0.001, regularization_for_underestimation=1, historical_data_for_sequence=None):
        self.seq_length = seq_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.regularization_for_underestimation = regularization_for_underestimation
        self.historical_data_for_sequence = historical_data_for_sequence
        self.model = None
        self.history = None
        
    def _build_model(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        self.num_features = X.shape[1]
        self.model = self._build_model()  # Manche Modelle brauchen die Anzahl der Features und deswegen wird das Modell nach Übergabe der Daten gebaut

        if isinstance(X, pd.Series):
            X = X.values  # Konvertiere zu numpy-Array

        if isinstance(y, pd.Series):
            y = y.values  # Konvertiere zu numpy-Array

        # Daten normalisieren
        scaled_data = self.scaler.fit_transform(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        X_scaled = scaled_data[:, :-1]
        y_scaled = scaled_data[:, -1]
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.seq_length)

        # Early Stopping Callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True)

        self.history = self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    def predict(self, X):
        if self.historical_data_for_sequence is not None:
            # Historische Daten verwenden
            X_with_historical = np.vstack((self.historical_data_for_sequence, X))
            scaled_data = self.scaler.transform(np.concatenate((X_with_historical, np.zeros((X_with_historical.shape[0], 1))), axis=1))
        else:
            scaled_data = self.scaler.transform(np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1))

        X_scaled = scaled_data[:, :-1]
        
        X_seq, _ = self.create_sequences(X_scaled, None, self.seq_length)
        predictions = self.model.predict(X_seq)
        
        # Rückskalieren der Vorhersagen
        predictions = self.scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], X.shape[1])), predictions), axis=1))[:, -1]
        return predictions[-len(X):]  # Nur die relevanten Vorhersagen zurückgeben

    def score(self, X, y):
        if isinstance(X, pd.Series):
            X = X.values  # Konvertiere zu numpy-Array

        if isinstance(y, pd.Series):
            y = y.values  # Konvertiere zu numpy-Array

        predictions = self.predict(X)
        
        # Rückskalieren der wahren Werte
        y_true = self.scaler.inverse_transform(np.concatenate((np.zeros((y.shape[0], X.shape[1])), y.reshape(-1, 1)), axis=1))[:, -1]
        
        # Sicherstellen, dass die Längen übereinstimmen
        min_len = min(len(y_true), len(predictions))
        y_true = y_true[-min_len:]
        predictions = predictions[-min_len:]
        
        return r2_score(y_true, predictions)

    def create_sequences(self, data, y, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            xs.append(x)
            if y is not None:
                ys.append(y[i+seq_length])
        return np.array(xs), np.array(ys) if y is not None else None
    
    def plot_loss(self):
        if self.history is None:
            raise ValueError("No training history found. Please train the model first.")
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Trainings loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.title('Modell loss per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def feature_importances(self):
        importances = np.abs(self.model.layers[0].get_weights()[0]).sum(axis=1)
        return pd.DataFrame(importances, columns=['Importance'])

    def get_config(self):
        return {
            'seq_length': self.seq_length,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'regularization_for_underestimation': self.regularization_for_underestimation,
            'historical_data_for_sequence': self.historical_data_for_sequence,
            'num_features': self.num_features,
            'scaler': self.scaler,
            'history': self.history.history if self.history else None
        }

    @classmethod
    def from_config(cls, config):
        instance = cls(
            seq_length=config['seq_length'],
            units=config['units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            regularization_for_underestimation=config['regularization_for_underestimation'],
            historical_data_for_sequence=config['historical_data_for_sequence']
        )
        instance.num_features = config['num_features']
        instance.scaler = config['scaler']
        if config['history'] is not None:
            instance.history = type('History', (object,), {'history': config['history']})()
        return instance

    def save(self, filepath):
        # Speichern des Modells
        self.model.save(filepath + '_model.keras')
        # Speichern der restlichen Attribute
        joblib.dump(self.get_config(), filepath + '_config.pkl')

    @classmethod
    def load(cls, filepath):
        # Laden der restlichen Attribute
        config = joblib.load(filepath + '_config.pkl')
        wrapper = cls.from_config(config)
        # Laden des Modells
        wrapper.model = load_model(filepath + '_model.keras', custom_objects={'CustomLoss': CustomLoss(wrapper.regularization_for_underestimation)})
        return wrapper

class LSTMModelWrapper(BaseModelWrapper):
    def _build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.units, return_sequences=True, input_shape=(self.seq_length, self.num_features))))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units=self.units)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model

class GRUModelWrapper(BaseModelWrapper):
    def _build_model(self):
        model = Sequential()
        model.add(Bidirectional(GRU(units=self.units, return_sequences=True, input_shape=(self.seq_length, self.num_features))))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(GRU(units=self.units)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model

class CNNModelWrapper(BaseModelWrapper):
    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.seq_length, self.num_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.dropout_rate))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model