#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten, GRU, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
import joblib
from keras.saving import register_keras_serializable
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models

@register_keras_serializable()
class CustomLoss:
    """
    Custom loss function class to penalize underestimation more than overestimation.
    
    Attributes:
        regularization_for_underestimation (float): Factor to penalize underestimation.
    """
    def __init__(self, regularization_for_underestimation):
        self.regularization_for_underestimation = regularization_for_underestimation

    def __call__(self, y_true, y_pred):
        """
        Calculate the custom loss.
        
        Args:
            y_true (tensor): True values.
            y_pred (tensor): Predicted values.
        
        Returns:
            tensor: Computed loss.
        """
        loss = tf.where(tf.greater(y_true, y_pred), 
                        tf.square(y_true - y_pred) * self.regularization_for_underestimation, 
                        tf.square(y_true - y_pred))
        return tf.reduce_mean(loss)

    def get_config(self):
        """
        Get the configuration of the custom loss.
        
        Returns:
            dict: Configuration dictionary.
        """
        return {'regularization_for_underestimation': self.regularization_for_underestimation}

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary.
        
        Returns:
            CustomLoss: Instance of CustomLoss.
        """
        return cls(**config)

class BaseModelWrapper:
    """
    Base model wrapper class for building and training models.
    
    Attributes:
        seq_length (int): Sequence length for the model.
        units (int): Number of units in the model layers.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        regularization_for_underestimation (float): Factor to penalize underestimation.
        historical_data_for_sequence (array): Historical data for sequence creation.
    """
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
        """
        Build the model. This method should be overridden by subclasses.
        
        Raises:
            NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Fit the model to the data.
        
        Args:
            X (array): Input features.
            y (array): Target values.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.
        """
        self.num_features = X.shape[1]
        self.model = self._build_model()  # Some models need the number of features, so the model is built after data is passed

        if isinstance(X, pd.Series):
            X = X.values  # Convert to numpy array

        if isinstance(y, pd.Series):
            y = y.values  # Convert to numpy array

        # Normalize data
        scaled_data = self.scaler.fit_transform(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        X_scaled = scaled_data[:, :-1]
        y_scaled = scaled_data[:, -1]
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.seq_length)

        # Early Stopping Callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True)

        self.history = self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    def predict(self, X):
        """
        Predict using the trained model.
        
        Args:
            X (array): Input features.
        
        Returns:
            array: Predicted values.
        """
        if self.historical_data_for_sequence is not None:
            # Use historical data
            X_with_historical = np.vstack((self.historical_data_for_sequence, X))
            scaled_data = self.scaler.transform(np.concatenate((X_with_historical, np.zeros((X_with_historical.shape[0], 1))), axis=1))
        else:
            scaled_data = self.scaler.transform(np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1))

        X_scaled = scaled_data[:, :-1]
        
        X_seq, _ = self.create_sequences(X_scaled, None, self.seq_length)
        predictions = self.model.predict(X_seq)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], X.shape[1])), predictions), axis=1))[:, -1]
        return predictions[-len(X):]  # Return only the relevant predictions

    def score(self, X, y):
        """
        Score the model using R^2 score.
        
        Args:
            X (array): Input features.
            y (array): True values.
        
        Returns:
            float: R^2 score.
        """
        if isinstance(X, pd.Series):
            X = X.values  # Convert to numpy array

        if isinstance(y, pd.Series):
            y = y.values  # Convert to numpy array

        predictions = self.predict(X)
        
        # Inverse transform true values
        y_true = self.scaler.inverse_transform(np.concatenate((np.zeros((y.shape[0], X.shape[1])), y.reshape(-1, 1)), axis=1))[:, -1]
        
        # Ensure lengths match
        min_len = min(len(y_true), len(predictions))
        y_true = y_true[-min_len:]
        predictions = predictions[-min_len:]
        
        return r2_score(y_true, predictions)

    def create_sequences(self, data, y, seq_length):
        """
        Create sequences from data.
        
        Args:
            data (array): Input data.
            y (array): Target values.
            seq_length (int): Sequence length.
        
        Returns:
            tuple: Sequences of input data and target values.
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            xs.append(x)
            if y is not None:
                ys.append(y[i+seq_length])
        return np.array(xs), np.array(ys) if y is not None else None
    
    def plot_loss(self):
        """
        Plot the training and validation loss.
        
        Raises:
            ValueError: If no training history is found.
        """
        if self.history is None:
            raise ValueError("No training history found. Please train the model first.")
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.title('Model loss per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def feature_importances(self):
        """
        Get feature importances.
        
        Returns:
            DataFrame: Feature importances.
        """
        importances = np.abs(self.model.layers[0].get_weights()[0]).sum(axis=1)
        return pd.DataFrame(importances, columns=['Importance'])

    def get_config(self):
        """
        Get the configuration of the model wrapper.
        
        Returns:
            dict: Configuration dictionary.
        """
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
        """
        Create an instance from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary.
        
        Returns:
            BaseModelWrapper: Instance of BaseModelWrapper.
        """
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
        """
        Save the model and its configuration.
        
        Args:
            filepath (str): Path to save the model and configuration.
        """
        # Save the model
        self.model.save(filepath + '_model.keras')
        # Save the remaining attributes
        joblib.dump(self.get_config(), filepath + '_config.pkl')

    @classmethod
    def load(cls, filepath):
        """
        Load the model and its configuration.
        
        Args:
            filepath (str): Path to load the model and configuration from.
        
        Returns:
            BaseModelWrapper: Instance of the model wrapper with loaded configuration and model.
        """
        # Load the remaining attributes
        config = joblib.load(filepath + '_config.pkl')
        wrapper = cls.from_config(config)
        # Load the model
        wrapper.model = load_model(filepath + '_model.keras', custom_objects={'CustomLoss': CustomLoss(wrapper.regularization_for_underestimation)})
        return wrapper

# Nice-To-Have To-Dos:
# Plot the built models
# Check if Transformer models can be saved without issues

class LSTMModelWrapper(BaseModelWrapper):
    """
    LSTM model wrapper class.
    """
    def _build_model(self):
        """
        Build the LSTM model.
        
        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(units=self.units, return_sequences=True, input_shape=(self.seq_length, self.num_features))))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units=self.units)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model

class GRUModelWrapper(BaseModelWrapper):
    """
    GRU model wrapper class.
    """
    def _build_model(self):
        """
        Build the GRU model.
        
        Returns:
            Sequential: Compiled GRU model.
        """
        model = Sequential()
        model.add(Bidirectional(GRU(units=self.units, return_sequences=True, input_shape=(self.seq_length, self.num_features))))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(GRU(units=self.units)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model

class CNNModelWrapper(BaseModelWrapper):
    """
    CNN model wrapper class.
    """
    def _build_model(self):
        """
        Build the CNN model.
        
        Returns:
            Sequential: Compiled CNN model.
        """
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

class TDNNModelWrapper(BaseModelWrapper):
    """
    TDNN model wrapper class.
    """
    def _build_model(self):
        """
        Build the TDNN model.
        
        Returns:
            Sequential: Compiled TDNN model.
        """
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.seq_length, self.num_features)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        return model

class TransformerModelWrapper(BaseModelWrapper):
    """
    Transformer model wrapper class.
    
    Attributes:
        num_transformer_blocks (int): Number of transformer blocks.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feedforward layer dimension.
    """
    def __init__(self, num_transformer_blocks=1, embed_dim=64, num_heads=8, ff_dim=512, **kwargs): 
        super().__init__(**kwargs)

        self.num_transformer_blocks = num_transformer_blocks
        self.embed_dim = embed_dim  # Embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.ff_dim = ff_dim  # Feedforward layer dimension

    def _build_model(self):
        """
        Build the Transformer model.
        
        Returns:
            Model: Compiled Transformer model.
        """
        class TransformerBlock(layers.Layer):
            """
            Transformer block layer.
            """
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
                super(TransformerBlock, self).__init__()
                self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = models.Sequential(
                    [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
                )
                self.layernorm1 = LayerNormalization(epsilon=1e-6)
                self.layernorm2 = LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(rate)
                self.dropout2 = layers.Dropout(rate)

            def call(self, inputs, training):
                """
                Forward pass for the transformer block.
                
                Args:
                    inputs (tensor): Input tensor.
                    training (bool): Training flag.
                
                Returns:
                    tensor: Output tensor.
                """
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        class TokenAndPositionEmbedding(layers.Layer):
            """
            Token and position embedding layer.
            """
            def __init__(self, seq_length, embed_dim):
                super(TokenAndPositionEmbedding, self).__init__()
                self.token_emb = layers.Dense(embed_dim)
                self.pos_emb = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)

            def call(self, x):
                """
                Forward pass for the token and position embedding.
                
                Args:
                    x (tensor): Input tensor.
                
                Returns:
                    tensor: Embedded tensor.
                """
                maxlen = tf.shape(x)[-2]
                positions = tf.range(start=0, limit=maxlen, delta=1)
                positions = self.pos_emb(positions)
                embedded_tokens = self.token_emb(x)
                return embedded_tokens + positions
        
        # Input layer
        inputs = layers.Input(shape=(self.seq_length, self.num_features))
        
        # Token and Position Embedding
        embedding_layer = TokenAndPositionEmbedding(self.seq_length, self.embed_dim)
        x = embedding_layer(inputs)
        
        # Add transformer blocks
        for _ in range(self.num_transformer_blocks):
            transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
            x = transformer_block(x, training=True)
        
        # GlobalAveragePooling and Dense layers
        x = GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        # Compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=CustomLoss(self.regularization_for_underestimation))
        
        return model
