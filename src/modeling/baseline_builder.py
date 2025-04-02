# baseline_builder.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
import pickle
from sklearn.ensemble import IsolationForest  # NEW: For ensemble approach
from src.utils.logger import setup_logger
from src.config import Config

logger = setup_logger()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

class BaselineBuilder:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.vae = self.build_vae()
        self.lstm_autoencoder = self.build_lstm_autoencoder()
        self.iso_forest = None  # Will hold IsolationForest model

    def build_vae(self):
        # [unchanged code...]
        inputs = layers.Input(shape=(self.input_dim,))
        h = layers.Dense(128, activation='relu')(inputs)
        h = layers.Dense(64, activation='relu')(h)

        z_mean = layers.Dense(32, name='z_mean')(h)
        z_log_var = layers.Dense(32, name='z_log_var')(h)

        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch_size = tf.shape(z_mean)[0]
                epsilon = tf.random.normal(shape=(batch_size, 32))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Sampling(name='z_sampling')([z_mean, z_log_var])

        decoder_h = layers.Dense(64, activation='relu')(z)
        decoder_h = layers.Dense(128, activation='relu')(decoder_h)
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(decoder_h)

        vae = keras.Model(inputs, outputs)
        vae.compile(optimizer='adam', loss='mse')
        logger.info("Variational Autoencoder built.")
        return vae

    def build_lstm_autoencoder(self):
        # [unchanged code...]
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(10, 32), return_sequences=True),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.RepeatVector(10),
            layers.LSTM(32, activation='relu', return_sequences=True),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.TimeDistributed(layers.Dense(32))
        ])
        model.compile(optimizer='adam', loss='mse')
        logger.info("LSTM Autoencoder built.")
        return model

    def build(self, benign_features):
        logger.info("Training VAE on benign data.")
        X_train = benign_features.values.astype(np.float32)
        
        # --- VAE Training ---
        self.vae.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.1, verbose=2)

        encoder = keras.Model(self.vae.input, self.vae.get_layer('z_sampling').output)
        latent_features = encoder.predict(X_train)

        # Reshape for LSTM
        time_steps = 10
        feature_size = 32
        num_samples = latent_features.shape[0]
        if num_samples % time_steps != 0:
            num_samples -= (num_samples % time_steps)

        latent_features = latent_features[:num_samples].reshape((-1, time_steps, feature_size))

        logger.info("Training LSTM Autoencoder on latent features.")
        self.lstm_autoencoder.fit(latent_features, latent_features,
                                  epochs=50, batch_size=256, validation_split=0.1,
                                  verbose=2)

        # [Optional] Additional fine-tuning with early_stopping
        self.vae.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.1,
                     callbacks=[early_stopping], verbose=2)
        self.lstm_autoencoder.fit(latent_features, latent_features, epochs=50,
                                  batch_size=256, validation_split=0.1,
                                  callbacks=[early_stopping], verbose=2)

        # NEW: Train IsolationForest on the same benign data
        logger.info("Training IsolationForest on benign data for ensemble.")
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.iso_forest.fit(X_train)  # train on raw features (or you can use latent_features if you prefer)

        # Save models
        self.vae.save(f'{Config.PROCESSED_DATA_PATH}_vae.keras', save_traces=True)
        self.lstm_autoencoder.save(f'{Config.PROCESSED_DATA_PATH}_lstm.keras', save_traces=True)
        
        # Save IsolationForest
        with open(f'{Config.PROCESSED_DATA_PATH}_iso_forest.pkl', 'wb') as f:
            pickle.dump(self.iso_forest, f)

        logger.info("All models (VAE, LSTM, IsolationForest) built and saved.")
