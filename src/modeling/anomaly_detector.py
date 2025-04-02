# anomaly_detector.py

from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest
import pickle
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()

class Sampling(keras.layers.Layer):
    # [unchanged...]
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, 32))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class AnomalyDetector:

    def __init__(self, time_steps=10, knowledge_manager=None, adaptive_threshold=True):
        """
        :param time_steps: The sequence length used by LSTM.
        :param knowledge_manager: An optional KnowledgeManager instance for storing anomalies.
        :param adaptive_threshold: Whether to use an adaptive threshold approach.
        """
        self.time_steps = time_steps
        self.knowledge_manager = knowledge_manager
        self.adaptive_threshold = adaptive_threshold

        try:
            # Load VAE
            self.vae_model = keras.models.load_model(
                f"{Config.PROCESSED_DATA_PATH}_vae.keras",
                custom_objects={'Sampling': Sampling},
                compile=False
            )
            self.vae_model.compile(optimizer='adam', loss='mse')
            
            # Extract encoder
            z_sampling_layer = self.vae_model.get_layer('z_sampling')
            self.encoder = keras.Model(inputs=self.vae_model.input, outputs=z_sampling_layer.output)
            logger.info("VAE model loaded and encoder extracted successfully.")
        except Exception as e:
            logger.error(f"Error loading VAE model: {e}")
            raise

        try:
            # Load LSTM Autoencoder
            self.lstm_model = keras.models.load_model(f"{Config.PROCESSED_DATA_PATH}_lstm.keras")
            logger.info("LSTM model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            raise

        try:
            # Load IsolationForest
            with open(f'{Config.PROCESSED_DATA_PATH}_iso_forest.pkl', 'rb') as f:
                self.iso_forest = pickle.load(f)
            logger.info("IsolationForest model loaded successfully.")
        except Exception as e:
            logger.warning(f"No IsolationForest found or error loading it: {e}")
            self.iso_forest = None

    def detect(self, features, original_data):
        """
        Detect anomalies using an ensemble of VAE+LSTM + IsolationForest (if available).
        """
        try:
            X = features.values.astype(np.float32)

            # 1) VAE Reconstruction Error
            vae_recon = self.vae_model.predict(X)
            if X.shape != vae_recon.shape:
                logger.error(f"Shape mismatch: Input {X.shape}, VAE Reconstructed {vae_recon.shape}")
                return pd.DataFrame()
            vae_errors = np.mean(np.square(X - vae_recon), axis=1)

            # 2) LSTM Reconstruction Error
            latent = self.encoder.predict(X)
            num_samples = latent.shape[0]
            if num_samples % self.time_steps != 0:
                num_samples -= (num_samples % self.time_steps)

            latent_seq = latent[:num_samples].reshape((-1, self.time_steps, 32))
            lstm_recon = self.lstm_model.predict(latent_seq).reshape(-1, 32)
            if lstm_recon.shape[0] != latent_seq.reshape(-1, 32).shape[0]:
                logger.error(f"LSTM shape mismatch: {lstm_recon.shape} vs {latent_seq.shape}")
                return pd.DataFrame()

            lstm_errors = np.mean(np.square(latent_seq.reshape(-1, 32) - lstm_recon), axis=1)

            # Trim vae_errors to match length
            vae_errors = vae_errors[:len(lstm_errors)]

            # 3) Combine VAE+LSTM Errors
            reconstruction_errors = (vae_errors + lstm_errors) / 2.0

            # 4) Isolation Forest Outlier Score (if available)
            if self.iso_forest is not None:
                # iso_forest gives 1 for inlier, -1 for outlier
                iso_labels = self.iso_forest.predict(X[:len(lstm_errors)])  
                # Convert -1/1 to 0 or 1 outlier score
                iso_outlier_score = np.where(iso_labels == -1, 1, 0).astype(float)
                # Weighted combination with reconstruction error
                # E.g. final_score = (reconstruction_errors + iso_outlier_score) / 2
                # Or you could do something more advanced
                combined_scores = reconstruction_errors + iso_outlier_score  
            else:
                logger.info("No IsolationForest loaded; using reconstruction errors only.")
                combined_scores = reconstruction_errors

            # 5) Adaptive Threshold
            # Simple approach: mean+3std, but you can condition on user/time if 'adaptive_threshold' is True
            mean_score = np.mean(combined_scores)
            std_score = np.std(combined_scores)
            static_threshold = mean_score + 3.0 * std_score

            if self.adaptive_threshold:
                # Example: if data has a timestamp, we can vary threshold at night vs. day.
                # Or apply user-specific threshold adjustments. (Simplified example below.)
                # For demonstration, we'll just do a small shift to the threshold.
                # Real logic might require user role, time, or knowledge graph data.
                static_threshold *= 0.95  # e.g., slightly more sensitive threshold

            anomaly_idx = np.where(combined_scores > static_threshold)[0]

            if len(anomaly_idx) == 0:
                return pd.DataFrame()

            # Build anomalies DF
            anomalies = features.iloc[anomaly_idx].copy()
            anomalies["Anomaly_Score"] = combined_scores[anomaly_idx]
            logger.warning(f"Detected {len(anomalies)} anomalies.")

            # Clean column names for original data
            original_data.columns = original_data.columns.str.strip()

            # 6) Optionally store anomalies in knowledge base
            for idx in anomalies.index:
                original_row = original_data.loc[idx]
                anomaly_score = anomalies.loc[idx, "Anomaly_Score"]

                anomaly_info = {
                    "Timestamp": original_row.get("Timestamp", "N/A"),
                    "Source IP": original_row.get("Source IP", "N/A"),
                    "Destination IP": original_row.get("Destination IP", "N/A"),
                    "Anomaly Score": anomaly_score,
                    "Label": original_row.get("Label", "Unknown")
                }
                logger.info(f"Anomaly Detected: {anomaly_info}")

                if self.knowledge_manager is not None:
                    # 6a) Create an anomaly node
                    anomaly_data = {
                        "timestamp": datetime.strptime(anomaly_info["Timestamp"], "%m/%d/%Y %H:%M").isoformat(),
                        "anomaly_score": float(anomaly_score),
                        "label": anomaly_info["Label"],
                        "source_ip": anomaly_info["Source IP"],
                        "destination_ip": anomaly_info["Destination IP"],
                        "index_id": idx,  # a unique ID if needed
                    }
                    if anomaly_data["label"] != "BENIGN":

                        anomaly_node = self.knowledge_manager.create_anomaly_node(anomaly_data)

                        # 6b) Link anomaly to source IP, destination IP
                        if anomaly_info["Source IP"]:
                            self.knowledge_manager.link_anomaly_to_host(anomaly_node, anomaly_info["Source IP"])
                        if anomaly_info["Destination IP"]:
                            self.knowledge_manager.link_anomaly_to_host(anomaly_node, anomaly_info["Destination IP"])

                        # 6c) Optional correlation with existing anomalies in the knowledge graph
                        similar_anomalies = self.knowledge_manager.correlate_anomalies(anomaly_node)
                        if similar_anomalies:
                            logger.info(f"Correlated {len(similar_anomalies)} similar anomalies in the knowledge base.")
                    else:
                        logger.info(f"Benign anomaly detected: {anomaly_info}")
            return anomalies

        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return pd.DataFrame()
