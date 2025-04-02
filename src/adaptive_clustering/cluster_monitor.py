import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import MiniBatchKMeans
from river import drift
from tensorflow.keras import layers, models
import pickle
from src.utils.logger import setup_logger
from src.config import Config

logger = setup_logger()

# ===========================
# Adaptive User Behavior Clustering
# ===========================

class AdaptiveUserBehaviorClustering:
    def __init__(self, n_clusters=5, batch_size=1000):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=42)
        self.drift_detector = drift.ADWIN()
        self.scaler = None

    def preprocess_data(self, df):
        logger.info("Preprocessing data for clustering.")
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = df.drop(columns=['Flow ID', 'Timestamp', 'Label'], errors='ignore')
        df = df.fillna(df.mean())

        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            df = self.scaler.fit_transform(df)
        else:
            df = self.scaler.transform(df)
        return df

    def detect_concept_drift(self, error):
        self.drift_detector.update(error)
        if self.drift_detector.change_detected:
            logger.warning("Concept drift detected. Reinitializing model.")
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=42)

    def train(self, df):
        logger.info("Training adaptive clustering model.")
        data = self.preprocess_data(df)
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            self.model.partial_fit(batch)
            labels = self.model.predict(batch)
            error = np.linalg.norm(batch - self.model.cluster_centers_[labels])
            self.detect_concept_drift(error)

    def save_model(self):
        model_path = f'{Config.PROCESSED_DATA_PATH}_adaptive_clustering.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"Adaptive clustering model saved to {model_path}")

# ===========================
# Network Mapping
# ===========================

class NetworkMapper:
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, df):
        logger.info("Building network graph from entity data.")
        if {'Source IP', 'Destination IP'}.issubset(df.columns):
            for _, row in df.iterrows():
                src_ip = row['Source IP']
                dest_ip = row['Destination IP']
                self.graph.add_edge(src_ip, dest_ip, weight=row.get('Flow Bytes/s', 1))
        logger.info(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

    def analyze_graph(self):
        logger.info("Analyzing network graph with graph metrics.")
        centrality = nx.betweenness_centrality(self.graph)
        return centrality

# ===========================
# User Behavior Fingerprinting with VAE and LSTM Autoencoder
# ===========================

class UserBehaviorFingerprint:
    def __init__(self):
        self.vae = models.load_model(f'{Config.PROCESSED_DATA_PATH}/cicids2017_features_vae.keras')
        self.lstm_autoencoder = models.load_model(f'{Config.PROCESSED_DATA_PATH}/cicids2017_features_lstm.keras')
        logger.info("VAE and LSTM Autoencoder models loaded for anomaly detection.")

    def generate_fingerprint(self, df):
        logger.info("Generating user behavior fingerprints using VAE.")
        data = df.drop(columns=['Source IP', 'Destination IP'], errors='ignore')
        reconstructed_data = self.vae.predict(data)
        reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=1)
        latent_features = self.vae.predict(data)
        latent_features = latent_features.reshape((-1, 10, 32))
        logger.info("Detecting anomalies using LSTM Autoencoder.")
        lstm_reconstructed = self.lstm_autoencoder.predict(latent_features)
        lstm_error = np.mean(np.square(latent_features - lstm_reconstructed), axis=(1, 2))
        return lstm_error

# ===========================
# Incident Response
# ===========================

class IncidentResponse:
    def __init__(self):
        self.incidents = []

    def trigger_response(self, incident):
        logger.warning(f"Incident detected: {incident}")
        self.incidents.append(incident)
        # Placeholder for integrating with SOAR platforms

# ===========================
# Main Execution
# ===========================

if __name__ == '__main__':
    logger.info("Loading data for analysis.")
    df = pd.read_csv(f'{Config.RAW_DATA_PATH}Monday-WorkingHours.pcap_ISCX.csv')

    # Adaptive Clustering
    clustering = AdaptiveUserBehaviorClustering(n_clusters=5)
    clustering.train(df)
    clustering.save_model()

    # Network Mapping
    mapper = NetworkMapper()
    mapper.build_graph(df)
    graph_metrics = mapper.analyze_graph()
    logger.info(f"Graph Analysis Complete. Centrality Metrics: {graph_metrics}")

    # Behavior Fingerprinting using VAE and LSTM
    fingerprint = UserBehaviorFingerprint()
    behavior_scores = fingerprint.generate_fingerprint(df)
    logger.info("User behavior fingerprints generated.")

    # Incident Response Simulation
    incident_response = IncidentResponse()
    if any(score > 0.9 for score in behavior_scores):
        incident_response.trigger_response("High-risk behavior detected!")

    logger.info("Network behavior analysis completed.")
