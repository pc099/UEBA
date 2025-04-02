from src.data_collection.cicids_loader import CICIDSLoader
from src.preprocessing.feature_engineer import preprocess_all_files
from src.modeling.baseline_builder import BaselineBuilder
from src.modeling.anomaly_detector import AnomalyDetector
from src.utils.logger import setup_logger
from src.config import Config
from src.Knowledge_pillar.knowledge_manager import KnowledgeManager
logger = setup_logger()

def run_ubea():
    logger.info("Starting UBEA-CyberGuard with provided CIC-IDS2017 files...")

    # Load data
    loader = CICIDSLoader()
    all_data = loader.load_all_files()

    # # Preprocess
    processed_data = preprocess_all_files(all_data)

    # # Build baseline with Monday (benign) data
    input_dim = processed_data["Monday-WorkingHours.pcap_ISCX"].drop(columns=["Label"], errors="ignore").shape[1]
    baseline_builder = BaselineBuilder(input_dim)
    baseline_builder.build(processed_data["Monday-WorkingHours.pcap_ISCX"].drop(columns=["Label"], errors="ignore"))

    # Detect anomalies in other files
    knowledge_manager_= KnowledgeManager(
            uri="Host", 
            user="neo4j", 
            password="password"
        )    
    detector = AnomalyDetector(knowledge_manager=knowledge_manager_)

    
    for filename in Config.DAYS[1:]:  # Skip Monday
        logger.info(f"Analyzing {filename}...")
        anomalies = detector.detect(
            processed_data[filename].drop(columns=["Label"], errors="ignore"),
            all_data[filename]
        )

        if not anomalies.empty:
            for idx, row in anomalies.iterrows():
            # Prepare data to store in the knowledge graph
                anomaly_data = {
                    "id": idx,
                    "timestamp": row.get("Timestamp", "N/A"),
                    "anomaly_score": float(row.get("Anomaly_Score", 0.0)),
                    "label": row.get("Label", "Unknown"),
                    "source_ip": row.get("Source IP", "N/A"),
                    "destination_ip": row.get("Destination IP", "N/A"),
                    "filename": filename,
                    "original_data": row.to_dict(),  # Store the original data for reference
                    }
                
                # Create a node in Neo4j for this anomaly
                anomaly_node = knowledge_manager_.create_anomaly_node(anomaly_data)

                # Link anomaly to source IP, destination IP
                src_ip = row.get("Source IP")
                if src_ip:
                    knowledge_manager_.link_anomaly_to_host(anomaly_node, src_ip)

                dest_ip = row.get("Destination IP")
                if dest_ip:
                    knowledge_manager_.link_anomaly_to_host(anomaly_node, dest_ip)

                # (Optional) correlate with similar anomalies
                similar_anomalies = knowledge_manager_.correlate_anomalies(anomaly_node)
                if similar_anomalies:
                    logger.info(f"Found {len(similar_anomalies)} similar anomalies for anomaly {idx}")



if __name__ == "__main__":
    run_ubea()