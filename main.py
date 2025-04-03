from src.data_collection.cicids_loader import CICIDSLoader
from src.preprocessing.feature_engineer import preprocess_all_files
from src.modeling.baseline_builder import BaselineBuilder
from src.modeling.anomaly_detector import AnomalyDetector
from src.utils.logger import setup_logger
from src.config import Config
from src.Knowledge_pillar.knowledge_manager import KnowledgeManager
import pandas as pd

logger = setup_logger()

def run_ubea():

    logger.info("Starting UBEA-CyberGuard with provided CIC-IDS2017 files...")

    # 1. Load all data
    loader = CICIDSLoader()
    all_data = loader.load_all_files()
    if not all_data:
        logger.error("No data was loaded. Please check your file paths and Config.")
        return

    # 2. Preprocess all data
    processed_data = preprocess_all_files(all_data)
    if not processed_data:
        logger.error("No data was successfully preprocessed. Check feature engineering.")
        return

    # 3. Build baseline with ALL CSV data (concatenate processed data from every day)
    all_processed_no_label = pd.concat(
        [df.drop(columns=["Label"], errors="ignore") for df in processed_data.values()],
        ignore_index=True
    )
    input_dim = all_processed_no_label.shape[1]

    baseline_builder = BaselineBuilder(input_dim)
    baseline_builder.build(all_processed_no_label)  # Pass the concatenated DataFrame

    # 4. Initialize Knowledge Manager & Anomaly Detector
    knowledge_manager_ = KnowledgeManager(
        uri="neo4j+s://37ffc2e9.databases.neo4j.io",
        user="neo4j",
        password="WNBJhza_cA7hoeXb5dXTY3MuzCilqulHtjFpKQMYACg"
    )
    detector = AnomalyDetector(knowledge_manager=knowledge_manager_)

    # 5. Detect anomalies in remaining files (skips Monday by default)
    for filename in Config.DAYS[1:]:
        logger.info(f"Analyzing {filename} for anomalies...")
        if filename not in processed_data:
            logger.warning(f"Skipping {filename} because it was not loaded.")
            continue

        # Drop "Label" column from features (if it exists), pass original data for logging
        anomalies = detector.detect(
            processed_data[filename].drop(columns=["Label"], errors="ignore"),
            all_data[filename]
        )
        if not anomalies.empty:
            logger.info(f"Detected {len(anomalies)} anomalies in {filename}.")

    logger.info("UEBA-CyberGuard pipeline completed.")




if __name__ == "__main__":
    run_ubea()
