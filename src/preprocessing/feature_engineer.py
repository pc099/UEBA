import pandas as pd
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def preprocess_data(df):
    df.columns = df.columns.str.strip()

    # Select only the available features
    available_features = [col for col in Config.CICIDS_FEATURES if col in df.columns]
    if not available_features:
        logger.error("No matching columns found for feature extraction.")
        return pd.DataFrame()

    features = df[available_features].copy()
    features.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    features.fillna(features.mean(), inplace=True)
    features = (features - features.mean()) / features.std()

    if "Label" in df.columns:
        features["Label"] = df["Label"].apply(lambda x: 1 if x != "BENIGN" else 0)

    logger.info(f"Preprocessed features: {features.shape}")
    return features


def preprocess_all_files(all_data):
    processed = {}
    for filename, df in all_data.items():
        processed_df = preprocess_data(df)
        if not processed_df.empty:
            processed[filename] = processed_df
    return processed
