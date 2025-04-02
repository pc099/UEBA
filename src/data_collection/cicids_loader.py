import pandas as pd
import os
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


class CICIDSLoader:
    def __init__(self):
        self.data_dir = Config.RAW_DATA_PATH

    def load_file(self, filename):
        file_path = os.path.join(self.data_dir, f"{filename}.csv")
        if not os.path.exists(file_path):
            logger.error(f"Data file {filename}.csv not found at {file_path}")
            return None

        # Define dtypes for columns with mixed types
        dtypes = {
            "Flow ID": str,
            "Source IP": str,
            "Destination IP": str,
            "Timestamp": str,
            "Label": str
        }

        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'windows-1252']
        for encoding in encodings:
            try:
                # Load the CSV with specified encoding and handle mixed types
                df = pd.read_csv(file_path, encoding=encoding, dtype=dtypes, low_memory=False)

                # Strip column names to remove any leading/trailing spaces
                df.columns = df.columns.str.strip()
                logger.info(f"Loaded {filename} with {encoding} encoding: {df.shape[0]} flows")

                if encoding != 'utf-8':
                    logger.warning(f"Non-UTF-8 encoding ({encoding}) used for {filename}. Verify data integrity.")

                return df
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to load {filename} with {encoding}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading {filename} with {encoding}: {str(e)}")

        # Fallback: Replace invalid characters using error replacement
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace', dtype=dtypes, low_memory=False)
            df.columns = df.columns.str.strip()
            logger.warning(f"Loaded {filename} with UTF-8 and error replacement: {df.shape[0]} flows")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filename} even with error replacement: {str(e)}")
            return None

    def load_all_files(self):
        all_data = {}
        for filename in Config.DAYS:
            df = self.load_file(filename)
            if df is not None:
                all_data[filename] = df
        return all_data
