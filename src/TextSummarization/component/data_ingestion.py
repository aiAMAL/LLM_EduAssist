import sys
from datasets import load_dataset
from src.TextSummarization.entity import DataIngestionConfig
from src.TextSummarization.exception import CustomException
from src.TextSummarization.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """
        Initiates the process of data ingestion by downloading the dataset and saving it to the specified directory.

        Raises:
            CustomException: If an error occurs during data ingestion.
        """
        try:
            logger.info(f"\n\n------- >>  Starting data ingestion for {self.config.source_data}...")

            # Load dataset
            dataset = load_dataset(self.config.source_data)
            logger.info(f'{self.config.source_data} dataset downloaded successfully!')

            # Save the DatasetDict to disk directly from the loaded dataset
            dataset.save_to_disk(self.config.dataset_dir)
            logger.info(f'Dataset saved to {self.config.dataset_dir} !')

        except (IOError, ValueError) as e:
            logger.error(f"Error occurred while downloading dataset: {str(e)}")
            raise CustomException(e, sys)
        except Exception as e:
            logger.error(f"An unexpected error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)
