from src.TextSummarization.config import ConfigurationManager
from src.TextSummarization.component.data_ingestion import DataIngestion
config = ConfigurationManager()

data_ingestion_config = config.get_data_ingestion_config()
data_ingestion = DataIngestion(data_ingestion_config)
data_ingestion.initiate_data_ingestion()

