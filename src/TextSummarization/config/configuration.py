from pathlib import Path
from src.TextSummarization.utils import read_yaml, create_directories
from src.TextSummarization.entity import (DataIngestionConfig,
                                          DataValidationConfig)

CONFIG_FILE_PATH = Path('config/config.yaml')


class ConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

        create_directories([Path(self.config.artifacts_root)])

    # ====================================================================
    # -------------------------- Data Ingestion --------------------------
    # ====================================================================

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_dir=config.dataset_dir,
            source_data=config.source_data,
        )

        return data_ingestion_config

    # ====================================================================
    # -------------------------- Data Validation --------------------------
    # ====================================================================

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([Path(config.root_dir)])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            dataset_dir=config.dataset_dir,
            status_file=config.status_file,
            required_data_folders=config.required_data_folders
        )

        return data_validation_config





