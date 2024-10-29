import sys
from src.TextSummarization.logger import logger
from src.TextSummarization.exception import CustomException
from src.TextSummarization.config import ConfigurationManager
from src.TextSummarization.component.data_ingestion import DataIngestion
from src.TextSummarization.component.data_validation import DataValidation
from src.TextSummarization.component.data_transformation import DataTransformation
from src.TextSummarization.component.model_training_peft import ModelTraining      # Use PEFT
# from src.TextSummarization.component.model_training import ModelTraining


try:
    # Configuration setup
    config = ConfigurationManager()

    # Data ingestion process
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()

    # Data validation process
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(data_validation_config)
    status = data_validation.validate_existing_files_data()
    if not status:
        raise CustomException("Data validation failed.")

    # Data transformation process
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    data_transformation.convert_and_save_dataset()

    # Model training process
    model_trainer_config = config.get_model_training_config()
    model_trainer = ModelTraining(model_trainer_config)
    model_trainer.train()

    # Model evaluation process
    # ...

    logger.info("Pipeline completed successfully.")

except Exception as e:
    logger.error(f"Error in the main pipeline: {str(e)}")
    raise CustomException(e, sys)
