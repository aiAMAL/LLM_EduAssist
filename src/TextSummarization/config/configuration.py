from pathlib import Path
from src.TextSummarization.utils import read_yaml, create_directories
from src.TextSummarization.entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainingConfig,
                                          ModelEvaluationConfig)

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path('config/params.yaml')


class ConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH, param_filepath: Path= PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(param_filepath)

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
    # -------------------------- Data Validation -------------------------
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

    # ====================================================================
    # ------------------------ Data Transformation -----------------------
    # ====================================================================

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            dataset_dir=config.dataset_dir,
            tokenized_dataset_dir=config.tokenized_dataset_dir,
            tokenizer_checkpoint=config.tokenizer_checkpoint
        )

        return data_transformation_config

    # ====================================================================
    # -------------------------- Model Training --------------------------
    # ====================================================================

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.training_arguments
        lora_params = self.params.lora_parameters
        create_directories([Path(config.root_dir)])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            dataset_dir=config.dataset_dir,
            model_dir=config.model_dir,
            tokenizer_dir=config.tokenizer_dir,
            model_checkpoint=config.model_checkpoint,
            num_train_epochs=params.num_train_epochs,
            auto_find_batch_size=params.auto_find_batch_size,
            learning_rate=params.learning_rate,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            load_best_model_at_end=params.load_best_model_at_end,
            report_to=params.report_to,
            save_total_limit=params.save_total_limit,
            lora_r=lora_params.lora_r,
            lora_alpha=lora_params.lora_alpha,
            target_modules=lora_params.target_modules,
            lora_dropout=lora_params.lora_dropout,
            bias=lora_params.bias,
            task_type=lora_params.task_type
        )

        return model_training_config

    # ====================================================================
    # ------------------------ Model Evaluation -----------------------
    # ====================================================================

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            dataset_dir=config.dataset_dir,
            tokenized_test_dataset_dir=config.tokenized_test_dataset_dir,
            model_dir=config.model_dir,
            tokenizer_dir=config.tokenizer_dir,
            metrics_file_name=config.metrics_file_name,
            base_model_checkpoint=config.base_model_checkpoint
        )

        return model_evaluation_config
